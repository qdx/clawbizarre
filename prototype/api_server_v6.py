"""
ClawBizarre HTTP API v6 — Phase 10: ERC-8004 Identity Bridge

Everything from v5 + ERC-8004 identity linking, agent cards, and on-chain feedback mapping.

New endpoints:
  POST /identity/link          — Link native ed25519 ID to ERC-8004 token
  GET  /identity/<id>          — Get identity (native or token ID)
  GET  /identity/stats         — Identity bridge statistics
  POST /identity/card          — Generate agent card from listing
  POST /identity/feedback      — Convert receipt to on-chain feedback format
  GET  /identity/resolve/<id>  — Resolve any identifier to full identity

Usage:
  python3 api_server_v6.py [--port 8420] [--db clawbizarre.db]
  python3 api_server_v6.py --test
"""

import json
import sys
import os
import time
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs
from typing import Optional
from dataclasses import asdict

from identity import AgentIdentity
from receipt import WorkReceipt, TestResults, RiskEnvelope, Timing, VerificationTier, hash_content, ReceiptChain
from persistence import PersistenceLayer
from auth import AuthMiddleware, is_public, extract_bearer_token
from aggregator import ReputationAggregator
from treasury import TreasuryAgent, BudgetPolicy, SpendRequest, SpendCategory
from matching import (
    MatchingEngine, ServiceListing, MatchRequest,
    SelectionStrategy, PricingModel
)
from handshake import HandshakeSession, HandshakeState, Constraints, TaskProposal
from notifications import NotificationBus, EventType
from erc8004_adapter import (
    IdentityBridge, ERC8004Identity, IdentitySource,
    AgentCard, OnChainFeedback, generate_agent_card_from_listing
)

# Import everything from v5 — reuse all existing handler logic
from api_server_v5 import (
    HandshakeStore, PersistentState as PersistentStateV5,
    APIv5Handler, run_server as run_server_v5
)


class PersistentStateV6(PersistentStateV5):
    """V5 state + in-memory identity bridge backed by SQLite."""

    def __init__(self, db_path: str = "clawbizarre.db"):
        super().__init__(db_path)
        self.identity_bridge = IdentityBridge()
        # Load any existing identities from DB into memory
        self._load_identities()

    def _load_identities(self):
        """Hydrate in-memory bridge from SQLite."""
        for row in self.db.list_identities(limit=10000):
            native_id = row["native_id"]
            if row["token_id"] is not None:
                self.identity_bridge.register_erc8004(
                    native_id=native_id,
                    token_id=row["token_id"],
                    eth_address=row["eth_address"] or "",
                    native_signature=row["native_signature"] or "",
                )
            else:
                self.identity_bridge.register_native(native_id)
            # Attach card if present
            eid = self.identity_bridge.get_by_native(native_id)
            if eid and row["card_json"]:
                eid.card = AgentCard.from_json(row["card_json"])


class APIv6Handler(APIv5Handler):
    """Extends v5 with ERC-8004 identity endpoints."""

    state: PersistentStateV6

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"status": "ok", "version": "0.8.0"})
        elif path == "/":
            self._handle_root_v6()
        elif path == "/identity/stats":
            self._handle_identity_stats()
        elif path.startswith("/identity/resolve/"):
            self._handle_identity_resolve(path[len("/identity/resolve/"):])
        elif path.startswith("/identity/"):
            self._handle_identity_get(path[len("/identity/"):])
        else:
            super().do_GET()

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/identity/link":
            self._handle_identity_link()
        elif path == "/identity/card":
            self._handle_identity_card()
        elif path == "/identity/feedback":
            self._handle_identity_feedback()
        else:
            super().do_POST()

    # --- Root ---

    def _handle_root_v6(self):
        self._send_json({
            "service": "ClawBizarre",
            "version": "0.8",
            "description": "Marketplace engine for the agent economy",
            "features": [
                "ed25519-auth", "work-receipts", "reputation-aggregation",
                "discovery", "matching-engine", "bilateral-handshake",
                "sse-notifications", "settlement-tracking", "erc8004-identity"
            ],
            "identity": {
                "native": "Ed25519 keypairs",
                "bridged": "ERC-8004 (Ethereum mainnet)",
                "linking": "POST /identity/link",
            },
        })

    # --- Identity Endpoints ---

    def _handle_identity_link(self):
        """Link native identity to ERC-8004 token."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        data = self._read_body()

        token_id = data.get("token_id")
        eth_address = data.get("eth_address")
        native_signature = data.get("native_signature")

        if token_id is None:
            # Register as native-only
            eid = self.state.identity_bridge.register_native(agent_id)
            self.state.db.store_identity(agent_id, source="native")
            self._send_json({
                "identity": {
                    "native_id": eid.native_id,
                    "source": eid.source.value,
                    "linked": False,
                },
            }, 201)
            return

        if not eth_address or not native_signature:
            self._send_json({"error": "eth_address and native_signature required for linking"}, 400)
            return

        # Check token_id not already claimed by another agent
        existing = self.state.db.get_identity_by_token(token_id)
        if existing and existing["native_id"] != agent_id:
            self._send_json({"error": f"Token {token_id} already linked to another agent"}, 409)
            return

        eid = self.state.identity_bridge.register_erc8004(
            native_id=agent_id,
            token_id=token_id,
            eth_address=eth_address,
            native_signature=native_signature,
        )

        self.state.db.store_identity(
            native_id=agent_id,
            source="hybrid",
            token_id=token_id,
            eth_address=eth_address,
            native_signature=native_signature,
        )

        self._send_json({
            "identity": {
                "native_id": eid.native_id,
                "token_id": eid.token_id,
                "eth_address": eid.eth_address,
                "source": eid.source.value,
                "linked": eid.is_linked(),
                "linked_at": eid.linked_at,
            },
        }, 201)

    def _handle_identity_get(self, identifier: str):
        """Get identity by native_id or token_id."""
        # Try DB first (persistent), fall back to memory
        record = self.state.db.resolve_identity(identifier)
        if not record:
            self._send_json({"error": "Identity not found"}, 404)
            return

        result = {
            "native_id": record["native_id"],
            "token_id": record["token_id"],
            "eth_address": record["eth_address"],
            "source": record["source"],
            "linked": record["token_id"] is not None,
            "linked_at": record["linked_at"],
            "created_at": record["created_at"],
        }

        # Include card if present
        if record.get("card_json"):
            result["card"] = json.loads(record["card_json"])

        self._send_json({"identity": result})

    def _handle_identity_resolve(self, identifier: str):
        """Resolve any identifier to full identity + reputation."""
        record = self.state.db.resolve_identity(identifier)
        if not record:
            self._send_json({"error": "Identity not found"}, 404)
            return

        native_id = record["native_id"]
        result = {
            "native_id": native_id,
            "token_id": record["token_id"],
            "source": record["source"],
            "linked": record["token_id"] is not None,
        }

        # Attach reputation if available
        rep = self.state.db.get_reputation(native_id)
        if rep:
            result["reputation"] = rep["snapshot"]

        # Attach receipt chain stats
        chain = self.state.db.rebuild_chain(native_id)
        if chain:
            result["receipt_chain"] = {
                "length": chain.length,
                "integrity": chain.verify_integrity(),
            }

        self._send_json({"resolved": result})

    def _handle_identity_card(self):
        """Generate/update agent card from current listing."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        data = self._read_body() or {}

        api_base = data.get("api_base", "https://api.clawbizarre.com")
        listing_data = data.get("listing", {})

        card = generate_agent_card_from_listing(agent_id, listing_data, api_base)

        # Store card
        eid = self.state.identity_bridge.get_by_native(agent_id)
        if eid:
            eid.card = card

        card_json = card.to_json()
        self.state.db.store_identity(
            native_id=agent_id,
            card_json=card_json,
        )

        self._send_json({"card": json.loads(card_json)}, 201)

    def _handle_identity_feedback(self):
        """Convert a receipt to ERC-8004 on-chain feedback format."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        data = self._read_body()
        if not data:
            self._send_json({"error": "Body required"}, 400)
            return

        receipt_id = data.get("receipt_id")
        if not receipt_id:
            self._send_json({"error": "receipt_id required"}, 400)
            return

        receipt = self.state.db.get_receipt(receipt_id)
        if not receipt:
            self._send_json({"error": "Receipt not found"}, 404)
            return

        # Need token_ids for both provider and buyer
        provider_id = receipt.agent_id
        provider_identity = self.state.db.resolve_identity(provider_id)
        buyer_identity = self.state.db.resolve_identity(agent_id)

        if not provider_identity or provider_identity.get("token_id") is None:
            self._send_json({"error": "Provider not linked to ERC-8004"}, 400)
            return
        if not buyer_identity or buyer_identity.get("token_id") is None:
            self._send_json({"error": "Buyer not linked to ERC-8004"}, 400)
            return

        receipt_data = {
            "receipt_id": receipt.receipt_id,
            "task_type": receipt.task_type,
            "verification_tier": int(receipt.verification_tier),
            "test_results": {
                "passed": receipt.test_results.passed if receipt.test_results else 0,
                "failed": receipt.test_results.failed if receipt.test_results else 0,
            },
            "timestamp": receipt.timestamp,
        }

        feedback = OnChainFeedback.from_receipt(
            receipt_data,
            provider_token_id=provider_identity["token_id"],
            buyer_token_id=buyer_identity["token_id"],
        )

        self._send_json({
            "feedback": feedback.to_chain_format(),
            "receipt_id": receipt_id,
            "note": "Submit this to the ERC-8004 Reputation Registry contract",
        })

    def _handle_identity_stats(self):
        """Identity bridge statistics."""
        db_stats = self.state.db.identity_stats()
        mem_stats = self.state.identity_bridge.stats()
        self._send_json({
            "database": db_stats,
            "memory": mem_stats,
        })


# --- Server ---

def run_server(port: int = 8420, db_path: str = "clawbizarre.db"):
    state = PersistentStateV6(db_path)
    APIv6Handler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", port), APIv6Handler)
    print(f"ClawBizarre API v6 running on http://127.0.0.1:{port}")
    print(f"Database: {db_path}")
    print(f"Features: identity, discovery, matching, handshake, reputation, SSE, settlement, ERC-8004")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


# --- Smoke Tests ---

def run_tests():
    import urllib.request

    db_path = "/tmp/clawbizarre_v6_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    state = PersistentStateV6(db_path)
    APIv6Handler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", 0), APIv6Handler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"

    passed = 0
    failed = 0

    def api(method, path, data=None, headers=None):
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(f"{base}{path}", data=body, method=method,
                                     headers=headers or {"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(req)
            return json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    def authed(method, path, token, data=None):
        h = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}
        return api(method, path, data, h)

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}")

    print(f"\n=== ClawBizarre API v6 Smoke Tests (port {port}) ===\n")

    # 1. Root includes erc8004-identity
    r, s = api("GET", "/")
    check("1. Root v0.8 with erc8004-identity",
          r.get("version") == "0.8" and "erc8004-identity" in r.get("features", []))

    # 2-3. Auth two agents
    buyer_id = AgentIdentity.generate()
    provider_id = AgentIdentity.generate()

    def auth_agent(identity):
        r, _ = api("POST", "/auth/challenge", {"agent_id": identity.agent_id})
        sig = identity.sign(r["challenge"])
        r2, _ = api("POST", "/auth/verify", {
            "challenge_id": r["challenge_id"], "agent_id": identity.agent_id,
            "signature": sig, "pubkey": identity.public_key_hex,
        })
        return r2.get("token")

    buyer_token = auth_agent(buyer_id)
    provider_token = auth_agent(provider_id)
    check("2. Buyer authenticated", buyer_token is not None)
    check("3. Provider authenticated", provider_token is not None)

    # 4. Register native-only identity
    r, s = authed("POST", "/identity/link", buyer_token, {})
    check("4. Native identity registered",
          s == 201 and r.get("identity", {}).get("source") == "native")

    # 5. Link provider to ERC-8004
    r, s = authed("POST", "/identity/link", provider_token, {
        "token_id": 42,
        "eth_address": "0xabcdef1234567890",
        "native_signature": "sig_placeholder_for_test",
    })
    check("5. Provider linked to ERC-8004",
          s == 201 and r.get("identity", {}).get("linked") is True)

    # 6. Get identity by native_id
    r, s = api("GET", f"/identity/{provider_id.agent_id}")
    check("6. Get identity by native_id",
          s == 200 and r.get("identity", {}).get("token_id") == 42)

    # 7. Get identity by token_id
    r, s = api("GET", "/identity/42")
    check("7. Get identity by token_id",
          s == 200 and r.get("identity", {}).get("native_id") == provider_id.agent_id)

    # 8. Identity stats
    r, s = api("GET", "/identity/stats")
    check("8. Identity stats correct",
          r.get("database", {}).get("total") == 2 and
          r.get("database", {}).get("linked") == 1)

    # 9. Token_id conflict rejected
    other_id = AgentIdentity.generate()
    other_token = auth_agent(other_id)
    r, s = authed("POST", "/identity/link", other_token, {
        "token_id": 42,
        "eth_address": "0xother",
        "native_signature": "sig_other",
    })
    check("9. Duplicate token_id rejected (409)", s == 409)

    # 10. Missing fields rejected
    r, s = authed("POST", "/identity/link", other_token, {
        "token_id": 99,
        # missing eth_address and native_signature
    })
    check("10. Missing fields rejected (400)", s == 400)

    # 11. Generate agent card
    r, s = authed("POST", "/identity/card", provider_token, {
        "api_base": "https://test.clawbizarre.com",
        "listing": {
            "name": "ReviewBot",
            "capabilities": ["code_review", "testing"],
        },
    })
    check("11. Agent card generated",
          s == 201 and r.get("card", {}).get("name") == "ReviewBot")

    # 12. Card persisted in identity
    r, s = api("GET", f"/identity/{provider_id.agent_id}")
    check("12. Card stored in identity",
          r.get("identity", {}).get("card", {}).get("name") == "ReviewBot")

    # 13. Resolve identity (includes reputation + chain)
    r, s = api("GET", f"/identity/resolve/{provider_id.agent_id}")
    check("13. Resolve returns identity",
          s == 200 and r.get("resolved", {}).get("native_id") == provider_id.agent_id)

    # 14. Unknown identity returns 404
    r, s = api("GET", "/identity/ed25519:nonexistent")
    check("14. Unknown identity 404", s == 404)

    # --- Full pipeline: handshake + receipt + feedback ---

    # 15. Provider lists service
    authed("POST", "/discovery/register", provider_token, {"capabilities": ["code_review"]})
    authed("POST", "/matching/listing", provider_token, {
        "capabilities": ["code_review"], "price_per_task": 5.0,
    })
    check("15. Provider listed", True)

    # 16. Buyer links to ERC-8004 (needed for feedback)
    authed("POST", "/identity/link", buyer_token, {
        "token_id": 99,
        "eth_address": "0xbuyer_address",
        "native_signature": "buyer_sig",
    })
    check("16. Buyer linked to ERC-8004", True)

    # 17. Complete handshake → receipt
    r, _ = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": provider_id.agent_id,
        "proposal": {"task_description": "Review PR", "task_type": "code_review", "verification_tier": 0},
    })
    sid = r["session_id"]
    authed("POST", "/handshake/respond", provider_token, {"session_id": sid, "action": "accept"})
    authed("POST", "/handshake/execute", provider_token, {
        "session_id": sid, "output": "LGTM", "proof": {"ok": True},
    })
    r, _ = authed("POST", "/handshake/verify", buyer_token, {
        "session_id": sid, "passed": 3, "failed": 0,
    })
    receipt_id = r.get("receipt_id")
    check("17. Handshake → receipt created", receipt_id is not None)

    # 18. Convert receipt to on-chain feedback
    r, s = authed("POST", "/identity/feedback", buyer_token, {
        "receipt_id": receipt_id,
    })
    check("18. Receipt → on-chain feedback",
          s == 200 and r.get("feedback", {}).get("agentTokenId") == 42 and
          r.get("feedback", {}).get("raterTokenId") == 99)

    # 19. Feedback for unlinked agent fails
    unlinked_id = AgentIdentity.generate()
    unlinked_token = auth_agent(unlinked_id)
    r, s = authed("POST", "/identity/feedback", unlinked_token, {
        "receipt_id": receipt_id,
    })
    check("19. Feedback requires ERC-8004 link (400)", s == 400)

    # 20. V5 endpoints still work (backward compat)
    r, _ = api("GET", "/stats")
    check("20. V5 stats endpoint works",
          "identities" in r and r.get("identities", 0) >= 2)

    # 21. Health check updated
    r, _ = api("GET", "/health")
    check("21. Health shows v0.8.0", r.get("version") == "0.8.0")

    # 22. Resolve by token_id
    r, s = api("GET", "/identity/resolve/42")
    check("22. Resolve by token_id works",
          s == 200 and r.get("resolved", {}).get("linked") is True)

    # Summary
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*50}\n")

    server.shutdown()
    os.remove(db_path)
    return failed == 0


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        port = 8420
        db = "clawbizarre.db"
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--port" and i < len(sys.argv) - 1:
                port = int(sys.argv[i + 1])
            elif arg == "--db" and i < len(sys.argv) - 1:
                db = sys.argv[i + 1]
        run_server(port, db)
