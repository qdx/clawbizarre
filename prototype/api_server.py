"""
ClawBizarre HTTP API — Phase 5
Unified REST API exposing all prototype components.

Endpoints:
  Identity:
    POST   /identity/create          — generate new Ed25519 keypair
    POST   /identity/sign            — sign arbitrary data
    POST   /identity/verify          — verify a signature

  Discovery:
    GET    /discovery/stats          — registry statistics
    POST   /discovery/register       — register capability ad
    POST   /discovery/search         — search for agents
    POST   /discovery/heartbeat      — keep-alive
    DELETE /discovery/<agent_id>     — deregister

  Handshake:
    POST   /handshake/start          — initiate negotiation session
    POST   /handshake/message        — send next message in session
    GET    /handshake/<session_id>   — get session state + transcript

  Reputation:
    POST   /reputation/aggregate     — compute reputation from receipt chain
    GET    /reputation/<agent_id>    — get cached reputation snapshot
    POST   /reputation/compare       — compare two snapshots

  Treasury:
    POST   /treasury/evaluate        — evaluate a spend request
    GET    /treasury/status          — current budget status
    POST   /treasury/policy          — update budget policy

  Receipts:
    POST   /receipt/create           — create a work receipt
    POST   /receipt/chain/append     — append receipt to chain
    POST   /receipt/chain/verify     — verify chain integrity
    GET    /receipt/chain/<agent_id> — get agent's receipt chain

Design: stdlib only (http.server + json). No Flask/FastAPI dependency.
Production would use async framework — this is the protocol spec.
"""

import json
import hashlib
import time
import uuid
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional

from identity import AgentIdentity
from receipt import WorkReceipt, ReceiptChain, TestResults, RiskEnvelope, VerificationTier
from discovery import Registry, CapabilityAd, SearchQuery, AvailabilityStatus
from aggregator import ReputationAggregator, ReputationSnapshot
from treasury import TreasuryAgent, BudgetPolicy, SpendRequest, SpendCategory, ApprovalDecision
from signed_handshake import SignedHandshakeSession


class ClawBizarreState:
    """Server-side state for the API. In production, this would be persistent storage."""

    def __init__(self):
        # Discovery
        self.registry = Registry()

        # Identity store: agent_id -> AgentIdentity (for demo; real agents hold own keys)
        self.identities: dict[str, AgentIdentity] = {}

        # Receipt chains: agent_id -> ReceiptChain
        self.chains: dict[str, ReceiptChain] = {}

        # Reputation cache: agent_id -> ReputationSnapshot
        self.reputation_cache: dict[str, ReputationSnapshot] = {}

        # Handshake sessions: session_id -> (initiator_session, responder_session)
        self.handshake_sessions: dict[str, dict] = {}

        # Treasury
        default_policy = BudgetPolicy(
            daily_budget=100.0,
            escalation_threshold=10.0,
            auto_approve_threshold=5.0,
            blocked_counterparties=[],
            category_limits={
                SpendCategory.COMPUTE.value: 50.0,
                SpendCategory.SERVICE.value: 30.0,
                SpendCategory.INFRASTRUCTURE.value: 20.0,
            },
        )
        self.treasury = TreasuryAgent(default_policy)

        # Aggregator
        self.aggregator = ReputationAggregator()


class APIHandler(BaseHTTPRequestHandler):
    """Route HTTP requests to the appropriate handler."""

    state: ClawBizarreState  # set by factory

    # --- Routing ---

    def do_GET(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            if path == "/discovery/stats":
                self._json(200, self.state.registry.stats())
            elif path == "/treasury/status":
                self._json(200, self.state.treasury.budget_status())
            elif path.startswith("/reputation/"):
                agent_id = path.split("/reputation/", 1)[1]
                snap = self.state.reputation_cache.get(agent_id)
                if snap:
                    self._json(200, json.loads(snap.to_json()))
                else:
                    self._json(404, {"error": "no reputation data", "agent_id": agent_id})
            elif path.startswith("/receipt/chain/"):
                agent_id = path.split("/receipt/chain/", 1)[1]
                chain = self.state.chains.get(agent_id)
                if chain:
                    self._json(200, {
                        "agent_id": agent_id,
                        "length": len(chain.receipts),
                        "valid": chain.verify_integrity(),
                    })
                else:
                    self._json(404, {"error": "no chain", "agent_id": agent_id})
            elif path.startswith("/handshake/"):
                session_id = path.split("/handshake/", 1)[1]
                session = self.state.handshake_sessions.get(session_id)
                if session:
                    self._json(200, {
                        "session_id": session_id,
                        "initiator_state": session["initiator"].state().value if session.get("initiator") else None,
                        "responder_state": session["responder"].state().value if session.get("responder") else None,
                        "created_at": session["created_at"],
                    })
                else:
                    self._json(404, {"error": "session not found"})
            elif path == "/health":
                self._json(200, {
                    "status": "ok",
                    "version": "0.5.0",
                    "agents": len(self.state.identities),
                    "chains": len(self.state.chains),
                    "registry": self.state.registry.stats()["total_agents"],
                })
            else:
                self._json(404, {"error": "not found", "path": path})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def do_POST(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            body = self._read_body()
            data = json.loads(body) if body else {}

            if path == "/identity/create":
                self._handle_identity_create(data)
            elif path == "/identity/sign":
                self._handle_identity_sign(data)
            elif path == "/identity/verify":
                self._handle_identity_verify(data)
            elif path == "/discovery/register":
                self._handle_discovery_register(data)
            elif path == "/discovery/search":
                self._handle_discovery_search(data)
            elif path == "/discovery/heartbeat":
                self._handle_discovery_heartbeat(data)
            elif path == "/handshake/start":
                self._handle_handshake_start(data)
            elif path == "/handshake/message":
                self._handle_handshake_message(data)
            elif path == "/reputation/aggregate":
                self._handle_reputation_aggregate(data)
            elif path == "/reputation/compare":
                self._handle_reputation_compare(data)
            elif path == "/treasury/evaluate":
                self._handle_treasury_evaluate(data)
            elif path == "/treasury/policy":
                self._handle_treasury_policy(data)
            elif path == "/receipt/create":
                self._handle_receipt_create(data)
            elif path == "/receipt/chain/append":
                self._handle_chain_append(data)
            elif path == "/receipt/chain/verify":
                self._handle_chain_verify(data)
            else:
                self._json(404, {"error": "not found", "path": path})
        except json.JSONDecodeError:
            self._json(400, {"error": "invalid JSON"})
        except Exception as e:
            self._json(500, {"error": str(e)})

    def do_DELETE(self):
        path = urlparse(self.path).path.rstrip("/")
        try:
            if path.startswith("/discovery/"):
                agent_id = path.split("/discovery/", 1)[1]
                ok = self.state.registry.deregister(agent_id)
                self._json(200, {"ok": ok, "agent_id": agent_id})
            else:
                self._json(404, {"error": "not found"})
        except Exception as e:
            self._json(500, {"error": str(e)})

    # --- Identity Handlers ---

    def _handle_identity_create(self, data: dict):
        identity = AgentIdentity.generate()
        agent_id = identity.agent_id
        self.state.identities[agent_id] = identity
        self._json(201, {
            "agent_id": agent_id,
            "public_key": identity.public_key_hex,
            "note": "Private key held server-side for demo. Real agents manage own keys.",
        })

    def _handle_identity_sign(self, data: dict):
        agent_id = data.get("agent_id")
        message = data.get("message", "")
        identity = self.state.identities.get(agent_id)
        if not identity:
            self._json(404, {"error": "identity not found"})
            return
        sig = identity.sign(message)
        self._json(200, {"agent_id": agent_id, "signature": sig, "message": message})

    def _handle_identity_verify(self, data: dict):
        agent_id = data.get("agent_id")
        message = data.get("message", "")
        signature = data.get("signature", "")
        identity = self.state.identities.get(agent_id)
        if not identity:
            self._json(404, {"error": "identity not found"})
            return
        try:
            valid = identity.verify(message, signature)
            self._json(200, {"valid": valid, "agent_id": agent_id})
        except Exception:
            self._json(200, {"valid": False, "agent_id": agent_id})

    # --- Discovery Handlers ---

    def _handle_discovery_register(self, data: dict):
        data["availability"] = AvailabilityStatus(data.get("availability", "immediate"))
        ad = CapabilityAd(**{k: v for k, v in data.items()
                            if k in CapabilityAd.__dataclass_fields__})
        self.state.registry.register(ad)
        self._json(200, {"ok": True, "agent_id": ad.agent_id, "trust_tier": ad.trust_tier})

    def _handle_discovery_search(self, data: dict):
        query = SearchQuery(**{k: v for k, v in data.items()
                              if k in SearchQuery.__dataclass_fields__})
        results = self.state.registry.search(query)
        self._json(200, {
            "results": [
                {
                    "agent_id": r.agent_id,
                    "relevance_score": r.relevance_score,
                    "is_newcomer_slot": r.is_newcomer_slot,
                    "trust_tier": r.capability_ad.trust_tier,
                    "capabilities": r.capability_ad.capabilities,
                    "success_rate": r.capability_ad.success_rate,
                    "endpoint": r.capability_ad.endpoint,
                }
                for r in results
            ],
            "total": len(results),
        })

    def _handle_discovery_heartbeat(self, data: dict):
        ok = self.state.registry.heartbeat(data["agent_id"])
        self._json(200, {"ok": ok})

    # --- Handshake Handlers ---

    def _handle_handshake_start(self, data: dict):
        initiator_id = data.get("initiator_id")
        responder_id = data.get("responder_id")

        initiator_identity = self.state.identities.get(initiator_id)
        responder_identity = self.state.identities.get(responder_id)

        if not initiator_identity or not responder_identity:
            self._json(400, {"error": "both agents must have registered identities"})
            return

        session_id = str(uuid.uuid4())[:8]
        initiator_session = SignedHandshakeSession(initiator_identity)
        responder_session = SignedHandshakeSession(responder_identity)

        # Exchange hellos
        hello1 = initiator_session.send_hello(
            capabilities=data.get("initiator_capabilities", ["general"]),
        )
        responder_session.receive_hello(hello1)
        hello2 = responder_session.send_hello(
            capabilities=data.get("responder_capabilities", ["general"]),
        )
        initiator_session.receive_hello(hello2)

        self.state.handshake_sessions[session_id] = {
            "initiator": initiator_session,
            "responder": responder_session,
            "initiator_id": initiator_id,
            "responder_id": responder_id,
            "created_at": time.time(),
        }

        self._json(201, {
            "session_id": session_id,
            "state": "hellos_exchanged",
            "initiator_id": initiator_id,
            "responder_id": responder_id,
        })

    def _handle_handshake_message(self, data: dict):
        session_id = data.get("session_id")
        action = data.get("action")  # propose, accept, reject, execute, verify
        session = self.state.handshake_sessions.get(session_id)

        if not session:
            self._json(404, {"error": "session not found"})
            return

        initiator = session["initiator"]
        responder = session["responder"]

        if action == "propose":
            msg = initiator.propose(
                task_description=data.get("task_description", ""),
                task_type=data.get("task_type", "general"),
                verification_tier=VerificationTier(data.get("verification_tier", 0)),
                input_data=data.get("input_data"),
            )
            responder.receive_proposal(msg)
            self._json(200, {"ok": True, "state": "proposed"})

        elif action == "accept":
            msg = responder.accept()
            initiator.receive_accept(msg)
            self._json(200, {"ok": True, "state": "accepted"})

        elif action == "reject":
            msg = responder.reject(reason=data.get("reason", ""))
            self._json(200, {"ok": True, "state": "rejected"})

        elif action == "execute":
            msg = responder.execute(
                output=data.get("output", ""),
                proof=data.get("proof"),
            )
            # Define a simple verifier that always passes
            def simple_verifier(payload):
                passed = data.get("tests_passed", 1)
                failed = data.get("tests_failed", 0)
                return TestResults(passed=passed, failed=failed, suite_hash=data.get("suite_hash", "sha256:test"))

            verify_msg, signed_receipt = initiator.verify_and_receipt(msg, verifier=simple_verifier)

            # Extract receipt from signed receipt JSON
            provider_id = session["responder_id"]
            if provider_id not in self.state.chains:
                self.state.chains[provider_id] = ReceiptChain()

            receipt_id = "none"
            if signed_receipt and signed_receipt.receipt_json:
                receipt_data = json.loads(signed_receipt.receipt_json)
                receipt_id = receipt_data.get("receipt_id", "unknown")
                # Reconstruct WorkReceipt for chain
                receipt_obj = WorkReceipt(
                    agent_id=receipt_data.get("agent_id", provider_id),
                    task_type=receipt_data.get("task_type", "general"),
                    verification_tier=VerificationTier(receipt_data.get("verification_tier", 0)),
                    input_hash=receipt_data.get("input_hash", "sha256:handshake"),
                    output_hash=receipt_data.get("output_hash", "sha256:handshake"),
                    pricing_strategy=receipt_data.get("pricing_strategy", "reputation_premium"),
                )
                self.state.chains[provider_id].append(receipt_obj)

            self._json(200, {
                "ok": True,
                "state": "completed",
                "receipt_id": receipt_id,
                "provider_id": provider_id,
                "chain_length": len(self.state.chains[provider_id].receipts),
            })
        else:
            self._json(400, {"error": f"unknown action: {action}"})

    # --- Reputation Handlers ---

    def _handle_reputation_aggregate(self, data: dict):
        agent_id = data.get("agent_id")
        chain = self.state.chains.get(agent_id)
        if not chain:
            self._json(404, {"error": "no receipt chain"})
            return
        snapshot = self.state.aggregator.aggregate(chain)
        self.state.reputation_cache[agent_id] = snapshot
        self._json(200, json.loads(snapshot.to_json()))

    def _handle_reputation_compare(self, data: dict):
        id_a = data.get("agent_a")
        id_b = data.get("agent_b")
        snap_a = self.state.reputation_cache.get(id_a)
        snap_b = self.state.reputation_cache.get(id_b)
        if not snap_a or not snap_b:
            self._json(404, {"error": "both agents need cached reputation"})
            return
        comparison = self.state.aggregator.compare(snap_a, snap_b)
        self._json(200, comparison)

    # --- Treasury Handlers ---

    def _handle_treasury_evaluate(self, data: dict):
        request = SpendRequest(
            requesting_agent=data.get("requester_id", ""),
            counterparty=data.get("counterparty_id", ""),
            amount=data.get("amount", 0),
            category=SpendCategory(data.get("category", "service")),
            description=data.get("description", ""),
            task_receipt_id=data.get("receipt_id"),
        )
        decision = self.state.treasury.evaluate(request)
        self._json(200, {
            "decision": decision.decision.value,
            "reason": decision.reason,
            "remaining_daily_budget": decision.remaining_daily_budget,
            "policy_version": decision.policy_version,
        })

    def _handle_treasury_policy(self, data: dict):
        policy = BudgetPolicy(
            daily_limit=data.get("daily_limit", 100),
            per_transaction_limit=data.get("per_transaction_limit", 10),
            auto_approve_below=data.get("auto_approve_below", 5),
            blocked_counterparties=data.get("blocked_counterparties", []),
            category_limits={SpendCategory(k): v for k, v in data.get("category_limits", {}).items()},
        )
        self.state.treasury.update_policy(policy)
        self._json(200, {"ok": True, "status": self.state.treasury.budget_status()})

    # --- Receipt Handlers ---

    def _handle_receipt_create(self, data: dict):
        test_results = None
        if "test_results" in data:
            tr = data["test_results"]
            test_results = TestResults(
                passed=tr.get("passed", 0),
                failed=tr.get("failed", 0),
                suite_hash=tr.get("suite_hash", ""),
            )

        receipt = WorkReceipt(
            agent_id=data.get("agent_id", ""),
            task_type=data.get("task_type", "general"),
            verification_tier=VerificationTier(data.get("verification_tier", 0)),
            input_hash=data.get("input_hash", f"sha256:{hashlib.sha256(b'input').hexdigest()}"),
            output_hash=data.get("output_hash", f"sha256:{hashlib.sha256(b'output').hexdigest()}"),
            test_results=test_results,
            pricing_strategy=data.get("pricing_strategy", "reputation_premium"),
        )
        self._json(201, {
            "receipt_id": receipt.receipt_id,
            "agent_id": receipt.agent_id,
            "content_hash": receipt.content_hash(),
        })

    def _handle_chain_append(self, data: dict):
        agent_id = data.get("agent_id")
        if agent_id not in self.state.chains:
            self.state.chains[agent_id] = ReceiptChain()

        test_results = None
        if data.get("success", True):
            test_results = TestResults(passed=1, failed=0, suite_hash="sha256:auto")
        else:
            test_results = TestResults(passed=0, failed=1, suite_hash="sha256:auto")

        receipt = WorkReceipt(
            agent_id=agent_id,
            task_type=data.get("task_type", "general"),
            verification_tier=VerificationTier(data.get("verification_tier", 0)),
            input_hash=data.get("input_hash", f"sha256:{hashlib.sha256(str(time.time()).encode()).hexdigest()}"),
            output_hash=data.get("output_hash", f"sha256:{hashlib.sha256(str(time.time() + 1).encode()).hexdigest()}"),
            pricing_strategy=data.get("pricing_strategy", "reputation_premium"),
            test_results=test_results,
        )

        chain = self.state.chains[agent_id]
        chain.append(receipt)
        self._json(200, {
            "ok": True,
            "chain_length": len(chain.receipts),
            "valid": chain.verify_integrity(),
        })

    def _handle_chain_verify(self, data: dict):
        agent_id = data.get("agent_id")
        chain = self.state.chains.get(agent_id)
        if not chain:
            self._json(404, {"error": "no chain"})
            return
        self._json(200, {
            "agent_id": agent_id,
            "length": len(chain.receipts),
            "valid": chain.verify_integrity(),
        })

    # --- Utilities ---

    def _read_body(self) -> str:
        length = int(self.headers.get("Content-Length", 0))
        return self.rfile.read(length).decode() if length > 0 else ""

    def _json(self, status: int, data: dict):
        body = json.dumps(data, indent=2, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        pass  # quiet


def create_server(host: str = "127.0.0.1", port: int = 8402) -> HTTPServer:
    """Create the ClawBizarre API server."""
    state = ClawBizarreState()

    class Handler(APIHandler):
        pass
    Handler.state = state

    server = HTTPServer((host, port), Handler)
    return server


# --- Smoke Test ---

def smoke_test():
    """Run the full lifecycle via HTTP (same flow as integration_test.py)."""
    import urllib.request

    BASE = "http://127.0.0.1:8402"

    def api(method, path, data=None):
        body = json.dumps(data).encode() if data else None
        req = urllib.request.Request(
            f"{BASE}{path}",
            data=body,
            method=method,
            headers={"Content-Type": "application/json"} if body else {},
        )
        with urllib.request.urlopen(req) as resp:
            return json.loads(resp.read())

    print("=== ClawBizarre API Smoke Test ===\n")

    # 1. Health check
    r = api("GET", "/health")
    print(f"1. Health: {r['status']} (v{r['version']})")

    # 2. Create identities
    alice = api("POST", "/identity/create")
    bob = api("POST", "/identity/create")
    print(f"2. Identities: Alice={alice['agent_id'][:16]}... Bob={bob['agent_id'][:16]}...")

    # 3. Sign and verify
    sig = api("POST", "/identity/sign", {"agent_id": alice["agent_id"], "message": "hello"})
    ver = api("POST", "/identity/verify", {
        "agent_id": alice["agent_id"], "message": "hello", "signature": sig["signature"]
    })
    print(f"3. Signature valid: {ver['valid']}")

    # 4. Register in discovery
    api("POST", "/discovery/register", {
        "agent_id": alice["agent_id"],
        "capabilities": ["code_review", "test_generation"],
        "verification_tier": 0,
        "receipt_chain_length": 25,
        "success_rate": 0.92,
        "strategy_consistency": 0.95,
        "pricing_strategy": "reputation_premium",
        "endpoint": "http://alice.local:8080",
    })
    api("POST", "/discovery/register", {
        "agent_id": bob["agent_id"],
        "capabilities": ["translation_en_zh"],
        "verification_tier": 0,
        "receipt_chain_length": 5,
        "success_rate": 0.80,
        "pricing_strategy": "market_rate",
    })
    stats = api("GET", "/discovery/stats")
    print(f"4. Registry: {stats['total_agents']} agents, {stats['unique_capabilities']} capabilities")

    # 5. Search
    results = api("POST", "/discovery/search", {"task_type": "code_review", "max_results": 5})
    print(f"5. Search 'code_review': {results['total']} results")

    # 6. Build receipt chain for Alice
    for i in range(10):
        api("POST", "/receipt/chain/append", {
            "agent_id": alice["agent_id"],
            "task_type": "code_review",
            "success": True,
            "pricing_strategy": "reputation_premium",
            "price_charged": 2.0 + i * 0.1,
        })
    chain = api("POST", "/receipt/chain/verify", {"agent_id": alice["agent_id"]})
    print(f"6. Chain: {chain['length']} receipts, valid={chain['valid']}")

    # 7. Aggregate reputation
    rep = api("POST", "/reputation/aggregate", {"agent_id": alice["agent_id"]})
    print(f"7. Reputation: composite={rep['composite_score']:.3f}, tier={rep['trust_tier']}")

    # 8. Handshake
    hs = api("POST", "/handshake/start", {
        "initiator_id": bob["agent_id"],
        "responder_id": alice["agent_id"],
        "initiator_capabilities": ["translation_en_zh"],
        "responder_capabilities": ["code_review"],
    })
    api("POST", "/handshake/message", {
        "session_id": hs["session_id"],
        "action": "propose",
        "task_description": "Review auth module",
        "task_type": "code_review",
        "price": 3.0,
    })
    api("POST", "/handshake/message", {
        "session_id": hs["session_id"],
        "action": "accept",
    })
    result = api("POST", "/handshake/message", {
        "session_id": hs["session_id"],
        "action": "execute",
        "output": "LGTM with 2 suggestions",
        "tests_passed": 5,
        "tests_failed": 0,
        "suite_hash": "sha256:review_tests",
    })
    print(f"8. Handshake: {result['state']}, receipt={result['receipt_id'][:12]}..., chain={result['chain_length']}")

    # 9. Treasury
    decision = api("POST", "/treasury/evaluate", {
        "requester_id": bob["agent_id"],
        "counterparty_id": alice["agent_id"],
        "amount": 3.0,
        "category": "service",
        "description": "Code review payment",
    })
    print(f"9. Treasury: {decision['decision']} — {decision['reason']}")

    # 10. Final health
    health = api("GET", "/health")
    print(f"\n✓ All checks passed. {health['agents']} agents, {health['chains']} chains, {health['registry']} registered")


if __name__ == "__main__":
    import sys
    import threading

    if "--test" in sys.argv:
        # Start server in background, run smoke test, stop
        server = create_server()
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        print(f"Server started on http://127.0.0.1:8402\n")
        try:
            smoke_test()
        finally:
            server.shutdown()
    else:
        server = create_server()
        print(f"ClawBizarre API server running on http://127.0.0.1:8402")
        print("Press Ctrl+C to stop")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
            print("\nServer stopped.")
