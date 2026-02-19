"""
ClawBizarre HTTP API v3 — Phase 8a Integration
Everything from v2 (persistence, auth) + matching engine endpoints.

New endpoints:
  POST /matching/listing       — Create/update a service listing (authed)
  DELETE /matching/listing      — Remove a listing (authed)
  POST /matching/match          — Find matching service providers (authed)
  GET  /matching/stats          — Matching engine statistics (public)
  GET  /matching/price-history  — Transparent price change log (public)

Usage:
  python3 api_server_v3.py [--port 8420] [--db clawbizarre.db]
  python3 api_server_v3.py --test
"""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
from typing import Optional

from identity import AgentIdentity
from receipt import WorkReceipt, TestResults, RiskEnvelope, Timing, VerificationTier, hash_content
from persistence import PersistenceLayer
from auth import AuthMiddleware, is_public, extract_bearer_token
from aggregator import ReputationAggregator
from treasury import TreasuryAgent, BudgetPolicy, SpendRequest, SpendCategory
from matching import (
    MatchingEngine, ServiceListing, MatchRequest,
    SelectionStrategy, PricingModel
)


class PersistentState:
    """Server state backed by SQLite + in-memory matching engine."""

    def __init__(self, db_path: str = "clawbizarre.db"):
        self.db = PersistenceLayer(db_path)
        self.auth = AuthMiddleware(self.db)
        self.aggregator = ReputationAggregator()
        self.matching = MatchingEngine()

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


class APIv3Handler(BaseHTTPRequestHandler):
    state: PersistentState

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def _authenticate(self) -> Optional[str]:
        return self.state.auth.authenticate(self.headers.get("Authorization"))

    def _require_auth(self) -> Optional[str]:
        agent_id = self._authenticate()
        if not agent_id:
            self._send_json({"error": "Authentication required", "hint": "POST /auth/challenge first"}, 401)
            return None
        return agent_id

    def log_message(self, format, *args):
        pass

    # --- Routing ---

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        if path == "/":
            self._handle_root()
        elif path == "/stats":
            self._handle_stats()
        elif path == "/discovery/stats":
            self._handle_discovery_stats()
        elif path.startswith("/receipt/chain/"):
            self._handle_get_chain(path[len("/receipt/chain/"):])
        elif path.startswith("/reputation/"):
            self._handle_get_reputation(path[len("/reputation/"):])
        elif path == "/treasury/status":
            self._handle_treasury_status()
        elif path == "/matching/stats":
            self._handle_matching_stats()
        elif path == "/matching/price-history":
            self._handle_price_history(parse_qs(parsed.query))
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/auth/challenge":
            self._handle_auth_challenge()
        elif path == "/auth/verify":
            self._handle_auth_verify()
        elif path == "/discovery/register":
            self._handle_discovery_register()
        elif path == "/discovery/search":
            self._handle_discovery_search()
        elif path == "/discovery/heartbeat":
            self._handle_discovery_heartbeat()
        elif path == "/receipt/create":
            self._handle_receipt_create()
        elif path == "/receipt/chain/append":
            self._handle_chain_append()
        elif path == "/reputation/aggregate":
            self._handle_reputation_aggregate()
        elif path == "/treasury/evaluate":
            self._handle_treasury_evaluate()
        elif path == "/matching/listing":
            self._handle_create_listing()
        elif path == "/matching/match":
            self._handle_match()
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/discovery/"):
            self._handle_discovery_deregister(path[len("/discovery/"):])
        elif path == "/matching/listing":
            self._handle_remove_listing()
        else:
            self._send_json({"error": "Not found"}, 404)

    # --- Root / Stats ---

    def _handle_root(self):
        self._send_json({
            "service": "ClawBizarre API v3",
            "version": "0.3",
            "features": ["persistence", "auth", "receipts", "discovery",
                         "reputation", "treasury", "matching"],
            "auth": "Ed25519 challenge-response → Bearer token",
        })

    def _handle_stats(self):
        db_stats = self.state.db.stats()
        db_stats["matching"] = self.state.matching.stats()
        self._send_json(db_stats)

    # --- Auth (unchanged from v2) ---

    def _handle_auth_challenge(self):
        body = self._read_body()
        agent_id = body.get("agent_id")
        if not agent_id:
            self._send_json({"error": "agent_id required"}, 400)
            return
        challenge = self.state.auth.create_challenge(agent_id)
        self._send_json(challenge)

    def _handle_auth_verify(self):
        body = self._read_body()
        required = ["challenge_id", "agent_id", "signature", "pubkey"]
        missing = [k for k in required if k not in body]
        if missing:
            self._send_json({"error": f"Missing fields: {missing}"}, 400)
            return
        token = self.state.auth.verify_and_issue_token(
            body["challenge_id"], body["agent_id"],
            body["signature"], body["pubkey"]
        )
        if not token:
            self._send_json({"error": "Verification failed"}, 403)
            return
        self._send_json({"token": token, "agent_id": body["agent_id"]})

    # --- Discovery (unchanged from v2) ---

    def _handle_discovery_stats(self):
        stats = self.state.db.stats()
        self._send_json({
            "registered_capabilities": stats["registered_capabilities"],
            "unique_agents": stats["unique_agents"],
            "total_receipts": stats["receipts"]
        })

    def _handle_discovery_register(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        capabilities = body.get("capabilities", [])
        if not capabilities:
            self._send_json({"error": "capabilities list required"}, 400)
            return
        self.state.db.register_capability(
            agent_id=agent_id,
            capabilities=capabilities,
            verification_tier=body.get("verification_tier", 0),
            pricing_strategy=body.get("pricing_strategy", "reputation_premium"),
            metadata=body.get("metadata")
        )
        self._send_json({"status": "registered", "agent_id": agent_id})

    def _handle_discovery_search(self):
        body = self._read_body()
        results = self.state.db.search_capabilities(
            task_type=body.get("task_type"),
            min_receipts=body.get("min_receipts", 0),
            limit=body.get("limit", 20)
        )
        for r in results:
            r["capabilities"] = json.loads(r.get("capabilities_json", "[]"))
            r.pop("capabilities_json", None)
            r.pop("metadata_json", None)
        self._send_json({"results": results, "count": len(results)})

    def _handle_discovery_heartbeat(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        self.state.db.heartbeat(agent_id)
        self._send_json({"status": "ok", "agent_id": agent_id})

    def _handle_discovery_deregister(self, path_agent_id: str):
        agent_id = self._require_auth()
        if not agent_id:
            return
        if agent_id != path_agent_id:
            self._send_json({"error": "Can only deregister own capability"}, 403)
            return
        self.state.db.deregister(agent_id)
        self._send_json({"status": "deregistered"})

    # --- Receipts (unchanged from v2) ---

    def _handle_receipt_create(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        body.setdefault("agent_id", agent_id)
        if body["agent_id"] != agent_id:
            self._send_json({"error": "agent_id mismatch"}, 403)
            return
        try:
            receipt = WorkReceipt(
                agent_id=body["agent_id"],
                task_type=body["task_type"],
                verification_tier=VerificationTier(body.get("verification_tier", 0)),
                input_hash=body["input_hash"],
                output_hash=body["output_hash"],
                pricing_strategy=body.get("pricing_strategy", "reputation_premium"),
                platform=body.get("platform", "clawbizarre"),
                environment_hash=body.get("environment_hash"),
                test_results=TestResults(**body["test_results"]) if body.get("test_results") else None,
                timing=Timing(**body["timing"]) if body.get("timing") else None,
                risk_envelope=RiskEnvelope(**body["risk_envelope"]) if body.get("risk_envelope") else None,
            )
            rid = self.state.db.store_receipt(receipt)
            self._send_json({"receipt_id": rid, "content_hash": receipt.content_hash})
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_chain_append(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        body.setdefault("agent_id", agent_id)
        if body["agent_id"] != agent_id:
            self._send_json({"error": "agent_id mismatch"}, 403)
            return
        try:
            receipt = WorkReceipt(
                agent_id=body["agent_id"],
                task_type=body["task_type"],
                verification_tier=VerificationTier(body.get("verification_tier", 0)),
                input_hash=body["input_hash"],
                output_hash=body["output_hash"],
                pricing_strategy=body.get("pricing_strategy", "reputation_premium"),
                platform=body.get("platform", "clawbizarre"),
                environment_hash=body.get("environment_hash"),
                test_results=TestResults(**body["test_results"]) if body.get("test_results") else None,
                timing=Timing(**body["timing"]) if body.get("timing") else None,
            )
            chain_hash = self.state.db.append_to_chain(agent_id, receipt)
            count = self.state.db.count_agent_receipts(agent_id)
            self._send_json({
                "receipt_id": receipt.receipt_id,
                "chain_hash": chain_hash,
                "chain_length": count
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_get_chain(self, agent_id: str):
        chain = self.state.db.rebuild_chain(agent_id)
        if chain.length == 0:
            self._send_json({"agent_id": agent_id, "chain_length": 0, "receipts": []})
            return
        self._send_json({
            "agent_id": agent_id,
            "chain_length": chain.length,
            "integrity": chain.verify_integrity(),
            "success_rate": chain.success_rate(),
            "strategy_consistency": chain.strategy_consistency(),
            "strategy_changes": chain.strategy_changes(),
            "on_time_rate": chain.on_time_rate(),
            "tier_breakdown": chain.tier_breakdown(),
        })

    # --- Reputation (unchanged from v2) ---

    def _handle_get_reputation(self, agent_id: str):
        cached = self.state.db.get_reputation(agent_id)
        if cached:
            self._send_json(cached)
            return
        chain = self.state.db.rebuild_chain(agent_id)
        if chain.length == 0:
            self._send_json({"agent_id": agent_id, "reputation": None, "reason": "no receipts"})
            return
        snapshot = self.state.aggregator.aggregate(chain)
        snap_dict = snapshot.to_dict() if hasattr(snapshot, 'to_dict') else str(snapshot)
        self.state.db.store_reputation(agent_id, json.dumps(snap_dict))
        self._send_json({"agent_id": agent_id, "reputation": snap_dict})

    def _handle_reputation_aggregate(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        chain = self.state.db.rebuild_chain(agent_id)
        if chain.length == 0:
            self._send_json({"error": "No receipts to aggregate"}, 400)
            return
        snapshot = self.state.aggregator.aggregate(chain)
        snap_dict = snapshot.to_dict() if hasattr(snapshot, 'to_dict') else str(snapshot)
        self.state.db.store_reputation(agent_id, json.dumps(snap_dict))
        self._send_json({"agent_id": agent_id, "reputation": snap_dict, "chain_length": chain.length})

    # --- Treasury (unchanged from v2) ---

    def _handle_treasury_status(self):
        self._send_json({
            "daily_budget": self.state.treasury.policy.daily_budget,
            "auto_approve_threshold": self.state.treasury.policy.auto_approve_threshold,
            "escalation_threshold": self.state.treasury.policy.escalation_threshold,
            "audit_log_entries": len(self.state.db.get_treasury_log(limit=1000)),
        })

    def _handle_treasury_evaluate(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        try:
            request = SpendRequest(
                requesting_agent=agent_id,
                counterparty=body.get("counterparty_id", "unknown"),
                amount=float(body["amount"]),
                category=SpendCategory(body.get("category", "service")),
                description=body.get("description", ""),
            )
            decision = self.state.treasury.evaluate(request)
            self.state.db.log_treasury_decision(
                {"requester": agent_id, "amount": request.amount, "category": request.category.value},
                decision.decision.value,
                decision.reason
            )
            self._send_json({
                "decision": decision.decision.value,
                "reason": decision.reason,
                "amount": request.amount,
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    # --- Matching Engine (NEW in v3) ---

    def _handle_create_listing(self):
        """POST /matching/listing — Create or update a service listing."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()

        capability = body.get("capability")
        base_rate = body.get("base_rate")
        unit = body.get("unit", "per_task")

        if not capability or base_rate is None:
            self._send_json({"error": "capability and base_rate required"}, 400)
            return

        # Pull reputation from aggregator if available
        rep_score = 0.0
        receipt_count = 0
        chain = self.state.db.rebuild_chain(agent_id)
        if chain.length > 0:
            receipt_count = chain.length
            snapshot = self.state.aggregator.aggregate(chain)
            if hasattr(snapshot, 'composite_score'):
                rep_score = snapshot.composite_score
            elif hasattr(snapshot, 'to_dict'):
                d = snapshot.to_dict()
                rep_score = d.get('composite_score', 0.0)

        try:
            listing = ServiceListing(
                agent_id=agent_id,
                capability=capability,
                base_rate=float(base_rate),
                unit=unit,
                pricing_model=PricingModel(body.get("pricing_model", "fixed")),
                verification_tier=body.get("verification_tier", 0),
                max_response_time_ms=body.get("max_response_time_ms", 60000),
                reputation_score=rep_score,
                uptime_fraction=body.get("uptime_fraction", 1.0),
                receipt_count=receipt_count,
            )
            listing_id = self.state.matching.add_listing(listing)
            self._send_json({
                "status": "listed",
                "listing_id": listing_id,
                "agent_id": agent_id,
                "capability": capability,
                "base_rate": float(base_rate),
                "reputation_score": rep_score,
                "receipt_count": receipt_count,
                "is_newcomer": listing.is_newcomer,
            })
        except ValueError as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_remove_listing(self):
        """DELETE /matching/listing — Remove a listing."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        capability = body.get("capability")
        if not capability:
            self._send_json({"error": "capability required"}, 400)
            return
        removed = self.state.matching.remove_listing(agent_id, capability)
        if removed:
            self._send_json({"status": "removed", "agent_id": agent_id, "capability": capability})
        else:
            self._send_json({"error": "Listing not found"}, 404)

    def _handle_match(self):
        """POST /matching/match — Find matching service providers."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()

        capability = body.get("capability")
        if not capability:
            self._send_json({"error": "capability required"}, 400)
            return

        strategy = SelectionStrategy.TOP3_RANDOM
        if body.get("selection_strategy"):
            try:
                strategy = SelectionStrategy(body["selection_strategy"])
            except ValueError:
                self._send_json({"error": f"Invalid strategy. Use: {[s.value for s in SelectionStrategy]}"}, 400)
                return

        request = MatchRequest(
            buyer_id=agent_id,
            capability=capability,
            max_price=body.get("max_price"),
            min_reputation=body.get("min_reputation", 0.0),
            max_response_time_ms=body.get("max_response_time_ms", 120000),
            verification_tier_required=body.get("verification_tier_required", 0),
            selection_strategy=strategy,
            max_results=body.get("max_results", 5),
            weight_reputation=body.get("weight_reputation", 0.5),
            weight_price=body.get("weight_price", 0.3),
            weight_reliability=body.get("weight_reliability", 0.2),
        )

        response = self.state.matching.match(request)

        self._send_json({
            "request_id": response.request_id,
            "total_available": response.total_available,
            "total_filtered": response.total_filtered,
            "newcomer_slots_used": response.newcomer_slots_used,
            "candidates": [
                {
                    "rank": c.rank,
                    "agent_id": c.listing.agent_id,
                    "capability": c.listing.capability,
                    "base_rate": c.listing.base_rate,
                    "unit": c.listing.unit,
                    "pricing_model": c.listing.pricing_model.value,
                    "reputation_score": c.listing.reputation_score,
                    "receipt_count": c.listing.receipt_count,
                    "uptime_fraction": c.listing.uptime_fraction,
                    "is_newcomer": c.listing.is_newcomer,
                    "score": c.score,
                    "match_reason": c.match_reason,
                }
                for c in response.candidates
            ]
        })

    def _handle_matching_stats(self):
        """GET /matching/stats — Public matching engine statistics."""
        self._send_json(self.state.matching.stats())

    def _handle_price_history(self, query: dict):
        """GET /matching/price-history?capability=X&agent_id=Y — Transparent price log."""
        capability = query.get("capability", [None])[0]
        agent_id = query.get("agent_id", [None])[0]
        since = query.get("since", [None])[0]
        if since:
            try:
                since = float(since)
            except ValueError:
                since = None
        history = self.state.matching.get_price_history(capability, agent_id, since)
        self._send_json({"price_changes": history, "count": len(history)})


def run_server(port: int = 8420, db_path: str = "clawbizarre.db"):
    state = PersistentState(db_path)
    APIv3Handler.state = state
    server = HTTPServer(("0.0.0.0", port), APIv3Handler)
    print(f"ClawBizarre API v3 running on http://0.0.0.0:{port}")
    print(f"Database: {db_path}")
    print(f"Features: persistence, auth, receipts, discovery, reputation, treasury, matching")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.db.close()
        server.server_close()


# --- Smoke Test ---

def smoke_test(port: int = 8422):
    """Full v3 smoke test: v2 tests + matching engine tests."""
    import urllib.request
    import tempfile

    db_path = tempfile.mktemp(suffix=".db")
    state = PersistentState(db_path)
    APIv3Handler.state = state
    server = HTTPServer(("127.0.0.1", port), APIv3Handler)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base = f"http://127.0.0.1:{port}"
    passed = 0
    total = 0

    def req(method, path, data=None, headers=None):
        body = json.dumps(data).encode() if data else None
        r = urllib.request.Request(f"{base}{path}", data=body, method=method,
                                   headers=headers or {"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(r)
            return json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")

    # --- v2 baseline tests ---

    # 1. Root
    resp, code = req("GET", "/")
    check("1. Root (v3)", code == 200 and resp["version"] == "0.3")

    # 2. Stats (empty)
    resp, code = req("GET", "/stats")
    check("2. Stats (empty)", resp["receipts"] == 0)

    # 3-4. Auth flow
    ident_a = AgentIdentity.generate()
    agent_a = f"ed25519:{ident_a.public_key_hex}"

    resp, code = req("POST", "/auth/challenge", {"agent_id": agent_a})
    check("3. Auth challenge", code == 200)
    sig = ident_a.sign(resp["challenge"])
    resp, code = req("POST", "/auth/verify", {
        "challenge_id": resp["challenge_id"], "agent_id": agent_a,
        "signature": sig, "pubkey": ident_a.public_key_hex
    })
    check("4. Auth verify", code == 200)
    token_a = resp["token"]
    auth_a = {"Content-Type": "application/json", "Authorization": f"Bearer {token_a}"}

    # 5. Discovery register
    resp, code = req("POST", "/discovery/register",
                     {"capabilities": ["code_review", "translation"]}, auth_a)
    check("5. Discovery register", code == 200)

    # 6. Chain append (build receipt history)
    for i in range(12):
        resp, code = req("POST", "/receipt/chain/append", {
            "task_type": "code_review",
            "input_hash": hash_content(f"input_{i}"),
            "output_hash": hash_content(f"output_{i}"),
            "test_results": {"passed": 5, "failed": 0, "suite_hash": hash_content("tests")},
        }, auth_a)
    check("6. Chain append (12 receipts)", code == 200 and resp["chain_length"] == 12)

    # --- Matching engine tests ---

    # 7. Create listing
    resp, code = req("POST", "/matching/listing", {
        "capability": "code_review",
        "base_rate": 0.05,
        "unit": "per_line",
        "pricing_model": "fixed",
        "verification_tier": 0,
    }, auth_a)
    check("7. Create listing", code == 200 and resp["status"] == "listed")
    check("8. Listing has reputation", resp["receipt_count"] == 12)

    # Create second agent (newcomer)
    ident_b = AgentIdentity.generate()
    agent_b = f"ed25519:{ident_b.public_key_hex}"
    resp, _ = req("POST", "/auth/challenge", {"agent_id": agent_b})
    sig_b = ident_b.sign(resp["challenge"])
    resp, _ = req("POST", "/auth/verify", {
        "challenge_id": resp["challenge_id"], "agent_id": agent_b,
        "signature": sig_b, "pubkey": ident_b.public_key_hex
    })
    token_b = resp["token"]
    auth_b = {"Content-Type": "application/json", "Authorization": f"Bearer {token_b}"}

    resp, code = req("POST", "/matching/listing", {
        "capability": "code_review",
        "base_rate": 0.03,
        "unit": "per_line",
    }, auth_b)
    check("9. Newcomer listing", code == 200 and resp["is_newcomer"] == True)

    # 10. Match request
    # Create a third agent as buyer
    ident_c = AgentIdentity.generate()
    agent_c = f"ed25519:{ident_c.public_key_hex}"
    resp, _ = req("POST", "/auth/challenge", {"agent_id": agent_c})
    sig_c = ident_c.sign(resp["challenge"])
    resp, _ = req("POST", "/auth/verify", {
        "challenge_id": resp["challenge_id"], "agent_id": agent_c,
        "signature": sig_c, "pubkey": ident_c.public_key_hex
    })
    token_c = resp["token"]
    auth_c = {"Content-Type": "application/json", "Authorization": f"Bearer {token_c}"}

    resp, code = req("POST", "/matching/match", {
        "capability": "code_review",
        "selection_strategy": "best_first",
    }, auth_c)
    check("10. Match returns candidates", code == 200 and len(resp["candidates"]) == 2)
    check("11. Match has both agents", resp["total_available"] == 2)

    # 12. Price filter
    resp, code = req("POST", "/matching/match", {
        "capability": "code_review",
        "max_price": 0.04,
    }, auth_c)
    check("12. Price filter", code == 200 and len(resp["candidates"]) == 1)

    # 13. Matching stats
    resp, code = req("GET", "/matching/stats")
    check("13. Matching stats", code == 200 and resp["total_listings"] == 2 and resp["total_matches"] >= 2)

    # 14. Price change → history
    resp, code = req("POST", "/matching/listing", {
        "capability": "code_review",
        "base_rate": 0.06,
        "unit": "per_line",
    }, auth_a)
    check("14. Price update", code == 200 and resp["base_rate"] == 0.06)

    resp, code = req("GET", "/matching/price-history")
    check("15. Price history logged", code == 200 and resp["count"] == 1)

    # 16. Compute cost floor rejection
    resp, code = req("POST", "/matching/listing", {
        "capability": "code_review",
        "base_rate": 0.0001,
    }, auth_b)
    check("16. Cost floor enforced", code == 400 and "floor" in resp.get("error", "").lower())

    # 17. Remove listing
    resp, code = req("DELETE", "/matching/listing",
                     {"capability": "code_review"}, auth_b)
    check("17. Remove listing", code == 200 and resp["status"] == "removed")

    # 18. Verify removal
    resp, code = req("GET", "/matching/stats")
    check("18. Listing count after removal", resp["total_listings"] == 1)

    # 19. Self-match prevention
    resp, code = req("POST", "/matching/match", {
        "capability": "code_review",
    }, auth_a)  # agent_a is both lister and buyer
    agent_ids = [c["agent_id"] for c in resp["candidates"]]
    check("19. Self-match prevented", agent_a not in agent_ids)

    # 20. Unauthenticated listing rejected
    resp, code = req("POST", "/matching/listing", {
        "capability": "code_review", "base_rate": 0.05
    })
    check("20. Unauthed listing rejected", code == 401)

    # 21. Treasury still works
    resp, code = req("GET", "/treasury/status")
    check("21. Treasury status", code == 200)

    # 22. Full stats
    resp, code = req("GET", "/stats")
    check("22. Combined stats", "matching" in resp and resp["matching"]["total_listings"] == 1)

    server.shutdown()
    state.db.close()
    os.unlink(db_path)
    for suffix in ["-wal", "-shm"]:
        try:
            os.unlink(db_path + suffix)
        except FileNotFoundError:
            pass

    print(f"\n=== v3 smoke tests: {passed}/{total} passed ===")
    return passed == total


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = smoke_test()
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
