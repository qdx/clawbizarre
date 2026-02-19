"""
ClawBizarre HTTP API v2 — Phase 7
Persistent storage (SQLite) + Ed25519 challenge-response auth.

Changes from v1:
- All state persisted to SQLite (survives restarts)
- Auth required on write endpoints (challenge-response → bearer token)
- Read endpoints remain public
- /auth/challenge and /auth/verify for login flow

Usage:
  python3 api_server_v2.py [--port 8420] [--db clawbizarre.db]
"""

import json
import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse
from typing import Optional

from identity import AgentIdentity
from receipt import WorkReceipt, TestResults, RiskEnvelope, Timing, VerificationTier
from persistence import PersistenceLayer
from auth import AuthMiddleware, is_public, extract_bearer_token
from aggregator import ReputationAggregator
from treasury import TreasuryAgent, BudgetPolicy, SpendRequest, SpendCategory


class PersistentState:
    """Server state backed by SQLite."""

    def __init__(self, db_path: str = "clawbizarre.db"):
        self.db = PersistenceLayer(db_path)
        self.auth = AuthMiddleware(self.db)
        self.aggregator = ReputationAggregator()

        # Treasury (in-memory for now, audit log persisted)
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


class APIv2Handler(BaseHTTPRequestHandler):
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
        """Returns agent_id if authenticated, None otherwise."""
        return self.state.auth.authenticate(self.headers.get("Authorization"))

    def _require_auth(self) -> Optional[str]:
        """Returns agent_id or sends 401 and returns None."""
        agent_id = self._authenticate()
        if not agent_id:
            self._send_json({"error": "Authentication required", "hint": "POST /auth/challenge first"}, 401)
            return None
        return agent_id

    def log_message(self, format, *args):
        pass  # suppress default logging

    # --- Routing ---

    def do_GET(self):
        path = urlparse(self.path).path
        if path == "/":
            self._handle_root()
        elif path == "/stats":
            self._handle_stats()
        elif path == "/discovery/stats":
            self._handle_discovery_stats()
        elif path.startswith("/receipt/chain/"):
            agent_id = path[len("/receipt/chain/"):]
            self._handle_get_chain(agent_id)
        elif path.startswith("/reputation/"):
            agent_id = path[len("/reputation/"):]
            self._handle_get_reputation(agent_id)
        elif path == "/treasury/status":
            self._handle_treasury_status()
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
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_DELETE(self):
        path = urlparse(self.path).path
        if path.startswith("/discovery/"):
            agent_id = path[len("/discovery/"):]
            self._handle_discovery_deregister(agent_id)
        else:
            self._send_json({"error": "Not found"}, 404)

    # --- Handlers ---

    def _handle_root(self):
        self._send_json({
            "service": "ClawBizarre API v2",
            "version": "0.2",
            "features": ["persistence", "auth", "receipts", "discovery", "reputation", "treasury"],
            "auth": "Ed25519 challenge-response → Bearer token",
            "docs": "https://github.com/qdx/rahcd (coming soon)"
        })

    def _handle_stats(self):
        self._send_json(self.state.db.stats())

    # --- Auth ---

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

    # --- Discovery ---

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
        # Parse capabilities JSON for response
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

    # --- Receipts ---

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

    # --- Reputation ---

    def _handle_get_reputation(self, agent_id: str):
        cached = self.state.db.get_reputation(agent_id)
        if cached:
            self._send_json(cached)
            return
        # Compute fresh
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

    # --- Treasury ---

    def _handle_treasury_status(self):
        status = {
            "daily_budget": self.state.treasury.policy.daily_budget,
            "auto_approve_threshold": self.state.treasury.policy.auto_approve_threshold,
            "escalation_threshold": self.state.treasury.policy.escalation_threshold,
            "audit_log_entries": len(self.state.db.get_treasury_log(limit=1000)),
        }
        self._send_json(status)

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


def run_server(port: int = 8420, db_path: str = "clawbizarre.db"):
    state = PersistentState(db_path)
    APIv2Handler.state = state

    server = HTTPServer(("0.0.0.0", port), APIv2Handler)
    print(f"ClawBizarre API v2 running on http://0.0.0.0:{port}")
    print(f"Database: {db_path}")
    print(f"Auth: Ed25519 challenge-response")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        state.db.close()
        server.server_close()


# --- Integration Test ---

def smoke_test(port: int = 8421):
    """Full auth + persistence smoke test."""
    import urllib.request
    import tempfile

    db_path = tempfile.mktemp(suffix=".db")
    state = PersistentState(db_path)
    APIv2Handler.state = state

    server = HTTPServer(("127.0.0.1", port), APIv2Handler)
    import threading
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()

    base = f"http://127.0.0.1:{port}"

    def req(method, path, data=None, headers=None):
        body = json.dumps(data).encode() if data else None
        r = urllib.request.Request(f"{base}{path}", data=body, method=method,
                                   headers=headers or {"Content-Type": "application/json"})
        try:
            resp = urllib.request.urlopen(r)
            return json.loads(resp.read()), resp.status
        except urllib.error.HTTPError as e:
            return json.loads(e.read()), e.code

    # 1. Root
    resp, code = req("GET", "/")
    assert code == 200 and resp["version"] == "0.2"
    print("✓ 1. Root endpoint")

    # 2. Stats (empty)
    resp, code = req("GET", "/stats")
    assert resp["receipts"] == 0
    print("✓ 2. Stats (empty)")

    # 3. Create identity + auth flow
    ident = AgentIdentity.generate()
    agent_id = f"ed25519:{ident.public_key_hex}"

    resp, code = req("POST", "/auth/challenge", {"agent_id": agent_id})
    assert code == 200
    challenge_id = resp["challenge_id"]
    challenge = resp["challenge"]
    print(f"✓ 3. Auth challenge")

    sig = ident.sign(challenge)
    resp, code = req("POST", "/auth/verify", {
        "challenge_id": challenge_id,
        "agent_id": agent_id,
        "signature": sig,
        "pubkey": ident.public_key_hex
    })
    assert code == 200
    token = resp["token"]
    print(f"✓ 4. Auth verify → token")

    auth_headers = {"Content-Type": "application/json", "Authorization": f"Bearer {token}"}

    # 5. Register capability (authed)
    resp, code = req("POST", "/discovery/register",
                     {"capabilities": ["code_review", "translation"]}, auth_headers)
    assert code == 200
    print(f"✓ 5. Discovery register (authed)")

    # 6. Search (public)
    resp, code = req("POST", "/discovery/search", {"task_type": "code_review"})
    assert code == 200 and resp["count"] == 1
    print(f"✓ 6. Discovery search (public)")

    # 7. Create receipt (authed)
    from receipt import hash_content
    resp, code = req("POST", "/receipt/chain/append", {
        "task_type": "code_review",
        "input_hash": hash_content("code to review"),
        "output_hash": hash_content("review result"),
        "test_results": {"passed": 5, "failed": 0, "suite_hash": hash_content("tests")},
    }, auth_headers)
    assert code == 200 and resp["chain_length"] == 1
    print(f"✓ 7. Chain append (authed)")

    # 8. Get chain (public)
    resp, code = req("GET", f"/receipt/chain/{agent_id}")
    assert resp["chain_length"] == 1 and resp["integrity"] == True
    print(f"✓ 8. Get chain (public, integrity verified)")

    # 9. Unauthenticated write → 401
    resp, code = req("POST", "/receipt/chain/append", {
        "task_type": "translation",
        "input_hash": "x", "output_hash": "y",
    })
    assert code == 401
    print(f"✓ 9. Unauthenticated write rejected (401)")

    # 10. Treasury evaluate (authed)
    resp, code = req("POST", "/treasury/evaluate", {
        "counterparty_id": "ed25519:other",
        "amount": 3.0,
        "category": "service",
        "description": "code review job"
    }, auth_headers)
    if code != 200:
        print(f"  Treasury response: {resp} (code {code})")
    assert code == 200, f"Treasury evaluate failed: {resp}"
    print(f"✓ 10. Treasury evaluate (authed)")

    # 11. Stats after operations
    resp, code = req("GET", "/stats")
    assert resp["receipts"] == 1 and resp["active_tokens"] == 1
    print(f"✓ 11. Stats: {resp}")

    server.shutdown()
    state.db.close()
    os.unlink(db_path)
    # Clean up WAL files
    for suffix in ["-wal", "-shm"]:
        try:
            os.unlink(db_path + suffix)
        except FileNotFoundError:
            pass

    print(f"\n=== All v2 smoke tests passed (11/11) ===")


if __name__ == "__main__":
    if "--test" in sys.argv:
        smoke_test()
    else:
        port = 8420
        db = "clawbizarre.db"
        for i, arg in enumerate(sys.argv[1:], 1):
            if arg == "--port" and i < len(sys.argv) - 1:
                port = int(sys.argv[i + 1])
            elif arg == "--db" and i < len(sys.argv) - 1:
                db = sys.argv[i + 1]
        run_server(port, db)
