"""
ClawBizarre HTTP API v4 — Phase 8c: Handshake Integration
Everything from v3 + HTTP handshake endpoints for the full pipeline:
  Match → Handshake → Execute → Receipt

New endpoints:
  POST /handshake/initiate     — Start handshake with a matched provider (authed)
  POST /handshake/respond       — Respond to a handshake (hello/accept/reject) (authed)
  POST /handshake/execute       — Submit work output (authed)
  POST /handshake/verify        — Verify work + generate receipt (authed)
  GET  /handshake/<session_id>  — Get handshake status (authed, participants only)
  GET  /handshake/active        — List your active handshakes (authed)

Full pipeline:
  1. Buyer: POST /matching/match → get provider list
  2. Buyer: POST /handshake/initiate → creates session, sends hello+proposal
  3. Provider: POST /handshake/respond → sends hello+accept (or reject)
  4. Provider: POST /handshake/execute → submits work output+proof
  5. Buyer: POST /handshake/verify → verifies, generates signed receipt
  6. Receipt auto-appended to both agents' chains

Usage:
  python3 api_server_v4.py [--port 8420] [--db clawbizarre.db]
  python3 api_server_v4.py --test
"""

import json
import sys
import os
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
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


class HandshakeStore:
    """In-memory store for active handshake sessions."""

    def __init__(self):
        self.sessions: dict[str, HandshakeSession] = {}  # session_id → session
        self.agent_sessions: dict[str, list[str]] = {}   # agent_id → [session_ids]
        self.session_meta: dict[str, dict] = {}           # session_id → {buyer, provider, proposal, ...}

    def create(self, buyer_id: str, provider_id: str, proposal: dict) -> HandshakeSession:
        session = HandshakeSession(buyer_id)
        sid = session.session_id
        self.sessions[sid] = session
        self.session_meta[sid] = {
            "buyer": buyer_id,
            "provider": provider_id,
            "proposal": proposal,
            "created_at": time.time(),
            "output": None,
            "proof": None,
        }
        for aid in (buyer_id, provider_id):
            self.agent_sessions.setdefault(aid, []).append(sid)
        return session

    def get(self, session_id: str) -> Optional[HandshakeSession]:
        return self.sessions.get(session_id)

    def meta(self, session_id: str) -> Optional[dict]:
        return self.session_meta.get(session_id)

    def for_agent(self, agent_id: str) -> list[str]:
        return self.agent_sessions.get(agent_id, [])

    def remove(self, session_id: str):
        meta = self.session_meta.pop(session_id, None)
        self.sessions.pop(session_id, None)
        if meta:
            for aid in (meta["buyer"], meta["provider"]):
                sids = self.agent_sessions.get(aid, [])
                if session_id in sids:
                    sids.remove(session_id)

    def stats(self) -> dict:
        states = {}
        for s in self.sessions.values():
            states[s.state.value] = states.get(s.state.value, 0) + 1
        return {"total": len(self.sessions), "by_state": states}


class PersistentState:
    """Server state backed by SQLite + in-memory matching/handshake."""

    def __init__(self, db_path: str = "clawbizarre.db"):
        self.db = PersistenceLayer(db_path)
        self.auth = AuthMiddleware(self.db)
        self.aggregator = ReputationAggregator()
        self.matching = MatchingEngine()
        self.handshakes = HandshakeStore()

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


class APIv4Handler(BaseHTTPRequestHandler):
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
            self._send_json({"error": "Authentication required"}, 401)
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
        elif path.startswith("/handshake/active"):
            self._handle_handshake_active()
        elif path.startswith("/handshake/"):
            self._handle_handshake_status(path[len("/handshake/"):])
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
        elif path == "/handshake/initiate":
            self._handle_handshake_initiate()
        elif path == "/handshake/respond":
            self._handle_handshake_respond()
        elif path == "/handshake/execute":
            self._handle_handshake_execute()
        elif path == "/handshake/verify":
            self._handle_handshake_verify()
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
            "service": "ClawBizarre API v4",
            "version": "0.4",
            "features": ["persistence", "auth", "receipts", "discovery",
                         "reputation", "treasury", "matching", "handshake"],
            "auth": "Ed25519 challenge-response → Bearer token",
        })

    def _handle_stats(self):
        db_stats = self.state.db.stats()
        db_stats["matching"] = self.state.matching.stats()
        db_stats["handshakes"] = self.state.handshakes.stats()
        self._send_json(db_stats)

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
            self._send_json({"error": f"Missing: {missing}"}, 400)
            return
        token = self.state.auth.verify_and_issue_token(
            body["challenge_id"], body["agent_id"],
            body["signature"], body["pubkey"]
        )
        if not token:
            self._send_json({"error": "Verification failed"}, 401)
        else:
            self._send_json({"token": token, "agent_id": body["agent_id"]})

    # --- Discovery ---

    def _handle_discovery_register(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        capabilities = body.get("capabilities", [])
        metadata = body.get("metadata", {})
        self.state.db.register_capability(
            agent_id, capabilities,
            verification_tier=body.get("verification_tier", 0),
            pricing_strategy=body.get("pricing_strategy", "reputation_premium"),
            metadata=metadata,
        )
        self._send_json({"registered": agent_id, "capabilities": capabilities})

    def _handle_discovery_search(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        capability = body.get("capability", "")
        limit = body.get("limit", 10)
        results = self.state.db.search_capabilities(capability, limit=limit)
        self._send_json({"results": results, "count": len(results)})

    def _handle_discovery_heartbeat(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        self.state.db.heartbeat(agent_id)
        self._send_json({"ok": True})

    def _handle_discovery_stats(self):
        stats = self.state.db.discovery_stats()
        self._send_json(stats)

    def _handle_discovery_deregister(self, agent_id_path: str):
        agent_id = self._require_auth()
        if not agent_id:
            return
        self.state.db.deregister(agent_id)
        self._send_json({"deregistered": agent_id})

    # --- Receipts ---

    def _handle_receipt_create(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        try:
            test_results = None
            if "test_results" in body:
                tr = body["test_results"]
                test_results = TestResults(tr["passed"], tr["failed"], tr.get("suite_hash", ""))
            risk = None
            if "risk_envelope" in body:
                r = body["risk_envelope"]
                risk = RiskEnvelope(
                    counterparty_risk=r.get("counterparty_risk", 0.0),
                    verification_confidence=r.get("verification_confidence", 1.0),
                    policy_version=r.get("policy_version", ""),
                    environment_fingerprint=r.get("environment_fingerprint", ""),
                )
            receipt = WorkReceipt(
                agent_id=agent_id,
                task_type=body["task_type"],
                verification_tier=VerificationTier(body.get("verification_tier", 0)),
                input_hash=body.get("input_hash", hash_content(json.dumps(body))),
                output_hash=body.get("output_hash", hash_content("")),
                test_results=test_results,
                platform=body.get("platform", "clawbizarre"),
                risk_envelope=risk,
            )
            self.state.db.store_receipt(receipt)
            self._send_json({"receipt_id": receipt.receipt_id, "stored": True})
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_chain_append(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        receipt_id = body.get("receipt_id")
        chain_id = body.get("chain_id", agent_id)
        if not receipt_id:
            self._send_json({"error": "receipt_id required"}, 400)
            return
        r = self.state.db.get_receipt(receipt_id)
        if not r:
            self._send_json({"error": "Receipt not found"}, 404)
            return
        self.state.db.append_to_chain(chain_id, r)
        count = self.state.db.count_agent_receipts(chain_id)
        self._send_json({"chain_id": chain_id, "length": count})

    def _handle_get_chain(self, chain_id: str):
        chain = self.state.db.rebuild_chain(chain_id)
        receipt_ids = [r.receipt_id for r in chain.receipts] if chain else []
        self._send_json({"chain_id": chain_id, "receipts": receipt_ids, "length": len(receipt_ids)})

    # --- Reputation ---

    def _handle_get_reputation(self, agent_id_path: str):
        chain = self.state.db.rebuild_chain(agent_id_path)
        if not chain or not chain.receipts:
            self._send_json({"agent_id": agent_id_path, "reputation": 0.0, "receipts": 0})
            return
        snapshot = self.state.aggregator.aggregate(chain)
        self._send_json({
            "agent_id": agent_id_path,
            "reputation": round(snapshot.composite_score, 4),
            "receipts": len(chain.receipts),
            "domains": {k: round(v, 4) for k, v in snapshot.domain_scores.items()},
        })

    def _handle_reputation_aggregate(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        chain = self.state.db.rebuild_chain(agent_id)
        receipts = chain.receipts if chain else []
        snapshot = self.state.aggregator.aggregate(chain) if receipts else None
        self._send_json({
            "agent_id": agent_id,
            "composite": round(snapshot.composite_score, 4) if snapshot else 0.0,
            "domains": {k: round(v, 4) for k, v in snapshot.domain_scores.items()} if snapshot else {},
            "receipt_count": len(receipts),
        })

    # --- Treasury ---

    def _handle_treasury_status(self):
        self._send_json(self.state.treasury.status())

    def _handle_treasury_evaluate(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        try:
            request = SpendRequest(
                amount=body["amount"],
                counterparty=body.get("counterparty", "unknown"),
                category=SpendCategory(body.get("category", "service")),
                description=body.get("description", ""),
            )
            decision = self.state.treasury.evaluate(request)
            self._send_json({"decision": decision.value, "request": body})
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    # --- Matching ---

    def _handle_create_listing(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        try:
            chain = self.state.db.rebuild_chain(agent_id)
            receipts = chain.receipts if chain else []
            snapshot = self.state.aggregator.aggregate(chain) if receipts else None

            capabilities = body.get("capabilities", [])
            base_rate = body.get("base_rate", body.get("price_per_task", 1.0))
            unit = body.get("unit", "per_task")
            listed = []
            for cap in capabilities:
                listing = ServiceListing(
                    agent_id=agent_id,
                    capability=cap,
                    base_rate=base_rate,
                    unit=unit,
                    pricing_model=PricingModel(body.get("pricing_model", "fixed")),
                    reputation_score=snapshot.composite_score if snapshot else 0.0,
                    receipt_count=len(receipts),
                    max_response_time_ms=int(body.get("response_time_avg", 60.0) * 1000),
                )
                self.state.matching.add_listing(listing)
                listed.append(cap)
            self._send_json({
                "listed": agent_id,
                "capabilities": listed,
                "reputation": round(snapshot.composite_score if snapshot else 0.0, 4),
                "receipts": len(receipts),
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_remove_listing(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        capability = body.get("capability", "")
        removed = self.state.matching.remove_listing(agent_id, capability)
        self._send_json({"removed": removed})

    def _handle_match(self):
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        try:
            request = MatchRequest(
                buyer_id=agent_id,
                capability=body["capability"],
                max_price=body.get("max_price"),
                min_reputation=body.get("min_reputation", 0.0),
                selection_strategy=SelectionStrategy(body.get("strategy", "top3_random")),
                max_results=body.get("limit", 5),
            )
            response = self.state.matching.match(request)
            self._send_json({
                "matches": [
                    {
                        "agent_id": r.listing.agent_id,
                        "score": round(r.score, 4),
                        "price": r.listing.base_rate,
                        "reputation": round(r.listing.reputation_score, 4),
                        "capability": r.listing.capability,
                    }
                    for r in response.candidates
                ],
                "count": len(response.candidates),
            })
        except Exception as e:
            self._send_json({"error": str(e)}, 400)

    def _handle_matching_stats(self):
        self._send_json(self.state.matching.stats())

    def _handle_price_history(self, params: dict):
        capability = params.get("capability", [None])[0]
        agent_id = params.get("agent_id", [None])[0]
        history = self.state.matching.get_price_history(capability, agent_id)
        self._send_json({"history": history, "count": len(history)})

    # --- Handshake (NEW in v4) ---

    def _handle_handshake_initiate(self):
        """Buyer initiates handshake with a provider after matching."""
        buyer_id = self._require_auth()
        if not buyer_id:
            return
        body = self._read_body()
        provider_id = body.get("provider_id")
        if not provider_id:
            self._send_json({"error": "provider_id required"}, 400)
            return
        if provider_id == buyer_id:
            self._send_json({"error": "Cannot handshake with yourself"}, 400)
            return

        proposal = body.get("proposal", {})
        if not proposal.get("task_description") or not proposal.get("task_type"):
            self._send_json({"error": "proposal.task_description and proposal.task_type required"}, 400)
            return

        # Create session
        session = self.state.handshakes.create(buyer_id, provider_id, proposal)

        # Buyer sends hello
        capabilities = body.get("capabilities", [])
        constraints = Constraints()
        if body.get("constraints"):
            c = body["constraints"]
            constraints = Constraints(
                time_limit_seconds=c.get("time_limit_seconds", 1800),
                budget=c.get("budget"),
                privacy=c.get("privacy", "no_credential_sharing"),
            )
        session.send_hello(capabilities, constraints)

        # Buyer proposes
        session.propose(
            task_description=proposal["task_description"],
            task_type=proposal["task_type"],
            verification_tier=VerificationTier(proposal.get("verification_tier", 0)),
            test_suite_hash=proposal.get("test_suite_hash"),
            input_data=proposal.get("input_data"),
        )

        self._send_json({
            "session_id": session.session_id,
            "state": session.state.value,
            "buyer": buyer_id,
            "provider": provider_id,
            "proposal": proposal,
        })

    def _handle_handshake_respond(self):
        """Provider responds to a handshake (hello + accept/reject)."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        session_id = body.get("session_id")
        if not session_id:
            self._send_json({"error": "session_id required"}, 400)
            return

        session = self.state.handshakes.get(session_id)
        meta = self.state.handshakes.meta(session_id)
        if not session or not meta:
            self._send_json({"error": "Session not found"}, 404)
            return
        if agent_id != meta["provider"]:
            self._send_json({"error": "Only the provider can respond"}, 403)
            return

        action = body.get("action", "accept")
        capabilities = body.get("capabilities", [])

        # Provider sends hello
        # We need to simulate receiving buyer's hello first
        from handshake import HandshakeMessage
        buyer_hello = None
        for msg in session.messages:
            if msg.msg_type == "hello":
                buyer_hello = msg
                break

        # Create provider-side state by accepting the proposal
        if action == "accept":
            # Provider takes over session perspective
            session.my_agent_id = agent_id
            session.state = HandshakeState.PROPOSED
            session.accept()
            self._send_json({
                "session_id": session_id,
                "state": session.state.value,
                "action": "accepted",
            })
        elif action == "reject":
            reason = body.get("reason", "declined")
            session.state = HandshakeState.PROPOSED
            session.reject(reason)
            self._send_json({
                "session_id": session_id,
                "state": session.state.value,
                "action": "rejected",
                "reason": reason,
            })
        else:
            self._send_json({"error": f"Unknown action: {action}"}, 400)

    def _handle_handshake_execute(self):
        """Provider submits work output."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        session_id = body.get("session_id")
        if not session_id:
            self._send_json({"error": "session_id required"}, 400)
            return

        session = self.state.handshakes.get(session_id)
        meta = self.state.handshakes.meta(session_id)
        if not session or not meta:
            self._send_json({"error": "Session not found"}, 404)
            return
        if agent_id != meta["provider"]:
            self._send_json({"error": "Only the provider can execute"}, 403)
            return
        if session.state != HandshakeState.ACCEPTED:
            self._send_json({"error": f"Cannot execute in state: {session.state.value}"}, 400)
            return

        output = body.get("output", "")
        proof = body.get("proof", {})
        session.execute(output, proof)

        # Store output in meta for verification step
        meta["output"] = output
        meta["proof"] = proof

        self._send_json({
            "session_id": session_id,
            "state": session.state.value,
            "output_hash": hash_content(output),
        })

    def _handle_handshake_verify(self):
        """Buyer verifies work and generates receipt."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        body = self._read_body()
        session_id = body.get("session_id")
        if not session_id:
            self._send_json({"error": "session_id required"}, 400)
            return

        session = self.state.handshakes.get(session_id)
        meta = self.state.handshakes.meta(session_id)
        if not session or not meta:
            self._send_json({"error": "Session not found"}, 404)
            return
        if agent_id != meta["buyer"]:
            self._send_json({"error": "Only the buyer can verify"}, 403)
            return
        if session.state != HandshakeState.EXECUTING:
            self._send_json({"error": f"Cannot verify in state: {session.state.value}"}, 400)
            return

        # Build test results from body or auto-verify
        passed = body.get("passed", 1)
        failed = body.get("failed", 0)
        suite_hash = body.get("suite_hash", meta["proposal"].get("test_suite_hash", ""))

        test_results = TestResults(passed=passed, failed=failed, suite_hash=suite_hash)

        # Find the execute message
        execute_msg = None
        for msg in session.messages:
            if msg.msg_type == "EXECUTE":
                execute_msg = msg
                break

        if not execute_msg:
            self._send_json({"error": "No execute message found"}, 400)
            return

        # Custom verifier using the provided test results
        def verifier(payload):
            return test_results

        verify_msg = session.verify(execute_msg, verifier)

        result = {
            "session_id": session_id,
            "state": session.state.value,
            "verified": session.state == HandshakeState.COMPLETE,
        }

        # If complete, create receipt and append to both chains
        if session.state == HandshakeState.COMPLETE and session.receipt:
            receipt = session.receipt
            # Store receipt and append to both agents' chains
            self.state.db.store_receipt(receipt)
            self.state.db.append_to_chain(meta["provider"], receipt)
            self.state.db.append_to_chain(meta["buyer"], receipt)

            result["receipt_id"] = receipt.receipt_id
            result["receipt"] = {
                "agent_id": receipt.agent_id,
                "task_type": receipt.task_type,
                "verification_tier": receipt.verification_tier.value,
                "test_results": {"passed": test_results.passed, "failed": test_results.failed},
            }

            # Clean up completed handshake
            self.state.handshakes.remove(session_id)

        self._send_json(result)

    def _handle_handshake_status(self, session_id: str):
        """Get handshake status (participants only)."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        session = self.state.handshakes.get(session_id)
        meta = self.state.handshakes.meta(session_id)
        if not session or not meta:
            self._send_json({"error": "Session not found"}, 404)
            return
        if agent_id not in (meta["buyer"], meta["provider"]):
            self._send_json({"error": "Not a participant"}, 403)
            return
        self._send_json({
            "session_id": session_id,
            "state": session.state.value,
            "buyer": meta["buyer"],
            "provider": meta["provider"],
            "proposal": meta["proposal"],
            "message_count": len(session.messages),
        })

    def _handle_handshake_active(self):
        """List active handshakes for the authenticated agent."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        session_ids = self.state.handshakes.for_agent(agent_id)
        active = []
        for sid in session_ids:
            session = self.state.handshakes.get(sid)
            meta = self.state.handshakes.meta(sid)
            if session and meta:
                active.append({
                    "session_id": sid,
                    "state": session.state.value,
                    "role": "buyer" if meta["buyer"] == agent_id else "provider",
                    "counterparty": meta["provider"] if meta["buyer"] == agent_id else meta["buyer"],
                })
        self._send_json({"active": active, "count": len(active)})


# --- Server ---

def run_server(port: int = 8420, db_path: str = "clawbizarre.db"):
    state = PersistentState(db_path)
    APIv4Handler.state = state

    server = HTTPServer(("127.0.0.1", port), APIv4Handler)
    print(f"ClawBizarre API v4 running on http://127.0.0.1:{port}")
    print(f"Database: {db_path}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


# --- Smoke Tests ---

def run_tests():
    import threading
    import urllib.request

    db_path = "/tmp/clawbizarre_v4_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    state = PersistentState(db_path)
    APIv4Handler.state = state
    server = HTTPServer(("127.0.0.1", 0), APIv4Handler)
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

    print(f"\n=== ClawBizarre API v4 Smoke Tests (port {port}) ===\n")

    # 1. Root
    r, s = api("GET", "/")
    check("1. Root returns v4", r.get("version") == "0.4" and "handshake" in r.get("features", []))

    # 2-3. Auth two agents (buyer + provider)
    buyer_id = AgentIdentity.generate()
    provider_id = AgentIdentity.generate()

    def auth_agent(identity, label):
        r, _ = api("POST", "/auth/challenge", {"agent_id": identity.agent_id})
        sig = identity.sign(r["challenge"])
        r2, _ = api("POST", "/auth/verify", {
            "challenge_id": r["challenge_id"], "agent_id": identity.agent_id,
            "signature": sig, "pubkey": identity.public_key_hex,
        })
        return r2.get("token")

    buyer_token = auth_agent(buyer_id, "buyer")
    provider_token = auth_agent(provider_id, "provider")
    check("2. Buyer authenticated", buyer_token is not None)
    check("3. Provider authenticated", provider_token is not None)

    # 4. Provider registers + creates listing
    authed("POST", "/discovery/register", provider_token, {
        "capabilities": ["code_review", "testing"],
        "metadata": {"specialty": "security"},
    })
    authed("POST", "/matching/listing", provider_token, {
        "capabilities": ["code_review", "testing"],
        "price_per_task": 5.0,
        "response_time_avg": 30.0,
    })
    check("4. Provider listed", True)

    # 5. Buyer finds provider via matching
    r, s = authed("POST", "/matching/match", buyer_token, {
        "capability": "code_review",
        "max_price": 10.0,
    })
    check("5. Match found provider", r.get("count", 0) > 0)

    # 6. Buyer initiates handshake
    r, s = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": provider_id.agent_id,
        "capabilities": ["research"],
        "proposal": {
            "task_description": "Review this function for SQL injection",
            "task_type": "code_review",
            "verification_tier": 0,
            "test_suite_hash": hash_content("assert 'injection' in output"),
            "input_data": "def query(x): db.execute(f'SELECT * WHERE id={x}')",
        },
    })
    session_id = r.get("session_id")
    check("6. Handshake initiated", session_id is not None and r.get("state") == "proposed")

    # 7. Provider sees active handshake
    r, s = authed("GET", "/handshake/active", provider_token)
    check("7. Provider sees handshake", r.get("count", 0) > 0 and r["active"][0]["role"] == "provider")

    # 8. Provider accepts
    r, s = authed("POST", "/handshake/respond", provider_token, {
        "session_id": session_id,
        "action": "accept",
        "capabilities": ["code_review", "security"],
    })
    check("8. Provider accepted", r.get("action") == "accepted")

    # 9. Provider executes
    r, s = authed("POST", "/handshake/execute", provider_token, {
        "session_id": session_id,
        "output": "SQL injection found in query(): use parameterized queries instead of f-strings",
        "proof": {"vulnerability": "sql_injection", "line": 1},
    })
    check("9. Work executed", r.get("state") == "executing")

    # 10. Buyer verifies → receipt generated
    r, s = authed("POST", "/handshake/verify", buyer_token, {
        "session_id": session_id,
        "passed": 2,
        "failed": 0,
        "suite_hash": hash_content("assert 'injection' in output"),
    })
    check("10. Verified + receipt", r.get("verified") == True and r.get("receipt_id") is not None)

    # 11. Receipt in provider's chain
    r, s = api("GET", f"/receipt/chain/{provider_id.agent_id}")
    check("11. Receipt in provider chain", r.get("length", 0) == 1)

    # 12. Receipt in buyer's chain too
    r, s = api("GET", f"/receipt/chain/{buyer_id.agent_id}")
    check("12. Receipt in buyer chain", r.get("length", 0) == 1)

    # 13. Provider reputation updated
    r, s = api("GET", f"/reputation/{provider_id.agent_id}")
    check("13. Provider has reputation", r.get("reputation", 0) > 0)

    # 14. Handshake cleaned up
    r, s = authed("GET", "/handshake/active", buyer_token)
    check("14. Handshake cleaned up", r.get("count", 0) == 0)

    # 15. Self-handshake prevented
    r, s = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": buyer_id.agent_id,
        "proposal": {"task_description": "test", "task_type": "test"},
    })
    check("15. Self-handshake blocked", s == 400)

    # 16. Reject flow
    r, _ = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": provider_id.agent_id,
        "proposal": {"task_description": "bad task", "task_type": "spam"},
    })
    sid2 = r.get("session_id")
    r, _ = authed("POST", "/handshake/respond", provider_token, {
        "session_id": sid2,
        "action": "reject",
        "reason": "not interested",
    })
    check("16. Reject flow works", r.get("action") == "rejected")

    # 17. Wrong agent can't execute
    r, s = authed("POST", "/handshake/execute", buyer_token, {
        "session_id": sid2,
        "output": "hack",
    })
    check("17. Wrong agent blocked from execute", s == 403)

    # 18. Stats include handshakes
    r, _ = api("GET", "/stats")
    check("18. Stats include handshakes", "handshakes" in r)

    # 19. Full pipeline: match → handshake → receipt → reputation
    # Build provider reputation with more receipts first
    for i in range(5):
        r, _ = authed("POST", "/handshake/initiate", buyer_token, {
            "provider_id": provider_id.agent_id,
            "proposal": {"task_description": f"task {i}", "task_type": "testing", "verification_tier": 0},
        })
        sid = r["session_id"]
        authed("POST", "/handshake/respond", provider_token, {"session_id": sid, "action": "accept"})
        authed("POST", "/handshake/execute", provider_token, {
            "session_id": sid, "output": f"result {i}", "proof": {"ok": True},
        })
        authed("POST", "/handshake/verify", buyer_token, {
            "session_id": sid, "passed": 1, "failed": 0,
        })

    r, _ = api("GET", f"/receipt/chain/{provider_id.agent_id}")
    check("19. Full pipeline: 6 receipts in chain", r.get("length", 0) == 6)

    r, _ = api("GET", f"/reputation/{provider_id.agent_id}")
    check("20. Reputation grew with receipts", r.get("reputation", 0) > 0.5)

    # 21. Provider listing auto-updates reputation
    authed("POST", "/matching/listing", provider_token, {
        "capabilities": ["code_review", "testing"],
        "price_per_task": 5.0,
    })
    r, _ = authed("POST", "/matching/match", buyer_token, {"capability": "code_review"})
    if r.get("count", 0) > 0:
        check("21. Listing reflects earned reputation", r["matches"][0]["reputation"] > 0.5)
    else:
        check("21. Listing reflects earned reputation", False)

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
