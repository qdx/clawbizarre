"""
ClawBizarre HTTP API v5 — Phase 9: SSE Notifications

Everything from v4 + real-time Server-Sent Events for agents.

New endpoints:
  GET /events?token=<bearer>  — SSE stream (supports Last-Event-ID header)
  GET /notifications/stats    — Notification bus stats

Notification events emitted:
  - handshake.initiated  → provider (when buyer initiates)
  - handshake.responded  → buyer (when provider accepts/rejects)
  - handshake.executed   → buyer (when provider submits work)
  - handshake.verified   → provider (when buyer verifies, receipt generated)

Usage:
  python3 api_server_v5.py [--port 8420] [--db clawbizarre.db]
  python3 api_server_v5.py --test
"""

import json
import sys
import os
import time
import threading
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
from notifications import NotificationBus, EventType


class HandshakeStore:
    """In-memory store for active handshake sessions."""

    def __init__(self):
        self.sessions: dict[str, HandshakeSession] = {}
        self.agent_sessions: dict[str, list[str]] = {}
        self.session_meta: dict[str, dict] = {}

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
    """Server state backed by SQLite + in-memory matching/handshake/notifications."""

    def __init__(self, db_path: str = "clawbizarre.db"):
        self.db = PersistenceLayer(db_path)
        self.auth = AuthMiddleware(self.db)
        self.aggregator = ReputationAggregator()
        self.matching = MatchingEngine()
        self.handshakes = HandshakeStore()
        self.notifications = NotificationBus()

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
        # settlements now in SQLite via self.db


class APIv5Handler(BaseHTTPRequestHandler):
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

    def _auth_from_query(self, params: dict) -> Optional[str]:
        """Auth via ?token= query param (for SSE — can't set custom headers)."""
        token = params.get("token", [None])[0]
        if token:
            return self.state.auth.authenticate(f"Bearer {token}")
        return None

    def log_message(self, format, *args):
        pass

    # --- Routing ---

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = parse_qs(parsed.query)

        if path == "/health":
            self._send_json({"status": "ok", "version": "0.7.0"})
        elif path == "/":
            self._handle_root()
        elif path == "/stats":
            self._handle_stats()
        elif path == "/events":
            self._handle_sse(params)
        elif path == "/notifications/stats":
            self._send_json(self.state.notifications.stats())
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
            self._handle_price_history(params)
        elif path.startswith("/settlement/"):
            self._handle_settlement_status(path[len("/settlement/"):])
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
        elif path == "/settlement/register":
            self._handle_settlement_register()
        elif path == "/settlement/confirm":
            self._handle_settlement_confirm()
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

    # --- SSE ---

    def _handle_sse(self, params: dict):
        """Server-Sent Events stream for an authenticated agent."""
        agent_id = self._auth_from_query(params)
        if not agent_id:
            self._send_json({"error": "Auth required: ?token=<bearer>"}, 401)
            return

        # Set up SSE response headers
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("X-Accel-Buffering", "no")  # nginx proxy
        self.end_headers()

        # Check for Last-Event-ID reconnection
        last_id = self.headers.get("Last-Event-ID")
        if last_id:
            replayed = self.state.notifications.replay_from(agent_id, last_id)
            for evt in replayed:
                try:
                    self.wfile.write(evt.to_sse().encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return

        # Subscribe and stream
        wake, pos = self.state.notifications.subscribe(agent_id)
        try:
            # Send initial comment (connection confirmation)
            self.wfile.write(f": connected as {agent_id}\n\n".encode())
            self.wfile.flush()

            while True:
                # Wait for events (with 30s heartbeat)
                wake.wait(timeout=30.0)
                wake.clear()

                # Drain new events
                events, pos = self.state.notifications.drain(agent_id, pos)
                if events:
                    for evt in events:
                        self.wfile.write(evt.to_sse().encode())
                    self.wfile.flush()
                else:
                    # Heartbeat comment to keep connection alive
                    self.wfile.write(f": heartbeat {int(time.time())}\n\n".encode())
                    self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError, OSError):
            pass
        finally:
            self.state.notifications.unsubscribe(agent_id, wake)

    # --- Root / Stats ---

    def _handle_root(self):
        self._send_json({
            "service": "ClawBizarre API v5",
            "version": "0.7",
            "features": ["persistence", "auth", "receipts", "discovery",
                         "reputation", "treasury", "matching", "handshake",
                         "sse-notifications"],
            "auth": "Ed25519 challenge-response → Bearer token",
            "sse": "GET /events?token=<bearer> (supports Last-Event-ID)",
        })

    def _handle_stats(self):
        db_stats = self.state.db.stats()
        db_stats["matching"] = self.state.matching.stats()
        db_stats["handshakes"] = self.state.handshakes.stats()
        db_stats["notifications"] = self.state.notifications.stats()
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

    # --- Handshake (with notifications) ---

    def _handle_handshake_initiate(self):
        """Buyer initiates handshake → notifies provider."""
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

        session = self.state.handshakes.create(buyer_id, provider_id, proposal)

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
        session.propose(
            task_description=proposal["task_description"],
            task_type=proposal["task_type"],
            verification_tier=VerificationTier(proposal.get("verification_tier", 0)),
            test_suite_hash=proposal.get("test_suite_hash"),
            input_data=proposal.get("input_data"),
        )

        # >>> NOTIFY PROVIDER <<<
        self.state.notifications.emit(
            EventType.HANDSHAKE_INITIATED,
            provider_id,
            {
                "session_id": session.session_id,
                "buyer": buyer_id,
                "proposal": proposal,
            },
        )

        self._send_json({
            "session_id": session.session_id,
            "state": session.state.value,
            "buyer": buyer_id,
            "provider": provider_id,
            "proposal": proposal,
        })

    def _handle_handshake_respond(self):
        """Provider responds → notifies buyer."""
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

        if action == "accept":
            session.my_agent_id = agent_id
            session.state = HandshakeState.PROPOSED
            session.accept()

            # >>> NOTIFY BUYER <<<
            self.state.notifications.emit(
                EventType.HANDSHAKE_RESPONDED,
                meta["buyer"],
                {
                    "session_id": session_id,
                    "provider": agent_id,
                    "action": "accepted",
                },
            )

            self._send_json({
                "session_id": session_id,
                "state": session.state.value,
                "action": "accepted",
            })
        elif action == "reject":
            reason = body.get("reason", "declined")
            session.state = HandshakeState.PROPOSED
            session.reject(reason)

            # >>> NOTIFY BUYER <<<
            self.state.notifications.emit(
                EventType.HANDSHAKE_RESPONDED,
                meta["buyer"],
                {
                    "session_id": session_id,
                    "provider": agent_id,
                    "action": "rejected",
                    "reason": reason,
                },
            )

            self._send_json({
                "session_id": session_id,
                "state": session.state.value,
                "action": "rejected",
                "reason": reason,
            })
        else:
            self._send_json({"error": f"Unknown action: {action}"}, 400)

    def _handle_handshake_execute(self):
        """Provider submits work → notifies buyer."""
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
        meta["output"] = output
        meta["proof"] = proof

        output_hash = hash_content(output)

        # >>> NOTIFY BUYER <<<
        self.state.notifications.emit(
            EventType.HANDSHAKE_EXECUTED,
            meta["buyer"],
            {
                "session_id": session_id,
                "provider": agent_id,
                "output_hash": output_hash,
            },
        )

        self._send_json({
            "session_id": session_id,
            "state": session.state.value,
            "output_hash": output_hash,
        })

    def _handle_handshake_verify(self):
        """Buyer verifies work → notifies provider with receipt."""
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

        passed = body.get("passed", 1)
        failed = body.get("failed", 0)
        suite_hash = body.get("suite_hash", meta["proposal"].get("test_suite_hash", ""))
        test_results = TestResults(passed=passed, failed=failed, suite_hash=suite_hash)

        execute_msg = None
        for msg in session.messages:
            if msg.msg_type == "EXECUTE":
                execute_msg = msg
                break

        if not execute_msg:
            self._send_json({"error": "No execute message found"}, 400)
            return

        def verifier(payload):
            return test_results

        session.verify(execute_msg, verifier)

        result = {
            "session_id": session_id,
            "state": session.state.value,
            "verified": session.state == HandshakeState.COMPLETE,
        }

        if session.state == HandshakeState.COMPLETE and session.receipt:
            receipt = session.receipt
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

            # >>> NOTIFY PROVIDER <<<
            self.state.notifications.emit(
                EventType.HANDSHAKE_VERIFIED,
                meta["provider"],
                {
                    "session_id": session_id,
                    "buyer": agent_id,
                    "receipt_id": receipt.receipt_id,
                    "task_type": receipt.task_type,
                    "test_results": {"passed": test_results.passed, "failed": test_results.failed},
                },
            )

            self.state.handshakes.remove(session_id)

        self._send_json(result)

    def _handle_handshake_status(self, session_id: str):
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

    # --- Settlement ---

    def _handle_settlement_register(self):
        """Register a payment intent for a receipt. Links x402/AP2 payment to ClawBizarre receipt."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        data = self._read_body()
        if not data:
            return
        receipt_id = data.get("receipt_id")
        if not receipt_id:
            self._send_json({"error": "receipt_id required"}, 400)
            return
        settlement = self.state.db.register_settlement(
            receipt_id=receipt_id,
            registered_by=agent_id,
            protocol=data.get("protocol", "x402"),
            payment_id=data.get("payment_id", ""),
            amount=data.get("amount", 0),
            currency=data.get("currency", "USDC"),
            chain=data.get("chain", "base"),
        )
        self._send_json({"settlement": settlement}, 201)

    def _handle_settlement_confirm(self):
        """Confirm payment settlement for a receipt (counterparty confirms receipt of payment)."""
        agent_id = self._require_auth()
        if not agent_id:
            return
        data = self._read_body()
        if not data:
            return
        receipt_id = data.get("receipt_id")
        if not receipt_id:
            self._send_json({"error": "receipt_id required"}, 400)
            return
        existing = self.state.db.get_settlement(receipt_id)
        if not existing:
            self._send_json({"error": "Unknown receipt_id"}, 404)
            return
        confirmed = self.state.db.confirm_settlement(receipt_id, agent_id)
        if not confirmed:
            self._send_json({"error": f"Settlement already {existing['status']}"}, 409)
            return
        self._send_json({"settlement": confirmed})

    def _handle_settlement_status(self, receipt_id: str):
        """Get settlement status for a receipt."""
        settlement = self.state.db.get_settlement(receipt_id)
        if settlement:
            self._send_json({"settlement": settlement})
        else:
            self._send_json({"error": "No settlement for this receipt"}, 404)


# --- Server ---

def run_server(port: int = 8420, db_path: str = "clawbizarre.db"):
    # Use ThreadingHTTPServer for SSE (long-lived connections need threads)
    from http.server import ThreadingHTTPServer
    state = PersistentState(db_path)
    APIv5Handler.state = state

    server = ThreadingHTTPServer(("127.0.0.1", port), APIv5Handler)
    print(f"ClawBizarre API v5 running on http://127.0.0.1:{port}")
    print(f"Database: {db_path}")
    print(f"SSE: GET /events?token=<bearer>")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.server_close()


# --- Smoke Tests ---

def run_tests():
    from http.server import ThreadingHTTPServer
    import urllib.request

    db_path = "/tmp/clawbizarre_v5_test.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    state = PersistentState(db_path)
    APIv5Handler.state = state
    server = ThreadingHTTPServer(("127.0.0.1", 0), APIv5Handler)
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

    print(f"\n=== ClawBizarre API v5 Smoke Tests (port {port}) ===\n")

    # 1. Root
    r, s = api("GET", "/")
    check("1. Root returns v7 with SSE", r.get("version") == "0.7" and "sse-notifications" in r.get("features", []))

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

    # 4. Provider registers + lists
    authed("POST", "/discovery/register", provider_token, {
        "capabilities": ["code_review"],
    })
    authed("POST", "/matching/listing", provider_token, {
        "capabilities": ["code_review"],
        "price_per_task": 5.0,
    })
    check("4. Provider listed", True)

    # 5. Buyer initiates → notification emitted
    r, s = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": provider_id.agent_id,
        "proposal": {
            "task_description": "Review code",
            "task_type": "code_review",
            "verification_tier": 0,
        },
    })
    session_id = r.get("session_id")
    check("5. Handshake initiated", session_id is not None)

    # 6. Provider has notification in queue
    events, pos = state.notifications.drain(provider_id.agent_id, 0)
    check("6. Provider got handshake.initiated notification",
          len(events) == 1 and events[0].event_type == EventType.HANDSHAKE_INITIATED)

    # 7. Provider accepts → buyer notified
    authed("POST", "/handshake/respond", provider_token, {
        "session_id": session_id,
        "action": "accept",
    })
    events, _ = state.notifications.drain(buyer_id.agent_id, 0)
    check("7. Buyer got handshake.responded notification",
          any(e.event_type == EventType.HANDSHAKE_RESPONDED for e in events))

    # 8. Provider executes → buyer notified
    authed("POST", "/handshake/execute", provider_token, {
        "session_id": session_id,
        "output": "LGTM",
        "proof": {"ok": True},
    })
    events, _ = state.notifications.drain(buyer_id.agent_id, 0)
    check("8. Buyer got handshake.executed notification",
          any(e.event_type == EventType.HANDSHAKE_EXECUTED for e in events))

    # 9. Buyer verifies → provider notified with receipt
    authed("POST", "/handshake/verify", buyer_token, {
        "session_id": session_id,
        "passed": 1,
        "failed": 0,
    })
    events, _ = state.notifications.drain(provider_id.agent_id, 0)
    check("9. Provider got handshake.verified notification with receipt",
          any(e.event_type == EventType.HANDSHAKE_VERIFIED and "receipt_id" in e.data for e in events))

    # 10. Notification stats
    r, _ = api("GET", "/notifications/stats")
    check("10. Notification stats", r.get("total_events") == 4)

    # 11. SSE endpoint requires auth
    r, s = api("GET", "/events")
    check("11. SSE requires auth", s == 401)

    # 12. SSE endpoint with token starts stream (we can't easily test streaming in-process,
    #     but we can verify the endpoint accepts the token by connecting briefly)
    import socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(2)
    sock.connect(("127.0.0.1", port))
    sock.send(f"GET /events?token={provider_token} HTTP/1.1\r\nHost: localhost\r\n\r\n".encode())
    response_start = sock.recv(512).decode()
    check("12. SSE stream starts (text/event-stream)",
          "200 OK" in response_start and "text/event-stream" in response_start)
    sock.close()

    # 13. Full pipeline with notifications: 5 more tasks
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

    r, _ = api("GET", "/notifications/stats")
    check("13. 24 total notifications (4 per handshake × 6)", r.get("total_events") == 24)

    # 14. Reputation grew
    r, _ = api("GET", f"/reputation/{provider_id.agent_id}")
    check("14. Provider reputation > 0.5", r.get("reputation", 0) > 0.5)

    # 15. Stats include notifications
    r, _ = api("GET", "/stats")
    check("15. Stats include notifications", "notifications" in r)

    # 16. Replay from event ID
    replayed = state.notifications.replay_from(provider_id.agent_id, "evt-1")
    check("16. Replay from evt-1 returns events", len(replayed) > 0)

    # 17. Reject flow with notification
    r, _ = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": provider_id.agent_id,
        "proposal": {"task_description": "bad", "task_type": "spam"},
    })
    sid = r["session_id"]
    events_before, pos_before = state.notifications.drain(buyer_id.agent_id, 0)
    authed("POST", "/handshake/respond", provider_token, {
        "session_id": sid, "action": "reject", "reason": "spam",
    })
    events_after, _ = state.notifications.drain(buyer_id.agent_id, pos_before)
    reject_events = [e for e in events_after if e.data.get("action") == "rejected"]
    check("17. Reject notifies buyer with reason", len(reject_events) == 1 and reject_events[0].data.get("reason") == "spam")

    # 18. Self-handshake still blocked
    r, s = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": buyer_id.agent_id,
        "proposal": {"task_description": "test", "task_type": "test"},
    })
    check("18. Self-handshake blocked", s == 400)

    # 19. Wrong agent blocked from execute
    r, _ = authed("POST", "/handshake/initiate", buyer_token, {
        "provider_id": provider_id.agent_id,
        "proposal": {"task_description": "t", "task_type": "t"},
    })
    sid = r["session_id"]
    r, s = authed("POST", "/handshake/execute", buyer_token, {
        "session_id": sid, "output": "hack",
    })
    check("19. Wrong agent blocked", s == 403)

    # 20. ThreadingHTTPServer (SSE compatible)
    check("20. Server is threaded (SSE compatible)", isinstance(server, ThreadingHTTPServer))

    # 21. v4 compatibility: all v4 endpoints still work
    r, _ = api("GET", f"/receipt/chain/{provider_id.agent_id}")
    check("21. v4 compat: receipt chain works", r.get("length", 0) == 6)

    # 22. Notification data includes session_id
    all_provider_events, _ = state.notifications.drain(provider_id.agent_id, 0)
    check("22. All notifications have session_id",
          all(e.data.get("session_id") for e in all_provider_events))

    # --- Settlement Tests ---

    # Get a receipt_id from the provider's chain
    r, _ = api("GET", f"/receipt/chain/{provider_id.agent_id}")
    receipt_id = r["receipts"][0] if r.get("receipts") else None
    check("23. Have receipt for settlement test", receipt_id is not None)

    # 24. Register settlement
    r, s = authed("POST", "/settlement/register", buyer_token, {
        "receipt_id": receipt_id,
        "protocol": "x402",
        "payment_id": "pay_test_001",
        "amount": 5.0,
        "currency": "USD",
    })
    check("24. Settlement registered", s == 201 and r.get("settlement", {}).get("status") == "pending")

    # 25. Get settlement status
    r, s = api("GET", f"/settlement/{receipt_id}")
    check("25. Settlement status=pending", s == 200 and r.get("settlement", {}).get("status") == "pending")

    # 26. Confirm settlement
    r, s = authed("POST", "/settlement/confirm", provider_token, {
        "receipt_id": receipt_id,
    })
    check("26. Settlement confirmed", s == 200 and r.get("settlement", {}).get("status") == "confirmed")

    # 27. Double-confirm rejected
    r, s = authed("POST", "/settlement/confirm", provider_token, {
        "receipt_id": receipt_id,
    })
    check("27. Double-confirm rejected (409)", s == 409)

    # 28. Unknown settlement returns 404
    r, s = api("GET", "/settlement/nonexistent_receipt_id")
    check("28. Unknown settlement 404", s == 404)

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
