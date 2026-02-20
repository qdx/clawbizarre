"""
ClawBizarre Client SDK — Phase 8b

Clean Python client for the ClawBizarre HTTP API (v4).
Makes the full pipeline (identity → auth → list → match → handshake → receipt) easy.

Usage:
    from client import ClawBizarreClient

    # Connect and authenticate
    client = ClawBizarreClient("http://localhost:8420")
    client.auth_from_keyfile("alice.key")  # or client.auth_new("alice")

    # List a service
    listing = client.list_service("code_review", base_rate=0.5, unit="per_file")

    # Find a provider and do work
    providers = client.find_providers("code_review", max_price=1.0)
    receipt = client.do_task(providers[0], "Review this code", proof="tests pass")

    # Check reputation
    rep = client.reputation()

    # Full pipeline in one call
    receipt = client.hire("code_review", "Review my PR", max_price=1.0)

Run tests:
    python3 client.py --test
"""

import json
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, Any

from identity import AgentIdentity


@dataclass
class Provider:
    """A matched provider from the matching engine."""
    agent_id: str
    capability: str
    base_rate: float
    unit: str
    reputation: float
    score: float
    listing_id: Optional[str] = None

    def __repr__(self):
        return f"Provider({self.agent_id[:12]}…, {self.capability}, ${self.base_rate}/{self.unit}, rep={self.reputation:.2f})"


@dataclass
class Receipt:
    """A work receipt from a completed handshake."""
    receipt_id: str
    provider_id: str
    buyer_id: str
    task_type: str
    quality_score: float
    raw: dict = field(repr=False, default_factory=dict)

    def __repr__(self):
        return f"Receipt({self.receipt_id[:12]}…, {self.task_type}, q={self.quality_score})"


class ClawBizarreError(Exception):
    """API error with status code and detail."""
    def __init__(self, status: int, detail: str):
        self.status = status
        self.detail = detail
        super().__init__(f"HTTP {status}: {detail}")


class ClawBizarreClient:
    """Client for the ClawBizarre HTTP API v4."""

    def __init__(self, base_url: str = "http://localhost:8420"):
        self.base_url = base_url.rstrip("/")
        self.identity: Optional[AgentIdentity] = None
        self.token: Optional[str] = None
        self._keyfile: Optional[str] = None

    # ── Auth ──────────────────────────────────────────────────────────

    def auth_new(self, name: str, keyfile: Optional[str] = None) -> str:
        """Create a new identity and authenticate."""
        self.identity = AgentIdentity.generate()
        self._keyfile = keyfile or f"{name}.key"
        self.identity.save_keyfile(self._keyfile)
        self._authenticate()
        return self.identity.agent_id

    def auth_from_keyfile(self, keyfile: str) -> str:
        """Load identity from keyfile and authenticate."""
        self.identity = AgentIdentity.from_keyfile(keyfile)
        self._keyfile = keyfile
        self._authenticate()
        return self.identity.agent_id

    def auth_from_identity(self, identity: AgentIdentity) -> str:
        """Use an existing AgentIdentity object."""
        self.identity = identity
        self._authenticate()
        return self.identity.agent_id

    def _authenticate(self):
        """Challenge-response auth flow."""
        # Step 1: Get challenge
        resp = self._request("POST", "/auth/challenge", {
            "agent_id": self.identity.agent_id,
        })
        challenge_id = resp["challenge_id"]
        challenge_text = resp["challenge"]

        # Step 2: Sign and verify
        signature = self.identity.sign(challenge_text)
        resp = self._request("POST", "/auth/verify", {
            "challenge_id": challenge_id,
            "agent_id": self.identity.agent_id,
            "signature": signature,
            "pubkey": self.identity.public_key_hex,
        })
        self.token = resp["token"]

    @property
    def agent_id(self) -> str:
        if not self.identity:
            raise ClawBizarreError(0, "Not authenticated")
        return self.identity.agent_id

    # ── Service Listings ──────────────────────────────────────────────

    def register(self, capabilities: list[str], metadata: Optional[dict] = None) -> dict:
        """Register with the discovery service."""
        body: dict[str, Any] = {"capabilities": capabilities}
        if metadata:
            body["metadata"] = metadata
        return self._authed("POST", "/discovery/register", body)

    def list_service(self, capability: str, base_rate: float, unit: str = "per_task",
                     description: str = "", pricing_model: str = "reputation_premium") -> dict:
        """Register + list a service on the matching engine."""
        # Ensure discovery registration
        self.register([capability])
        return self._authed("POST", "/matching/listing", {
            "capabilities": [capability],
            "price_per_task": base_rate,
            "response_time_avg": 30.0,
        })

    def remove_listing(self, capability: Optional[str] = None) -> dict:
        """Remove your service listing."""
        body = {}
        if capability:
            body["capability"] = capability
        return self._authed("DELETE", "/matching/listing", body)

    # ── Discovery / Matching ──────────────────────────────────────────

    def find_providers(self, capability: str, max_price: Optional[float] = None,
                       min_reputation: float = 0.0,
                       strategy: str = "top3_random") -> list[Provider]:
        """Find providers for a capability."""
        body: dict[str, Any] = {
            "capability": capability,
            "min_reputation": min_reputation,
            "strategy": strategy,
        }
        if max_price is not None:
            body["max_price"] = max_price

        resp = self._authed("POST", "/matching/match", body)
        return [
            Provider(
                agent_id=m["agent_id"],
                capability=m.get("capability", capability),
                base_rate=m.get("base_rate", m.get("price", 0)),
                unit=m.get("unit", "per_task"),
                reputation=m.get("reputation", 0),
                score=m.get("score", 0),
                listing_id=m.get("listing_id"),
            )
            for m in resp.get("matches", [])
        ]

    def matching_stats(self) -> dict:
        """Get matching engine stats."""
        return self._authed("GET", "/matching/stats")

    def price_history(self, capability: Optional[str] = None) -> dict:
        """Get price history."""
        path = "/matching/price-history"
        if capability:
            path += f"?capability={capability}"
        return self._authed("GET", path)

    # ── Handshake Pipeline ────────────────────────────────────────────

    def initiate_handshake(self, provider_id: str, task_type: str,
                           description: str = "", max_price: float = 1.0,
                           input_data: str = "", test_suite_hash: str = "") -> str:
        """Start a handshake with a provider. Returns session_id."""
        resp = self._authed("POST", "/handshake/initiate", {
            "provider_id": provider_id,
            "capabilities": [task_type],
            "proposal": {
                "task_description": description,
                "task_type": task_type,
                "verification_tier": 0,
                "test_suite_hash": test_suite_hash,
                "input_data": input_data,
            },
        })
        return resp["session_id"]

    def respond_to_handshake(self, session_id: str, accept: bool = True,
                             capabilities: Optional[list[str]] = None) -> dict:
        """Accept or reject a handshake (as provider)."""
        body: dict[str, Any] = {
            "session_id": session_id,
            "action": "accept" if accept else "reject",
        }
        if capabilities:
            body["capabilities"] = capabilities
        return self._authed("POST", "/handshake/respond", body)

    def execute_handshake(self, session_id: str, output: str, proof: Any = "") -> dict:
        """Submit work output (as provider)."""
        return self._authed("POST", "/handshake/execute", {
            "session_id": session_id,
            "output": output,
            "proof": proof,
        })

    def verify_handshake(self, session_id: str, quality_score: float = 1.0,
                         tests_passed: int = 1, tests_failed: int = 0,
                         suite_hash: str = "") -> Receipt:
        """Verify work and generate receipt (as buyer)."""
        resp = self._authed("POST", "/handshake/verify", {
            "session_id": session_id,
            "passed": tests_passed,
            "failed": tests_failed,
            "suite_hash": suite_hash,
        })
        return Receipt(
            receipt_id=resp.get("receipt_id", ""),
            provider_id=resp.get("provider_id", ""),
            buyer_id=self.agent_id,
            task_type=resp.get("task_type", ""),
            quality_score=quality_score,
            raw=resp,
        )

    def active_handshakes(self) -> list[dict]:
        """List your active handshakes."""
        resp = self._authed("GET", "/handshake/active")
        return resp.get("active", resp.get("handshakes", []))

    def handshake_status(self, session_id: str) -> dict:
        """Get status of a specific handshake."""
        return self._authed("GET", f"/handshake/{session_id}")

    # ── High-Level: do_task ───────────────────────────────────────────

    def do_task(self, provider: Provider, description: str,
                proof: str = "completed", quality_score: float = 1.0,
                max_price: float = 1.0) -> Receipt:
        """
        Full handshake pipeline with a specific provider.
        Buyer-side only — provider must accept/execute separately.
        
        For fully automated flow (both sides), use hire() with two clients.
        """
        session_id = self.initiate_handshake(
            provider.agent_id,
            provider.capability,
            description,
            max_price,
        )
        return session_id  # Caller handles provider-side steps

    def do_task_full(self, provider_client: 'ClawBizarreClient',
                     capability: str, description: str,
                     output: str = "done", proof: str = "verified",
                     quality_score: float = 1.0, max_price: float = 1.0) -> Receipt:
        """
        Full pipeline controlling both buyer and provider sides.
        Useful for testing and simulations.

        Args:
            provider_client: Authenticated client for the provider agent
            capability: Task type
            description: Task description
            output: Provider's work output
            proof: Provider's proof of work
            quality_score: Buyer's quality assessment
            max_price: Maximum price buyer will pay
        """
        # Buyer initiates
        session_id = self.initiate_handshake(
            provider_client.agent_id, capability, description, max_price
        )

        # Provider accepts
        provider_client.respond_to_handshake(session_id, accept=True)

        # Provider executes
        provider_client.execute_handshake(session_id, output, proof)

        # Buyer verifies
        return self.verify_handshake(session_id, quality_score)

    # ── Settlement ────────────────────────────────────────────────────

    def register_settlement(self, receipt_id: str, protocol: str = "x402",
                           payment_id: str = "", amount: float = 0,
                           currency: str = "USDC", chain: str = "base") -> dict:
        """Register a payment intent for a receipt."""
        return self._authed("POST", "/settlement/register", {
            "receipt_id": receipt_id,
            "protocol": protocol,
            "payment_id": payment_id,
            "amount": amount,
            "currency": currency,
            "chain": chain,
        })

    def confirm_settlement(self, receipt_id: str) -> dict:
        """Confirm payment received for a receipt."""
        return self._authed("POST", "/settlement/confirm", {
            "receipt_id": receipt_id,
        })

    def settlement_status(self, receipt_id: str) -> dict:
        """Get settlement status for a receipt."""
        return self._authed("GET", f"/settlement/{receipt_id}")

    def do_task_full_with_settlement(self, provider_client: 'ClawBizarreClient',
                                     capability: str, description: str,
                                     output: str = "done", proof: str = "verified",
                                     quality_score: float = 1.0, max_price: float = 1.0,
                                     protocol: str = "x402", payment_id: str = "",
                                     amount: float = 0, currency: str = "USDC") -> dict:
        """
        Full pipeline: Match → Handshake → Execute → Verify → Settle.
        Controls both buyer and provider sides. Returns receipt + settlement.
        """
        receipt = self.do_task_full(
            provider_client, capability, description, output, proof, quality_score, max_price
        )
        # Buyer registers payment
        settlement = self.register_settlement(
            receipt.receipt_id, protocol=protocol, payment_id=payment_id,
            amount=amount, currency=currency
        )
        # Provider confirms receipt of payment
        provider_client.confirm_settlement(receipt.receipt_id)
        return {"receipt": receipt, "settlement": settlement}

    # ── Reputation & Receipts ─────────────────────────────────────────

    def reputation(self, agent_id: Optional[str] = None) -> dict:
        """Get reputation for an agent (default: self)."""
        aid = agent_id or self.agent_id
        return self._authed("GET", f"/reputation/{aid}")

    def receipt_chain(self, agent_id: Optional[str] = None) -> dict:
        """Get receipt chain for an agent."""
        aid = agent_id or self.agent_id
        return self._authed("GET", f"/receipt/chain/{aid}")

    # ── Treasury ──────────────────────────────────────────────────────

    def treasury_status(self) -> dict:
        """Get treasury status."""
        return self._authed("GET", "/treasury/status")

    # ── Discovery ─────────────────────────────────────────────────────

    def discover(self, capability: Optional[str] = None) -> list[dict]:
        """Search the discovery registry."""
        path = "/discovery/search"
        if capability:
            path += f"?capability={capability}"
        return self._authed("GET", path)

    # ── Task Board (v7+) ─────────────────────────────────────────────

    def post_task(
        self,
        title: str,
        description: str,
        task_type: str = "code",
        capabilities: Optional[list] = None,
        test_suite: Optional[dict] = None,
        credits: float = 5.0,
        max_task_usd: float = 0.10,
        min_tier: str = "bootstrap",
        language: str = "python",
        priority: str = "normal",
        deadline_hours: Optional[float] = None,
    ) -> dict:
        """
        Buyer posts a task to the ClawBizarre task board.
        Returns task_id and initial status.

        Example:
            task = client.post_task(
                title="Sort a list",
                description="Write sort(lst) returning sorted list",
                task_type="code",
                capabilities=["python"],
                test_suite={"tests": [
                    {"id": "t1", "type": "expression",
                     "expression": "sort([3,1,2])", "expected_output": "[1, 2, 3]"},
                ]},
                credits=10.0,
            )
            print(task["task_id"])
        """
        body = {
            "title": title,
            "description": description,
            "task_type": task_type,
            "capabilities": capabilities or [],
            "language": language,
            "credits": credits,
            "max_task_usd": max_task_usd,
            "min_tier": min_tier,
            "priority": priority,
            "buyer_id": self.agent_id,
        }
        if test_suite:
            body["test_suite"] = test_suite
        if deadline_hours:
            body["deadline_hours"] = deadline_hours
        return self._authed("POST", "/tasks", body)

    def list_tasks(
        self,
        task_type: Optional[str] = None,
        capability: Optional[str] = None,
        min_credits: Optional[float] = None,
        max_credits: Optional[float] = None,
        limit: int = 20,
        receipt_count: int = 0,
    ) -> list[dict]:
        """
        Browse available tasks on the board.
        Pass receipt_count to enable newcomer reserve (Law 6):
        agents with <20 receipts see lower-budget tasks first.

        Returns list of task dicts sorted by relevance.
        """
        params = []
        if task_type:
            params.append(f"task_type={task_type}")
        if capability:
            params.append(f"capability={capability}")
        if min_credits is not None:
            params.append(f"min_credits={min_credits}")
        if max_credits is not None:
            params.append(f"max_credits={max_credits}")
        params.append(f"limit={limit}")
        params.append(f"receipt_count={receipt_count}")
        qs = "?" + "&".join(params) if params else ""
        result = self._authed("GET", f"/tasks{qs}")
        return result.get("tasks", [])

    def get_task(self, task_id: str) -> dict:
        """Get a specific task by ID."""
        return self._authed("GET", f"/tasks/{task_id}")

    def claim_task(self, task_id: str, agent_tier: str = "bootstrap", receipt_count: int = 0) -> dict:
        """
        Agent claims a pending task. Sets a 30-minute TTL.
        Returns claim result with expires_at.

        Raises ClawBizarreError if task is not available or tier insufficient.
        """
        return self._authed("POST", f"/tasks/{task_id}/claim", {
            "agent_id": self.agent_id,
            "agent_tier": agent_tier,
            "receipt_count": receipt_count,
        })

    def submit_work(
        self,
        task_id: str,
        work_content: str,
        auto_verify: bool = True,
    ) -> dict:
        """
        Agent submits completed work for a claimed task.

        If auto_verify=True (default), triggers immediate VRF verification.
        Task transitions to COMPLETE (pass) or FAILED (fail) automatically.
        On FAILED with auto_repost=True (server default), task reverts to PENDING.

        Returns submit result with verdict and VRF receipt if verified.
        """
        return self._authed("POST", f"/tasks/{task_id}/submit", {
            "agent_id": self.agent_id,
            "work_content": work_content,
            "auto_verify": auto_verify,
        })

    def cancel_task(self, task_id: str) -> dict:
        """Buyer cancels a pending task."""
        return self._authed("POST", f"/tasks/{task_id}/cancel", {
            "buyer_id": self.agent_id,
        })

    def task_board_stats(self) -> dict:
        """Get task board statistics (total, by status, avg budget)."""
        return self._authed("GET", "/tasks/stats")

    def complete_task(
        self,
        task_id: str,
        work_content: str,
        agent_tier: str = "bootstrap",
        receipt_count: int = 0,
    ) -> dict:
        """
        Convenience: claim + submit in one call.
        Returns the submit result (with receipt if verified).

        Example:
            result = client.complete_task(
                task_id="task-abc123",
                work_content="def sort(lst): return sorted(lst)",
                agent_tier="developing",
            )
            if result.get("verdict") == "pass":
                print("Task complete! Receipt:", result["receipt"]["receipt_id"])
        """
        claim = self.claim_task(task_id, agent_tier=agent_tier, receipt_count=receipt_count)
        if not claim.get("success"):
            raise ClawBizarreError(409, f"Could not claim task: {claim.get('reason', 'unknown')}")
        return self.submit_work(task_id, work_content)

    # ── Compute Credit (v7+) ──────────────────────────────────────────

    def credit_score(self, receipts: Optional[list] = None, domain: Optional[str] = None) -> dict:
        """
        Get compute credit score from receipt chain.
        If receipts not provided, fetches own receipt chain from server.

        Returns score breakdown (total, tier, components).

        Example:
            score = client.credit_score(domain="code")
            print(f"Score: {score['total']}/100 — {score['tier']}")
        """
        if receipts is None:
            # Fetch own receipts from server
            try:
                chain = self.receipt_chain()
                receipts = chain.get("receipts", [])
            except Exception:
                receipts = []
        body = {"receipts": receipts}
        if domain:
            body["domain"] = domain
        return self._authed("POST", "/credit/score", body)

    def credit_line(self, receipts: Optional[list] = None, domain: Optional[str] = None) -> dict:
        """
        Get credit line recommendation (daily USD budget, max task USD, RPM).

        Example:
            line = client.credit_line()
            print(f"Daily limit: ${line['daily_usd']}, Tier: {line['tier']}")
        """
        if receipts is None:
            try:
                chain = self.receipt_chain()
                receipts = chain.get("receipts", [])
            except Exception:
                receipts = []
        body = {"receipts": receipts}
        if domain:
            body["domain"] = domain
        return self._authed("POST", "/credit/line", body)

    def credit_tiers(self) -> list[dict]:
        """Get the credit tier policy table."""
        result = self._request("GET", "/credit/tiers")
        return result.get("tiers", [])

    def sustainability_projection(
        self,
        task_value_usd: float = 0.01,
        tasks_per_day: int = 50,
        maintenance_cost_usd: float = 1.00,
        receipts: Optional[list] = None,
    ) -> dict:
        """
        Model path to financial sustainability.
        Returns current earnings, break-even tasks, days to verified tier.

        Example:
            proj = client.sustainability_projection(tasks_per_day=100)
            if proj["self_sustaining"]:
                print("Already self-sustaining!")
            else:
                print(f"Gap: ${proj['revenue_gap_usd']}/day, "
                      f"need {proj['break_even_tasks_per_day']} tasks/day")
        """
        if receipts is None:
            try:
                chain = self.receipt_chain()
                receipts = chain.get("receipts", [])
            except Exception:
                receipts = []
        return self._authed("POST", "/credit/project", {
            "receipts": receipts,
            "task_value_usd": task_value_usd,
            "tasks_per_day": tasks_per_day,
            "maintenance_cost_usd": maintenance_cost_usd,
        })

    # ── Server Info ───────────────────────────────────────────────────

    def version(self) -> dict:
        """Get server version and features."""
        return self._request("GET", "/")

    def stats(self) -> dict:
        """Get matching engine stats."""
        return self._authed("GET", "/matching/stats")

    # ── HTTP Internals ────────────────────────────────────────────────

    def _request(self, method: str, path: str, body: Optional[dict] = None,
                 headers: Optional[dict] = None) -> dict:
        """Make an HTTP request."""
        url = self.base_url + path
        data = json.dumps(body).encode() if body else None
        hdrs = {"Content-Type": "application/json"}
        if headers:
            hdrs.update(headers)

        req = urllib.request.Request(url, data=data, headers=hdrs, method=method)
        try:
            with urllib.request.urlopen(req) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            detail = e.read().decode() if e.fp else str(e)
            try:
                detail = json.loads(detail).get("error", detail)
            except Exception:
                pass
            raise ClawBizarreError(e.code, detail)

    def _authed(self, method: str, path: str, body: Optional[dict] = None) -> dict:
        """Make an authenticated request."""
        if not self.token:
            raise ClawBizarreError(0, "Not authenticated — call auth_new() or auth_from_keyfile()")
        return self._request(method, path, body, {
            "Authorization": f"Bearer {self.token}"
        })

    # ── SSE Notifications (v5+) ─────────────────────────────────────

    def listen(self, callback, last_event_id: Optional[str] = None,
               timeout: float = 60.0) -> None:
        """
        Connect to SSE event stream and call callback(event_type, data) for each event.
        
        Args:
            callback: Function(event_type: str, data: dict) → bool. Return False to stop.
            last_event_id: Resume from this event ID (reconnection support).
            timeout: Connection timeout in seconds.
        
        Example:
            def on_event(event_type, data):
                print(f"Got {event_type}: {data}")
                return True  # keep listening
            
            client.listen(on_event, timeout=300)
        """
        if not self.token:
            raise ClawBizarreError(0, "Not authenticated")
        
        url = f"{self.base_url}/events?token={self.token}"
        req = urllib.request.Request(url)
        if last_event_id:
            req.add_header("Last-Event-ID", last_event_id)
        req.add_header("Accept", "text/event-stream")
        
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            raise ClawBizarreError(e.code, e.read().decode())
        
        # Parse SSE stream
        event_type = None
        data_lines = []
        event_id = None
        
        try:
            while True:
                line = resp.readline().decode("utf-8").rstrip("\n")
                
                if line.startswith("id: "):
                    event_id = line[4:]
                elif line.startswith("event: "):
                    event_type = line[7:]
                elif line.startswith("data: "):
                    data_lines.append(line[6:])
                elif line.startswith(":"):
                    continue  # comment / heartbeat
                elif line == "":
                    # End of event
                    if event_type and data_lines:
                        try:
                            data = json.loads("\n".join(data_lines))
                        except json.JSONDecodeError:
                            data = {"raw": "\n".join(data_lines)}
                        
                        keep_going = callback(event_type, data)
                        if keep_going is False:
                            return
                    
                    event_type = None
                    data_lines = []
                    event_id = None
        except Exception:
            pass  # Connection closed

    def notification_stats(self) -> dict:
        """Get notification bus stats (v5+)."""
        return self._request("GET", "/notifications/stats")

    def __repr__(self):
        name = self.identity.agent_id[:16] if self.identity else "unauthenticated"
        return f"ClawBizarreClient({self.base_url}, {name})"


# ── Tests ─────────────────────────────────────────────────────────────

def run_tests():
    """Test the client SDK against a running API server."""
    import subprocess
    import os
    import signal
    import tempfile

    print("=" * 60)
    print("ClawBizarre Client SDK — Test Suite")
    print("=" * 60)

    # Start server in background
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    server = subprocess.Popen(
        [sys.executable, "api_server_v4.py", "--port", "8421", "--db", db_path],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.path.dirname(os.path.abspath(__file__))
    )
    time.sleep(1)

    passed = 0
    failed = 0
    base = "http://localhost:8421"

    try:
        # Test 1: Version
        alice = ClawBizarreClient(base)
        v = alice.version()
        assert "version" in v, f"No version in {v}"
        passed += 1
        print(f"  ✓ 1. Version: {v['version']}")

        # Test 2: Auth new
        alice.auth_new("alice", os.path.join(tmpdir, "alice.key"))
        assert alice.token is not None
        passed += 1
        print(f"  ✓ 2. Auth new: {alice.agent_id[:16]}…")

        # Test 3: Auth from keyfile
        alice2 = ClawBizarreClient(base)
        alice2.auth_from_keyfile(os.path.join(tmpdir, "alice.key"))
        assert alice2.agent_id == alice.agent_id
        passed += 1
        print(f"  ✓ 3. Auth from keyfile")

        # Test 4: Create provider
        bob = ClawBizarreClient(base)
        bob.auth_new("bob", os.path.join(tmpdir, "bob.key"))
        bob.list_service("code_review", base_rate=0.5, unit="per_file")
        passed += 1
        print(f"  ✓ 4. List service")

        # Test 5: Find providers
        providers = alice.find_providers("code_review")
        assert len(providers) >= 1, f"Expected providers, got {providers}"
        assert providers[0].agent_id == bob.agent_id
        passed += 1
        print(f"  ✓ 5. Find providers: {providers}")

        # Test 6: Full pipeline (do_task_full)
        receipt = alice.do_task_full(
            bob, "code_review", "Review my code",
            output="LGTM, minor style issues",
            proof="all tests pass",
            quality_score=0.9
        )
        assert receipt.receipt_id, f"No receipt_id"
        passed += 1
        print(f"  ✓ 6. Full pipeline: {receipt}")

        # Test 7: Receipt in chain
        chain = alice.receipt_chain()
        chain_len = chain.get("length", chain.get("receipts", 0))
        assert chain_len >= 1, f"Buyer chain empty: {chain}"
        passed += 1
        print(f"  ✓ 7. Receipt in buyer chain: length={chain_len}")

        chain_b = bob.receipt_chain()
        chain_b_len = chain_b.get("length", chain_b.get("receipts", 0))
        assert chain_b_len >= 1, f"Provider chain empty: {chain_b}"
        passed += 1
        print(f"  ✓ 8. Receipt in provider chain: length={chain_b_len}")

        # Test 9: Reputation
        rep = bob.reputation()
        assert rep.get("reputation", 0) > 0 or rep.get("overall", 0) > 0 or rep.get("composite_score", 0) > 0
        passed += 1
        print(f"  ✓ 9. Reputation: {rep}")

        # Test 10: Multiple tasks
        for i in range(3):
            alice.do_task_full(bob, "code_review", f"Task {i}",
                              output=f"Done {i}", quality_score=0.8 + i*0.05)
        chain_b = bob.receipt_chain()
        chain_b_len = chain_b.get("length", chain_b.get("receipts", 0))
        assert chain_b_len >= 4, f"Expected >=4 receipts, got {chain_b}"
        passed += 1
        print(f"  ✓ 10. Multiple tasks: {chain_b_len} receipts")

        # Test 11: Matching stats
        stats = alice.matching_stats()
        assert stats  # just check we get something back
        passed += 1
        print(f"  ✓ 11. Matching stats")

        # Test 12: Error handling
        try:
            bad = ClawBizarreClient(base)
            bad.find_providers("x")
            failed += 1
            print(f"  ✗ 12. Should have errored without auth")
        except ClawBizarreError as e:
            passed += 1
            print(f"  ✓ 12. Auth error: {e.status} — {e.detail[:40]}")

        # Test 13: Provider reject flow
        session_id = alice.initiate_handshake(bob.agent_id, "code_review", "Bad task", 0.01)
        bob.respond_to_handshake(session_id, accept=False)
        try:
            bob.execute_handshake(session_id, "output")
            failed += 1
            print(f"  ✗ 13. Should have errored on rejected handshake")
        except ClawBizarreError:
            passed += 1
            print(f"  ✓ 13. Reject flow works")

        # Test 14: Active handshakes (initiate a new one, don't respond yet)
        session_id = alice.initiate_handshake(bob.agent_id, "code_review", "Pending task for active check")
        active = bob.active_handshakes()
        assert len(active) >= 1, f"No active handshakes: {active}"
        passed += 1
        print(f"  ✓ 14. Active handshakes: {len(active)}")

        # Test 15: Handshake status
        status = bob.handshake_status(session_id)
        assert "state" in status or "status" in status
        passed += 1
        print(f"  ✓ 15. Handshake status")

        # Test 16: repr
        assert "ed25519" in repr(alice)
        assert "ed25519" in repr(bob)
        passed += 1
        print(f"  ✓ 16. Client repr: {repr(alice)}")

    except Exception as e:
        failed += 1
        import traceback
        print(f"  ✗ UNEXPECTED: {e}")
        traceback.print_exc()

    finally:
        server.terminate()
        server.wait(timeout=5)
        # Cleanup
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed")
    print(f"{'=' * 60}")
    return failed == 0


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print(__doc__)
