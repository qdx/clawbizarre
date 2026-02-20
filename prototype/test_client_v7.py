"""
test_client_v7.py — Tests for v7 client extensions.

Tests the task board + credit client methods added in Phase 30.
Uses a mock HTTP server to avoid needing a running api_server_v7 instance.
"""

import sys
import os
import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from client import ClawBizarreClient, ClawBizarreError


# ── Mock v7 Server ─────────────────────────────────────────────────────────────

class MockV7Handler(BaseHTTPRequestHandler):
    """
    Minimal mock HTTP server implementing v7 endpoints for testing.
    Returns realistic-looking responses without real logic.
    """

    # In-memory task store for mock
    _tasks = {}
    _task_counter = 0

    @classmethod
    def _next_id(cls) -> str:
        cls._task_counter += 1
        return f"task-mock-{cls._task_counter:04d}"

    def do_GET(self):
        path = self.path.split("?")[0]

        if path == "/health":
            self._json({"status": "ok", "version": "0.9.0"})
        elif path in ("/tasks", "/tasks/"):
            tasks = list(self.__class__._tasks.values())
            self._json({"tasks": tasks, "count": len(tasks)})
        elif path == "/tasks/stats":
            by_status = {}
            for t in self.__class__._tasks.values():
                s = t["status"]
                by_status[s] = by_status.get(s, 0) + 1
            self._json({"total_tasks": len(self.__class__._tasks), "by_status": by_status})
        elif path.startswith("/tasks/"):
            task_id = path[len("/tasks/"):]
            task = self.__class__._tasks.get(task_id)
            if task:
                self._json(task)
            else:
                self._error(404, f"Task {task_id} not found")
        elif path == "/credit/tiers":
            self._json({"tiers": [
                {"name": "verified", "min_score": 80, "max_score": 100, "daily_usd": 10.0, "rpm": 1000},
                {"name": "established", "min_score": 60, "max_score": 80, "daily_usd": 5.0, "rpm": 500},
                {"name": "developing", "min_score": 40, "max_score": 60, "daily_usd": 2.0, "rpm": 200},
                {"name": "new", "min_score": 20, "max_score": 40, "daily_usd": 0.5, "rpm": 50},
                {"name": "bootstrap", "min_score": 0, "max_score": 20, "daily_usd": 0.1, "rpm": 10},
            ]})
        elif path.startswith("/receipt/chain/"):
            self._json({"receipts": [], "length": 0})
        else:
            self._error(404, f"Not found: {path}")

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(content_length)) if content_length > 0 else {}
        path = self.path.split("?")[0]

        # Auth endpoints (minimal mock)
        if path == "/auth/new":
            self._json({"token": "mock-token-" + body.get("name", "anon"), "agent_id": f"agent:{body.get('name', 'anon')}"})
        elif path == "/auth/verify":
            self._json({"agent_id": "agent:tester", "valid": True})

        elif path in ("/tasks", "/tasks/"):
            task_id = self.__class__._next_id()
            task = {
                "task_id": task_id,
                "title": body.get("title", "Untitled"),
                "description": body.get("description", ""),
                "status": "pending",
                "buyer_id": body.get("buyer_id", "anon"),
                "requirements": {
                    "task_type": body.get("task_type", "code"),
                    "capabilities": body.get("capabilities", []),
                    "language": body.get("language", "python"),
                    "test_suite": body.get("test_suite"),
                    "schema": body.get("schema"),
                    "min_tier": body.get("min_tier", "bootstrap"),
                    "verification_tier": body.get("verification_tier", 0),
                },
                "budget": {
                    "credits": body.get("credits", 1.0),
                    "max_task_usd": body.get("max_task_usd", 0.10),
                    "payment_protocol": "credits",
                    "escrow": False,
                    "refund_on_failure": True,
                },
                "priority": body.get("priority", "normal"),
                "posted_at": datetime.now(timezone.utc).isoformat(),
                "deadline_at": None,
                "claimed_by": None,
                "claimed_at": None,
                "claim_expires_at": None,
                "submitted_at": None,
                "completed_at": None,
                "work_content": None,
                "receipt": None,
                "failure_reason": None,
                "re_post_count": 0,
            }
            self.__class__._tasks[task_id] = task
            self._json({"task_id": task_id, "status": "pending", "task": task})

        elif path.endswith("/claim"):
            task_id = path.split("/")[-2]
            task = self.__class__._tasks.get(task_id)
            if not task:
                self._error(404, "Task not found")
                return
            if task["status"] != "pending":
                self._json({"success": False, "task_id": task_id, "agent_id": body.get("agent_id"), "reason": f"Task is {task['status']}", "expires_at": None})
                return
            agent_id = body.get("agent_id", "anon")
            expires = datetime.now(timezone.utc).replace(minute=datetime.now().minute + 30).isoformat()
            task["status"] = "claimed"
            task["claimed_by"] = agent_id
            task["claimed_at"] = datetime.now(timezone.utc).isoformat()
            task["claim_expires_at"] = expires
            self._json({"success": True, "task_id": task_id, "agent_id": agent_id, "expires_at": expires, "reason": None})

        elif path.endswith("/submit"):
            task_id = path.split("/")[-2]
            task = self.__class__._tasks.get(task_id)
            if not task:
                self._error(404, "Task not found")
                return
            work_content = body.get("work_content", "")
            task["work_content"] = work_content
            task["submitted_at"] = datetime.now(timezone.utc).isoformat()
            receipt = {
                "receipt_id": f"rcpt-mock-{task_id}",
                "verdict": "pass",
                "verified_at": datetime.now(timezone.utc).isoformat(),
                "results": {"total": 2, "passed": 2, "failed": 0, "errors": 0},
                "vrf_version": "1.0",
            }
            task["status"] = "complete"
            task["completed_at"] = datetime.now(timezone.utc).isoformat()
            task["receipt"] = receipt
            self._json({"success": True, "task_id": task_id, "status": "complete",
                        "verdict": "pass", "receipt": receipt, "reason": None})

        elif path.endswith("/cancel"):
            task_id = path.split("/")[-2]
            task = self.__class__._tasks.get(task_id)
            if not task:
                self._error(404, "Task not found")
                return
            if task.get("buyer_id") != body.get("buyer_id"):
                self._json({"success": False, "reason": "Only buyer can cancel"})
                return
            task["status"] = "cancelled"
            self._json({"success": True, "task_id": task_id, "status": "cancelled"})

        elif path == "/credit/score":
            receipts = body.get("receipts", [])
            total = min(100, len(receipts) * 5 + 20)  # Mock score
            self._json({"total": total, "tier": "new" if total < 40 else "developing",
                        "components": {"volume": min(25, len(receipts) * 0.5),
                                        "quality": 30.0, "consistency": 15.0,
                                        "recency": 8.0, "diversity": 2.0}})

        elif path == "/credit/line":
            receipts = body.get("receipts", [])
            total = min(100, len(receipts) * 5 + 20)
            tier = "developing" if total >= 40 else "new"
            self._json({"score": total, "tier": tier, "daily_usd": 2.0 if tier == "developing" else 0.5,
                        "max_task_usd": 0.50, "requests_per_day": 200,
                        "justification": "Mock justification.", "sponsor_required": True,
                        "breakdown": {"total": total, "tier": tier, "components": {}}})

        elif path == "/credit/project":
            tasks_per_day = body.get("tasks_per_day", 50)
            task_value = body.get("task_value_usd", 0.01)
            maintenance = body.get("maintenance_cost_usd", 1.00)
            earnings = tasks_per_day * task_value
            self._json({"self_sustaining": earnings >= maintenance,
                        "current_earnings_usd": round(earnings, 2),
                        "maintenance_cost_usd": maintenance,
                        "break_even_tasks_per_day": int(maintenance / task_value),
                        "revenue_gap_usd": round(max(0, maintenance - earnings), 2),
                        "score": 50, "tier": "developing"})
        else:
            self._error(404, f"Not found: {path}")

    def _json(self, data: dict, status: int = 200):
        resp = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def _error(self, status: int, msg: str):
        self._json({"error": msg}, status)

    def log_message(self, *args):
        pass  # Suppress logs


def start_mock_v7(port: int = 18799) -> tuple:
    """Start mock v7 server. Returns (server, url)."""
    MockV7Handler._tasks = {}
    MockV7Handler._task_counter = 0
    server = HTTPServer(("localhost", port), MockV7Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    return server, f"http://localhost:{port}"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_server():
    server, url = start_mock_v7(18799)
    yield url
    server.shutdown()


class _MockIdentity:
    """Minimal mock identity for testing without real keypairs."""
    def __init__(self, agent_id: str):
        self.agent_id = agent_id


@pytest.fixture
def client(mock_server):
    """Authenticated client against mock server."""
    MockV7Handler._tasks = {}
    MockV7Handler._task_counter = 0
    c = ClawBizarreClient(mock_server)
    c.token = "mock-token-tester"
    c.identity = _MockIdentity("agent:tester")
    return c


# ── Task Board Client Tests ───────────────────────────────────────────────────

class TestPostTask:

    def test_post_task_returns_task_id(self, client):
        result = client.post_task("Sort integers", "Write sort(lst)")
        assert "task_id" in result
        assert result["task_id"].startswith("task-")

    def test_post_task_with_test_suite(self, client):
        result = client.post_task(
            "Sort test",
            "Write a sort function",
            task_type="code",
            capabilities=["python"],
            test_suite={"tests": [
                {"id": "t1", "type": "expression",
                 "expression": "sort([3,1,2])", "expected_output": "[1, 2, 3]"}
            ]},
            credits=10.0,
        )
        assert result["status"] == "pending"
        task = result["task"]
        assert task["requirements"]["task_type"] == "code"
        assert task["budget"]["credits"] == 10.0

    def test_post_task_sets_buyer_id_to_agent_id(self, client):
        result = client.post_task("My task", "desc")
        assert result["task"]["buyer_id"] == "agent:tester"

    def test_post_multiple_tasks(self, client):
        client.post_task("Task A", "desc A")
        client.post_task("Task B", "desc B")
        tasks = client.list_tasks()
        assert len(tasks) >= 2


class TestListTasks:

    def test_list_returns_tasks(self, client):
        client.post_task("Browseable task", "desc")
        tasks = client.list_tasks()
        assert isinstance(tasks, list)
        assert len(tasks) >= 1

    def test_list_empty_initially(self, mock_server):
        """Fresh client with reset task store."""
        MockV7Handler._tasks = {}
        MockV7Handler._task_counter = 0
        c = ClawBizarreClient(mock_server)
        c.token = "mock-token-fresh"
        c.identity = _MockIdentity("agent:fresh")
        tasks = c.list_tasks()
        assert tasks == []

    def test_list_with_limit(self, client):
        for i in range(5):
            client.post_task(f"Task {i}", f"desc {i}")
        tasks = client.list_tasks(limit=3)
        # Mock returns all; in real server this would be limited
        assert isinstance(tasks, list)

    def test_list_with_task_type_filter(self, client):
        tasks = client.list_tasks(task_type="code")
        assert isinstance(tasks, list)


class TestGetTask:

    def test_get_task_by_id(self, client):
        result = client.post_task("Get me", "desc")
        task_id = result["task_id"]
        task = client.get_task(task_id)
        assert task["task_id"] == task_id

    def test_get_nonexistent_raises(self, client):
        with pytest.raises(ClawBizarreError) as exc_info:
            client.get_task("task-does-not-exist")
        assert exc_info.value.status == 404


class TestClaimTask:

    def test_claim_pending_task(self, client):
        result = client.post_task("Claim me", "desc")
        task_id = result["task_id"]
        claim = client.claim_task(task_id, agent_tier="bootstrap")
        assert claim["success"] is True
        assert claim["expires_at"] is not None

    def test_double_claim_fails(self, client):
        result = client.post_task("Once only", "desc")
        task_id = result["task_id"]
        client.claim_task(task_id, "bootstrap")
        # Second claim should fail (task no longer pending)
        claim2 = client.claim_task(task_id, "bootstrap")
        assert claim2["success"] is False

    def test_claim_returns_expires_at(self, client):
        result = client.post_task("Expiry test", "desc")
        claim = client.claim_task(result["task_id"])
        assert "expires_at" in claim


class TestSubmitWork:

    def test_submit_claimed_task(self, client):
        result = client.post_task("Submit me", "desc")
        task_id = result["task_id"]
        client.claim_task(task_id)
        submit = client.submit_work(task_id, "def sort(lst): return sorted(lst)")
        assert submit["success"] is True
        assert submit["status"] == "complete"
        assert submit["verdict"] == "pass"
        assert submit["receipt"] is not None

    def test_submit_returns_receipt(self, client):
        result = client.post_task("Get receipt", "desc")
        task_id = result["task_id"]
        client.claim_task(task_id)
        submit = client.submit_work(task_id, "code here")
        receipt = submit["receipt"]
        assert "receipt_id" in receipt
        assert receipt["verdict"] == "pass"


class TestCompleteTask:

    def test_complete_task_convenience(self, client):
        result = client.post_task("Complete me", "desc")
        task_id = result["task_id"]
        submit = client.complete_task(task_id, "def sort(lst): return sorted(lst)")
        assert submit["verdict"] == "pass"
        assert submit["receipt"] is not None

    def test_complete_already_claimed_raises(self, client):
        result = client.post_task("Already claimed", "desc")
        task_id = result["task_id"]
        client.claim_task(task_id)  # Claim it first

        # Other client tries to complete_task (task no longer pending)
        other = ClawBizarreClient(client.base_url)
        other.token = "other-token"
        other.identity = _MockIdentity("agent:other")
        with pytest.raises(ClawBizarreError):
            other.complete_task(task_id, "code")


class TestCancelTask:

    def test_cancel_pending_task(self, client):
        result = client.post_task("Cancel me", "desc")
        task_id = result["task_id"]
        cancel = client.cancel_task(task_id)
        assert cancel["success"] is True

    def test_cancel_wrong_buyer_fails(self, client):
        result = client.post_task("Dont cancel", "desc")
        task_id = result["task_id"]
        other = ClawBizarreClient(client.base_url)
        other.token = "other-token"
        other.identity = _MockIdentity("agent:other")
        cancel = other.cancel_task(task_id)
        assert cancel["success"] is False


class TestTaskBoardStats:

    def test_stats_returns_total(self, client):
        client.post_task("Stat task", "desc")
        stats = client.task_board_stats()
        assert "total_tasks" in stats
        assert stats["total_tasks"] >= 1

    def test_stats_has_by_status(self, client):
        stats = client.task_board_stats()
        assert "by_status" in stats


# ── Credit Client Tests ───────────────────────────────────────────────────────

class TestCreditScore:

    def test_credit_score_no_receipts(self, client):
        score = client.credit_score(receipts=[])
        assert "total" in score
        assert "tier" in score
        assert "components" in score

    def test_credit_score_with_receipts(self, client):
        receipts = [{"verdict": "pass", "pass_rate": 1.0, "task_type": "code",
                     "verified_at": "2026-02-20T00:00:00+00:00", "domain": "code"} for _ in range(10)]
        score = client.credit_score(receipts=receipts)
        assert score["total"] >= 0

    def test_credit_score_with_domain(self, client):
        score = client.credit_score(receipts=[], domain="code")
        assert "total" in score


class TestCreditLine:

    def test_credit_line_returns_daily_usd(self, client):
        line = client.credit_line(receipts=[])
        assert "daily_usd" in line
        assert "tier" in line
        assert "max_task_usd" in line

    def test_credit_line_has_justification(self, client):
        line = client.credit_line(receipts=[])
        assert "justification" in line
        assert len(line["justification"]) > 0

    def test_credit_line_sponsor_required(self, client):
        line = client.credit_line(receipts=[])
        assert line.get("sponsor_required") is True


class TestCreditTiers:

    def test_credit_tiers_returns_five(self, client):
        tiers = client.credit_tiers()
        assert len(tiers) == 5

    def test_credit_tiers_has_expected_names(self, client):
        tiers = client.credit_tiers()
        names = {t["name"] for t in tiers}
        assert names == {"verified", "established", "developing", "new", "bootstrap"}

    def test_credit_tiers_verified_highest(self, client):
        tiers = client.credit_tiers()
        by_name = {t["name"]: t for t in tiers}
        assert by_name["verified"]["daily_usd"] > by_name["bootstrap"]["daily_usd"]


class TestSustainabilityProjection:

    def test_not_self_sustaining_at_5_tasks(self, client):
        proj = client.sustainability_projection(tasks_per_day=5, task_value_usd=0.01, maintenance_cost_usd=1.00)
        assert proj["self_sustaining"] is False
        assert proj["revenue_gap_usd"] > 0

    def test_self_sustaining_at_200_tasks(self, client):
        proj = client.sustainability_projection(tasks_per_day=200, task_value_usd=0.01, maintenance_cost_usd=1.00)
        assert proj["self_sustaining"] is True

    def test_break_even_calculation(self, client):
        proj = client.sustainability_projection(task_value_usd=0.01, maintenance_cost_usd=1.00)
        assert proj["break_even_tasks_per_day"] == 100


# ── Integration: Post → Claim → Submit ───────────────────────────────────────

class TestClientWorkflow:

    def test_full_workflow_in_client(self, client):
        """Post a task, list it, claim it, submit work, check receipt."""
        # Post
        result = client.post_task(
            "Integration test task",
            "Write double(x) = x * 2",
            task_type="code",
            capabilities=["python"],
            credits=5.0,
        )
        task_id = result["task_id"]

        # List - should appear
        tasks = client.list_tasks()
        assert any(t["task_id"] == task_id for t in tasks)

        # Claim
        claim = client.claim_task(task_id, agent_tier="bootstrap")
        assert claim["success"] is True

        # Submit
        submit = client.submit_work(task_id, "def double(x): return x * 2")
        assert submit["verdict"] == "pass"
        assert submit["receipt"]["receipt_id"].startswith("rcpt-")

        # Stats
        stats = client.task_board_stats()
        assert stats["by_status"].get("complete", 0) >= 1

    def test_buyer_and_agent_workflow(self, mock_server):
        """Two clients: buyer posts, agent completes."""
        MockV7Handler._tasks = {}
        MockV7Handler._task_counter = 0

        buyer = ClawBizarreClient(mock_server)
        buyer.token = "buyer-token"
        buyer.identity = _MockIdentity("agent:buyer")

        agent = ClawBizarreClient(mock_server)
        agent.token = "agent-token"
        agent.identity = _MockIdentity("agent:worker")

        # Buyer posts
        result = buyer.post_task("Two-party task", "desc", credits=10.0)
        task_id = result["task_id"]

        # Agent finds and completes
        tasks = agent.list_tasks()
        found = next((t for t in tasks if t["task_id"] == task_id), None)
        assert found is not None

        result = agent.complete_task(task_id, "def solve(): return 42")
        assert result["verdict"] == "pass"

        # Buyer checks task is done
        task = buyer.get_task(task_id)
        assert task["status"] == "complete"
        assert task["receipt"] is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
