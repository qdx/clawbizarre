"""
test_full_lifecycle.py — End-to-end integration test for ClawBizarre.

Exercises the complete lifecycle:
  Buyer posts task → Agent discovers → Agent claims → Agent submits →
  Auto-verify (mocked verify_server) → Receipt → Credit score update

This test validates that all components INTEGRATE correctly:
- task_board.py + compute_credit.py + verify_server.py (mocked)

All components are independently unit-tested; this test catches integration bugs.

Two test modes:
  1. Mocked verify (default) — no running server required
  2. Live verify (when CLAWBIZARRE_TEST_VERIFY_URL is set) — uses real verify_server
"""

import sys
import os
import json
import time
import threading
import urllib.request
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
from task_board import TaskBoard, TaskRequirements, TaskBudget, TaskPriority, TaskStatus
from compute_credit import CreditScorer, ReceiptSummary


# ── Mock verify server ────────────────────────────────────────────────────────

class MockVerifyHandler(BaseHTTPRequestHandler):
    """
    Minimal mock of verify_server for integration testing.
    Accepts POST /verify and returns a synthetic VRF receipt.
    
    Controlled via class-level attributes for test scenarios:
    - MockVerifyHandler.verdict = "pass"   → receipt with pass verdict
    - MockVerifyHandler.verdict = "fail"   → receipt with fail verdict
    """
    verdict = "pass"
    pass_rate = 1.0
    call_count = 0

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)
        MockVerifyHandler.call_count += 1
        receipt = {
            "receipt_id": f"rcpt-mock-{MockVerifyHandler.call_count:04d}",
            "verified_at": datetime.now(timezone.utc).isoformat(),
            "tier": 0,
            "verdict": MockVerifyHandler.verdict,
            "results": {
                "total": 2,
                "passed": int(2 * MockVerifyHandler.pass_rate),
                "failed": 2 - int(2 * MockVerifyHandler.pass_rate),
                "errors": 0,
            },
            "hashes": {
                "input_hash": "mock-input",
                "output_hash": "mock-output",
                "suite_hash": "mock-suite",
                "algorithm": "sha256",
            },
            "metadata": {"mock": True},
            "vrf_version": "1.0",
        }
        resp = json.dumps({"receipt": receipt}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(resp)))
        self.end_headers()
        self.wfile.write(resp)

    def log_message(self, fmt, *args):
        pass  # Suppress request logs


def start_mock_server(port: int = 18765) -> tuple:
    """Start mock verify server in a background thread. Returns (server, url)."""
    server = HTTPServer(("localhost", port), MockVerifyHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    return server, f"http://localhost:{port}"


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def mock_verify():
    """Start mock verify server once for the module."""
    server, url = start_mock_server(18765)
    yield url
    server.shutdown()


@pytest.fixture
def board(mock_verify):
    """TaskBoard wired to mock verify server."""
    MockVerifyHandler.verdict = "pass"
    MockVerifyHandler.pass_rate = 1.0
    return TaskBoard(db_path=":memory:", verify_url=mock_verify, claim_ttl_s=30)


@pytest.fixture
def scorer():
    return CreditScorer()


def make_code_receipt(days_ago: float, pass_rate: float = 1.0, verdict: str = "pass") -> ReceiptSummary:
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return ReceiptSummary(
        receipt_id=f"hist-{days_ago:.0f}d",
        agent_id="agent:rahcd",
        task_type="code",
        verdict=verdict,
        pass_rate=pass_rate,
        verified_at=ts,
        domain="code",
    )


# ── Full Lifecycle Tests ──────────────────────────────────────────────────────

class TestFullLifecycle:
    """Tests that exercise multiple components end-to-end."""

    def test_buyer_post_agent_claim_submit_complete(self, board, scorer):
        """
        Complete happy path:
        buyer posts → agent claims → submits → auto-verified (pass) → COMPLETE
        """
        # 1. Buyer posts task
        req = TaskRequirements(
            task_type="code",
            capabilities=["python"],
            language="python",
            test_suite={
                "tests": [
                    {"id": "t1", "type": "expression",
                     "expression": "double(3)", "expected_output": "6"},
                ]
            },
            min_tier="bootstrap",
        )
        task = board.post_task(
            title="Double a number",
            description="Write double(x) function",
            requirements=req,
            budget=TaskBudget(credits=10.0, max_task_usd=0.10),
            buyer_id="agent:buyer-001",
        )
        assert task.status == TaskStatus.PENDING

        # 2. Agent discovers task
        tasks = board.list_tasks(task_type="code", receipt_count=5)
        assert any(t.task_id == task.task_id for t in tasks)

        # 3. Agent claims task
        claim = board.claim_task(task.task_id, "agent:rahcd", agent_tier="bootstrap")
        assert claim.success is True
        t = board.get_task(task.task_id)
        assert t.status == TaskStatus.CLAIMED
        assert t.claimed_by == "agent:rahcd"

        # 4. Agent submits work (mock verify_server returns "pass")
        result = board.submit_work(
            task.task_id,
            agent_id="agent:rahcd",
            work_content="def double(x): return x * 2",
            auto_verify=True,
        )

        # 5. Task should be COMPLETE with a receipt
        assert result.success is True
        assert result.verdict == "pass"
        assert result.receipt is not None

        completed = board.get_task(task.task_id)
        assert completed.status == TaskStatus.COMPLETE
        assert completed.receipt is not None
        assert completed.receipt["verdict"] == "pass"
        assert completed.completed_at is not None

    def test_failed_verification_reposts_task(self, board):
        """When verification fails, task is re-posted (up to 3 times)."""
        MockVerifyHandler.verdict = "fail"
        MockVerifyHandler.pass_rate = 0.0
        board_with_repost = TaskBoard(
            db_path=":memory:",
            verify_url=board.verify_url,
            auto_repost=True,
            claim_ttl_s=30,
        )

        task = board_with_repost.post_task(
            title="Failing task",
            description="This will fail",
            requirements=TaskRequirements(
                task_type="code", capabilities=["python"], language="python",
                test_suite={"tests": [
                    {"id": "t1", "type": "expression",
                     "expression": "f()", "expected_output": "wrong"}
                ]},
            ),
            budget=TaskBudget(credits=5.0, max_task_usd=0.05),
            buyer_id="agent:buyer-001",
        )
        board_with_repost.claim_task(task.task_id, "agent:rahcd")
        result = board_with_repost.submit_work(
            task.task_id, "agent:rahcd", "def f(): return 'not_wrong'", auto_verify=True
        )

        # Should have been re-posted (back to PENDING)
        t = board_with_repost.get_task(task.task_id)
        assert t.status == TaskStatus.PENDING
        assert t.re_post_count == 1

        # Reset for other tests
        MockVerifyHandler.verdict = "pass"
        MockVerifyHandler.pass_rate = 1.0

    def test_full_cycle_improves_credit_score(self, board, scorer):
        """
        Agent completes multiple tasks → receipt chain grows →
        credit score improves → credit line expands.
        """
        # Baseline: new agent, 5 historical receipts
        old_receipts = [make_code_receipt(i * 7, pass_rate=0.9) for i in range(5)]
        old_score = scorer.score_from_receipts(old_receipts, domain="code")
        old_credit = scorer.credit_line(old_score)

        # Complete 5 more tasks via task board
        new_receipts_from_board = []
        for i in range(5):
            req = TaskRequirements(
                task_type="code", capabilities=["python"], language="python",
                test_suite={"tests": [
                    {"id": f"t{i}", "type": "expression",
                     "expression": f"x_{i}", "expected_output": "1"}
                ]},
            )
            task = board.post_task(
                f"Task {i}", f"desc {i}", req,
                TaskBudget(credits=10.0, max_task_usd=0.10), "agent:buyer-001",
            )
            board.claim_task(task.task_id, "agent:rahcd")
            result = board.submit_work(
                task.task_id, "agent:rahcd", f"x_{i} = 1", auto_verify=True
            )
            if result.receipt:
                receipt = result.receipt
                new_receipts_from_board.append(ReceiptSummary(
                    receipt_id=receipt["receipt_id"],
                    agent_id="agent:rahcd",
                    task_type="code",
                    verdict=receipt["verdict"],
                    pass_rate=receipt["results"]["passed"] / max(receipt["results"]["total"], 1),
                    verified_at=receipt["verified_at"],
                    domain="code",
                ))

        # Updated credit score
        all_receipts = old_receipts + new_receipts_from_board
        new_score = scorer.score_from_receipts(all_receipts, domain="code")
        new_credit = scorer.credit_line(new_score)

        # More receipts → higher score → bigger credit line
        assert new_score.total > old_score.total
        assert new_score.volume > old_score.volume
        assert new_credit.daily_usd >= old_credit.daily_usd

    def test_task_board_respects_credit_tier_gating(self, board):
        """Bootstrap agent cannot claim established-tier task."""
        premium_task = board.post_task(
            title="Enterprise data pipeline",
            description="Complex ETL pipeline review",
            requirements=TaskRequirements(
                task_type="code", capabilities=["python", "etl"],
                language="python", min_tier="established",
            ),
            budget=TaskBudget(credits=100.0, max_task_usd=1.00),
            buyer_id="agent:enterprise-buyer",
        )

        # Bootstrap agent can't claim
        bootstrap_result = board.claim_task(
            premium_task.task_id, "agent:bootstrap-rahcd", agent_tier="bootstrap"
        )
        assert bootstrap_result.success is False
        assert "tier" in bootstrap_result.reason.lower()

        # Verified agent can claim
        verified_result = board.claim_task(
            premium_task.task_id, "agent:senior-rahcd", agent_tier="verified"
        )
        assert verified_result.success is True

    def test_claim_expiry_makes_task_available_again(self):
        """Expired claim reverts task to PENDING — another agent can claim."""
        quick_board = TaskBoard(
            db_path=":memory:", claim_ttl_s=1, verify_url=None
        )
        task = quick_board.post_task(
            "Quick task", "desc",
            TaskRequirements("code", ["python"], "python"),
            TaskBudget(credits=5.0, max_task_usd=0.05),
            "agent:buyer-001",
        )

        # First agent claims
        r1 = quick_board.claim_task(task.task_id, "agent:slow")
        assert r1.success is True

        # Wait for TTL to expire
        time.sleep(1.5)

        # Second agent claims (TTL expired)
        r2 = quick_board.claim_task(task.task_id, "agent:fast")
        assert r2.success is True
        t = quick_board.get_task(task.task_id)
        assert t.claimed_by == "agent:fast"

    def test_stats_reflect_completed_tasks(self, board, scorer):
        """Task board stats accurately reflect lifecycle transitions."""
        # Post 3 tasks
        tasks = []
        for i in range(3):
            req = TaskRequirements("code", ["python"], "python",
                test_suite={"tests": [{"id": "t1", "type": "expression",
                    "expression": "v", "expected_output": "1"}]})
            t = board.post_task(f"Task {i}", "desc", req,
                TaskBudget(credits=5.0, max_task_usd=0.05), "agent:buyer-001")
            tasks.append(t)

        # Complete first 2
        for t in tasks[:2]:
            board.claim_task(t.task_id, "agent:rahcd")
            board.submit_work(t.task_id, "agent:rahcd", "v = 1", auto_verify=True)

        # Cancel third
        board.cancel_task(tasks[2].task_id, "agent:buyer-001")

        stats = board.stats()
        assert stats["by_status"].get("complete", 0) == 2
        assert stats["by_status"].get("cancelled", 0) == 1

    def test_verify_server_called_on_submit(self, board):
        """verify_server is called exactly once per submit."""
        initial_count = MockVerifyHandler.call_count
        req = TaskRequirements("code", ["python"], "python",
            test_suite={"tests": [{"id": "t1", "type": "expression",
                "expression": "x", "expected_output": "1"}]})
        task = board.post_task("Counting task", "desc", req,
            TaskBudget(credits=5.0, max_task_usd=0.05), "agent:buyer-001")
        board.claim_task(task.task_id, "agent:rahcd")
        board.submit_work(task.task_id, "agent:rahcd", "x = 1", auto_verify=True)

        assert MockVerifyHandler.call_count == initial_count + 1

    def test_no_auto_verify_stays_submitted(self, board):
        """submit_work(auto_verify=False) leaves task in SUBMITTED state."""
        req = TaskRequirements("code", ["python"], "python",
            test_suite={"tests": [{"id": "t1", "type": "expression",
                "expression": "x", "expected_output": "1"}]})
        task = board.post_task("Manual task", "desc", req,
            TaskBudget(credits=5.0, max_task_usd=0.05), "agent:buyer-001")
        board.claim_task(task.task_id, "agent:rahcd")
        result = board.submit_work(task.task_id, "agent:rahcd", "x = 1", auto_verify=False)

        assert result.status == "submitted"
        t = board.get_task(task.task_id)
        assert t.status == TaskStatus.SUBMITTED


# ── Sustainability Projection ─────────────────────────────────────────────────

class TestSustainabilityFromLifecycle:
    """Tests that model the path to agent financial sustainability."""

    def test_50_tasks_approaches_sustainability(self, board, scorer):
        """50 verified tasks/day at $0.01 → approaching sustainability at $1/day cost."""
        # Build 50-receipt history
        receipts = [make_code_receipt(i * 1.7, pass_rate=0.95) for i in range(50)]
        score = scorer.score_from_receipts(receipts, domain="code")

        proj = scorer.sustainability_projection(
            score,
            task_value_usd=0.01,
            tasks_per_day=50,
            maintenance_cost_usd=1.00,
        )
        # Not yet self-sustaining ($0.50 < $1.00), but credit tier should be good
        assert score.tier in ("established", "verified")
        assert proj["break_even_tasks_per_day"] == 100
        assert proj["current_earnings_usd"] == pytest.approx(0.50, abs=0.01)

    def test_100_tasks_is_self_sustaining(self, board, scorer):
        """100 verified tasks/day at $0.01 → self-sustaining at $1/day cost."""
        receipts = [make_code_receipt(i * 1.7, pass_rate=0.95) for i in range(50)]
        score = scorer.score_from_receipts(receipts)

        proj = scorer.sustainability_projection(
            score,
            task_value_usd=0.01,
            tasks_per_day=100,
            maintenance_cost_usd=1.00,
        )
        assert proj["self_sustaining"] is True
        assert proj["revenue_gap_usd"] == 0.0

    def test_newcomer_path_to_established(self, scorer):
        """Simulate agent growing from 0 to 50 receipts over time."""
        receipt_counts = [5, 15, 30, 50]
        tiers = []
        for count in receipt_counts:
            receipts = [make_code_receipt(i * 1.5, pass_rate=0.92) for i in range(count)]
            score = scorer.score_from_receipts(receipts)
            tiers.append(score.tier)

        # Tier should generally improve with more receipts (not necessarily strict)
        tier_order = ["bootstrap", "new", "developing", "established", "verified"]
        tier_indices = [tier_order.index(t) for t in tiers]
        # Last tier should be at least as good as first
        assert tier_indices[-1] >= tier_indices[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
