"""
Tests for task_board.py — ClawBizarre Task Board (Phase 29)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pytest
import time
from datetime import datetime, timezone, timedelta
from task_board import (
    TaskBoard, Task, TaskStatus, TaskPriority,
    TaskRequirements, TaskBudget, ClaimResult, SubmitResult,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def board():
    """In-memory task board with short claim TTL for testing."""
    return TaskBoard(db_path=":memory:", claim_ttl_s=10)


@pytest.fixture
def short_ttl_board():
    """Board with very short claim TTL for expiry tests."""
    return TaskBoard(db_path=":memory:", claim_ttl_s=1)


def make_code_task_req(**kwargs) -> TaskRequirements:
    defaults = dict(
        task_type="code",
        capabilities=["python"],
        language="python",
        test_suite={
            "tests": [
                {"id": "t1", "type": "expression", "expression": "double(3)", "expected_output": "6"},
            ]
        },
        min_tier="bootstrap",
    )
    defaults.update(kwargs)
    return TaskRequirements(**defaults)


def make_budget(**kwargs) -> TaskBudget:
    defaults = dict(credits=5.0, max_task_usd=0.05)
    defaults.update(kwargs)
    return TaskBudget(**defaults)


def post_task(board, title="Test task", buyer="buyer:001", description="A test task", **kwargs):
    """Helper: post a task with sensible defaults."""
    return board.post_task(
        title=title,
        description=description,
        requirements=make_code_task_req(**kwargs.pop("requirements", {})),
        budget=make_budget(**kwargs.pop("budget", {})),
        buyer_id=buyer,
        **kwargs
    )


# ── Post Task ─────────────────────────────────────────────────────────────────

class TestPostTask:

    def test_post_creates_pending_task(self, board):
        task = post_task(board)
        assert task.task_id.startswith("task-")
        assert task.status == TaskStatus.PENDING

    def test_post_stores_title_and_description(self, board):
        task = post_task(board, title="Sort integers", description="Sort them ascending.")
        assert task.title == "Sort integers"
        assert task.description == "Sort them ascending."

    def test_post_stores_buyer_id(self, board):
        task = post_task(board, buyer="buyer:xyz")
        assert task.buyer_id == "buyer:xyz"

    def test_post_assigns_unique_ids(self, board):
        t1 = post_task(board)
        t2 = post_task(board)
        assert t1.task_id != t2.task_id

    def test_post_sets_posted_at(self, board):
        before = datetime.now(timezone.utc).isoformat()
        task = post_task(board)
        after = datetime.now(timezone.utc).isoformat()
        assert before <= task.posted_at <= after

    def test_post_with_deadline(self, board):
        task = post_task(board, deadline_hours=24)
        assert task.deadline_at is not None
        # Deadline should be ~24h from now
        dl = datetime.fromisoformat(task.deadline_at)
        now = datetime.now(timezone.utc)
        diff_h = (dl - now).total_seconds() / 3600
        assert 23.5 < diff_h < 24.5

    def test_post_priority_urgent(self, board):
        task = post_task(board, priority=TaskPriority.URGENT)
        assert task.priority == TaskPriority.URGENT

    def test_post_persists_to_db(self, board):
        task = post_task(board)
        retrieved = board.get_task(task.task_id)
        assert retrieved is not None
        assert retrieved.task_id == task.task_id
        assert retrieved.title == task.title


# ── List Tasks ────────────────────────────────────────────────────────────────

class TestListTasks:

    def test_list_returns_pending_tasks(self, board):
        post_task(board, title="Task 1")
        post_task(board, title="Task 2")
        tasks = board.list_tasks()
        assert len(tasks) == 2

    def test_list_filters_by_task_type(self, board):
        post_task(board, requirements={"task_type": "code"})
        post_task(board, requirements={"task_type": "translation"})
        code_tasks = board.list_tasks(task_type="code")
        assert all(t.requirements.task_type == "code" for t in code_tasks)
        assert len(code_tasks) == 1

    def test_list_filters_by_min_credits(self, board):
        post_task(board, budget={"credits": 5.0})
        post_task(board, budget={"credits": 50.0})
        expensive = board.list_tasks(min_credits=20.0)
        assert len(expensive) == 1
        assert expensive[0].budget.credits >= 20.0

    def test_list_filters_by_max_credits(self, board):
        post_task(board, budget={"credits": 5.0})
        post_task(board, budget={"credits": 50.0})
        cheap = board.list_tasks(max_credits=10.0)
        assert len(cheap) == 1
        assert cheap[0].budget.credits <= 10.0

    def test_list_filters_by_capability(self, board):
        board.post_task(
            "Task A", "desc",
            TaskRequirements("code", ["python", "algorithms"], "python"),
            make_budget(), "buyer:001",
        )
        board.post_task(
            "Task B", "desc",
            TaskRequirements("code", ["golang"], "python"),
            make_budget(), "buyer:001",
        )
        python_tasks = board.list_tasks(required_capability="python")
        assert len(python_tasks) == 1
        assert "python" in python_tasks[0].requirements.capabilities

    def test_list_respects_limit(self, board):
        for i in range(10):
            post_task(board, title=f"Task {i}")
        tasks = board.list_tasks(limit=3)
        assert len(tasks) <= 3

    def test_empty_board_returns_empty_list(self, board):
        tasks = board.list_tasks()
        assert tasks == []


# ── Claim Task ────────────────────────────────────────────────────────────────

class TestClaimTask:

    def test_claim_pending_task_succeeds(self, board):
        task = post_task(board)
        result = board.claim_task(task.task_id, "agent:rahcd")
        assert result.success is True
        assert result.task_id == task.task_id

    def test_claim_sets_claimed_status(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        t = board.get_task(task.task_id)
        assert t.status == TaskStatus.CLAIMED

    def test_claim_stores_agent_id(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        t = board.get_task(task.task_id)
        assert t.claimed_by == "agent:rahcd"

    def test_claim_sets_expiry(self, board):
        task = post_task(board)
        result = board.claim_task(task.task_id, "agent:rahcd")
        assert result.expires_at is not None
        expires = datetime.fromisoformat(result.expires_at)
        now = datetime.now(timezone.utc)
        # Should expire ~10s from now (fixture ttl=10)
        assert 5 < (expires - now).total_seconds() < 15

    def test_double_claim_fails(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        result2 = board.claim_task(task.task_id, "agent:other")
        assert result2.success is False
        assert "claimed" in result2.reason.lower()

    def test_claim_nonexistent_task_fails(self, board):
        result = board.claim_task("task-doesnotexist", "agent:rahcd")
        assert result.success is False

    def test_claim_requires_min_tier(self, board):
        task = board.post_task(
            "Premium task", "desc",
            TaskRequirements("code", ["python"], "python", min_tier="established"),
            make_budget(credits=100.0), "buyer:001",
        )
        # Bootstrap agent can't claim established-tier task
        result = board.claim_task(task.task_id, "agent:bootstrap", agent_tier="bootstrap")
        assert result.success is False
        assert "tier" in result.reason.lower()

    def test_claim_with_sufficient_tier(self, board):
        task = board.post_task(
            "Premium task", "desc",
            TaskRequirements("code", ["python"], "python", min_tier="developing"),
            make_budget(credits=20.0), "buyer:001",
        )
        result = board.claim_task(task.task_id, "agent:pro", agent_tier="established")
        assert result.success is True

    def test_expired_claim_reverts_to_pending(self, short_ttl_board):
        task = post_task(short_ttl_board)
        short_ttl_board.claim_task(task.task_id, "agent:rahcd")
        # Wait for TTL to expire
        time.sleep(1.5)
        # Calling any method triggers expiry enforcement
        t = short_ttl_board.get_task(task.task_id)
        assert t.status == TaskStatus.PENDING
        assert t.claimed_by is None


# ── Submit Work ───────────────────────────────────────────────────────────────

class TestSubmitWork:

    def test_submit_transitions_to_submitted(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        result = board.submit_work(task.task_id, "agent:rahcd", "def double(x): return x * 2")
        assert result.success is True
        assert result.status == TaskStatus.SUBMITTED.value

    def test_submit_stores_work_content(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        board.submit_work(task.task_id, "agent:rahcd", "def double(x): return x * 2")
        t = board.get_task(task.task_id)
        assert t.work_content == "def double(x): return x * 2"

    def test_submit_without_claim_fails(self, board):
        task = post_task(board)
        result = board.submit_work(task.task_id, "agent:rahcd", "code")
        assert result.success is False
        assert "claimed" in result.reason.lower()

    def test_wrong_agent_cannot_submit(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        result = board.submit_work(task.task_id, "agent:imposter", "code")
        assert result.success is False

    def test_submit_nonexistent_task_fails(self, board):
        result = board.submit_work("task-ghost", "agent:rahcd", "code")
        assert result.success is False

    def test_submit_sets_submitted_at(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        board.submit_work(task.task_id, "agent:rahcd", "code")
        t = board.get_task(task.task_id)
        assert t.submitted_at is not None


# ── Cancel Task ───────────────────────────────────────────────────────────────

class TestCancelTask:

    def test_cancel_pending_task(self, board):
        task = post_task(board, buyer="buyer:001")
        result = board.cancel_task(task.task_id, buyer_id="buyer:001")
        assert result["success"] is True

    def test_cancel_changes_status(self, board):
        task = post_task(board, buyer="buyer:001")
        board.cancel_task(task.task_id, buyer_id="buyer:001")
        t = board.get_task(task.task_id)
        assert t.status == TaskStatus.CANCELLED

    def test_only_buyer_can_cancel(self, board):
        task = post_task(board, buyer="buyer:001")
        result = board.cancel_task(task.task_id, buyer_id="buyer:impostor")
        assert result["success"] is False

    def test_cannot_cancel_claimed_task(self, board):
        task = post_task(board, buyer="buyer:001")
        board.claim_task(task.task_id, "agent:rahcd")
        result = board.cancel_task(task.task_id, buyer_id="buyer:001")
        assert result["success"] is False

    def test_cancel_nonexistent_fails(self, board):
        result = board.cancel_task("task-ghost", buyer_id="buyer:001")
        assert result["success"] is False


# ── Stats ─────────────────────────────────────────────────────────────────────

class TestStats:

    def test_stats_empty_board(self, board):
        stats = board.stats()
        assert stats["total_tasks"] == 0

    def test_stats_counts_by_status(self, board):
        t1 = post_task(board, buyer="buyer:001")
        t2 = post_task(board, buyer="buyer:001")
        t3 = post_task(board, buyer="buyer:001")
        board.claim_task(t1.task_id, "agent:rahcd")
        board.cancel_task(t3.task_id, buyer_id="buyer:001")
        stats = board.stats()
        assert stats["by_status"].get("pending", 0) == 1
        assert stats["by_status"].get("claimed", 0) == 1
        assert stats["by_status"].get("cancelled", 0) == 1

    def test_stats_total_includes_all_statuses(self, board):
        for _ in range(5):
            post_task(board)
        assert board.stats()["total_tasks"] == 5

    def test_stats_pending_budget_average(self, board):
        post_task(board, budget={"credits": 10.0})
        post_task(board, budget={"credits": 20.0})
        stats = board.stats()
        assert stats["pending_avg_budget_credits"] == pytest.approx(15.0, abs=0.1)


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class TestHTTPHandler:

    def test_post_tasks(self, board):
        result = board.handle_request("POST", "/tasks", {
            "title": "Test task",
            "description": "desc",
            "task_type": "code",
            "capabilities": ["python"],
            "credits": 5.0,
            "buyer_id": "buyer:001",
        })
        assert "task_id" in result
        assert result["status"] == "pending"

    def test_get_tasks(self, board):
        post_task(board)
        post_task(board)
        result = board.handle_request("GET", "/tasks", {})
        assert "tasks" in result
        assert result["count"] == 2

    def test_get_task_by_id(self, board):
        task = post_task(board)
        result = board.handle_request("GET", f"/tasks/{task.task_id}", {})
        assert result["task_id"] == task.task_id

    def test_get_nonexistent_task(self, board):
        result = board.handle_request("GET", "/tasks/task-ghost", {})
        assert "error" in result

    def test_claim_via_handler(self, board):
        task = post_task(board)
        result = board.handle_request("POST", f"/tasks/{task.task_id}/claim", {
            "agent_id": "agent:rahcd",
            "agent_tier": "bootstrap",
        })
        assert result["success"] is True

    def test_submit_via_handler(self, board):
        task = post_task(board)
        board.claim_task(task.task_id, "agent:rahcd")
        result = board.handle_request("POST", f"/tasks/{task.task_id}/submit", {
            "agent_id": "agent:rahcd",
            "work_content": "def double(x): return x * 2",
            "auto_verify": False,
        })
        assert result["success"] is True

    def test_cancel_via_handler(self, board):
        task = post_task(board, buyer="buyer:001")
        result = board.handle_request("POST", f"/tasks/{task.task_id}/cancel", {
            "buyer_id": "buyer:001",
        })
        assert result["success"] is True

    def test_stats_via_handler(self, board):
        post_task(board)
        result = board.handle_request("GET", "/tasks/stats", {})
        assert "total_tasks" in result

    def test_unknown_route(self, board):
        result = board.handle_request("GET", "/unknown/route", {})
        assert "error" in result


# ── Newcomer Reserve ──────────────────────────────────────────────────────────

class TestNewcomerReserve:

    def test_newcomer_sees_smaller_tasks_first(self, board):
        """Newcomer agent (<20 receipts) should see smaller tasks prominently."""
        for credits in [5.0, 10.0, 15.0, 50.0, 100.0]:
            post_task(board, budget={"credits": credits})
        tasks_newcomer = board.list_tasks(receipt_count=5, limit=5)
        tasks_veteran = board.list_tasks(receipt_count=100, limit=5)
        # Newcomer list first item should have lower credits than veteran list first item
        if tasks_newcomer and tasks_veteran:
            assert tasks_newcomer[0].budget.credits <= tasks_veteran[0].budget.credits

    def test_both_see_tasks(self, board):
        for i in range(10):
            post_task(board, budget={"credits": float(i + 1)})
        newcomer_tasks = board.list_tasks(receipt_count=3, limit=10)
        veteran_tasks = board.list_tasks(receipt_count=100, limit=10)
        assert len(newcomer_tasks) > 0
        assert len(veteran_tasks) > 0


# ── Serialization ─────────────────────────────────────────────────────────────

class TestSerialization:

    def test_task_to_dict_roundtrip(self, board):
        task = post_task(board)
        d = task.to_dict()
        restored = Task.from_dict(d)
        assert restored.task_id == task.task_id
        assert restored.status == task.status
        assert restored.requirements.task_type == task.requirements.task_type

    def test_task_status_is_string_in_dict(self, board):
        task = post_task(board)
        d = task.to_dict()
        assert isinstance(d["status"], str)

    def test_task_survives_db_roundtrip(self, board):
        task = post_task(board, title="Roundtrip test")
        retrieved = board.get_task(task.task_id)
        assert retrieved.title == "Roundtrip test"
        assert retrieved.requirements.task_type == task.requirements.task_type


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
