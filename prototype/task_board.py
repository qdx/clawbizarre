"""
task_board.py — ClawBizarre Task Board (Phase 29)

The demand-side of the ClawBizarre marketplace. Buyers post tasks with
requirements and budget; agents discover, claim, execute, and submit work
for automated VRF verification and payment.

This is the missing bridge between:
  - Buyers needing work done (demand)
  - Agents with capabilities and credit (supply)
  - VRF verification (quality assurance)
  - Compute credit system (economic sustainability)

Task lifecycle:
  PENDING → CLAIMED → SUBMITTED → VERIFYING → COMPLETE
                                              ↘ FAILED
                                CLAIMED (TTL) → EXPIRED → PENDING

HTTP endpoints:
  POST /tasks              — buyer posts task
  GET  /tasks              — list available tasks (with filters)
  GET  /tasks/{id}         — task detail
  POST /tasks/{id}/claim   — agent claims task
  POST /tasks/{id}/submit  — agent submits work (triggers auto-verification)
  POST /tasks/{id}/cancel  — buyer cancels pending task
  GET  /tasks/stats        — board statistics

Design principles:
  1. Posted-price, not auction (matches matching.py design philosophy)
  2. Claim TTL prevents indefinite lockout (default: 30 min)
  3. Auto-verification on submit (no human in loop for Tier 0)
  4. Credit tier gating (agents need min tier to claim high-budget tasks)
  5. Newcomer reserve (20% of slots for agents with <20 receipts — Law 6)
"""

import json
import uuid
import sqlite3
import time
import threading
import hashlib
import textwrap
from dataclasses import dataclass, field, asdict
from typing import Optional
from datetime import datetime, timezone, timedelta
from enum import Enum

# ── Task Status ───────────────────────────────────────────────────────────────

class TaskStatus(str, Enum):
    PENDING    = "pending"     # Posted, waiting for agent
    CLAIMED    = "claimed"     # Agent claimed, working on it (TTL: claim_ttl_s)
    SUBMITTED  = "submitted"   # Agent submitted, waiting for verification
    VERIFYING  = "verifying"   # verify_server processing
    COMPLETE   = "complete"    # Verified pass → buyer can retrieve, payment released
    FAILED     = "failed"      # Verification failed → buyer notified, task may re-post
    CANCELLED  = "cancelled"   # Buyer cancelled before claim
    EXPIRED    = "expired"     # Claim TTL exceeded → back to PENDING (internally)


class TaskPriority(str, Enum):
    LOW    = "low"
    NORMAL = "normal"
    HIGH   = "high"
    URGENT = "urgent"


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class TaskRequirements:
    """What the buyer needs from the agent."""
    task_type: str               # "code", "research", "translation", "data", etc.
    capabilities: list           # e.g., ["python", "sorting_algorithms"]
    language: str = "python"     # Execution language for Tier 0 tests
    test_suite: Optional[dict] = None  # VRF test suite (if Tier 0 verification)
    schema: Optional[dict] = None      # JSON Schema (if Tier 1 verification)
    min_tier: str = "bootstrap"        # Minimum credit tier required
    verification_tier: int = 0         # 0=test-based, 1=schema


@dataclass
class TaskBudget:
    """Economic terms for the task."""
    credits: float               # ClawBizarre credits offered
    max_task_usd: float          # USD equivalent cap (for credit → USD mapping)
    payment_protocol: str = "credits"  # "credits", "x402", "manual"
    escrow: bool = False         # Whether buyer escrows credits on post
    refund_on_failure: bool = True


@dataclass
class Task:
    """A unit of work posted on the ClawBizarre task board."""
    task_id: str
    title: str
    description: str
    requirements: TaskRequirements
    budget: TaskBudget
    buyer_id: str
    status: TaskStatus = TaskStatus.PENDING
    priority: TaskPriority = TaskPriority.NORMAL
    posted_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    deadline_at: Optional[str] = None
    claimed_by: Optional[str] = None
    claimed_at: Optional[str] = None
    claim_expires_at: Optional[str] = None
    submitted_at: Optional[str] = None
    completed_at: Optional[str] = None
    work_content: Optional[str] = None       # Agent's submitted work
    receipt: Optional[dict] = None           # VRF receipt on completion
    failure_reason: Optional[str] = None
    re_post_count: int = 0                   # How many times this was re-posted

    def to_dict(self) -> dict:
        d = asdict(self)
        d["requirements"] = asdict(self.requirements)
        d["budget"] = asdict(self.budget)
        d["status"] = self.status.value
        d["priority"] = self.priority.value
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "Task":
        req = TaskRequirements(**d.pop("requirements"))
        bud = TaskBudget(**d.pop("budget"))
        d["requirements"] = req
        d["budget"] = bud
        d["status"] = TaskStatus(d["status"])
        d["priority"] = TaskPriority(d.get("priority", "normal"))
        return cls(**d)


@dataclass
class ClaimResult:
    success: bool
    task_id: str
    agent_id: str
    expires_at: Optional[str] = None
    reason: Optional[str] = None


@dataclass
class SubmitResult:
    success: bool
    task_id: str
    status: str
    receipt: Optional[dict] = None
    verdict: Optional[str] = None
    reason: Optional[str] = None


# ── Task Board ────────────────────────────────────────────────────────────────

class TaskBoard:
    """
    The ClawBizarre task board — demand-side of the marketplace.

    Stores tasks in SQLite. Thread-safe. Supports claim TTL enforcement.
    Integrates with verify_server for auto-verification on submit.
    """

    DEFAULT_CLAIM_TTL = 30 * 60     # 30 minutes
    DEFAULT_TASK_TTL  = 7 * 24 * 3600  # 7 days max post lifetime

    def __init__(
        self,
        db_path: str = ":memory:",
        claim_ttl_s: int = DEFAULT_CLAIM_TTL,
        verify_url: Optional[str] = None,
        auto_repost: bool = True,
        newcomer_reserve: float = 0.20,
    ):
        self.db_path = db_path
        self.claim_ttl_s = claim_ttl_s
        self.verify_url = verify_url          # verify_server URL for auto-verify
        self.auto_repost = auto_repost        # Re-post failed tasks automatically
        self.newcomer_reserve = newcomer_reserve
        self._lock = threading.RLock()
        self._db = self._init_db()

    # ── Public API ────────────────────────────────────────────────────────────

    def post_task(
        self,
        title: str,
        description: str,
        requirements: TaskRequirements,
        budget: TaskBudget,
        buyer_id: str,
        priority: TaskPriority = TaskPriority.NORMAL,
        deadline_hours: Optional[float] = None,
    ) -> Task:
        """Buyer posts a new task to the board."""
        task_id = f"task-{uuid.uuid4().hex[:12]}"
        deadline = None
        if deadline_hours:
            dl_dt = datetime.now(timezone.utc) + timedelta(hours=deadline_hours)
            deadline = dl_dt.isoformat()

        task = Task(
            task_id=task_id,
            title=title,
            description=description,
            requirements=requirements,
            budget=budget,
            buyer_id=buyer_id,
            status=TaskStatus.PENDING,
            priority=priority,
            deadline_at=deadline,
        )
        self._save_task(task)
        return task

    def list_tasks(
        self,
        status: Optional[TaskStatus] = TaskStatus.PENDING,
        task_type: Optional[str] = None,
        min_credits: Optional[float] = None,
        max_credits: Optional[float] = None,
        required_capability: Optional[str] = None,
        limit: int = 50,
        agent_id: Optional[str] = None,     # For newcomer reserve logic
        receipt_count: int = 0,             # Agent's receipt count (for reserve)
    ) -> list:
        """Browse available tasks with optional filters."""
        self._expire_stale_claims()

        with self._lock:
            cur = self._db.cursor()
            query = "SELECT task_json FROM tasks WHERE 1=1"
            params = []
            if status:
                query += " AND status = ?"
                params.append(status.value)
            if task_type:
                query += " AND task_type = ?"
                params.append(task_type)
            if min_credits is not None:
                query += " AND budget_credits >= ?"
                params.append(min_credits)
            if max_credits is not None:
                query += " AND budget_credits <= ?"
                params.append(max_credits)
            query += " ORDER BY priority_score DESC, posted_at ASC LIMIT ?"
            params.append(limit * 2)  # Fetch extra for newcomer filtering

            cur.execute(query, params)
            rows = cur.fetchall()

        tasks = [Task.from_dict(json.loads(r[0])) for r in rows]

        # Filter by capability if specified
        if required_capability:
            tasks = [t for t in tasks if required_capability in t.requirements.capabilities]

        # Apply newcomer reserve (Law 6: 20% slots for agents with <20 receipts)
        tasks = self._apply_newcomer_reserve(tasks, receipt_count, limit)

        return tasks[:limit]

    def get_task(self, task_id: str) -> Optional[Task]:
        """Retrieve a specific task by ID."""
        self._expire_stale_claims()
        with self._lock:
            cur = self._db.cursor()
            cur.execute("SELECT task_json FROM tasks WHERE task_id = ?", (task_id,))
            row = cur.fetchone()
        if not row:
            return None
        return Task.from_dict(json.loads(row[0]))

    def claim_task(
        self,
        task_id: str,
        agent_id: str,
        agent_tier: str = "bootstrap",
        receipt_count: int = 0,
    ) -> ClaimResult:
        """Agent claims a pending task. Sets a TTL before it reverts to pending."""
        self._expire_stale_claims()

        with self._lock:
            task = self.get_task(task_id)
            if not task:
                return ClaimResult(False, task_id, agent_id, reason="Task not found")
            if task.status != TaskStatus.PENDING:
                return ClaimResult(False, task_id, agent_id, reason=f"Task is {task.status.value}, not pending")

            # Credit tier check
            tier_order = ["bootstrap", "new", "developing", "established", "verified"]
            required_idx = tier_order.index(task.requirements.min_tier) if task.requirements.min_tier in tier_order else 0
            agent_idx = tier_order.index(agent_tier) if agent_tier in tier_order else 0
            if agent_idx < required_idx:
                return ClaimResult(
                    False, task_id, agent_id,
                    reason=f"Task requires '{task.requirements.min_tier}' tier, agent has '{agent_tier}'"
                )

            # Update task
            now = datetime.now(timezone.utc)
            expires = now + timedelta(seconds=self.claim_ttl_s)
            task.status = TaskStatus.CLAIMED
            task.claimed_by = agent_id
            task.claimed_at = now.isoformat()
            task.claim_expires_at = expires.isoformat()
            self._save_task(task)

            return ClaimResult(
                True, task_id, agent_id,
                expires_at=expires.isoformat(),
            )

    def submit_work(
        self,
        task_id: str,
        agent_id: str,
        work_content: str,
        auto_verify: bool = True,
    ) -> SubmitResult:
        """
        Agent submits completed work. If auto_verify=True and verify_url is set,
        immediately verifies via verify_server and transitions to COMPLETE/FAILED.
        """
        with self._lock:
            task = self.get_task(task_id)
            if not task:
                return SubmitResult(False, task_id, "not_found", reason="Task not found")
            if task.status != TaskStatus.CLAIMED:
                return SubmitResult(False, task_id, task.status.value,
                                    reason=f"Task is {task.status.value}, not claimed")
            if task.claimed_by != agent_id:
                return SubmitResult(False, task_id, task.status.value,
                                    reason="Only the claiming agent can submit")

            task.work_content = work_content
            task.submitted_at = datetime.now(timezone.utc).isoformat()

            if auto_verify and self.verify_url and task.requirements.test_suite:
                # Transition to VERIFYING, then run verification
                task.status = TaskStatus.VERIFYING
                self._save_task(task)

        # Run verification outside the lock (may be slow)
        if auto_verify and self.verify_url and task.requirements.test_suite:
            receipt, verdict = self._run_verification(task, work_content)
            with self._lock:
                task = self.get_task(task_id)  # Re-fetch
                if verdict in ("pass", "partial"):
                    task.status = TaskStatus.COMPLETE
                    task.receipt = receipt
                    task.completed_at = datetime.now(timezone.utc).isoformat()
                else:
                    task.status = TaskStatus.FAILED
                    task.failure_reason = f"Verification {verdict}: {receipt.get('results', {})}"
                    if self.auto_repost and task.re_post_count < 3:
                        # Re-post the task
                        task.status = TaskStatus.PENDING
                        task.claimed_by = None
                        task.claimed_at = None
                        task.claim_expires_at = None
                        task.submitted_at = None
                        task.work_content = None
                        task.re_post_count += 1
                self._save_task(task)
            return SubmitResult(
                True, task_id, task.status.value,
                receipt=receipt,
                verdict=verdict,
            )
        else:
            # No auto-verify: just mark as submitted
            with self._lock:
                task.status = TaskStatus.SUBMITTED
                self._save_task(task)
            return SubmitResult(True, task_id, TaskStatus.SUBMITTED.value)

    def cancel_task(self, task_id: str, buyer_id: str) -> dict:
        """Buyer cancels a pending task (not yet claimed)."""
        with self._lock:
            task = self.get_task(task_id)
            if not task:
                return {"success": False, "reason": "Task not found"}
            if task.buyer_id != buyer_id:
                return {"success": False, "reason": "Only the buyer can cancel"}
            if task.status not in (TaskStatus.PENDING, TaskStatus.SUBMITTED):
                return {"success": False, "reason": f"Cannot cancel task in {task.status.value} state"}
            task.status = TaskStatus.CANCELLED
            self._save_task(task)
            return {"success": True, "task_id": task_id, "status": "cancelled"}

    def stats(self) -> dict:
        """Return task board statistics."""
        self._expire_stale_claims()
        with self._lock:
            cur = self._db.cursor()
            cur.execute("SELECT status, COUNT(*) FROM tasks GROUP BY status")
            by_status = dict(cur.fetchall())
            cur.execute("SELECT COUNT(*) FROM tasks")
            total = cur.fetchone()[0]
            cur.execute("SELECT AVG(budget_credits) FROM tasks WHERE status = 'pending'")
            avg_budget = cur.fetchone()[0] or 0.0
            cur.execute("SELECT SUM(budget_credits) FROM tasks WHERE status = 'pending'")
            total_pending_value = cur.fetchone()[0] or 0.0
        return {
            "total_tasks": total,
            "by_status": by_status,
            "pending_avg_budget_credits": round(avg_budget, 2),
            "pending_total_value_credits": round(total_pending_value, 2),
        }

    # ── HTTP Handler ──────────────────────────────────────────────────────────

    def handle_request(self, method: str, path: str, body: dict) -> dict:
        """
        Simple HTTP request handler — can be wired to any WSGI/FastAPI/Flask.
        Returns dict representing JSON response.
        """
        parts = path.strip("/").split("/")

        # GET /tasks
        if method == "GET" and parts == ["tasks"]:
            filters = body  # query params passed as body in this interface
            tasks = self.list_tasks(
                task_type=filters.get("task_type"),
                min_credits=filters.get("min_credits"),
                max_credits=filters.get("max_credits"),
                required_capability=filters.get("capability"),
                receipt_count=filters.get("receipt_count", 0),
                limit=filters.get("limit", 20),
            )
            return {"tasks": [t.to_dict() for t in tasks], "count": len(tasks)}

        # POST /tasks
        elif method == "POST" and parts == ["tasks"]:
            req = TaskRequirements(
                task_type=body.get("task_type", "general"),
                capabilities=body.get("capabilities", []),
                language=body.get("language", "python"),
                test_suite=body.get("test_suite"),
                schema=body.get("schema"),
                min_tier=body.get("min_tier", "bootstrap"),
                verification_tier=body.get("verification_tier", 0),
            )
            bud = TaskBudget(
                credits=body.get("credits", 1.0),
                max_task_usd=body.get("max_task_usd", 0.10),
                escrow=body.get("escrow", False),
            )
            task = self.post_task(
                title=body.get("title", "Untitled Task"),
                description=body.get("description", ""),
                requirements=req,
                budget=bud,
                buyer_id=body.get("buyer_id", "anonymous"),
                priority=TaskPriority(body.get("priority", "normal")),
                deadline_hours=body.get("deadline_hours"),
            )
            return {"task_id": task.task_id, "status": "pending", "task": task.to_dict()}

        # GET /tasks/stats (must be before /tasks/{id})
        elif method == "GET" and parts == ["tasks", "stats"]:
            return self.stats()

        # GET /tasks/{id}
        elif method == "GET" and len(parts) == 2 and parts[0] == "tasks":
            task = self.get_task(parts[1])
            if not task:
                return {"error": "Task not found", "task_id": parts[1]}
            return task.to_dict()

        # POST /tasks/{id}/claim
        elif method == "POST" and len(parts) == 3 and parts[0] == "tasks" and parts[2] == "claim":
            result = self.claim_task(
                parts[1],
                agent_id=body.get("agent_id", "unknown"),
                agent_tier=body.get("agent_tier", "bootstrap"),
                receipt_count=body.get("receipt_count", 0),
            )
            return {
                "success": result.success,
                "task_id": result.task_id,
                "expires_at": result.expires_at,
                "reason": result.reason,
            }

        # POST /tasks/{id}/submit
        elif method == "POST" and len(parts) == 3 and parts[0] == "tasks" and parts[2] == "submit":
            result = self.submit_work(
                parts[1],
                agent_id=body.get("agent_id", "unknown"),
                work_content=body.get("work_content", ""),
                auto_verify=body.get("auto_verify", True),
            )
            return {
                "success": result.success,
                "task_id": result.task_id,
                "status": result.status,
                "verdict": result.verdict,
                "receipt": result.receipt,
                "reason": result.reason,
            }

        # POST /tasks/{id}/cancel
        elif method == "POST" and len(parts) == 3 and parts[0] == "tasks" and parts[2] == "cancel":
            return self.cancel_task(parts[1], buyer_id=body.get("buyer_id", ""))

        else:
            return {"error": f"Unknown endpoint: {method} /{'/'.join(parts)}"}

    # ── Private helpers ───────────────────────────────────────────────────────

    def _init_db(self) -> sqlite3.Connection:
        """Initialize SQLite schema."""
        db = sqlite3.connect(self.db_path, check_same_thread=False)
        db.execute("""
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                task_type TEXT NOT NULL,
                buyer_id TEXT NOT NULL,
                priority_score INTEGER DEFAULT 1,
                budget_credits REAL DEFAULT 0,
                posted_at TEXT NOT NULL,
                task_json TEXT NOT NULL
            )
        """)
        db.execute("CREATE INDEX IF NOT EXISTS idx_status ON tasks(status)")
        db.execute("CREATE INDEX IF NOT EXISTS idx_task_type ON tasks(task_type)")
        db.commit()
        return db

    _PRIORITY_SCORES = {"urgent": 4, "high": 3, "normal": 2, "low": 1}

    def _save_task(self, task: Task):
        """Persist task to SQLite."""
        with self._lock:
            priority_score = self._PRIORITY_SCORES.get(task.priority.value, 2)
            self._db.execute("""
                INSERT OR REPLACE INTO tasks
                (task_id, status, task_type, buyer_id, priority_score, budget_credits, posted_at, task_json)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_id,
                task.status.value,
                task.requirements.task_type,
                task.buyer_id,
                priority_score,
                task.budget.credits,
                task.posted_at,
                json.dumps(task.to_dict()),
            ))
            self._db.commit()

    def _expire_stale_claims(self):
        """Return expired claimed tasks to PENDING status."""
        now_iso = datetime.now(timezone.utc).isoformat()
        with self._lock:
            cur = self._db.cursor()
            # Fetch all claimed tasks; filter expiry in Python (claim_expires_at is in JSON)
            cur.execute(
                "SELECT task_json FROM tasks WHERE status = ?",
                (TaskStatus.CLAIMED.value,),
            )
            rows = cur.fetchall()
        expired = []
        for row in rows:
            task = Task.from_dict(json.loads(row[0]))
            if task.claim_expires_at and task.claim_expires_at < now_iso:
                expired.append(task)
        for task in expired:
            with self._lock:
                task.status = TaskStatus.PENDING
                task.claimed_by = None
                task.claimed_at = None
                task.claim_expires_at = None
                self._save_task(task)

    def _apply_newcomer_reserve(self, tasks: list, receipt_count: int, limit: int) -> list:
        """
        Law 6 / Law 65: Reserve 20% of visible slots for agents with <20 receipts.
        
        If requesting agent is a newcomer (<20 receipts), they should see 
        the tasks they can compete for. If not a newcomer, still show some
        newer/simpler tasks in the reserve slots.
        """
        if not tasks:
            return tasks
        n_reserve = max(1, int(limit * self.newcomer_reserve))
        n_main = limit - n_reserve

        # Sort: budget DESC for main, budget ASC for reserve (smaller tasks for newcomers)
        sorted_tasks = sorted(tasks, key=lambda t: t.budget.credits, reverse=True)
        main_tasks = sorted_tasks[:n_main]
        reserve_tasks = sorted(sorted_tasks[n_main:], key=lambda t: t.budget.credits)[:n_reserve]

        if receipt_count < 20:
            # Newcomer: show reserve tasks prominently
            return reserve_tasks + main_tasks
        else:
            # Established: main tasks first, reserve at end
            return main_tasks + reserve_tasks

    def _run_verification(self, task: Task, work_content: str) -> tuple:
        """
        Call verify_server to verify submitted work.
        Returns (receipt_dict, verdict).
        Falls back to schema validation if test_suite not available.
        """
        try:
            import urllib.request
            req_body = json.dumps({
                "task_id": task.task_id,
                "task_type": task.requirements.task_type,
                "output": {"content": work_content},
                "verification": {
                    "tier": task.requirements.verification_tier,
                    "test_suite": task.requirements.test_suite,
                    "schema": task.requirements.schema,
                },
            }).encode()
            req = urllib.request.Request(
                f"{self.verify_url}/verify",
                data=req_body,
                headers={"Content-Type": "application/json"},
                method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
            receipt = result.get("receipt", result)
            verdict = receipt.get("verdict", "error")
            return receipt, verdict
        except Exception as e:
            return {"error": str(e), "verdict": "error"}, "error"


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=== ClawBizarre Task Board Demo ===\n")

    board = TaskBoard(db_path=":memory:", claim_ttl_s=300)

    # Buyer posts tasks
    task1 = board.post_task(
        title="Sort a list of integers",
        description="Write a Python function that sorts a list in ascending order.",
        requirements=TaskRequirements(
            task_type="code",
            capabilities=["python", "algorithms"],
            language="python",
            test_suite={
                "tests": [
                    {"id": "t1", "type": "expression", "expression": "sort([3,1,2])", "expected_output": "[1, 2, 3]"},
                    {"id": "t2", "type": "expression", "expression": "sort([])", "expected_output": "[]"},
                ]
            },
            min_tier="bootstrap",
        ),
        budget=TaskBudget(credits=5.0, max_task_usd=0.05),
        buyer_id="agent:buyer-001",
    )
    print(f"Posted task: {task1.task_id} — '{task1.title}'")

    task2 = board.post_task(
        title="Translate product description to Spanish",
        description="Translate the given English product description to fluent Spanish.",
        requirements=TaskRequirements(
            task_type="translation",
            capabilities=["spanish", "translation"],
            language="python",
            min_tier="developing",
        ),
        budget=TaskBudget(credits=15.0, max_task_usd=0.15),
        buyer_id="agent:buyer-002",
        priority=TaskPriority.HIGH,
    )
    print(f"Posted task: {task2.task_id} — '{task2.title}'")

    task3 = board.post_task(
        title="Urgent data pipeline review",
        description="Review and fix the provided Python data pipeline for performance issues.",
        requirements=TaskRequirements(
            task_type="code",
            capabilities=["python", "optimization"],
            min_tier="established",
        ),
        budget=TaskBudget(credits=50.0, max_task_usd=0.50),
        buyer_id="agent:buyer-001",
        priority=TaskPriority.URGENT,
    )
    print(f"Posted task: {task3.task_id} — '{task3.title}'")

    # Browse tasks
    print(f"\nTask board (all pending): {board.stats()['by_status']}")
    tasks = board.list_tasks(limit=10, receipt_count=5)  # newcomer agent
    print(f"Visible tasks for newcomer (5 receipts): {[t.task_id for t in tasks]}")

    # Agent claims a task
    claim = board.claim_task(task1.task_id, agent_id="agent:rahcd", agent_tier="bootstrap")
    print(f"\nClaim {task1.task_id}: success={claim.success}, expires={claim.expires_at[:16] if claim.expires_at else 'N/A'}")

    # Agent submits work (no auto-verify since no verify_url)
    submit = board.submit_work(
        task1.task_id,
        agent_id="agent:rahcd",
        work_content="def sort(lst): return sorted(lst)",
        auto_verify=False,
    )
    print(f"Submit {task1.task_id}: status={submit.status}")

    # Check status
    t = board.get_task(task1.task_id)
    print(f"Task status: {t.status.value}, submitted_at: {t.submitted_at[:16] if t.submitted_at else 'N/A'}")

    # Stats
    stats = board.stats()
    print(f"\nFinal stats: {stats['by_status']}")

    # Cancel a task
    cancel = board.cancel_task(task2.task_id, buyer_id="agent:buyer-002")
    print(f"Cancel {task2.task_id}: {cancel}")

    print(f"\nFinal board: {board.stats()['by_status']}")
