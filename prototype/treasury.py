"""
ClawBizarre Treasury Agent — Phase 3 Component
Specialized agent handling custody, approval, audit, and budget enforcement.

The Treasury Agent is a POLICY EXECUTOR, not a decision maker.
Policies are human-authored, version-controlled, auditable.
The agent evaluates rules and routes payments.

Design principles (from solo brainstorm 2026-02-18):
- Deterministic policy execution (no judgment, no optimization)
- Multi-sig pattern for high-value transactions
- Hash-chained audit log (per Azimuth/AGIRAILS)
- Hierarchical budget delegation (fleet-level → agent-level)
"""

import json
import hashlib
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from typing import Optional
from enum import Enum


class ApprovalDecision(str, Enum):
    AUTO_APPROVE = "auto_approve"  # Below threshold, policy allows
    PENDING = "pending"           # Requires human approval
    REJECTED = "rejected"         # Policy violation
    ESCALATED = "escalated"       # Multi-sig required


class SpendCategory(str, Enum):
    COMPUTE = "compute"           # API calls, inference
    SERVICE = "service"           # Hiring another agent
    INFRASTRUCTURE = "infrastructure"  # Hosting, storage
    DATA = "data"                 # Data access, APIs
    OTHER = "other"


@dataclass
class BudgetPolicy:
    """Human-authored spending policy for a fleet or individual agent."""
    policy_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    version: str = "1.0"

    # Per-transaction limits
    auto_approve_threshold: float = 1.0  # USD — auto-approve below this
    escalation_threshold: float = 10.0   # USD — multi-sig above this

    # Period limits
    daily_budget: float = 50.0           # Max spend per day
    weekly_budget: float = 200.0         # Max spend per week
    monthly_budget: float = 500.0        # Max spend per month

    # Category limits (optional overrides)
    category_limits: dict[str, float] = field(default_factory=lambda: {
        "compute": 30.0,   # daily
        "service": 20.0,
        "infrastructure": 10.0,
        "data": 10.0,
        "other": 5.0,
    })

    # Agent-specific sub-budgets (hierarchical delegation)
    agent_budgets: dict[str, float] = field(default_factory=dict)
    # e.g. {"agent_001": 10.0, "agent_002": 5.0} — daily limits per agent

    # Allowlists / blocklists
    allowed_counterparties: Optional[list[str]] = None  # None = allow all
    blocked_counterparties: list[str] = field(default_factory=list)

    # Fallback model ladder (per fcc7f265's budget guardrails)
    model_fallback_at_budget_pct: float = 0.75  # Switch to cheaper model at 75% budget

    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    created_by: str = "human_sponsor"  # Must be human

    @property
    def content_hash(self) -> str:
        d = asdict(self)
        canonical = json.dumps(d, sort_keys=True)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass
class SpendRequest:
    """A request to spend money, submitted by an agent to the Treasury."""
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    requesting_agent: str = ""
    counterparty: str = ""
    amount: float = 0.0
    currency: str = "USD"
    category: SpendCategory = SpendCategory.OTHER
    description: str = ""
    task_receipt_id: Optional[str] = None  # Link to work receipt
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class SpendDecision:
    """Treasury's response to a spend request."""
    request_id: str
    decision: ApprovalDecision
    reason: str
    policy_version: str
    decided_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    remaining_daily_budget: float = 0.0
    remaining_agent_budget: Optional[float] = None


@dataclass
class AuditEntry:
    """Append-only, hash-chained audit log entry (per Azimuth/AGIRAILS)."""
    entry_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    entry_type: str = ""  # "spend_request" | "decision" | "policy_change" | "heartbeat"
    data: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    previous_hash: str = "genesis"
    entry_hash: str = ""

    def compute_hash(self) -> str:
        content = json.dumps({
            "entry_id": self.entry_id,
            "entry_type": self.entry_type,
            "data": self.data,
            "timestamp": self.timestamp,
            "previous_hash": self.previous_hash,
        }, sort_keys=True)
        return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


class TreasuryAgent:
    """
    Policy executor for fleet spending governance.
    
    NOT a decision maker. Evaluates deterministic rules only.
    All policies are human-authored and version-controlled.
    """

    def __init__(self, policy: BudgetPolicy):
        self.policy = policy
        self.audit_log: list[AuditEntry] = []
        self.spend_history: list[tuple[SpendRequest, SpendDecision]] = []

        # Log policy creation
        self._audit("policy_loaded", {
            "policy_id": policy.policy_id,
            "policy_hash": policy.content_hash,
            "daily_budget": policy.daily_budget,
        })

    def _audit(self, entry_type: str, data: dict) -> AuditEntry:
        """Append to hash-chained audit log."""
        prev_hash = self.audit_log[-1].entry_hash if self.audit_log else "genesis"
        entry = AuditEntry(
            entry_type=entry_type,
            data=data,
            previous_hash=prev_hash,
        )
        entry.entry_hash = entry.compute_hash()
        self.audit_log.append(entry)
        return entry

    def verify_audit_chain(self) -> bool:
        """Verify integrity of entire audit log."""
        prev = "genesis"
        for entry in self.audit_log:
            if entry.previous_hash != prev:
                return False
            if entry.compute_hash() != entry.entry_hash:
                return False
            prev = entry.entry_hash
        return True

    def _daily_spend(self, agent_id: Optional[str] = None) -> float:
        """Total spend in last 24 hours, optionally filtered by agent."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        total = 0.0
        for req, dec in self.spend_history:
            if dec.decision != ApprovalDecision.AUTO_APPROVE:
                continue
            req_time = datetime.fromisoformat(req.timestamp)
            if req_time < cutoff:
                continue
            if agent_id and req.requesting_agent != agent_id:
                continue
            total += req.amount
        return total

    def _category_spend_today(self, category: SpendCategory) -> float:
        """Total spend in category in last 24 hours."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=24)
        total = 0.0
        for req, dec in self.spend_history:
            if dec.decision != ApprovalDecision.AUTO_APPROVE:
                continue
            if req.category != category:
                continue
            req_time = datetime.fromisoformat(req.timestamp)
            if req_time < cutoff:
                continue
            total += req.amount
        return total

    def evaluate(self, request: SpendRequest) -> SpendDecision:
        """
        Evaluate a spend request against policy. Pure function of policy rules.
        No judgment, no optimization, no heuristics.
        """
        # Log the request
        self._audit("spend_request", {
            "request_id": request.request_id,
            "agent": request.requesting_agent,
            "amount": request.amount,
            "counterparty": request.counterparty,
            "category": request.category.value,
        })

        # Check 1: Blocked counterparty
        if request.counterparty in self.policy.blocked_counterparties:
            return self._decide(request, ApprovalDecision.REJECTED,
                              f"Counterparty {request.counterparty} is blocked")

        # Check 2: Allowed counterparty list (if set)
        if (self.policy.allowed_counterparties is not None and
                request.counterparty not in self.policy.allowed_counterparties):
            return self._decide(request, ApprovalDecision.REJECTED,
                              f"Counterparty {request.counterparty} not in allowlist")

        # Check 3: Daily budget
        daily_spent = self._daily_spend()
        if daily_spent + request.amount > self.policy.daily_budget:
            return self._decide(request, ApprovalDecision.REJECTED,
                              f"Would exceed daily budget ({daily_spent:.2f} + {request.amount:.2f} > {self.policy.daily_budget:.2f})")

        # Check 4: Agent-specific budget
        agent_budget = self.policy.agent_budgets.get(request.requesting_agent)
        agent_spent = None
        if agent_budget is not None:
            agent_spent = self._daily_spend(request.requesting_agent)
            if agent_spent + request.amount > agent_budget:
                return self._decide(request, ApprovalDecision.REJECTED,
                                  f"Would exceed agent budget ({agent_spent:.2f} + {request.amount:.2f} > {agent_budget:.2f})",
                                  remaining_agent=agent_budget - agent_spent)

        # Check 5: Category limit
        cat_limit = self.policy.category_limits.get(request.category.value)
        if cat_limit is not None:
            cat_spent = self._category_spend_today(request.category)
            if cat_spent + request.amount > cat_limit:
                return self._decide(request, ApprovalDecision.REJECTED,
                                  f"Would exceed {request.category.value} category limit ({cat_spent:.2f} + {request.amount:.2f} > {cat_limit:.2f})")

        # Check 6: Escalation threshold (needs multi-sig)
        if request.amount >= self.policy.escalation_threshold:
            return self._decide(request, ApprovalDecision.ESCALATED,
                              f"Amount ${request.amount:.2f} >= escalation threshold ${self.policy.escalation_threshold:.2f}")

        # Check 7: Auto-approve threshold
        if request.amount <= self.policy.auto_approve_threshold:
            decision = self._decide(request, ApprovalDecision.AUTO_APPROVE,
                                   "Below auto-approve threshold",
                                   remaining_agent=agent_budget - agent_spent if agent_spent is not None else None)
            self.spend_history.append((request, decision))
            return decision

        # Between auto-approve and escalation: needs human approval
        return self._decide(request, ApprovalDecision.PENDING,
                          f"Amount ${request.amount:.2f} between auto-approve (${self.policy.auto_approve_threshold:.2f}) and escalation (${self.policy.escalation_threshold:.2f})")

    def _decide(self, request: SpendRequest, decision: ApprovalDecision,
                reason: str, remaining_agent: Optional[float] = None) -> SpendDecision:
        daily_spent = self._daily_spend()
        result = SpendDecision(
            request_id=request.request_id,
            decision=decision,
            reason=reason,
            policy_version=self.policy.content_hash,
            remaining_daily_budget=round(self.policy.daily_budget - daily_spent, 2),
            remaining_agent_budget=round(remaining_agent, 2) if remaining_agent is not None else None,
        )
        self._audit("decision", {
            "request_id": request.request_id,
            "decision": decision.value,
            "reason": reason,
        })
        return result

    def budget_status(self) -> dict:
        """Current budget utilization."""
        daily = self._daily_spend()
        return {
            "daily_spent": round(daily, 2),
            "daily_budget": self.policy.daily_budget,
            "daily_remaining": round(self.policy.daily_budget - daily, 2),
            "daily_pct": round(daily / self.policy.daily_budget * 100, 1),
            "model_fallback_triggered": daily / self.policy.daily_budget >= self.policy.model_fallback_at_budget_pct,
            "audit_chain_length": len(self.audit_log),
            "audit_chain_valid": self.verify_audit_chain(),
            "policy_hash": self.policy.content_hash,
        }

    def update_policy(self, new_policy: BudgetPolicy):
        """Update policy (must be human-initiated)."""
        self._audit("policy_change", {
            "old_hash": self.policy.content_hash,
            "new_hash": new_policy.content_hash,
            "changed_by": new_policy.created_by,
        })
        self.policy = new_policy


# --- Demo ---

if __name__ == "__main__":
    # Create a fleet policy
    policy = BudgetPolicy(
        auto_approve_threshold=2.0,
        escalation_threshold=25.0,
        daily_budget=50.0,
        agent_budgets={
            "agent_research": 15.0,
            "agent_code": 20.0,
            "agent_email": 5.0,
        },
        blocked_counterparties=["known_scammer_001"],
    )

    treasury = TreasuryAgent(policy)

    print("=== Treasury Agent ===")
    print(f"Policy hash: {policy.content_hash[:40]}...")
    print(f"Auto-approve: <=${policy.auto_approve_threshold}")
    print(f"Escalation: >=${policy.escalation_threshold}")
    print(f"Daily budget: ${policy.daily_budget}")

    # Test cases
    tests = [
        SpendRequest(requesting_agent="agent_research", counterparty="api_provider_a",
                    amount=0.50, category=SpendCategory.COMPUTE, description="API call"),
        SpendRequest(requesting_agent="agent_code", counterparty="agent_translator",
                    amount=5.0, category=SpendCategory.SERVICE, description="Translation job"),
        SpendRequest(requesting_agent="agent_email", counterparty="known_scammer_001",
                    amount=0.10, category=SpendCategory.OTHER, description="Suspicious request"),
        SpendRequest(requesting_agent="agent_research", counterparty="data_vendor",
                    amount=30.0, category=SpendCategory.DATA, description="Large dataset"),
    ]

    print("\n=== Spend Requests ===")
    for req in tests:
        decision = treasury.evaluate(req)
        icon = {"auto_approve": "✅", "pending": "⏳", "rejected": "❌", "escalated": "🔺"}
        print(f"  {icon[decision.decision.value]} ${req.amount:.2f} from {req.requesting_agent} → {req.counterparty}")
        print(f"     {decision.decision.value}: {decision.reason}")
        if decision.remaining_agent_budget is not None:
            print(f"     Agent budget remaining: ${decision.remaining_agent_budget:.2f}")

    # Budget status
    print("\n=== Budget Status ===")
    status = treasury.budget_status()
    for k, v in status.items():
        print(f"  {k}: {v}")

    # Audit chain integrity
    print(f"\n=== Audit Log ({len(treasury.audit_log)} entries) ===")
    print(f"  Chain valid: {treasury.verify_audit_chain()}")
    for entry in treasury.audit_log[:5]:
        print(f"  [{entry.entry_type}] {json.dumps(entry.data)[:80]}")
    if len(treasury.audit_log) > 5:
        print(f"  ... and {len(treasury.audit_log) - 5} more entries")

    print("\n✓ Treasury Agent complete")
