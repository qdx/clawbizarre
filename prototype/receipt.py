"""
ClawBizarre Work Receipt v0.2
Structural, self-verifying work receipts for agent-to-agent commerce.
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from typing import Optional


class PricingStrategy:
    REPUTATION_PREMIUM = "reputation_premium"
    QUALITY_PREMIUM = "quality_premium"
    MARKET_RATE = "market_rate"
    UNDERCUT = "undercut"


class VerificationTier(IntEnum):
    SELF_VERIFYING = 0  # Output proves itself (tests pass, code compiles)
    MECHANICAL = 1       # Checkable by machine (format, timing, uptime)
    PEER_REVIEW = 2      # Requires another agent's judgment
    HUMAN_ONLY = 3       # Requires human evaluation


@dataclass
class TestResults:
    passed: int
    failed: int
    suite_hash: str  # sha256 of the test suite itself

    @property
    def success(self) -> bool:
        return self.failed == 0 and self.passed > 0


@dataclass
class RiskEnvelope:
    """Decision context at time of execution (from juanciclawbot's proposal)."""
    counterparty_risk_at_start: float
    counterparty_risk_at_completion: float
    policy_decision: str  # "approve" | "review" | "block"
    policy_version: str   # sha256 of policy file

    @property
    def risk_delta(self) -> float:
        return self.counterparty_risk_at_completion - self.counterparty_risk_at_start


@dataclass
class Timing:
    """Task timing for reliability reputation (timestamps are Tier 0 self-verifying)."""
    proposed_at: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    deadline: Optional[str] = None

    @property
    def on_time(self) -> Optional[bool]:
        """Was the task completed before deadline? None if no deadline."""
        if not self.deadline or not self.completed_at:
            return None
        return self.completed_at <= self.deadline

    @property
    def duration_seconds(self) -> Optional[float]:
        """Execution duration in seconds."""
        if not self.started_at or not self.completed_at:
            return None
        start = datetime.fromisoformat(self.started_at)
        end = datetime.fromisoformat(self.completed_at)
        return (end - start).total_seconds()


@dataclass
class PaymentReference:
    """Link to x402 or other payment protocol settlement."""
    protocol: str = "x402"           # "x402" | "ap2" | "lightning" | "manual"
    payment_id: Optional[str] = None  # x402 payment ID or tx hash
    amount: Optional[str] = None      # e.g. "0.05"
    currency: Optional[str] = None    # e.g. "USDC"
    chain: Optional[str] = None       # e.g. "base", "solana"
    settled: bool = False


@dataclass
class Attestation:
    agent_id: str
    attestation_type: str  # "peer_review" | "mechanical_check" | "co_sign"
    result: str            # "approved" | "rejected" | "conditional"
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    details: Optional[str] = None


@dataclass
class WorkReceipt:
    """A structural, self-verifying record of agent work."""
    agent_id: str
    task_type: str
    verification_tier: VerificationTier
    input_hash: str
    output_hash: str
    version: str = "0.3"
    pricing_strategy: str = PricingStrategy.REPUTATION_PREMIUM
    platform: str = "direct"
    receipt_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    test_results: Optional[TestResults] = None
    risk_envelope: Optional[RiskEnvelope] = None
    timing: Optional[Timing] = None
    environment_hash: Optional[str] = None  # nix/docker hash for reproducibility
    agent_config_hash: Optional[str] = None
    payment: Optional[PaymentReference] = None
    attestations: list[Attestation] = field(default_factory=list)

    def add_attestation(self, attestation: Attestation):
        self.attestations.append(attestation)

    def verify_tier0(self) -> bool:
        """Tier 0: self-verifying. Tests must exist and pass."""
        if self.verification_tier != VerificationTier.SELF_VERIFYING:
            return False
        if self.test_results is None:
            return False
        return self.test_results.success

    def to_json(self) -> str:
        d = asdict(self)
        d["verification_tier"] = int(d["verification_tier"])
        return json.dumps(d, indent=2)

    @property
    def content_hash(self) -> str:
        """Hash of receipt content (excluding attestations) for signing."""
        d = asdict(self)
        d.pop("attestations", None)
        d["verification_tier"] = int(d["verification_tier"])
        canonical = json.dumps(d, sort_keys=True)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"

    @classmethod
    def from_json(cls, data: str) -> "WorkReceipt":
        d = json.loads(data)
        d["verification_tier"] = VerificationTier(d["verification_tier"])
        if d.get("test_results"):
            d["test_results"] = TestResults(**d["test_results"])
        if d.get("risk_envelope"):
            d["risk_envelope"] = RiskEnvelope(**d["risk_envelope"])
        if d.get("timing"):
            d["timing"] = Timing(**d["timing"])
        if d.get("attestations"):
            d["attestations"] = [Attestation(**a) for a in d["attestations"]]
        return cls(**d)


def hash_content(content: str) -> str:
    return f"sha256:{hashlib.sha256(content.encode()).hexdigest()}"


# --- Receipt Chain (append-only log, Azimuth's pattern) ---

@dataclass
class ReceiptChain:
    """Append-only chain of receipts with hash integrity (per Azimuth/AGIRAILS)."""
    receipts: list[WorkReceipt] = field(default_factory=list)
    chain_hashes: list[str] = field(default_factory=list)

    def append(self, receipt: WorkReceipt):
        prev = self.chain_hashes[-1] if self.chain_hashes else "genesis"
        entry = f"{prev}:{receipt.content_hash}"
        chain_hash = f"sha256:{hashlib.sha256(entry.encode()).hexdigest()}"
        self.receipts.append(receipt)
        self.chain_hashes.append(chain_hash)

    def verify_integrity(self) -> bool:
        """Verify the entire chain hasn't been tampered with."""
        prev = "genesis"
        for receipt, expected_hash in zip(self.receipts, self.chain_hashes):
            entry = f"{prev}:{receipt.content_hash}"
            computed = f"sha256:{hashlib.sha256(entry.encode()).hexdigest()}"
            if computed != expected_hash:
                return False
            prev = expected_hash
        return True

    @property
    def length(self) -> int:
        return len(self.receipts)

    def tier_breakdown(self) -> dict[int, int]:
        counts: dict[int, int] = {}
        for r in self.receipts:
            t = int(r.verification_tier)
            counts[t] = counts.get(t, 0) + 1
        return counts

    def success_rate(self) -> float:
        if not self.receipts:
            return 0.0
        verified = sum(1 for r in self.receipts if r.verify_tier0())
        return verified / len(self.receipts)

    def strategy_consistency(self) -> float:
        """Fraction of receipts with same pricing strategy as majority.
        Low consistency = frequent strategy switching = buyer discount (Law 1)."""
        if not self.receipts:
            return 1.0
        strategies = [r.pricing_strategy for r in self.receipts]
        from collections import Counter
        most_common_count = Counter(strategies).most_common(1)[0][1]
        return most_common_count / len(strategies)

    def strategy_changes(self) -> int:
        """Number of times pricing strategy changed between consecutive receipts."""
        if len(self.receipts) < 2:
            return 0
        changes = 0
        for i in range(1, len(self.receipts)):
            if self.receipts[i].pricing_strategy != self.receipts[i-1].pricing_strategy:
                changes += 1
        return changes

    def on_time_rate(self) -> Optional[float]:
        """Fraction of timed tasks completed before deadline."""
        timed = [r for r in self.receipts if r.timing and r.timing.on_time is not None]
        if not timed:
            return None
        return sum(1 for r in timed if r.timing.on_time) / len(timed)


# --- Demo ---

if __name__ == "__main__":
    # Create a Tier 0 receipt for a code review task
    receipt = WorkReceipt(
        agent_id="sigil:rahcd_pubkey_abc123",
        task_type="code_review",
        verification_tier=VerificationTier.SELF_VERIFYING,
        input_hash=hash_content("def foo(): return 1"),
        output_hash=hash_content("3 issues found: ..."),
        platform="moltbook",
        test_results=TestResults(
            passed=12,
            failed=0,
            suite_hash=hash_content("test_suite_v1")
        ),
        risk_envelope=RiskEnvelope(
            counterparty_risk_at_start=0.12,
            counterparty_risk_at_completion=0.08,
            policy_decision="approve",
            policy_version=hash_content("policy_v2.yaml")
        ),
        environment_hash="nix:sha256:abc123def456"
    )

    # Add a peer attestation
    receipt.add_attestation(Attestation(
        agent_id="sigil:peer_agent_xyz",
        attestation_type="co_sign",
        result="approved",
        details="Reviewed output, confirms findings"
    ))

    print("=== Work Receipt v0.2 ===")
    print(receipt.to_json())
    print(f"\nContent hash: {receipt.content_hash}")
    print(f"Tier 0 verified: {receipt.verify_tier0()}")

    # Build a receipt chain
    chain = ReceiptChain()
    chain.append(receipt)

    # Add another receipt
    receipt2 = WorkReceipt(
        agent_id="sigil:rahcd_pubkey_abc123",
        task_type="translation",
        verification_tier=VerificationTier.SELF_VERIFYING,
        input_hash=hash_content("Hello world"),
        output_hash=hash_content("你好世界"),
        test_results=TestResults(passed=5, failed=0, suite_hash=hash_content("translation_tests")),
    )
    chain.append(receipt2)

    print(f"\n=== Receipt Chain ===")
    print(f"Length: {chain.length}")
    print(f"Integrity: {chain.verify_integrity()}")
    print(f"Tier breakdown: {chain.tier_breakdown()}")
    print(f"Success rate: {chain.success_rate():.0%}")

    # Round-trip test
    json_str = receipt.to_json()
    restored = WorkReceipt.from_json(json_str)
    assert restored.content_hash == receipt.content_hash
    print(f"\nRound-trip: ✓")
