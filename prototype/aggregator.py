"""
ClawBizarre Reputation Aggregator v0.1
Phase 4: Bridges receipt chains → reputation scores.

Takes raw receipt chains and produces:
1. Domain-specific reputation with decay
2. Strategy consistency penalty
3. Reliability score (on-time rate)
4. Composite trust score for discovery ranking
5. Portable reputation snapshots (signed, timestamped)

This is the missing link between receipts (Phase 1) and discovery (Phase 2).
"""

import hashlib
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
from collections import defaultdict

from receipt import WorkReceipt, ReceiptChain, VerificationTier, PricingStrategy
from reputation import DecayingReputation, DomainReputation, MerkleTree


# --- Aggregation Config ---

# Default domain correlations (from simulation findings)
DEFAULT_DOMAIN_CORRELATIONS = {
    ("code_review", "code_generation"): 0.7,
    ("code_review", "research"): 0.4,
    ("code_generation", "research"): 0.3,
    ("translation", "research"): 0.2,
    ("data_validation", "code_review"): 0.5,
    ("data_validation", "research"): 0.3,
    ("monitoring", "data_validation"): 0.4,
}

# Composite score weights (from discovery.py scoring, empirically validated in simulations)
DEFAULT_WEIGHTS = {
    "success_rate": 0.40,
    "strategy_consistency": 0.25,
    "on_time_rate": 0.20,
    "chain_length_bonus": 0.15,
}

# Strategy switching penalty (Law 1 from simulations: switching is poison)
STRATEGY_SWITCH_PENALTY = 0.30  # 30% rep penalty per switch (v10 finding)


@dataclass
class ReputationSnapshot:
    """A portable, timestamped reputation summary.
    
    Can be shared across platforms without exposing full receipt chain.
    Includes Merkle root for verifiability.
    """
    agent_id: str
    timestamp: str
    merkle_root: str
    chain_length: int
    
    # Composite scores
    composite_score: float
    domain_scores: dict[str, float]
    
    # Component signals
    success_rate: float
    strategy_consistency: float
    on_time_rate: Optional[float]
    strategy_changes: int
    
    # Trust tier (from discovery.py)
    trust_tier: str  # "newcomer" | "established" | "veteran"
    
    # Verification breakdown
    tier_breakdown: dict[str, int]  # tier -> count
    
    # Snapshot metadata
    snapshot_version: str = "0.1"
    
    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> "ReputationSnapshot":
        return cls(**json.loads(data))
    
    @property
    def content_hash(self) -> str:
        d = asdict(self)
        canonical = json.dumps(d, sort_keys=True)
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


class ReputationAggregator:
    """Processes receipt chains into reputation scores.
    
    Design principles:
    - Deterministic: same chain always produces same scores
    - Structural: based on receipt metadata, not subjective ratings
    - Decaying: recent work weighs more (half_life from DecayingReputation)
    - Strategy-aware: penalizes switching (Law 1)
    - Domain-specific: cross-domain transfer with configurable correlations
    """
    
    def __init__(
        self,
        half_life_days: float = 30.0,
        domain_correlations: Optional[dict] = None,
        weights: Optional[dict] = None,
    ):
        self.half_life_days = half_life_days
        self.domain_correlations = domain_correlations or DEFAULT_DOMAIN_CORRELATIONS
        self.weights = weights or DEFAULT_WEIGHTS
    
    def aggregate(self, chain: ReceiptChain, now: Optional[float] = None) -> ReputationSnapshot:
        """Process a receipt chain into a ReputationSnapshot."""
        now = now or time.time()
        
        if not chain.receipts:
            return self._empty_snapshot("unknown", now)
        
        agent_id = chain.receipts[0].agent_id
        
        # 1. Build domain reputation from receipts
        domain_rep = DomainReputation(
            half_life_days=self.half_life_days,
            domain_correlations=self.domain_correlations,
        )
        
        for receipt in chain.receipts:
            ts = self._receipt_timestamp(receipt)
            success = receipt.verify_tier0() if receipt.verification_tier == VerificationTier.SELF_VERIFYING else True
            # Weight by verification tier (higher tier = more trust signal)
            weight = 1.0 + 0.2 * int(receipt.verification_tier)
            domain_rep.record(receipt.task_type, success, timestamp=ts, weight=weight)
        
        # 2. Compute component signals
        success_rate = chain.success_rate()
        consistency = chain.strategy_consistency()
        on_time = chain.on_time_rate()
        changes = chain.strategy_changes()
        
        # 3. Apply strategy switching penalty
        adjusted_consistency = consistency * (1 - STRATEGY_SWITCH_PENALTY) ** changes
        
        # 4. Chain length bonus (logarithmic — diminishing returns)
        import math
        length_bonus = min(1.0, math.log1p(chain.length) / math.log1p(50))  # caps at 50 receipts
        
        # 5. Composite score
        w = self.weights
        composite = (
            w["success_rate"] * success_rate
            + w["strategy_consistency"] * adjusted_consistency
            + w["on_time_rate"] * (on_time if on_time is not None else 0.8)  # default assumption
            + w["chain_length_bonus"] * length_bonus
        )
        
        # 6. Domain scores
        domains = set(r.task_type for r in chain.receipts)
        domain_scores = {d: domain_rep.score(d, now) for d in domains}
        
        # 7. Trust tier
        if chain.length < 20:
            trust_tier = "newcomer"
        elif chain.length < 50:
            trust_tier = "established"
        else:
            trust_tier = "veteran"
        
        # 8. Merkle root for verifiability
        leaf_hashes = [r.content_hash.replace("sha256:", "") for r in chain.receipts]
        merkle = MerkleTree(leaf_hashes)
        
        # 9. Tier breakdown
        tb = chain.tier_breakdown()
        tier_breakdown = {f"tier_{k}": v for k, v in tb.items()}
        
        from datetime import datetime, timezone
        return ReputationSnapshot(
            agent_id=agent_id,
            timestamp=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            merkle_root=merkle.root,
            chain_length=chain.length,
            composite_score=round(composite, 4),
            domain_scores={k: round(v, 4) for k, v in domain_scores.items()},
            success_rate=round(success_rate, 4),
            strategy_consistency=round(adjusted_consistency, 4),
            on_time_rate=round(on_time, 4) if on_time is not None else None,
            strategy_changes=changes,
            trust_tier=trust_tier,
            tier_breakdown=tier_breakdown,
        )
    
    def _empty_snapshot(self, agent_id: str, now: float) -> ReputationSnapshot:
        from datetime import datetime, timezone
        return ReputationSnapshot(
            agent_id=agent_id,
            timestamp=datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            merkle_root="",
            chain_length=0,
            composite_score=0.0,
            domain_scores={},
            success_rate=0.0,
            strategy_consistency=1.0,
            on_time_rate=None,
            strategy_changes=0,
            trust_tier="newcomer",
            tier_breakdown={},
        )
    
    def _receipt_timestamp(self, receipt: WorkReceipt) -> float:
        """Extract unix timestamp from receipt."""
        from datetime import datetime, timezone
        try:
            dt = datetime.fromisoformat(receipt.timestamp)
            return dt.timestamp()
        except (ValueError, AttributeError):
            return time.time()
    
    def compare(self, snap_a: ReputationSnapshot, snap_b: ReputationSnapshot) -> dict:
        """Compare two snapshots (useful for reputation change tracking)."""
        return {
            "composite_delta": snap_b.composite_score - snap_a.composite_score,
            "chain_growth": snap_b.chain_length - snap_a.chain_length,
            "success_rate_delta": snap_b.success_rate - snap_a.success_rate,
            "consistency_delta": snap_b.strategy_consistency - snap_a.strategy_consistency,
            "tier_change": None if snap_a.trust_tier == snap_b.trust_tier 
                          else f"{snap_a.trust_tier} → {snap_b.trust_tier}",
            "new_domains": list(set(snap_b.domain_scores.keys()) - set(snap_a.domain_scores.keys())),
        }


# --- Tests ---

def test_basic_aggregation():
    """Test aggregation of a simple receipt chain."""
    from receipt import TestResults, Timing, hash_content
    from datetime import datetime, timezone, timedelta
    
    chain = ReceiptChain()
    now = datetime.now(timezone.utc)
    
    # 10 successful code reviews over 20 days
    for i in range(10):
        ts = (now - timedelta(days=20-i*2)).isoformat()
        r = WorkReceipt(
            agent_id="sigil:test_agent_001",
            task_type="code_review",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content(f"input_{i}"),
            output_hash=hash_content(f"output_{i}"),
            timestamp=ts,
            pricing_strategy=PricingStrategy.REPUTATION_PREMIUM,
            test_results=TestResults(passed=5, failed=0, suite_hash=hash_content("suite")),
            timing=Timing(
                started_at=ts,
                completed_at=ts,
                deadline=(now - timedelta(days=20-i*2) + timedelta(hours=1)).isoformat(),
            ),
        )
        chain.append(r)
    
    agg = ReputationAggregator()
    snap = agg.aggregate(chain)
    
    print("=== Basic Aggregation ===")
    print(f"Agent: {snap.agent_id}")
    print(f"Composite: {snap.composite_score}")
    print(f"Success rate: {snap.success_rate}")
    print(f"Consistency: {snap.strategy_consistency}")
    print(f"On-time: {snap.on_time_rate}")
    print(f"Trust tier: {snap.trust_tier}")
    print(f"Domain scores: {snap.domain_scores}")
    print(f"Merkle root: {snap.merkle_root[:16]}...")
    print(f"Chain length: {snap.chain_length}")
    
    assert snap.composite_score > 0.7, f"Expected high score, got {snap.composite_score}"
    assert snap.success_rate == 1.0
    assert snap.strategy_consistency == 1.0
    assert snap.trust_tier == "newcomer"  # 10 < 20
    assert "code_review" in snap.domain_scores
    
    print("✓ Basic aggregation passed\n")
    return snap


def test_strategy_switching_penalty():
    """Test that strategy switching tanks reputation (Law 1)."""
    from receipt import TestResults, hash_content
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    
    # Chain with strategy switches
    switching_chain = ReceiptChain()
    strategies = [
        PricingStrategy.REPUTATION_PREMIUM,
        PricingStrategy.UNDERCUT,  # switch 1
        PricingStrategy.REPUTATION_PREMIUM,  # switch 2
        PricingStrategy.MARKET_RATE,  # switch 3
        PricingStrategy.UNDERCUT,  # switch 4
    ] * 2  # 10 receipts, 8 switches total
    
    for i, strat in enumerate(strategies):
        r = WorkReceipt(
            agent_id="sigil:switcher",
            task_type="code_review",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content(f"in_{i}"),
            output_hash=hash_content(f"out_{i}"),
            timestamp=(now - timedelta(days=10-i)).isoformat(),
            pricing_strategy=strat,
            test_results=TestResults(passed=3, failed=0, suite_hash=hash_content("s")),
        )
        switching_chain.append(r)
    
    # Chain without switches
    stable_chain = ReceiptChain()
    for i in range(10):
        r = WorkReceipt(
            agent_id="sigil:stable",
            task_type="code_review",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content(f"in_{i}"),
            output_hash=hash_content(f"out_{i}"),
            timestamp=(now - timedelta(days=10-i)).isoformat(),
            pricing_strategy=PricingStrategy.REPUTATION_PREMIUM,
            test_results=TestResults(passed=3, failed=0, suite_hash=hash_content("s")),
        )
        stable_chain.append(r)
    
    agg = ReputationAggregator()
    switch_snap = agg.aggregate(switching_chain)
    stable_snap = agg.aggregate(stable_chain)
    
    print("=== Strategy Switching Penalty ===")
    print(f"Stable agent:    composite={stable_snap.composite_score}, consistency={stable_snap.strategy_consistency}, changes={stable_snap.strategy_changes}")
    print(f"Switching agent:  composite={switch_snap.composite_score}, consistency={switch_snap.strategy_consistency}, changes={switch_snap.strategy_changes}")
    
    assert switch_snap.composite_score < stable_snap.composite_score, "Switcher should score lower"
    assert switch_snap.strategy_changes > 0
    assert stable_snap.strategy_changes == 0
    
    diff = agg.compare(switch_snap, stable_snap)
    print(f"Delta: {diff}")
    
    print("✓ Strategy switching penalty passed\n")


def test_multi_domain():
    """Test domain-specific aggregation with cross-domain transfer."""
    from receipt import TestResults, hash_content
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    chain = ReceiptChain()
    
    # 15 code reviews + 5 research tasks
    for i in range(15):
        r = WorkReceipt(
            agent_id="sigil:specialist",
            task_type="code_review",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content(f"cr_{i}"),
            output_hash=hash_content(f"cr_out_{i}"),
            timestamp=(now - timedelta(days=20-i)).isoformat(),
            test_results=TestResults(passed=5, failed=0, suite_hash=hash_content("s")),
        )
        chain.append(r)
    
    for i in range(5):
        r = WorkReceipt(
            agent_id="sigil:specialist",
            task_type="research",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content(f"rs_{i}"),
            output_hash=hash_content(f"rs_out_{i}"),
            timestamp=(now - timedelta(days=5-i)).isoformat(),
            test_results=TestResults(passed=3, failed=0, suite_hash=hash_content("s")),
        )
        chain.append(r)
    
    agg = ReputationAggregator()
    snap = agg.aggregate(chain)
    
    print("=== Multi-Domain Aggregation ===")
    print(f"Composite: {snap.composite_score}")
    print(f"Domain scores: {snap.domain_scores}")
    print(f"Trust tier: {snap.trust_tier}")
    
    assert "code_review" in snap.domain_scores
    assert "research" in snap.domain_scores
    assert snap.domain_scores["code_review"] > 0.8
    assert snap.trust_tier == "established"  # 20 receipts
    
    print("✓ Multi-domain aggregation passed\n")


def test_snapshot_portability():
    """Test that snapshots serialize/deserialize correctly."""
    from receipt import TestResults, hash_content
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    chain = ReceiptChain()
    for i in range(5):
        r = WorkReceipt(
            agent_id="sigil:portable",
            task_type="translation",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content(f"in_{i}"),
            output_hash=hash_content(f"out_{i}"),
            timestamp=(now - timedelta(days=5-i)).isoformat(),
            test_results=TestResults(passed=2, failed=0, suite_hash=hash_content("s")),
        )
        chain.append(r)
    
    agg = ReputationAggregator()
    snap = agg.aggregate(chain)
    
    # Round-trip
    json_str = snap.to_json()
    restored = ReputationSnapshot.from_json(json_str)
    
    assert restored.composite_score == snap.composite_score
    assert restored.merkle_root == snap.merkle_root
    assert restored.domain_scores == snap.domain_scores
    assert restored.content_hash == snap.content_hash
    
    print("=== Snapshot Portability ===")
    print(f"Original hash: {snap.content_hash[:40]}...")
    print(f"Restored hash: {restored.content_hash[:40]}...")
    print("✓ Snapshot round-trip passed\n")


def test_veteran_chain():
    """Test a large chain hitting veteran tier."""
    from receipt import TestResults, hash_content
    from datetime import datetime, timezone, timedelta
    
    now = datetime.now(timezone.utc)
    chain = ReceiptChain()
    
    for i in range(60):
        success = i != 15 and i != 42  # 2 failures out of 60
        r = WorkReceipt(
            agent_id="sigil:veteran",
            task_type="data_validation" if i % 3 == 0 else "code_review",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content(f"in_{i}"),
            output_hash=hash_content(f"out_{i}"),
            timestamp=(now - timedelta(days=90-i*1.5)).isoformat(),
            test_results=TestResults(
                passed=5 if success else 3,
                failed=0 if success else 2,
                suite_hash=hash_content("s"),
            ),
        )
        chain.append(r)
    
    agg = ReputationAggregator()
    snap = agg.aggregate(chain)
    
    print("=== Veteran Chain ===")
    print(f"Composite: {snap.composite_score}")
    print(f"Success rate: {snap.success_rate} (58/60)")
    print(f"Trust tier: {snap.trust_tier}")
    print(f"Tier breakdown: {snap.tier_breakdown}")
    print(f"Chain length: {snap.chain_length}")
    
    assert snap.trust_tier == "veteran"
    assert snap.chain_length == 60
    assert 0.95 < snap.success_rate < 0.98  # 58/60
    
    print("✓ Veteran chain passed\n")


if __name__ == "__main__":
    test_basic_aggregation()
    test_strategy_switching_penalty()
    test_multi_domain()
    test_snapshot_portability()
    test_veteran_chain()
    print("=== All aggregator tests passed ===")
