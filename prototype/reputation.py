"""
ClawBizarre Reputation Engine v0.1
Implements reputation models with decay, domain tracking, and Merkle-based verification.

Models:
1. Bayesian with exponential decay (recent work matters more)
2. Domain-specific reputation (per task type)
3. Merkle proof for receipt chain verification (verify without downloading full chain)
"""

import hashlib
import math
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from receipt import WorkReceipt, ReceiptChain


# --- Reputation with Decay ---

@dataclass
class DecayingReputation:
    """Bayesian reputation with exponential time decay.
    
    Recent work weighs more than old work. Inactive agents' reputation
    decays toward the prior (0.5 = unknown).
    
    half_life_days: time for a receipt's weight to halve
    prior_strength: how many phantom observations to assume (regularization)
    """
    half_life_days: float = 30.0  # receipt weight halves every 30 days
    prior_strength: float = 3.0   # equivalent to 3 phantom observations (2 success, 1 fail → 0.67 prior)
    prior_success_ratio: float = 0.67
    
    # Internal state
    _successes: list[tuple[float, float]] = field(default_factory=list)  # (timestamp, weight)
    _failures: list[tuple[float, float]] = field(default_factory=list)
    
    def record(self, success: bool, timestamp: Optional[float] = None, weight: float = 1.0):
        """Record a completed task."""
        ts = timestamp or time.time()
        if success:
            self._successes.append((ts, weight))
        else:
            self._failures.append((ts, weight))
    
    def score(self, now: Optional[float] = None) -> float:
        """Calculate decayed reputation score."""
        now = now or time.time()
        decay_rate = math.log(2) / (self.half_life_days * 86400)  # per-second decay
        
        weighted_success = sum(w * math.exp(-decay_rate * (now - ts)) 
                               for ts, w in self._successes)
        weighted_failure = sum(w * math.exp(-decay_rate * (now - ts)) 
                               for ts, w in self._failures)
        
        # Add prior
        prior_success = self.prior_strength * self.prior_success_ratio
        prior_failure = self.prior_strength * (1 - self.prior_success_ratio)
        
        total_success = weighted_success + prior_success
        total_failure = weighted_failure + prior_failure
        
        return total_success / (total_success + total_failure)
    
    def effective_observations(self, now: Optional[float] = None) -> float:
        """How many 'effective' observations remain after decay."""
        now = now or time.time()
        decay_rate = math.log(2) / (self.half_life_days * 86400)
        total = sum(w * math.exp(-decay_rate * (now - ts)) 
                    for ts, w in self._successes + self._failures)
        return total + self.prior_strength
    
    def confidence(self, now: Optional[float] = None) -> float:
        """Confidence in the score (0-1). More observations = higher confidence."""
        eff = self.effective_observations(now)
        # Logistic curve: 50% confidence at 10 effective obs, 90% at 30
        return 1 / (1 + math.exp(-0.2 * (eff - 10)))


# --- Domain-Specific Reputation ---

@dataclass
class DomainReputation:
    """Tracks reputation per task type with cross-domain correlation.
    
    An agent trusted in code_review gets partial credit for related domains.
    Domain correlations are configurable.
    """
    half_life_days: float = 30.0
    prior_strength: float = 3.0
    # How much reputation transfers between domains (0-1)
    domain_correlations: dict[tuple[str, str], float] = field(default_factory=dict)
    _domains: dict[str, DecayingReputation] = field(default_factory=dict)
    
    def _get_domain(self, domain: str) -> DecayingReputation:
        if domain not in self._domains:
            self._domains[domain] = DecayingReputation(
                half_life_days=self.half_life_days,
                prior_strength=self.prior_strength,
            )
        return self._domains[domain]
    
    def record(self, domain: str, success: bool, timestamp: Optional[float] = None, weight: float = 1.0):
        self._get_domain(domain).record(success, timestamp, weight)
    
    def score(self, domain: str, now: Optional[float] = None) -> float:
        """Score for a specific domain, including cross-domain transfer."""
        direct = self._get_domain(domain)
        direct_score = direct.score(now)
        direct_conf = direct.confidence(now)
        
        if direct_conf > 0.8:
            # High confidence in direct domain — no need for cross-domain transfer
            return direct_score
        
        # Blend in correlated domains
        transfer_scores = []
        for other_domain, rep in self._domains.items():
            if other_domain == domain:
                continue
            correlation = self.domain_correlations.get(
                (other_domain, domain),
                self.domain_correlations.get((domain, other_domain), 0.0)
            )
            if correlation > 0:
                other_score = rep.score(now)
                other_conf = rep.confidence(now)
                transfer_scores.append((other_score, correlation * other_conf))
        
        if not transfer_scores:
            return direct_score
        
        # Weighted blend: direct score weighted by its confidence,
        # transferred scores weighted by correlation * their confidence
        total_weight = direct_conf
        blended = direct_score * direct_conf
        for score, weight in transfer_scores:
            blended += score * weight * (1 - direct_conf)  # transfer only fills confidence gap
            total_weight += weight * (1 - direct_conf)
        
        return blended / total_weight if total_weight > 0 else 0.5
    
    def summary(self, now: Optional[float] = None) -> dict:
        """Summary of all domain scores."""
        return {
            domain: {
                "score": rep.score(now),
                "confidence": rep.confidence(now),
                "effective_obs": rep.effective_observations(now),
            }
            for domain, rep in self._domains.items()
        }


# --- Merkle Tree for Receipt Chain Verification ---

def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class MerkleTree:
    """Merkle tree over receipt hashes for efficient verification.
    
    Allows proving a specific receipt exists in a chain without
    downloading the entire chain. O(log n) proof size.
    """
    
    def __init__(self, leaves: list[str]):
        """Build tree from leaf hashes (receipt content hashes)."""
        self.leaves = leaves
        self.layers: list[list[str]] = []
        self._build()
    
    def _build(self):
        if not self.leaves:
            self.layers = [[sha256(b"empty")]]
            return
        
        # Pad to power of 2
        layer = list(self.leaves)
        while len(layer) & (len(layer) - 1):  # not power of 2
            layer.append(layer[-1])  # duplicate last
        
        self.layers = [layer]
        while len(layer) > 1:
            next_layer = []
            for i in range(0, len(layer), 2):
                combined = sha256((layer[i] + layer[i+1]).encode())
                next_layer.append(combined)
            layer = next_layer
            self.layers.append(layer)
    
    @property
    def root(self) -> str:
        return self.layers[-1][0] if self.layers else ""
    
    def proof(self, index: int) -> list[tuple[str, str]]:
        """Generate Merkle proof for leaf at index.
        Returns list of (hash, side) where side is 'left' or 'right'.
        """
        if index >= len(self.leaves):
            raise IndexError(f"Leaf index {index} out of range")
        
        proof_path = []
        idx = index
        # Adjust for padding
        padded_len = len(self.layers[0])
        
        for layer in self.layers[:-1]:
            if idx % 2 == 0:
                sibling_idx = idx + 1
                side = "right"
            else:
                sibling_idx = idx - 1
                side = "left"
            
            if sibling_idx < len(layer):
                proof_path.append((layer[sibling_idx], side))
            idx //= 2
        
        return proof_path
    
    @staticmethod
    def verify_proof(leaf_hash: str, proof: list[tuple[str, str]], root: str) -> bool:
        """Verify a Merkle proof against a known root."""
        current = leaf_hash
        for sibling_hash, side in proof:
            if side == "left":
                current = sha256((sibling_hash + current).encode())
            else:
                current = sha256((current + sibling_hash).encode())
        return current == root


# --- Tests ---

def test_decaying_reputation():
    """Test reputation decay over time."""
    rep = DecayingReputation(half_life_days=30)
    
    now = time.time()
    day = 86400
    
    # Record 10 successes 60 days ago
    for _ in range(10):
        rep.record(True, timestamp=now - 60 * day)
    
    # Record 3 failures recently
    for _ in range(3):
        rep.record(False, timestamp=now - 1 * day)
    
    score = rep.score(now)
    # Old successes should be heavily decayed (60 days = 2 half-lives → weight ~0.25 each)
    # Recent failures at full weight
    # Score should be pulled down significantly
    print(f"Decay test: score={score:.3f} (old successes + recent failures)")
    assert score < 0.6, f"Expected <0.6 with recent failures, got {score}"
    
    # Compare to non-decayed (lifetime) version
    lifetime_score = (10 + 2) / (10 + 3 + 3)  # Bayesian: 12/16 = 0.75
    print(f"Lifetime equivalent: {lifetime_score:.3f}")
    assert score < lifetime_score, "Decayed score should be lower than lifetime"
    
    # Confidence should be moderate (decayed observations)
    conf = rep.confidence(now)
    print(f"Confidence: {conf:.3f}")
    
    print("✓ Decaying reputation test passed\n")


def test_domain_reputation():
    """Test domain-specific reputation with cross-domain transfer."""
    rep = DomainReputation(
        half_life_days=30,
        domain_correlations={
            ("code_review", "research"): 0.5,
            ("translation", "research"): 0.2,
        }
    )
    
    now = time.time()
    
    # Build strong code_review reputation
    for _ in range(20):
        rep.record("code_review", True, timestamp=now)
    
    # No research history — should get partial credit from code_review
    code_score = rep.score("code_review", now)
    research_score = rep.score("research", now)
    translation_score = rep.score("translation", now)  # no correlation, no data
    
    print(f"Code review score: {code_score:.3f} (20 successes)")
    print(f"Research score: {research_score:.3f} (transferred from code_review)")
    print(f"Translation score: {translation_score:.3f} (no data, no transfer)")
    
    assert code_score > 0.9
    assert research_score > translation_score, "Research should benefit from code_review correlation"
    
    print(f"Summary: {rep.summary(now)}")
    print("✓ Domain reputation test passed\n")


def test_merkle_tree():
    """Test Merkle tree construction and proof verification."""
    # Build tree from receipt hashes
    receipts = [sha256(f"receipt_{i}".encode()) for i in range(10)]
    tree = MerkleTree(receipts)
    
    print(f"Merkle tree: {len(receipts)} leaves, root={tree.root[:16]}...")
    
    # Generate and verify proof for each leaf
    for i in range(len(receipts)):
        proof = tree.proof(i)
        valid = MerkleTree.verify_proof(receipts[i], proof, tree.root)
        assert valid, f"Proof failed for leaf {i}"
    
    print(f"All {len(receipts)} proofs verified ✓")
    
    # Tampered proof should fail
    fake_hash = sha256(b"fake_receipt")
    proof = tree.proof(0)
    assert not MerkleTree.verify_proof(fake_hash, proof, tree.root), "Fake proof should fail"
    print("Tampered proof correctly rejected ✓")
    
    # Single leaf
    single = MerkleTree([receipts[0]])
    assert single.root == receipts[0]
    print("Single-leaf tree ✓")
    
    print("✓ Merkle tree test passed\n")


def test_reputation_decay_scenarios():
    """Test specific economic scenarios that emerged from brainstorming."""
    now = time.time()
    day = 86400
    
    print("=== Economic Scenario Tests ===\n")
    
    # Scenario 1: The ghost agent (inactive for 90 days)
    ghost = DecayingReputation(half_life_days=30)
    for _ in range(50):
        ghost.record(True, timestamp=now - 90 * day)
    ghost_score = ghost.score(now)
    ghost_conf = ghost.confidence(now)
    print(f"Ghost agent (50 successes 90 days ago): score={ghost_score:.3f}, confidence={ghost_conf:.3f}")
    # 50 successes at 3 half-lives = ~6.25 effective. Prior=3. So ~9.25 obs, score ~(6.25*0.67+2)/(6.25+3) ≈ 0.89
    # Key insight: score stays high but CONFIDENCE drops — that's the useful signal
    assert ghost_score > 0.5, "Ghost should still have positive score from history"
    assert ghost_conf < 0.5, "Ghost should have low confidence (decayed observations)"
    
    # Scenario 2: The comeback (inactive then recent burst)
    comeback = DecayingReputation(half_life_days=30)
    for _ in range(20):
        comeback.record(True, timestamp=now - 90 * day)
    for _ in range(5):
        comeback.record(True, timestamp=now - 1 * day)
    comeback_score = comeback.score(now)
    comeback_conf = comeback.confidence(now)
    print(f"Comeback agent (20 old + 5 recent): score={comeback_score:.3f}, confidence={comeback_conf:.3f}")
    
    # Scenario 3: The declining agent (was good, getting worse)
    declining = DecayingReputation(half_life_days=30)
    for _ in range(20):
        declining.record(True, timestamp=now - 60 * day)
    for _ in range(5):
        declining.record(False, timestamp=now - 2 * day)
    declining_score = declining.score(now)
    print(f"Declining agent (20 old successes + 5 recent failures): score={declining_score:.3f}")
    assert declining_score < 0.7, "Declining agent should have low score from recent failures"
    
    # Scenario 4: Specialist vs generalist with decay
    specialist = DomainReputation(half_life_days=30)
    generalist = DomainReputation(half_life_days=30)
    
    # Specialist: 30 data_validation tasks
    for _ in range(30):
        specialist.record("data_validation", True, timestamp=now - 5 * day)
    
    # Generalist: 10 each of 3 types
    for domain in ["code_review", "translation", "data_validation"]:
        for _ in range(10):
            generalist.record(domain, True, timestamp=now - 5 * day)
    
    spec_score = specialist.score("data_validation", now)
    gen_score = generalist.score("data_validation", now)
    print(f"Specialist data_validation: {spec_score:.3f} vs Generalist: {gen_score:.3f}")
    assert spec_score > gen_score, "Specialist should score higher in their domain"
    
    print("\n✓ All economic scenario tests passed")


if __name__ == "__main__":
    test_decaying_reputation()
    test_domain_reputation()
    test_merkle_tree()
    test_reputation_decay_scenarios()
    print("\n=== All reputation tests passed ===")
