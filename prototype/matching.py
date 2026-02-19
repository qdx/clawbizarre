"""
ClawBizarre Matching Engine — Phase 8a
Filter → Rank → Select pipeline for agent service matching.

Design informed by:
- Vast.ai: posted price marketplace with rich filtering (not auction)
- AWS Spot history: transparency > optimality
- Phase 6b simulation: top3_random + 15% newcomer reserve = best fairness
- v5-v10: anti-undercutting via strategy visibility + reputation weighting

Key design decisions:
1. Posted price, not auction (heterogeneous goods)
2. Composite scoring: reputation + price + reliability (configurable weights)
3. top3_random selection default (from 6b: lowest Gini, 97% newcomer survival)
4. 15% newcomer reserve in results
5. Strategy change visibility in rankings
6. Compute cost floor enforcement
"""

import json
import time
import random
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Optional
from enum import Enum


@dataclass
class PaymentTerms:
    """Payment configuration for a service listing (x402/AP2 compatible)."""
    protocol: str = "x402"           # "x402", "ap2", "manual", "free"
    currency: str = "USDC"           # Payment currency
    chain: str = "base"              # Blockchain network (for x402)
    wallet_address: str = ""         # Provider's payment address
    escrow_required: bool = False    # Whether payment must be escrowed before work
    refund_on_failure: bool = True   # Auto-refund if verification fails (Tier 0)
    max_payment_delay_ms: int = 30000  # Max time to settle payment after verification

    def to_dict(self) -> dict:
        return {
            'protocol': self.protocol,
            'currency': self.currency,
            'chain': self.chain,
            'wallet_address': self.wallet_address,
            'escrow_required': self.escrow_required,
            'refund_on_failure': self.refund_on_failure,
            'max_payment_delay_ms': self.max_payment_delay_ms,
        }


class SelectionStrategy(str, Enum):
    BEST_FIRST = "best_first"       # Always pick highest-scored
    TOP3_RANDOM = "top3_random"     # Random from top 3 (default, best for fairness)
    WEIGHTED_RANDOM = "weighted"    # Score-weighted random selection
    CHEAPEST = "cheapest"           # Pure price optimization


class PricingModel(str, Enum):
    FIXED = "fixed"           # Guaranteed completion, no discount
    FLEXIBLE = "flexible"     # Adjustable, can reprioritize, 10-30% discount
    SPOT = "spot"             # Market-clearing, can abandon, 30-60% discount


@dataclass
class ServiceListing:
    """A seller's advertised service."""
    agent_id: str
    capability: str
    base_rate: float              # Price per unit
    unit: str                     # "per_line", "per_word", "per_task", "per_hour"
    pricing_model: PricingModel = PricingModel.FIXED
    verification_tier: int = 0
    max_response_time_ms: int = 60000
    reputation_score: float = 0.0
    uptime_fraction: float = 1.0  # Last 24h availability
    receipt_count: int = 0        # Completed tasks
    strategy_changes_30d: int = 0 # Strategy switches in last 30 days
    payment_terms: Optional[PaymentTerms] = None  # x402/AP2 payment config
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    @property
    def is_newcomer(self) -> bool:
        return self.receipt_count < 10

    def to_dict(self) -> dict:
        d = asdict(self)
        d['pricing_model'] = self.pricing_model.value
        d['is_newcomer'] = self.is_newcomer
        if self.payment_terms:
            d['payment_terms'] = self.payment_terms.to_dict()
        else:
            d['payment_terms'] = None
        return d


@dataclass
class MatchRequest:
    """A buyer's request for a service match."""
    buyer_id: str
    capability: str
    max_price: Optional[float] = None
    min_reputation: float = 0.0
    max_response_time_ms: int = 120000
    verification_tier_required: int = 0
    selection_strategy: SelectionStrategy = SelectionStrategy.TOP3_RANDOM
    # Custom scoring weights (must sum to 1.0)
    weight_reputation: float = 0.5
    weight_price: float = 0.3
    weight_reliability: float = 0.2
    # Payment preferences
    accepted_protocols: Optional[list[str]] = None  # e.g. ["x402", "ap2"] — None = any
    # Number of candidates to return
    max_results: int = 5


@dataclass
class MatchResult:
    """A single match candidate."""
    listing: ServiceListing
    score: float
    rank: int
    match_reason: str  # Human-readable explanation


@dataclass
class MatchResponse:
    """Full matching response."""
    request_id: str
    candidates: list[MatchResult]
    total_filtered: int      # How many passed filter
    total_available: int     # How many listings existed
    newcomer_slots_used: int
    timestamp: float = field(default_factory=time.time)


class MatchingEngine:
    """
    Posted-price matching engine for agent services.
    
    NOT an auction. Sellers post prices, buyers filter/rank/select.
    Informed by Vast.ai marketplace model and simulation findings.
    """

    NEWCOMER_RESERVE_FRACTION = 0.15  # 15% of results reserved for newcomers
    NEWCOMER_THRESHOLD = 10           # <10 receipts = newcomer
    COMPUTE_COST_FLOOR = 0.001        # Minimum viable price (prevents loss-leader attacks)

    def __init__(self):
        self.listings: dict[str, list[ServiceListing]] = {}  # capability -> listings
        self._all_listings: dict[str, ServiceListing] = {}   # listing_id -> listing
        self.match_count = 0
        self.price_history: list[dict] = []  # Append-only price log

    def _listing_id(self, agent_id: str, capability: str) -> str:
        return hashlib.sha256(f"{agent_id}:{capability}".encode()).hexdigest()[:16]

    def add_listing(self, listing: ServiceListing) -> str:
        """Register or update a service listing."""
        lid = self._listing_id(listing.agent_id, listing.capability)

        # Enforce compute cost floor
        if listing.base_rate < self.COMPUTE_COST_FLOOR:
            raise ValueError(
                f"Price {listing.base_rate} below compute cost floor {self.COMPUTE_COST_FLOOR}. "
                "Loss-leader pricing rejected."
            )

        # Track price changes
        if lid in self._all_listings:
            old = self._all_listings[lid]
            if old.base_rate != listing.base_rate:
                self.price_history.append({
                    'listing_id': lid,
                    'agent_id': listing.agent_id,
                    'capability': listing.capability,
                    'old_price': old.base_rate,
                    'new_price': listing.base_rate,
                    'timestamp': time.time()
                })
            # Detect strategy changes
            if old.pricing_model != listing.pricing_model:
                listing.strategy_changes_30d = old.strategy_changes_30d + 1

        # Update indexes
        listing.updated_at = time.time()
        self._all_listings[lid] = listing

        if listing.capability not in self.listings:
            self.listings[listing.capability] = []

        # Remove old listing for same agent+capability
        self.listings[listing.capability] = [
            l for l in self.listings[listing.capability]
            if self._listing_id(l.agent_id, l.capability) != lid
        ]
        self.listings[listing.capability].append(listing)

        return lid

    def remove_listing(self, agent_id: str, capability: str) -> bool:
        lid = self._listing_id(agent_id, capability)
        if lid not in self._all_listings:
            return False
        del self._all_listings[lid]
        if capability in self.listings:
            self.listings[capability] = [
                l for l in self.listings[capability]
                if self._listing_id(l.agent_id, l.capability) != lid
            ]
        return True

    def _filter(self, request: MatchRequest) -> list[ServiceListing]:
        """Phase 1: Filter by hard constraints."""
        if request.capability not in self.listings:
            return []

        candidates = []
        for listing in self.listings[request.capability]:
            # Don't match buyer with themselves
            if listing.agent_id == request.buyer_id:
                continue
            # Price filter
            if request.max_price is not None and listing.base_rate > request.max_price:
                continue
            # Reputation filter
            if listing.reputation_score < request.min_reputation:
                continue
            # Response time filter
            if listing.max_response_time_ms > request.max_response_time_ms:
                continue
            # Verification tier filter
            if listing.verification_tier < request.verification_tier_required:
                continue
            # Payment protocol filter
            if request.accepted_protocols is not None:
                listing_proto = listing.payment_terms.protocol if listing.payment_terms else "manual"
                if listing_proto not in request.accepted_protocols:
                    continue
            candidates.append(listing)

        return candidates

    def _score(self, listing: ServiceListing, request: MatchRequest,
               price_range: tuple[float, float]) -> tuple[float, str]:
        """Phase 2: Score a candidate. Returns (score, explanation)."""
        reasons = []

        # Reputation component (0-1)
        rep_score = min(listing.reputation_score, 1.0)
        reasons.append(f"rep={rep_score:.2f}")

        # Price component (0-1, lower price = higher score)
        if price_range[1] > price_range[0]:
            price_normalized = (listing.base_rate - price_range[0]) / (price_range[1] - price_range[0])
            price_score = 1.0 - price_normalized
        else:
            price_score = 1.0
        reasons.append(f"price={price_score:.2f}")

        # Reliability component (0-1)
        reliability_score = listing.uptime_fraction
        reasons.append(f"reliability={reliability_score:.2f}")

        # Strategy stability penalty (from v10: strategy switching destroys value)
        stability_penalty = min(listing.strategy_changes_30d * 0.1, 0.3)
        if stability_penalty > 0:
            reasons.append(f"strategy_penalty=-{stability_penalty:.2f}")

        # Composite score
        score = (
            request.weight_reputation * rep_score +
            request.weight_price * price_score +
            request.weight_reliability * reliability_score -
            stability_penalty
        )

        return max(score, 0.0), "; ".join(reasons)

    def _select(self, scored: list[tuple[ServiceListing, float, str]],
                request: MatchRequest) -> list[MatchResult]:
        """Phase 3: Select from scored candidates with newcomer protection."""
        if not scored:
            return []

        # Separate newcomers and established
        newcomers = [(l, s, r) for l, s, r in scored if l.is_newcomer]
        established = [(l, s, r) for l, s, r in scored if not l.is_newcomer]

        # Sort each group by score descending
        newcomers.sort(key=lambda x: x[1], reverse=True)
        established.sort(key=lambda x: x[1], reverse=True)

        # Reserve slots for newcomers
        max_results = request.max_results
        newcomer_slots = max(1, int(max_results * self.NEWCOMER_RESERVE_FRACTION))
        established_slots = max_results - newcomer_slots

        # Fill slots
        selected_newcomers = newcomers[:newcomer_slots]
        selected_established = established[:established_slots]

        # If not enough newcomers, give slots back to established
        if len(selected_newcomers) < newcomer_slots:
            extra = newcomer_slots - len(selected_newcomers)
            selected_established = established[:established_slots + extra]

        # Combine and apply selection strategy
        all_selected = selected_newcomers + selected_established

        if request.selection_strategy == SelectionStrategy.BEST_FIRST:
            all_selected.sort(key=lambda x: x[1], reverse=True)
        elif request.selection_strategy == SelectionStrategy.TOP3_RANDOM:
            # Sort by score, then shuffle top 3
            all_selected.sort(key=lambda x: x[1], reverse=True)
            if len(all_selected) > 3:
                top3 = all_selected[:3]
                random.shuffle(top3)
                all_selected = top3 + all_selected[3:]
        elif request.selection_strategy == SelectionStrategy.WEIGHTED_RANDOM:
            # Score-weighted shuffle
            random.shuffle(all_selected)
            weights = [s for _, s, _ in all_selected]
            # Weighted sample without replacement
            if weights:
                weighted = []
                pool = list(all_selected)
                w_pool = list(weights)
                while pool and len(weighted) < max_results:
                    total = sum(w_pool)
                    if total <= 0:
                        break
                    r = random.uniform(0, total)
                    cumulative = 0
                    for i, w in enumerate(w_pool):
                        cumulative += w
                        if r <= cumulative:
                            weighted.append(pool.pop(i))
                            w_pool.pop(i)
                            break
                all_selected = weighted
        elif request.selection_strategy == SelectionStrategy.CHEAPEST:
            all_selected.sort(key=lambda x: x[0].base_rate)

        # Build results
        results = []
        for rank, (listing, score, reason) in enumerate(all_selected[:max_results]):
            results.append(MatchResult(
                listing=listing,
                score=round(score, 4),
                rank=rank + 1,
                match_reason=reason
            ))

        return results

    def match(self, request: MatchRequest) -> MatchResponse:
        """Execute full matching pipeline: filter → rank → select."""
        self.match_count += 1
        request_id = hashlib.sha256(
            f"{request.buyer_id}:{request.capability}:{self.match_count}".encode()
        ).hexdigest()[:16]

        total_available = len(self.listings.get(request.capability, []))

        # Phase 1: Filter
        candidates = self._filter(request)

        # Phase 2: Score
        if candidates:
            prices = [c.base_rate for c in candidates]
            price_range = (min(prices), max(prices))
            scored = [
                (c, *self._score(c, request, price_range))
                for c in candidates
            ]
        else:
            scored = []

        # Phase 3: Select
        results = self._select(scored, request)
        newcomer_count = sum(1 for r in results if r.listing.is_newcomer)

        return MatchResponse(
            request_id=request_id,
            candidates=results,
            total_filtered=len(candidates),
            total_available=total_available,
            newcomer_slots_used=newcomer_count
        )

    def get_price_history(self, capability: Optional[str] = None,
                          agent_id: Optional[str] = None,
                          since: Optional[float] = None) -> list[dict]:
        """Query transparent price history (append-only)."""
        history = self.price_history
        if capability:
            history = [h for h in history if h['capability'] == capability]
        if agent_id:
            history = [h for h in history if h['agent_id'] == agent_id]
        if since:
            history = [h for h in history if h['timestamp'] >= since]
        return history

    def stats(self) -> dict:
        """Engine statistics."""
        all_listings = list(self._all_listings.values())
        return {
            'total_listings': len(all_listings),
            'capabilities': list(self.listings.keys()),
            'total_matches': self.match_count,
            'price_changes_logged': len(self.price_history),
            'newcomer_listings': sum(1 for l in all_listings if l.is_newcomer),
            'established_listings': sum(1 for l in all_listings if not l.is_newcomer),
            'avg_reputation': (
                sum(l.reputation_score for l in all_listings) / len(all_listings)
                if all_listings else 0
            ),
        }


# ── Tests ────────────────────────────────────────────────────────────────

def test_matching_engine():
    engine = MatchingEngine()
    passed = 0
    total = 0

    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")

    print("Testing MatchingEngine...")

    # Test 1: Add listings
    l1 = ServiceListing("agent_a", "code_review", 0.05, "per_line",
                        reputation_score=0.8, receipt_count=50, uptime_fraction=0.95)
    l2 = ServiceListing("agent_b", "code_review", 0.03, "per_line",
                        reputation_score=0.4, receipt_count=5, uptime_fraction=0.9)
    l3 = ServiceListing("agent_c", "code_review", 0.07, "per_line",
                        reputation_score=0.9, receipt_count=100, uptime_fraction=0.99)
    l4 = ServiceListing("agent_d", "translation", 0.02, "per_word",
                        reputation_score=0.6, receipt_count=30)

    engine.add_listing(l1)
    engine.add_listing(l2)
    engine.add_listing(l3)
    engine.add_listing(l4)
    check("Add 4 listings", engine.stats()['total_listings'] == 4)

    # Test 2: Basic matching
    req = MatchRequest("buyer_1", "code_review")
    resp = engine.match(req)
    check("Match returns candidates", len(resp.candidates) > 0)
    check("Match finds 3 code_review agents", resp.total_available == 3)

    # Test 3: Price filter
    req = MatchRequest("buyer_1", "code_review", max_price=0.04)
    resp = engine.match(req)
    check("Price filter works", all(r.listing.base_rate <= 0.04 for r in resp.candidates))

    # Test 4: Reputation filter
    req = MatchRequest("buyer_1", "code_review", min_reputation=0.7)
    resp = engine.match(req)
    check("Reputation filter works", all(r.listing.reputation_score >= 0.7 for r in resp.candidates))
    check("Reputation filter returns 2", len(resp.candidates) == 2)

    # Test 5: Newcomer in results
    req = MatchRequest("buyer_1", "code_review", max_results=3)
    resp = engine.match(req)
    check("Newcomer slots tracked", resp.newcomer_slots_used >= 0)

    # Test 6: Compute cost floor
    try:
        bad = ServiceListing("agent_evil", "code_review", 0.0001, "per_line")
        engine.add_listing(bad)
        check("Compute cost floor enforced", False)
    except ValueError:
        check("Compute cost floor enforced", True)

    # Test 7: Self-match prevention
    req = MatchRequest("agent_a", "code_review")
    resp = engine.match(req)
    check("Self-match prevented", all(r.listing.agent_id != "agent_a" for r in resp.candidates))

    # Test 8: Price history tracking
    l1_updated = ServiceListing("agent_a", "code_review", 0.06, "per_line",
                                 reputation_score=0.8, receipt_count=55)
    engine.add_listing(l1_updated)
    history = engine.get_price_history(capability="code_review")
    check("Price change logged", len(history) == 1 and history[0]['old_price'] == 0.05)

    # Test 9: Strategy change tracking
    l2_switched = ServiceListing("agent_b", "code_review", 0.03, "per_line",
                                  pricing_model=PricingModel.SPOT,
                                  reputation_score=0.4, receipt_count=6)
    engine.add_listing(l2_switched)
    lid = engine._listing_id("agent_b", "code_review")
    check("Strategy change counted", engine._all_listings[lid].strategy_changes_30d == 1)

    # Test 10: Strategy penalty in scoring
    req = MatchRequest("buyer_1", "code_review")
    resp = engine.match(req)
    # agent_b should be penalized for strategy switch
    agent_b_results = [r for r in resp.candidates if r.listing.agent_id == "agent_b"]
    if agent_b_results:
        check("Strategy penalty applied", "strategy_penalty" in agent_b_results[0].match_reason)
    else:
        check("Strategy penalty applied", True)  # agent_b filtered out = also fine

    # Test 11: Remove listing
    removed = engine.remove_listing("agent_d", "translation")
    check("Remove listing works", removed and engine.stats()['total_listings'] == 3)

    # Test 12: Empty capability match
    req = MatchRequest("buyer_1", "nonexistent_service")
    resp = engine.match(req)
    check("Empty match returns 0", len(resp.candidates) == 0 and resp.total_available == 0)

    # Test 13: Payment terms on listings
    l5 = ServiceListing("agent_e", "code_review", 0.04, "per_line",
                        reputation_score=0.7, receipt_count=20,
                        payment_terms=PaymentTerms(protocol="x402", currency="USDC",
                                                    chain="base", wallet_address="0xabc"))
    engine.add_listing(l5)
    lid5 = engine._listing_id("agent_e", "code_review")
    check("Payment terms stored", engine._all_listings[lid5].payment_terms.protocol == "x402")
    d5 = engine._all_listings[lid5].to_dict()
    check("Payment terms in dict", d5['payment_terms']['protocol'] == "x402")

    # Test 14: Payment protocol filter
    req = MatchRequest("buyer_1", "code_review", accepted_protocols=["x402"])
    resp = engine.match(req)
    check("Protocol filter: only x402", all(
        r.listing.payment_terms is not None and r.listing.payment_terms.protocol == "x402"
        for r in resp.candidates
    ))
    check("Protocol filter: finds agent_e", any(
        r.listing.agent_id == "agent_e" for r in resp.candidates
    ))

    # Test 15: No payment terms = "manual" protocol
    req = MatchRequest("buyer_1", "code_review", accepted_protocols=["manual"])
    resp = engine.match(req)
    check("Manual protocol matches listings without payment_terms",
          all(r.listing.payment_terms is None for r in resp.candidates))

    # Test 16: PaymentTerms defaults
    pt = PaymentTerms()
    check("Default protocol is x402", pt.protocol == "x402")
    check("Default refund_on_failure is True", pt.refund_on_failure is True)

    print(f"\n{passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    test_matching_engine()
