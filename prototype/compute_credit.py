"""
compute_credit.py — Compute Credit Protocol for ClawBizarre.

Translates VRF receipt chains into compute credit scores and credit lines.
Closes the original ClawBizarre loop: verified work → compute access → sustained operation.

Credit score algorithm (FICO-analogous):
  - Volume score:      0-25 points (receipt chain depth)
  - Quality score:     0-40 points (recency-weighted pass rate)
  - Consistency score: 0-20 points (variance in pass rate over time)
  - Recency score:     0-10 points (days since last receipt)
  - Diversity bonus:   0-5 points (unique task types)
  Total:               0-100 points

Credit tiers:
  80-100: Verified     — uncapped (sponsor policy)
  60-79:  Established  — $5/day
  40-59:  Developing   — $2/day
  20-39:  New          — $0.50/day
  0-19:   Bootstrap    — free tier only

Usage:
  scorer = CreditScorer()
  score = scorer.score_from_receipts(receipt_list, domain="code")
  credit = scorer.credit_line(score)
  print(f"Score: {score.total}, Daily limit: ${credit.daily_usd}")
"""

import math
import json
import time
from dataclasses import dataclass, field
from typing import Optional
from datetime import datetime, timezone


# ── Data Classes ──────────────────────────────────────────────────────────────

@dataclass
class ReceiptSummary:
    """Minimal receipt data needed for credit scoring."""
    receipt_id: str
    agent_id: str
    task_type: str
    verdict: str          # "pass" | "fail" | "partial" | "error"
    pass_rate: float      # 0.0-1.0 (tests_passed / tests_total)
    verified_at: str      # ISO 8601 timestamp
    domain: Optional[str] = None


@dataclass
class CreditScoreBreakdown:
    """Detailed breakdown of credit score components."""
    volume: float         # 0-25
    quality: float        # 0-40
    consistency: float    # 0-20
    recency: float        # 0-10
    diversity: float      # 0-5

    @property
    def total(self) -> float:
        return round(self.volume + self.quality + self.consistency + self.recency + self.diversity, 1)

    @property
    def tier(self) -> str:
        t = self.total
        if t >= 80: return "verified"
        if t >= 60: return "established"
        if t >= 40: return "developing"
        if t >= 20: return "new"
        return "bootstrap"

    def to_dict(self) -> dict:
        return {
            "total": self.total,
            "tier": self.tier,
            "components": {
                "volume": self.volume,
                "quality": self.quality,
                "consistency": self.consistency,
                "recency": self.recency,
                "diversity": self.diversity,
            },
        }


@dataclass
class CreditLine:
    """Compute credit line recommendation."""
    score: float
    tier: str
    daily_usd: float              # Daily compute budget in USD
    max_task_usd: float           # Max spend per single task
    requests_per_day: int         # For rate-limited providers
    justification: str            # Human-readable explanation
    sponsor_required: bool = True  # True until Era 3 (direct provider integration)


# ── Credit Tier Policy ────────────────────────────────────────────────────────

TIER_POLICIES = {
    "verified":    {"daily_usd": 10.0,  "max_task_usd": 2.00, "rpm": 1000},
    "established": {"daily_usd":  5.0,  "max_task_usd": 1.00, "rpm": 500},
    "developing":  {"daily_usd":  2.0,  "max_task_usd": 0.50, "rpm": 200},
    "new":         {"daily_usd":  0.5,  "max_task_usd": 0.15, "rpm": 50},
    "bootstrap":   {"daily_usd":  0.1,  "max_task_usd": 0.05, "rpm": 10},
}


# ── Core Scorer ───────────────────────────────────────────────────────────────

class CreditScorer:
    """
    Computes agent compute credit scores from VRF receipt chains.
    
    Algorithm is deterministic and auditable — all inputs are VRF receipts,
    all outputs are explainable. No black-box ML.
    """

    def __init__(
        self,
        lookback_days: int = 90,
        recency_half_life_days: float = 30.0,
        min_receipts_for_full_volume: int = 50,
        max_credit_usd_per_day: float = 10.0,
    ):
        self.lookback_days = lookback_days
        self.recency_half_life_days = recency_half_life_days
        self.min_receipts_for_full_volume = min_receipts_for_full_volume
        self.max_credit_usd_per_day = max_credit_usd_per_day

    def score_from_receipts(
        self,
        receipts: list,
        domain: Optional[str] = None,
        now: Optional[datetime] = None,
    ) -> CreditScoreBreakdown:
        """
        Compute credit score from a list of ReceiptSummary or receipt dicts.

        Args:
            receipts: List of ReceiptSummary objects or raw receipt dicts
            domain: Task domain to weight (if specified, domain receipts get 2x weight)
            now: Reference time (defaults to UTC now)
        
        Returns:
            CreditScoreBreakdown with total score and component breakdown
        """
        if now is None:
            now = datetime.now(timezone.utc)

        # Normalize input
        normalized = [self._normalize(r) for r in receipts]

        # Filter to lookback window
        cutoff_ts = now.timestamp() - (self.lookback_days * 86400)
        in_window = [r for r in normalized if self._ts(r.verified_at) >= cutoff_ts]

        if not in_window:
            return CreditScoreBreakdown(0, 0, 0, 0, 0)

        # Component calculations
        volume = self._volume_score(in_window, domain)
        quality = self._quality_score(in_window, domain, now)
        consistency = self._consistency_score(in_window, now)
        recency = self._recency_score(in_window, now)
        diversity = self._diversity_score(in_window)

        return CreditScoreBreakdown(
            volume=round(volume, 1),
            quality=round(quality, 1),
            consistency=round(consistency, 1),
            recency=round(recency, 1),
            diversity=round(diversity, 1),
        )

    def credit_line(self, score: CreditScoreBreakdown) -> CreditLine:
        """Convert a credit score to a compute credit line recommendation."""
        tier = score.tier
        policy = TIER_POLICIES[tier]

        # Scale daily budget by score within tier
        tier_min = {"verified": 80, "established": 60, "developing": 40, "new": 20, "bootstrap": 0}[tier]
        tier_max = {"verified": 100, "established": 80, "developing": 60, "new": 40, "bootstrap": 20}[tier]
        tier_range = tier_max - tier_min
        position_in_tier = (score.total - tier_min) / tier_range if tier_range > 0 else 1.0
        
        # Daily budget scales from policy min to max within tier
        daily_scale = 0.5 + 0.5 * position_in_tier  # 50%-100% of tier max
        daily_usd = round(policy["daily_usd"] * daily_scale, 2)

        justification = self._justify(score, daily_usd)

        return CreditLine(
            score=score.total,
            tier=tier,
            daily_usd=daily_usd,
            max_task_usd=policy["max_task_usd"],
            requests_per_day=policy["rpm"],
            justification=justification,
            sponsor_required=True,  # Until Era 3 direct provider integration
        )

    def bootstrap_credit(self, voucher_score: "CreditScoreBreakdown") -> CreditLine:
        """
        Compute bootstrap credit for a new agent being introduced by a voucher.
        
        Staked introduction: voucher stakes 10% of their credit line; 
        new agent gets 20% of voucher's credit line as bootstrap.
        """
        voucher_line = self.credit_line(voucher_score)
        bootstrap_daily = round(voucher_line.daily_usd * 0.20, 2)
        staked = round(voucher_line.daily_usd * 0.10, 2)

        return CreditLine(
            score=0.0,
            tier="bootstrap",
            daily_usd=bootstrap_daily,
            max_task_usd=min(0.10, bootstrap_daily * 0.25),
            requests_per_day=max(10, int(TIER_POLICIES["bootstrap"]["rpm"] * (bootstrap_daily / 0.1))),
            justification=(
                f"Bootstrap via staked introduction. Voucher credit: ${voucher_line.daily_usd}/day, "
                f"staked: ${staked}/day, bootstrap grant: ${bootstrap_daily}/day."
            ),
            sponsor_required=True,
        )

    def sustainability_projection(
        self,
        score: CreditScoreBreakdown,
        task_value_usd: float = 0.01,
        tasks_per_day: int = 50,
        maintenance_cost_usd: float = 1.00,
    ) -> dict:
        """
        Project days to financial sustainability given current score and task volume.
        
        Returns:
          - current_earnings: earnings/day at current task volume
          - break_even_tasks: tasks/day needed for maintenance
          - days_to_verified_tier: estimated days to reach Verified tier
          - self_sustaining: whether current pace is self-sustaining
        """
        credit = self.credit_line(score)
        current_earnings = tasks_per_day * task_value_usd
        break_even_tasks = math.ceil(maintenance_cost_usd / task_value_usd)
        self_sustaining = current_earnings >= maintenance_cost_usd

        # Simple projection: to reach Verified tier (score 80), need ~50 quality receipts
        receipts_needed = max(0, 80 - score.total)  # rough: 1 receipt ≈ 1 score point at early stage
        days_to_verified = math.ceil(receipts_needed / max(1, tasks_per_day)) if not self_sustaining else 0

        return {
            "score": score.total,
            "tier": score.tier,
            "daily_credit_usd": credit.daily_usd,
            "current_earnings_usd": round(current_earnings, 2),
            "maintenance_cost_usd": maintenance_cost_usd,
            "self_sustaining": self_sustaining,
            "break_even_tasks_per_day": break_even_tasks,
            "days_to_verified_tier": days_to_verified,
            "revenue_gap_usd": round(max(0, maintenance_cost_usd - current_earnings), 2),
        }

    # ── Private helpers ───────────────────────────────────────────────────────

    def _normalize(self, r) -> ReceiptSummary:
        """Convert receipt dict or ReceiptSummary to ReceiptSummary."""
        if isinstance(r, ReceiptSummary):
            return r
        # Handle raw VRF receipt dict format
        task_type = r.get("task_type") or r.get("metadata", {}).get("task_type", "unknown")
        results = r.get("results", {})
        total = results.get("total", 0) or results.get("tests_total", 0)
        passed = results.get("passed", 0) or results.get("tests_passed", 0)
        pass_rate = (passed / total) if total > 0 else (1.0 if r.get("verdict") == "pass" else 0.0)
        return ReceiptSummary(
            receipt_id=r.get("receipt_id", ""),
            agent_id=r.get("agent_id", r.get("metadata", {}).get("agent_id", "")),
            task_type=task_type,
            verdict=r.get("verdict", "error"),
            pass_rate=pass_rate,
            verified_at=r.get("verified_at", r.get("timestamp", datetime.now(timezone.utc).isoformat())),
            domain=r.get("domain", task_type),
        )

    def _ts(self, iso: str) -> float:
        """Parse ISO timestamp to Unix timestamp."""
        try:
            if iso.endswith("Z"):
                iso = iso[:-1] + "+00:00"
            return datetime.fromisoformat(iso).timestamp()
        except Exception:
            return time.time()

    def _decay(self, receipt: ReceiptSummary, now: datetime) -> float:
        """Recency decay weight: exp(-age_days / half_life)."""
        age_days = (now.timestamp() - self._ts(receipt.verified_at)) / 86400
        return math.exp(-age_days / self.recency_half_life_days)

    def _volume_score(self, receipts: list, domain: Optional[str]) -> float:
        """0-25 points: receipt chain depth, domain receipts 2x weight."""
        if domain:
            weighted = sum(2.0 if r.domain == domain else 1.0 for r in receipts)
        else:
            weighted = float(len(receipts))
        return min(25.0, weighted * 25.0 / self.min_receipts_for_full_volume)

    def _quality_score(self, receipts: list, domain: Optional[str], now: datetime) -> float:
        """0-40 points: recency-weighted pass rate."""
        if not receipts:
            return 0.0
        
        total_weight = 0.0
        weighted_pass = 0.0
        
        for r in receipts:
            w = self._decay(r, now)
            if domain:
                w *= 2.0 if r.domain == domain else 1.0
            total_weight += w
            # pass=1.0, partial=0.5, fail/error=0.0
            quality = {"pass": r.pass_rate, "partial": r.pass_rate * 0.5, "fail": 0.0, "error": 0.0}.get(r.verdict, r.pass_rate)
            weighted_pass += w * quality

        weighted_rate = weighted_pass / total_weight if total_weight > 0 else 0.0
        return weighted_rate * 40.0

    def _consistency_score(self, receipts: list, now: datetime) -> float:
        """0-20 points: lower variance in pass rate = more consistent = higher score."""
        if len(receipts) < 3:
            # Not enough data — give partial score (benefit of the doubt)
            return 10.0

        pass_rates = [r.pass_rate for r in receipts]
        mean = sum(pass_rates) / len(pass_rates)
        variance = sum((p - mean) ** 2 for p in pass_rates) / len(pass_rates)
        std_dev = math.sqrt(variance)

        # High std_dev (0.5) → 0 points. Low std_dev (0.0) → 20 points
        variance_penalty = std_dev * 40.0  # 0.5 std_dev = 20 points penalty = 0 score
        return max(0.0, 20.0 - variance_penalty)

    def _recency_score(self, receipts: list, now: datetime) -> float:
        """0-10 points: days since most recent receipt (0 today, 0 after 10 days)."""
        if not receipts:
            return 0.0
        most_recent_ts = max(self._ts(r.verified_at) for r in receipts)
        days_ago = (now.timestamp() - most_recent_ts) / 86400
        return max(0.0, 10.0 - days_ago)

    def _diversity_score(self, receipts: list) -> float:
        """0-5 points: unique task types."""
        unique_types = len(set(r.task_type for r in receipts))
        return min(5.0, unique_types * 1.5)

    def _justify(self, score: CreditScoreBreakdown, daily_usd: float) -> str:
        """Generate human-readable justification."""
        parts = [f"Score {score.total}/100 ({score.tier} tier) → ${daily_usd}/day compute credit."]
        if score.volume < 15:
            parts.append("Low volume: build more receipt history to increase credit.")
        if score.quality < 30:
            parts.append("Quality below threshold: improve pass rate to unlock higher limits.")
        if score.consistency < 10:
            parts.append("High variance in pass rates: consistency builds trust.")
        if score.recency < 5:
            parts.append("Low recency: inactive period detected; credit decaying.")
        if score.total >= 80:
            parts.append("Full credit tier achieved; limit set by sponsor policy.")
        return " ".join(parts)


# ── HTTP API extension ────────────────────────────────────────────────────────

def make_credit_handler(scorer: CreditScorer):
    """
    Returns an HTTP request handler function for credit scoring endpoints.
    
    Endpoints:
      POST /credit/score    — compute score from receipt list
      POST /credit/line     — compute credit line from receipts  
      POST /credit/project  — sustainability projection
      GET  /credit/tiers    — policy table
    """
    def handle(path: str, body: dict) -> dict:
        if path == "/credit/tiers":
            return {
                "tiers": [
                    {
                        "name": name,
                        "min_score": {"verified": 80, "established": 60, "developing": 40, "new": 20, "bootstrap": 0}[name],
                        "max_score": {"verified": 100, "established": 80, "developing": 60, "new": 40, "bootstrap": 20}[name],
                        **policy,
                    }
                    for name, policy in TIER_POLICIES.items()
                ]
            }

        receipts = body.get("receipts", [])
        domain = body.get("domain")
        score = scorer.score_from_receipts(receipts, domain=domain)

        if path == "/credit/score":
            return score.to_dict()

        elif path == "/credit/line":
            line = scorer.credit_line(score)
            return {
                "score": line.score,
                "tier": line.tier,
                "daily_usd": line.daily_usd,
                "max_task_usd": line.max_task_usd,
                "requests_per_day": line.requests_per_day,
                "justification": line.justification,
                "sponsor_required": line.sponsor_required,
                "breakdown": score.to_dict(),
            }

        elif path == "/credit/project":
            task_value = body.get("task_value_usd", 0.01)
            tasks_per_day = body.get("tasks_per_day", 50)
            maintenance = body.get("maintenance_cost_usd", 1.00)
            return scorer.sustainability_projection(score, task_value, tasks_per_day, maintenance)

        else:
            return {"error": f"Unknown path: {path}"}

    return handle


# ── Demo ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import random

    print("=== ClawBizarre Compute Credit Protocol Demo ===\n")

    scorer = CreditScorer()
    now = datetime.now(timezone.utc)

    def make_receipt(days_ago: float, verdict: str, pass_rate: float, task_type: str) -> ReceiptSummary:
        ts = datetime.fromtimestamp(now.timestamp() - days_ago * 86400, tz=timezone.utc).isoformat()
        return ReceiptSummary(
            receipt_id=f"rcpt-{days_ago:.0f}-{task_type[:4]}",
            agent_id="agent:test",
            task_type=task_type,
            verdict=verdict,
            pass_rate=pass_rate,
            verified_at=ts,
            domain=task_type,
        )

    # Scenario 1: Established agent (40 receipts, good pass rate)
    print("Scenario 1: Established agent — 40 receipts, 88% avg pass rate")
    receipts_1 = (
        [make_receipt(i * 2, "pass", 0.85 + random.uniform(-0.1, 0.15), "code") for i in range(20)]
        + [make_receipt(i * 3 + 1, "pass", 0.90, "research") for i in range(10)]
        + [make_receipt(i * 5 + 2, "partial", 0.60, "translation") for i in range(8)]
        + [make_receipt(60, "fail", 0.0, "code")]
        + [make_receipt(70, "fail", 0.0, "code")]
    )
    score_1 = scorer.score_from_receipts(receipts_1, domain="code")
    line_1 = scorer.credit_line(score_1)
    print(f"  Score: {score_1.total}/100  Tier: {score_1.tier}")
    print(f"  Components: volume={score_1.volume}, quality={score_1.quality}, "
          f"consistency={score_1.consistency}, recency={score_1.recency}, diversity={score_1.diversity}")
    print(f"  Credit line: ${line_1.daily_usd}/day, max ${line_1.max_task_usd}/task")
    print(f"  Justification: {line_1.justification[:100]}...")

    projection_1 = scorer.sustainability_projection(score_1)
    print(f"  Self-sustaining at 50 tasks/day: {projection_1['self_sustaining']}")
    print()

    # Scenario 2: New agent, 5 receipts
    print("Scenario 2: New agent — 5 receipts, 100% pass rate")
    receipts_2 = [make_receipt(i, "pass", 1.0, "code") for i in range(5)]
    score_2 = scorer.score_from_receipts(receipts_2)
    line_2 = scorer.credit_line(score_2)
    print(f"  Score: {score_2.total}/100  Tier: {score_2.tier}")
    print(f"  Credit line: ${line_2.daily_usd}/day")
    bootstrap = scorer.bootstrap_credit(score_1)  # established agent vouches
    print(f"  With voucher (established agent): ${bootstrap.daily_usd}/day bootstrap")
    print()

    # Scenario 3: Struggling agent, high variance
    print("Scenario 3: Struggling agent — 30 receipts, inconsistent quality")
    receipts_3 = [make_receipt(i * 3, "pass" if i % 3 != 0 else "fail", 1.0 if i % 3 != 0 else 0.0, "code")
                  for i in range(30)]
    score_3 = scorer.score_from_receipts(receipts_3)
    line_3 = scorer.credit_line(score_3)
    print(f"  Score: {score_3.total}/100  Tier: {score_3.tier}")
    print(f"  Credit line: ${line_3.daily_usd}/day")
    print(f"  Note: High variance penalty = {round(20 - score_3.consistency, 1)} points lost")
    print()

    # Tier table
    print("Credit Tier Policy:")
    handler = make_credit_handler(scorer)
    tiers = handler("/credit/tiers", {})["tiers"]
    for t in tiers:
        print(f"  {t['name']:12s} {t['min_score']:3d}-{t['max_score']:3d} pts | "
              f"${t['daily_usd']:5.2f}/day | {t['rpm']:5d} rpm")
