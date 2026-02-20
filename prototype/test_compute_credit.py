"""
Tests for compute_credit.py — Compute Credit Protocol.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import math
from datetime import datetime, timezone, timedelta
from compute_credit import (
    CreditScorer, CreditScoreBreakdown, CreditLine, ReceiptSummary,
    TIER_POLICIES, make_credit_handler
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def scorer():
    return CreditScorer()


@pytest.fixture
def now():
    return datetime.now(timezone.utc)


def make_receipt(days_ago: float, verdict: str = "pass", pass_rate: float = 1.0,
                 task_type: str = "code", agent_id: str = "agent:test") -> ReceiptSummary:
    """Create a test receipt."""
    ts = (datetime.now(timezone.utc) - timedelta(days=days_ago)).isoformat()
    return ReceiptSummary(
        receipt_id=f"rcpt-{days_ago:.1f}-{task_type[:4]}",
        agent_id=agent_id,
        task_type=task_type,
        verdict=verdict,
        pass_rate=pass_rate,
        verified_at=ts,
        domain=task_type,
    )


# ── Score Components ──────────────────────────────────────────────────────────

class TestVolumeScore:

    def test_zero_receipts_zero_score(self, scorer):
        score = scorer.score_from_receipts([])
        assert score.total == 0.0
        assert score.volume == 0.0

    def test_full_volume_at_50_receipts(self, scorer):
        receipts = [make_receipt(i * 1.5) for i in range(50)]
        score = scorer.score_from_receipts(receipts)
        assert score.volume == 25.0

    def test_partial_volume_at_25_receipts(self, scorer):
        receipts = [make_receipt(i) for i in range(25)]
        score = scorer.score_from_receipts(receipts)
        assert score.volume == pytest.approx(12.5, abs=0.1)

    def test_domain_receipts_get_2x_weight(self, scorer):
        domain_receipts = [make_receipt(i, task_type="code") for i in range(15)]
        score = scorer.score_from_receipts(domain_receipts, domain="code")
        # 15 receipts * 2x weight = 30 "weighted" → min(25, 30 * 25/50) = min(25, 15) = 15
        assert score.volume == pytest.approx(15.0, abs=0.5)

    def test_volume_capped_at_25(self, scorer):
        receipts = [make_receipt(i * 0.5) for i in range(100)]
        score = scorer.score_from_receipts(receipts)
        assert score.volume == 25.0


class TestQualityScore:

    def test_perfect_pass_rate_max_quality(self, scorer):
        receipts = [make_receipt(i) for i in range(20)]  # all pass, pass_rate=1.0
        score = scorer.score_from_receipts(receipts)
        assert score.quality == pytest.approx(40.0, abs=1.0)

    def test_zero_pass_rate_zero_quality(self, scorer):
        receipts = [make_receipt(i, verdict="fail", pass_rate=0.0) for i in range(20)]
        score = scorer.score_from_receipts(receipts)
        assert score.quality == pytest.approx(0.0, abs=0.5)

    def test_50pct_pass_rate_half_quality(self, scorer):
        receipts = [make_receipt(i, verdict="pass", pass_rate=0.5) for i in range(20)]
        score = scorer.score_from_receipts(receipts)
        assert score.quality == pytest.approx(20.0, abs=2.0)

    def test_older_receipts_less_weight(self, scorer):
        """Recent receipts should weigh more than old ones."""
        recent = [make_receipt(1, verdict="pass", pass_rate=1.0)]
        old = [make_receipt(60, verdict="fail", pass_rate=0.0)]
        
        score_recent_first = scorer.score_from_receipts(recent + old)
        score_old_first = scorer.score_from_receipts(old + recent)
        
        # Both should be same (order doesn't matter), and quality should be > 0
        # because recent pass > old fail due to recency weighting
        assert score_recent_first.quality == score_old_first.quality
        assert score_recent_first.quality > 15.0  # recent pass dominates

    def test_partial_verdict_half_credit(self, scorer):
        """Partial receipts should get 50% quality credit."""
        partial_receipts = [make_receipt(i, verdict="partial", pass_rate=0.8) for i in range(20)]
        pass_receipts = [make_receipt(i, verdict="pass", pass_rate=0.8) for i in range(20)]
        
        score_partial = scorer.score_from_receipts(partial_receipts)
        score_pass = scorer.score_from_receipts(pass_receipts)
        
        assert score_partial.quality < score_pass.quality


class TestConsistencyScore:

    def test_perfect_consistency(self, scorer):
        """All same pass rate = zero variance = full consistency."""
        receipts = [make_receipt(i, verdict="pass", pass_rate=0.9) for i in range(20)]
        score = scorer.score_from_receipts(receipts)
        assert score.consistency == pytest.approx(20.0, abs=0.5)

    def test_high_variance_low_consistency(self, scorer):
        """Alternating 1.0 and 0.0 pass rates = high variance."""
        receipts = [
            make_receipt(i, verdict=("pass" if i % 2 == 0 else "fail"),
                        pass_rate=(1.0 if i % 2 == 0 else 0.0))
            for i in range(20)
        ]
        score = scorer.score_from_receipts(receipts)
        assert score.consistency < 5.0  # High variance = low consistency

    def test_few_receipts_partial_consistency(self, scorer):
        """< 3 receipts → partial score (benefit of the doubt)."""
        receipts = [make_receipt(i) for i in range(2)]
        score = scorer.score_from_receipts(receipts)
        assert score.consistency == pytest.approx(10.0, abs=0.1)


class TestRecencyScore:

    def test_receipt_today_max_recency(self, scorer):
        receipts = [make_receipt(0.0)]
        score = scorer.score_from_receipts(receipts)
        assert score.recency == pytest.approx(10.0, abs=0.1)

    def test_receipt_5_days_ago_half_recency(self, scorer):
        receipts = [make_receipt(5.0)]
        score = scorer.score_from_receipts(receipts)
        assert score.recency == pytest.approx(5.0, abs=0.5)

    def test_receipt_10_days_ago_zero_recency(self, scorer):
        receipts = [make_receipt(10.0)]
        score = scorer.score_from_receipts(receipts)
        assert score.recency == pytest.approx(0.0, abs=0.5)

    def test_receipt_30_days_ago_zero_recency(self, scorer):
        receipts = [make_receipt(30.0)]
        score = scorer.score_from_receipts(receipts)
        assert score.recency == 0.0


class TestDiversityScore:

    def test_single_task_type_low_diversity(self, scorer):
        receipts = [make_receipt(i, task_type="code") for i in range(20)]
        score = scorer.score_from_receipts(receipts)
        assert score.diversity == pytest.approx(1.5, abs=0.1)

    def test_multiple_task_types_diversity(self, scorer):
        receipts = (
            [make_receipt(i, task_type="code") for i in range(10)]
            + [make_receipt(i, task_type="research") for i in range(5)]
            + [make_receipt(i, task_type="translation") for i in range(5)]
        )
        score = scorer.score_from_receipts(receipts)
        assert score.diversity == pytest.approx(4.5, abs=0.1)

    def test_diversity_capped_at_5(self, scorer):
        receipts = [make_receipt(i, task_type=f"type_{i}") for i in range(10)]
        score = scorer.score_from_receipts(receipts)
        assert score.diversity == 5.0


# ── Credit Tiers ─────────────────────────────────────────────────────────────

class TestCreditTiers:

    def test_empty_receipts_bootstrap_tier(self, scorer):
        score = scorer.score_from_receipts([])
        assert score.tier == "bootstrap"

    def test_new_agent_tier(self, scorer):
        """Few receipts with good quality → new tier."""
        receipts = [make_receipt(i, verdict="pass", pass_rate=1.0) for i in range(3)]
        score = scorer.score_from_receipts(receipts)
        # Volume low, but quality/recency/consistency help
        assert score.tier in ("new", "developing", "established")  # depends on exact scores

    def test_established_agent_tier(self, scorer):
        """40 receipts with good pass rate → established or verified."""
        receipts = [make_receipt(i * 2, verdict="pass", pass_rate=0.9) for i in range(40)]
        score = scorer.score_from_receipts(receipts)
        assert score.tier in ("established", "verified")

    def test_perfect_agent_verified_tier(self, scorer):
        """50+ receipts with 100% pass rate → verified."""
        receipts = [make_receipt(i * 1.5, verdict="pass", pass_rate=1.0, task_type=f"type_{i%3}") for i in range(50)]
        score = scorer.score_from_receipts(receipts)
        assert score.tier == "verified"
        assert score.total >= 80.0


# ── Credit Lines ──────────────────────────────────────────────────────────────

class TestCreditLines:

    def test_credit_line_for_bootstrap(self, scorer):
        score = scorer.score_from_receipts([])
        line = scorer.credit_line(score)
        assert line.tier == "bootstrap"
        assert line.daily_usd <= TIER_POLICIES["bootstrap"]["daily_usd"]
        assert line.daily_usd >= 0

    def test_credit_line_scales_with_score(self, scorer):
        """Higher score → higher credit line."""
        low_score = CreditScoreBreakdown(2, 10, 5, 3, 0)   # ~20 total
        high_score = CreditScoreBreakdown(25, 38, 18, 9, 5) # ~95 total
        
        low_line = scorer.credit_line(low_score)
        high_line = scorer.credit_line(high_score)
        
        assert high_line.daily_usd > low_line.daily_usd

    def test_credit_line_has_justification(self, scorer):
        receipts = [make_receipt(i) for i in range(10)]
        score = scorer.score_from_receipts(receipts)
        line = scorer.credit_line(score)
        assert len(line.justification) > 20

    def test_sponsor_required_always_true(self, scorer):
        """Until Era 3, sponsor is always required."""
        score = scorer.score_from_receipts([])
        line = scorer.credit_line(score)
        assert line.sponsor_required is True


# ── Bootstrap Credit ─────────────────────────────────────────────────────────

class TestBootstrapCredit:

    def test_bootstrap_credit_from_voucher(self, scorer):
        """New agent gets 20% of voucher's credit line."""
        voucher_receipts = [make_receipt(i * 2, verdict="pass", pass_rate=0.9) for i in range(50)]
        voucher_score = scorer.score_from_receipts(voucher_receipts)
        
        bootstrap = scorer.bootstrap_credit(voucher_score)
        voucher_line = scorer.credit_line(voucher_score)
        
        assert bootstrap.daily_usd == pytest.approx(voucher_line.daily_usd * 0.20, abs=0.01)

    def test_bootstrap_lower_than_voucher(self, scorer):
        """Bootstrap credit always less than voucher's own credit line."""
        voucher_receipts = [make_receipt(i * 2, verdict="pass", pass_rate=0.9) for i in range(50)]
        voucher_score = scorer.score_from_receipts(voucher_receipts)
        
        bootstrap = scorer.bootstrap_credit(voucher_score)
        voucher_line = scorer.credit_line(voucher_score)
        
        assert bootstrap.daily_usd < voucher_line.daily_usd


# ── Sustainability Projection ────────────────────────────────────────────────

class TestSustainabilityProjection:

    def test_high_volume_self_sustaining(self, scorer):
        """Agent doing 200 tasks/day at $0.01 each earns $2/day > $1 maintenance."""
        receipts = [make_receipt(i * 2, verdict="pass", pass_rate=0.9) for i in range(50)]
        score = scorer.score_from_receipts(receipts)
        proj = scorer.sustainability_projection(score, task_value_usd=0.01, tasks_per_day=200, maintenance_cost_usd=1.00)
        assert proj["self_sustaining"] is True

    def test_low_volume_not_self_sustaining(self, scorer):
        """5 tasks/day at $0.01 = $0.05 < $1 maintenance."""
        receipts = [make_receipt(i) for i in range(5)]
        score = scorer.score_from_receipts(receipts)
        proj = scorer.sustainability_projection(score, task_value_usd=0.01, tasks_per_day=5, maintenance_cost_usd=1.00)
        assert proj["self_sustaining"] is False
        assert proj["revenue_gap_usd"] > 0

    def test_break_even_calculation(self, scorer):
        """Break-even at task_value=0.01, maintenance=1.00 → 100 tasks/day."""
        receipts = [make_receipt(i) for i in range(20)]
        score = scorer.score_from_receipts(receipts)
        proj = scorer.sustainability_projection(score, task_value_usd=0.01, maintenance_cost_usd=1.00)
        assert proj["break_even_tasks_per_day"] == 100


# ── HTTP Handler ──────────────────────────────────────────────────────────────

class TestHTTPHandler:

    @pytest.fixture
    def handler(self, scorer):
        return make_credit_handler(scorer)

    def test_tiers_endpoint(self, handler):
        result = handler("/credit/tiers", {})
        assert "tiers" in result
        assert len(result["tiers"]) == 5

    def test_score_endpoint(self, handler):
        receipts = [make_receipt(i).__dict__ for i in range(10)]
        result = handler("/credit/score", {"receipts": receipts})
        assert "total" in result
        assert "tier" in result
        assert "components" in result

    def test_line_endpoint(self, handler):
        receipts = [make_receipt(i).__dict__ for i in range(10)]
        result = handler("/credit/line", {"receipts": receipts})
        assert "daily_usd" in result
        assert "tier" in result
        assert "justification" in result

    def test_project_endpoint(self, handler):
        receipts = [make_receipt(i).__dict__ for i in range(20)]
        result = handler("/credit/project", {
            "receipts": receipts,
            "task_value_usd": 0.01,
            "tasks_per_day": 50,
        })
        assert "self_sustaining" in result
        assert "break_even_tasks_per_day" in result

    def test_unknown_path_returns_error(self, handler):
        result = handler("/credit/unknown", {})
        assert "error" in result


# ── Raw dict normalization ────────────────────────────────────────────────────

class TestNormalization:

    def test_normalize_receipt_dict(self, scorer):
        """Raw VRF receipt dict should be normalized correctly."""
        raw = {
            "receipt_id": "rcpt-001",
            "agent_id": "agent:test",
            "task_type": "code",
            "verdict": "pass",
            "results": {"total": 10, "passed": 9},
            "verified_at": datetime.now(timezone.utc).isoformat(),
        }
        score = scorer.score_from_receipts([raw])
        assert score.total > 0

    def test_normalize_receipt_summary(self, scorer):
        """ReceiptSummary should pass through unchanged."""
        r = make_receipt(1.0)
        score = scorer.score_from_receipts([r])
        assert score.total > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
