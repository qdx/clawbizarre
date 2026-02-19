"""
ClawBizarre Economy Simulation v5 — Price Competition & Dynamic Pricing

New features over v4:
- Agents observe market prices and adjust their own pricing strategy
- Three pricing strategies: reputation-premium (v4 default), undercut, adaptive
- Price floors based on compute cost (can't go below cost)
- Demand elasticity: lower prices attract more tasks
- Price wars and equilibrium analysis
- Margin tracking: revenue vs compute cost
"""

import random
import math
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from reputation import DecayingReputation, DomainReputation, MerkleTree, sha256
import hashlib

class _FastIdentity:
    """Lightweight identity stub for simulation (skips Ed25519 keygen)."""
    _counter = 0
    def __init__(self):
        _FastIdentity._counter += 1
        self.agent_id = f"sim:{_FastIdentity._counter}"
    @classmethod
    def generate(cls):
        return cls()

AgentIdentity = _FastIdentity


# --- Market Configuration ---

TASK_CATALOG = {
    "code_review": {"base_cost": 0.50, "complexity": 0.7, "compute_cost": 0.15},
    "translation": {"base_cost": 0.30, "complexity": 0.5, "compute_cost": 0.08},
    "summarization": {"base_cost": 0.20, "complexity": 0.3, "compute_cost": 0.05},
    "data_validation": {"base_cost": 0.15, "complexity": 0.2, "compute_cost": 0.03},
    "research": {"base_cost": 1.00, "complexity": 0.9, "compute_cost": 0.30},
    "monitoring": {"base_cost": 0.10, "complexity": 0.1, "compute_cost": 0.02},
}

DOMAIN_CORRELATIONS = {
    ("code_review", "research"): 0.5,
    ("code_review", "data_validation"): 0.3,
    ("translation", "summarization"): 0.4,
    ("research", "summarization"): 0.6,
    ("monitoring", "data_validation"): 0.3,
}

ALL_DOMAINS = list(TASK_CATALOG.keys())


# --- Pricing Strategies ---

class PricingStrategy:
    """Base pricing strategy."""
    name = "base"

    def price(self, agent, domain: str, sim_time: float, market_prices: dict) -> float:
        raise NotImplementedError


class ReputationPremium(PricingStrategy):
    """v4 default: price = base * (1 + rep_premium). No market awareness."""
    name = "reputation"

    def price(self, agent, domain: str, sim_time: float, market_prices: dict) -> float:
        base = TASK_CATALOG[domain]["base_cost"]
        floor = TASK_CATALOG[domain]["compute_cost"]
        rep_score = agent.reputation.score(domain, sim_time)
        conf = agent.reputation._get_domain(domain).confidence(sim_time) if domain in agent.reputation._domains else 0.0
        rep_premium = rep_score * conf * 2.0
        return max(base * (1 + rep_premium), floor)


class UndercutStrategy(PricingStrategy):
    """Always price X% below market average. Race to the bottom."""
    name = "undercut"

    def __init__(self, undercut_pct: float = 0.15):
        self.undercut_pct = undercut_pct

    def price(self, agent, domain: str, sim_time: float, market_prices: dict) -> float:
        floor = TASK_CATALOG[domain]["compute_cost"]
        if domain in market_prices and market_prices[domain]:
            avg_price = sum(market_prices[domain]) / len(market_prices[domain])
            return max(avg_price * (1 - self.undercut_pct), floor)
        return max(TASK_CATALOG[domain]["base_cost"] * 0.85, floor)


class AdaptiveStrategy(PricingStrategy):
    """Adjust based on utilization. High demand → raise prices. Low demand → cut prices."""
    name = "adaptive"

    def __init__(self, target_utilization: float = 0.6):
        self.target_utilization = target_utilization

    def price(self, agent, domain: str, sim_time: float, market_prices: dict) -> float:
        base = TASK_CATALOG[domain]["base_cost"]
        floor = TASK_CATALOG[domain]["compute_cost"]
        rep_score = agent.reputation.score(domain, sim_time)
        conf = agent.reputation._get_domain(domain).confidence(sim_time) if domain in agent.reputation._domains else 0.0

        # Calculate utilization: tasks completed / rounds alive
        rounds_alive = max(agent.tasks_completed + agent.tasks_failed, 1)
        total_opportunities = max(agent.rounds_active, 1)
        utilization = rounds_alive / total_opportunities

        # Price adjustment: above target util → raise, below → lower
        util_ratio = utilization / self.target_utilization
        price_mult = 0.7 + 0.6 * min(util_ratio, 2.0)  # range: 0.7x to 1.9x

        # Reputation still matters
        rep_premium = rep_score * conf * 1.5
        return max(base * price_mult * (1 + rep_premium), floor)


class QualityPremium(PricingStrategy):
    """Price based on success rate. Higher reliability → higher price. Quality signal."""
    name = "quality"

    def price(self, agent, domain: str, sim_time: float, market_prices: dict) -> float:
        base = TASK_CATALOG[domain]["base_cost"]
        floor = TASK_CATALOG[domain]["compute_cost"]
        total = agent.tasks_completed + agent.tasks_failed
        if total < 10:
            return base  # not enough data, charge base
        success_rate = agent.tasks_completed / total
        # Premium scales with reliability: 90% → 1.8x, 70% → 1.4x, 50% → 1.0x
        premium = 1.0 + (success_rate - 0.5) * 2.0
        return max(base * max(premium, 0.5), floor)


# --- Coordination Penalty ---

def quadratic_penalty(fleet_size: int) -> float:
    pairs = fleet_size * (fleet_size - 1) / 2
    return min(pairs * 0.002, 0.50)


# --- Agent ---

@dataclass
class MarketAgent:
    name: str
    identity: AgentIdentity
    capabilities: list[str]
    reliability: dict[str, float]
    pricing_strategy: PricingStrategy
    reputation: DomainReputation = field(default_factory=lambda: DomainReputation(
        half_life_days=30, domain_correlations=DOMAIN_CORRELATIONS
    ))
    fleet_id: Optional[str] = None

    # Economics
    balance: float = 0.0
    spent: float = 0.0
    compute_costs: float = 0.0  # actual compute spent doing work
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_requested: int = 0
    revenue_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    costs_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    prices_charged: list[float] = field(default_factory=list)
    receipt_hashes: list[str] = field(default_factory=list)

    # Lifecycle
    joined_round: int = 0
    exited_round: Optional[int] = None
    active: bool = True
    identity_cost_per_round: float = 0.01
    rounds_active: int = 0

    # Newcomer
    is_newcomer: bool = False

    def price_for(self, domain: str, sim_time: float, market_prices: dict) -> float:
        return self.pricing_strategy.price(self, domain, sim_time, market_prices)

    def accept_task(self, domain: str) -> bool:
        return random.random() < self.reliability.get(domain, 0.5)

    def net_profit(self) -> float:
        return self.balance - self.spent - self.compute_costs

    def margin(self) -> float:
        if self.balance <= 0:
            return 0.0
        return (self.balance - self.compute_costs) / self.balance


def make_agent(idx: int, joined_round: int = 0, strategy: Optional[PricingStrategy] = None) -> MarketAgent:
    archetype = random.choice(["specialist", "generalist", "midtier", "newbie"])
    if archetype == "specialist":
        caps = [random.choice(ALL_DOMAINS)]
        rel_range = (0.90, 0.99)
    elif archetype == "generalist":
        caps = random.sample(ALL_DOMAINS, k=random.randint(3, 5))
        rel_range = (0.70, 0.85)
    elif archetype == "midtier":
        caps = random.sample(ALL_DOMAINS, k=random.randint(2, 3))
        rel_range = (0.75, 0.90)
    else:
        caps = random.sample(ALL_DOMAINS, k=random.randint(1, 2))
        rel_range = (0.55, 0.75)

    if strategy is None:
        strategy = random.choice([
            ReputationPremium(),
            UndercutStrategy(random.uniform(0.10, 0.25)),
            AdaptiveStrategy(random.uniform(0.4, 0.8)),
            QualityPremium(),
        ])

    return MarketAgent(
        name=f"Agent_{idx}",
        identity=AgentIdentity.generate(),
        capabilities=caps,
        reliability={d: random.uniform(*rel_range) for d in caps},
        pricing_strategy=strategy,
        joined_round=joined_round,
        is_newcomer=(joined_round > 0),
    )


# --- Discovery (with price sensitivity) ---

def discover_worker(agents: list[MarketAgent], requester: MarketAgent,
                    domain: str, sim_time: float, round_num: int,
                    market_prices: dict, price_sensitivity: float = 1.0) -> Optional[MarketAgent]:
    """
    Discovery with price awareness. Higher price_sensitivity = buyers care more about price.
    """
    candidates = [a for a in agents
                  if a.active and a.name != requester.name
                  and domain in a.capabilities]
    if not candidates:
        return None

    scored = []
    for c in candidates:
        rep = c.reputation.score(domain, sim_time)
        conf = c.reputation._get_domain(domain).confidence(sim_time) if domain in c.reputation._domains else 0.0
        price = c.price_for(domain, sim_time, market_prices)

        exploration_bonus = 0.3 if conf < 0.2 else 0.1 if conf < 0.4 else 0.0

        # Value = quality / price^sensitivity
        quality = rep * max(conf, 0.1) + exploration_bonus
        price_factor = max(price, 0.01) ** price_sensitivity
        value = quality / price_factor

        scored.append((c, value, price))

    temperature = 0.5
    values = [v for _, v, _ in scored]
    max_v = max(values)
    exp_values = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]

    chosen = random.choices([c for c, _, _ in scored], weights=probs, k=1)[0]
    return chosen


# --- Market Price Tracker ---

class MarketPriceTracker:
    """Rolling window of recent transaction prices per domain."""

    def __init__(self, window: int = 100):
        self.window = window
        self.prices: dict[str, list[float]] = defaultdict(list)

    def record(self, domain: str, price: float):
        self.prices[domain].append(price)
        if len(self.prices[domain]) > self.window:
            self.prices[domain] = self.prices[domain][-self.window:]

    def get_prices(self) -> dict[str, list[float]]:
        return dict(self.prices)

    def avg(self, domain: str) -> float:
        if domain not in self.prices or not self.prices[domain]:
            return TASK_CATALOG[domain]["base_cost"]
        return sum(self.prices[domain]) / len(self.prices[domain])

    def spread(self, domain: str) -> tuple[float, float]:
        """Min and max recent price."""
        if domain not in self.prices or not self.prices[domain]:
            base = TASK_CATALOG[domain]["base_cost"]
            return base, base
        return min(self.prices[domain]), max(self.prices[domain])


# --- Simulation ---

def run_simulation(label: str, num_agents: int = 50, num_rounds: int = 2000,
                   entry_rate: float = 0.05, exit_threshold: float = -5.0,
                   price_sensitivity: float = 1.0,
                   strategy_mix: Optional[dict] = None,
                   seed: int = 42) -> dict:
    """
    v5 simulation with price competition.

    strategy_mix: dict mapping strategy name to fraction, e.g.
        {"reputation": 0.25, "undercut": 0.25, "adaptive": 0.25, "quality": 0.25}
    """
    random.seed(seed)
    sim_start = time.time() - 84 * 86400
    tracker = MarketPriceTracker(window=200)

    agent_idx = 0
    agents = []

    strategies_pool = {
        "reputation": ReputationPremium,
        "undercut": lambda: UndercutStrategy(random.uniform(0.10, 0.25)),
        "adaptive": lambda: AdaptiveStrategy(random.uniform(0.4, 0.8)),
        "quality": QualityPremium,
    }

    def pick_strategy() -> PricingStrategy:
        if strategy_mix:
            roll = random.random()
            cumulative = 0.0
            for name, frac in strategy_mix.items():
                cumulative += frac
                if roll < cumulative:
                    factory = strategies_pool[name]
                    return factory() if callable(factory) and not isinstance(factory, type) else factory()
            # fallback
            return ReputationPremium()
        return random.choice([
            ReputationPremium(),
            UndercutStrategy(random.uniform(0.10, 0.25)),
            AdaptiveStrategy(random.uniform(0.4, 0.8)),
            QualityPremium(),
        ])

    # Create initial agents
    for i in range(num_agents):
        a = make_agent(agent_idx, strategy=pick_strategy())
        a.is_newcomer = False
        agents.append(a)
        agent_idx += 1

    # Tracking
    snapshots = []
    strategy_earnings = defaultdict(lambda: {"agents": 0, "total_balance": 0.0, "total_margin": 0.0,
                                              "tasks": 0, "survived": 0})
    price_history = []  # (round, {domain: avg_price})

    for round_num in range(1, num_rounds + 1):
        sim_time = sim_start + round_num * 3600
        active_agents = [a for a in agents if a.active]

        # Identity cost
        for a in active_agents:
            a.balance -= a.identity_cost_per_round
            a.rounds_active += 1

        # Update newcomer status
        for a in active_agents:
            if a.is_newcomer and (round_num - a.joined_round) > 200:
                a.is_newcomer = False

        # Agent entry
        num_entries = 0
        while random.random() < entry_rate:
            num_entries += 1
        for _ in range(num_entries):
            new_agent = make_agent(agent_idx, joined_round=round_num, strategy=pick_strategy())
            agents.append(new_agent)
            agent_idx += 1

        # Agent exit
        for a in active_agents:
            if a.balance < exit_threshold and round_num - a.joined_round > 100:
                a.active = False
                a.exited_round = round_num

        active_agents = [a for a in agents if a.active]
        if len(active_agents) < 2:
            continue

        # Demand scales with number of active agents (network effect)
        base_tasks = random.randint(3, 6)
        demand_mult = 1.0 + 0.01 * len(active_agents)  # slight network effect
        num_tasks = int(base_tasks * demand_mult)

        market_prices = tracker.get_prices()

        for _ in range(num_tasks):
            requester = random.choice(active_agents)
            domain = random.choice(ALL_DOMAINS)

            worker = discover_worker(
                active_agents, requester, domain, sim_time, round_num,
                market_prices, price_sensitivity
            )
            if not worker:
                continue

            price = worker.price_for(domain, sim_time, market_prices)
            compute_cost = TASK_CATALOG[domain]["compute_cost"]

            # Coordination penalty for fleets
            if worker.fleet_id:
                fleet_members = sum(1 for a in active_agents if a.fleet_id == worker.fleet_id)
                penalty = quadratic_penalty(fleet_members)
                effective_reliability = worker.reliability.get(domain, 0.5) * (1 - penalty)
                success = random.random() < effective_reliability
            else:
                success = worker.accept_task(domain)

            worker.reputation.record(domain, success, timestamp=sim_time)
            tracker.record(domain, price)
            worker.prices_charged.append(price)

            # Compute cost is always paid (attempted work costs compute)
            worker.compute_costs += compute_cost
            worker.costs_by_domain[domain] += compute_cost

            if success:
                worker.balance += price
                worker.revenue_by_domain[domain] += price
                worker.tasks_completed += 1
                worker.receipt_hashes.append(sha256(f"{round_num}:{worker.name}:{domain}".encode()))
            else:
                worker.tasks_failed += 1
                # Failed work: no payment but compute cost was incurred

            requester.spent += price if success else 0  # only pay on success
            requester.tasks_requested += 1

        # Price history snapshot
        if round_num % 100 == 0:
            price_snap = {}
            for d in ALL_DOMAINS:
                price_snap[d] = tracker.avg(d)
            price_history.append({"round": round_num, "prices": price_snap})

        # Periodic snapshot
        if round_num % 200 == 0:
            earnings = sorted(a.balance for a in active_agents)
            n = len(earnings)
            s = sum(earnings)
            gini = sum((2*i - n + 1) * e for i, e in enumerate(earnings)) / (n * s) if s > 0 and n > 1 else 0

            newcomer_count = sum(1 for a in agents if a.joined_round > 0)
            newcomer_alive = sum(1 for a in agents if a.joined_round > 0 and a.active)
            surv_rate = newcomer_alive / max(newcomer_count, 1)

            snapshots.append({
                "round": round_num,
                "active": len(active_agents),
                "total": len(agents),
                "gini": gini,
                "newcomer_survival": surv_rate,
                "avg_prices": {d: tracker.avg(d) for d in ALL_DOMAINS},
            })

    # Final analysis
    active_agents = [a for a in agents if a.active]
    incumbents = [a for a in active_agents if a.joined_round == 0]
    newcomer_survivors = [a for a in active_agents if a.joined_round > 0]

    # Strategy performance
    for a in agents:
        sname = a.pricing_strategy.name
        strategy_earnings[sname]["agents"] += 1
        strategy_earnings[sname]["total_balance"] += a.balance
        strategy_earnings[sname]["tasks"] += a.tasks_completed
        if a.active:
            strategy_earnings[sname]["survived"] += 1
            strategy_earnings[sname]["total_margin"] += a.margin()

    # Cohort analysis
    cohort_size = 400
    cohorts = []
    for start in range(0, num_rounds, cohort_size):
        cohort = [a for a in agents if start < a.joined_round <= start + cohort_size]
        if not cohort:
            continue
        alive = sum(1 for a in cohort if a.active)
        avg_bal = sum(a.balance for a in cohort) / len(cohort)
        cohorts.append({
            "range": f"{start+1}-{start+cohort_size}",
            "entered": len(cohort),
            "survived": alive,
            "survival_pct": alive / len(cohort) * 100,
            "avg_balance": avg_bal,
        })

    # Price convergence analysis
    early_prices = price_history[:5] if len(price_history) >= 5 else price_history
    late_prices = price_history[-5:] if len(price_history) >= 5 else price_history

    price_convergence = {}
    for d in ALL_DOMAINS:
        early_avg = sum(p["prices"][d] for p in early_prices) / max(len(early_prices), 1)
        late_avg = sum(p["prices"][d] for p in late_prices) / max(len(late_prices), 1)
        compute = TASK_CATALOG[d]["compute_cost"]
        price_convergence[d] = {
            "early_avg": early_avg,
            "late_avg": late_avg,
            "compute_cost": compute,
            "late_margin": (late_avg - compute) / max(late_avg, 0.01),
            "price_change_pct": (late_avg - early_avg) / max(early_avg, 0.01) * 100,
        }

    result = {
        "label": label,
        "total_agents": len(agents),
        "active": len(active_agents),
        "exited": len(agents) - len(active_agents),
        "incumbents_alive": len(incumbents),
        "incumbents_total": num_agents,
        "newcomer_survivors": len(newcomer_survivors),
        "newcomer_total": len(agents) - num_agents,
        "incumbent_avg_balance": sum(a.balance for a in incumbents) / max(len(incumbents), 1),
        "newcomer_avg_balance": sum(a.balance for a in newcomer_survivors) / max(len(newcomer_survivors), 1),
        "final_gini": snapshots[-1]["gini"] if snapshots else 0,
        "cohorts": cohorts,
        "snapshots": snapshots,
        "strategy_performance": dict(strategy_earnings),
        "price_convergence": price_convergence,
        "price_history": price_history,
    }

    return result


def print_result(r: dict):
    print(f"\n{'='*70}")
    print(f"  {r['label']}")
    print(f"{'='*70}")
    print(f"  Agents: {r['total_agents']} total, {r['active']} active, {r['exited']} exited")
    print(f"  Incumbents: {r['incumbents_alive']}/{r['incumbents_total']} survived")
    nc_total = max(r['newcomer_total'], 1)
    print(f"  Newcomers: {r['newcomer_survivors']}/{r['newcomer_total']} survived "
          f"({r['newcomer_survivors']/nc_total*100:.0f}%)")
    print(f"  Incumbent avg: ${r['incumbent_avg_balance']:.2f}")
    print(f"  Newcomer avg:  ${r['newcomer_avg_balance']:.2f}")
    gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
    print(f"  Earnings gap: {gap:.1f}x")
    print(f"  Final Gini: {r['final_gini']:.3f}")

    # Strategy performance
    print(f"\n  Strategy Performance:")
    print(f"  {'Strategy':<15} {'Agents':>7} {'Survived':>9} {'Avg $':>10} {'Avg Margin':>11}")
    print(f"  {'─'*55}")
    for sname, data in sorted(r["strategy_performance"].items()):
        avg_bal = data["total_balance"] / max(data["agents"], 1)
        avg_margin = data["total_margin"] / max(data["survived"], 1)
        surv_pct = data["survived"] / max(data["agents"], 1) * 100
        print(f"  {sname:<15} {data['agents']:>7} {data['survived']:>5} ({surv_pct:>4.0f}%) "
              f"${avg_bal:>8.2f} {avg_margin:>10.1%}")

    # Price convergence
    print(f"\n  Price Convergence (early → late):")
    print(f"  {'Domain':<20} {'Early':>8} {'Late':>8} {'Compute':>8} {'Margin':>8} {'Δ%':>8}")
    print(f"  {'─'*55}")
    for d, pc in sorted(r["price_convergence"].items()):
        print(f"  {d:<20} ${pc['early_avg']:>6.3f} ${pc['late_avg']:>6.3f} "
              f"${pc['compute_cost']:>6.3f} {pc['late_margin']:>7.1%} {pc['price_change_pct']:>+7.1f}%")

    # Cohort survival
    print(f"\n  Cohort Survival:")
    for c in r["cohorts"]:
        bar = "█" * int(c["survival_pct"] / 5)
        print(f"    Rounds {c['range']:>10}: {c['survived']:>3}/{c['entered']:>3} "
              f"({c['survival_pct']:>5.1f}%) ${c['avg_balance']:>7.2f}  {bar}")


# ─────────────────────────────────────────────────────────────
# Experiment 1: Strategy Competition (mixed population)
# ─────────────────────────────────────────────────────────────

def experiment_strategy_competition():
    """Equal mix of all 4 strategies competing head-to-head."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 1: Strategy Competition (equal mix)             ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    r = run_simulation(
        label="MIXED STRATEGIES (25% each)",
        num_agents=60, num_rounds=2000, entry_rate=0.05,
        price_sensitivity=1.0,
        strategy_mix={"reputation": 0.25, "undercut": 0.25, "adaptive": 0.25, "quality": 0.25},
        seed=42,
    )
    print_result(r)
    return r


# ─────────────────────────────────────────────────────────────
# Experiment 2: Homogeneous Strategy Populations
# ─────────────────────────────────────────────────────────────

def experiment_homogeneous():
    """What happens when ALL agents use the same strategy?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 2: Homogeneous Populations                      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = {}
    for strat_name in ["reputation", "undercut", "adaptive", "quality"]:
        print(f"  Running all-{strat_name}...")
        r = run_simulation(
            label=f"ALL {strat_name.upper()}",
            num_agents=50, num_rounds=2000, entry_rate=0.05,
            price_sensitivity=1.0,
            strategy_mix={strat_name: 1.0},
            seed=42,
        )
        print_result(r)
        results[strat_name] = r

    # Summary comparison
    print(f"\n{'='*70}")
    print(f"  HOMOGENEOUS STRATEGY COMPARISON")
    print(f"{'='*70}")
    print(f"  {'Strategy':<15} {'Active':>7} {'Gini':>7} {'Inc.Avg$':>10} {'New.Avg$':>10} {'Gap':>6}")
    print(f"  {'─'*60}")
    for name, r in results.items():
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        print(f"  {name:<15} {r['active']:>7} {r['final_gini']:>7.3f} "
              f"${r['incumbent_avg_balance']:>8.2f} ${r['newcomer_avg_balance']:>8.2f} {gap:>5.1f}x")

    return results


# ─────────────────────────────────────────────────────────────
# Experiment 3: Price Sensitivity Sweep
# ─────────────────────────────────────────────────────────────

def experiment_price_sensitivity():
    """How much does buyer price-sensitivity affect market dynamics?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 3: Price Sensitivity Sweep                      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = {}
    for ps in [0.0, 0.5, 1.0, 1.5, 2.0, 3.0]:
        print(f"  Running price_sensitivity={ps}...")
        r = run_simulation(
            label=f"Price Sensitivity = {ps}",
            num_agents=50, num_rounds=2000, entry_rate=0.05,
            price_sensitivity=ps,
            strategy_mix={"reputation": 0.25, "undercut": 0.25, "adaptive": 0.25, "quality": 0.25},
            seed=42,
        )
        results[ps] = r

    print(f"\n{'='*70}")
    print(f"  PRICE SENSITIVITY SWEEP")
    print(f"{'='*70}")
    print(f"  {'Sensitivity':>12} {'Active':>7} {'Gini':>7} {'Avg Price':>10} {'Gap':>6} {'Best Strategy':<15}")
    print(f"  {'─'*65}")
    for ps, r in sorted(results.items()):
        # Find which strategy has highest avg balance
        best_strat = max(r["strategy_performance"].items(),
                         key=lambda x: x[1]["total_balance"] / max(x[1]["agents"], 1))
        avg_price = sum(pc["late_avg"] for pc in r["price_convergence"].values()) / len(r["price_convergence"])
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        print(f"  {ps:>12.1f} {r['active']:>7} {r['final_gini']:>7.3f} "
              f"${avg_price:>8.3f} {gap:>5.1f}x  {best_strat[0]:<15}")

    return results


# ─────────────────────────────────────────────────────────────
# Experiment 4: Race to the Bottom
# ─────────────────────────────────────────────────────────────

def experiment_race_to_bottom():
    """75% undercutters vs 25% reputation-premium. Do prices collapse?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 4: Race to the Bottom (75% undercutters)        ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    r = run_simulation(
        label="75% UNDERCUT / 25% REPUTATION",
        num_agents=60, num_rounds=2000, entry_rate=0.05,
        price_sensitivity=1.5,  # buyers are price-sensitive
        strategy_mix={"undercut": 0.75, "reputation": 0.25},
        seed=42,
    )
    print_result(r)

    # Show price trajectory
    print(f"\n  Price Trajectory (code_review):")
    for ph in r["price_history"][::3]:  # every 300 rounds
        p = ph["prices"]["code_review"]
        compute = TASK_CATALOG["code_review"]["compute_cost"]
        margin = (p - compute) / max(p, 0.01)
        bar = "█" * int(p * 20)
        print(f"    Round {ph['round']:>5}: ${p:.3f} (margin {margin:.0%})  {bar}")

    return r


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v5                  ║")
    print("║  Price Competition & Dynamic Pricing                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    exp1 = experiment_strategy_competition()
    exp2 = experiment_homogeneous()
    exp3 = experiment_price_sensitivity()
    exp4 = experiment_race_to_bottom()

    print("\n\n" + "═" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
