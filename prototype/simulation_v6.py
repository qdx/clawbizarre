"""
ClawBizarre Economy Simulation v6 — Strategy Evolution & Information Asymmetry

New features over v5:
- Agents evaluate their performance and SWITCH pricing strategies periodically
- Evolutionary dynamics: do strategies converge? What's the Nash equilibrium?
- Information asymmetry: buyers can't observe quality until after purchase (lemons problem)
- Quality revelation through repeat interactions (relationship capital)
- Service bundling: agents can offer multi-domain packages at discount

Open questions from v5:
1. Strategy evolution — do agents converge on one strategy? ✅ THIS FILE
2. Information asymmetry — lemons problem ✅ THIS FILE
3. Bundling — avoid per-task commodity pricing ✅ THIS FILE
"""

import sys
import random
import math
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

from reputation import DecayingReputation, DomainReputation, MerkleTree, sha256
import hashlib


# --- Market Configuration (same as v5) ---

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


# --- Pricing Strategies (from v5) ---

class PricingStrategy:
    name = "base"
    def price(self, agent, domain, sim_time, market_prices):
        raise NotImplementedError

class ReputationPremium(PricingStrategy):
    name = "reputation"
    def price(self, agent, domain, sim_time, market_prices):
        base = TASK_CATALOG[domain]["base_cost"]
        floor = TASK_CATALOG[domain]["compute_cost"]
        rep_score = agent.reputation.score(domain, sim_time)
        conf = agent.reputation._get_domain(domain).confidence(sim_time) if domain in agent.reputation._domains else 0.0
        rep_premium = rep_score * conf * 2.0
        return max(base * (1 + rep_premium), floor)

class UndercutStrategy(PricingStrategy):
    name = "undercut"
    def __init__(self, undercut_pct=0.15):
        self.undercut_pct = undercut_pct
    def price(self, agent, domain, sim_time, market_prices):
        floor = TASK_CATALOG[domain]["compute_cost"]
        if domain in market_prices and market_prices[domain]:
            avg_price = sum(market_prices[domain]) / len(market_prices[domain])
            return max(avg_price * (1 - self.undercut_pct), floor)
        return max(TASK_CATALOG[domain]["base_cost"] * 0.85, floor)

class AdaptiveStrategy(PricingStrategy):
    name = "adaptive"
    def __init__(self, target_utilization=0.6):
        self.target_utilization = target_utilization
    def price(self, agent, domain, sim_time, market_prices):
        base = TASK_CATALOG[domain]["base_cost"]
        floor = TASK_CATALOG[domain]["compute_cost"]
        rep_score = agent.reputation.score(domain, sim_time)
        conf = agent.reputation._get_domain(domain).confidence(sim_time) if domain in agent.reputation._domains else 0.0
        rounds_alive = max(agent.tasks_completed + agent.tasks_failed, 1)
        total_opportunities = max(agent.rounds_active, 1)
        utilization = rounds_alive / total_opportunities
        util_ratio = utilization / self.target_utilization
        price_mult = 0.7 + 0.6 * min(util_ratio, 2.0)
        rep_premium = rep_score * conf * 1.5
        return max(base * price_mult * (1 + rep_premium), floor)

class QualityPremium(PricingStrategy):
    name = "quality"
    def price(self, agent, domain, sim_time, market_prices):
        base = TASK_CATALOG[domain]["base_cost"]
        floor = TASK_CATALOG[domain]["compute_cost"]
        total = agent.tasks_completed + agent.tasks_failed
        if total < 10:
            return base
        success_rate = agent.tasks_completed / total
        premium = 1.0 + (success_rate - 0.5) * 2.0
        return max(base * max(premium, 0.5), floor)


STRATEGY_FACTORIES = {
    "reputation": lambda: ReputationPremium(),
    "undercut": lambda: UndercutStrategy(random.uniform(0.10, 0.25)),
    "adaptive": lambda: AdaptiveStrategy(random.uniform(0.4, 0.8)),
    "quality": lambda: QualityPremium(),
}


# --- Identity stub ---

class _FastIdentity:
    _counter = 0
    def __init__(self):
        _FastIdentity._counter += 1
        self.agent_id = f"sim:{_FastIdentity._counter}"
    @classmethod
    def generate(cls):
        return cls()

AgentIdentity = _FastIdentity


# --- Strategy Evaluator (NEW in v6) ---

class StrategyEvaluator:
    """
    Tracks per-strategy performance windows. Agents use this to decide
    whether to switch strategies.
    """
    def __init__(self, window_size=100):
        self.window_size = window_size
        # Per strategy: list of (revenue, cost) tuples
        self.history: dict[str, list[tuple[float, float]]] = defaultdict(list)

    def record(self, strategy_name: str, revenue: float, cost: float):
        self.history[strategy_name].append((revenue, cost))
        if len(self.history[strategy_name]) > self.window_size:
            self.history[strategy_name] = self.history[strategy_name][-self.window_size:]

    def avg_profit(self, strategy_name: str) -> float:
        entries = self.history[strategy_name]
        if not entries:
            return 0.0
        return sum(r - c for r, c in entries) / len(entries)

    def avg_margin(self, strategy_name: str) -> float:
        entries = self.history[strategy_name]
        if not entries:
            return 0.0
        total_rev = sum(r for r, _ in entries)
        total_cost = sum(c for _, c in entries)
        if total_rev <= 0:
            return -1.0
        return (total_rev - total_cost) / total_rev


# --- Relationship Tracker (NEW in v6 — for information asymmetry) ---

class RelationshipTracker:
    """
    Tracks past interactions between agent pairs.
    Repeat buyers have private quality information that reduces information asymmetry.
    """
    def __init__(self):
        # (buyer_id, seller_id) → list of (success: bool, price: float)
        self.interactions: dict[tuple[str, str], list[tuple[bool, float]]] = defaultdict(list)

    def record(self, buyer_id: str, seller_id: str, success: bool, price: float):
        self.interactions[(buyer_id, seller_id)].append((success, price))

    def history_count(self, buyer_id: str, seller_id: str) -> int:
        return len(self.interactions.get((buyer_id, seller_id), []))

    def private_quality_estimate(self, buyer_id: str, seller_id: str) -> Optional[float]:
        """Buyer's private estimate of seller quality based on past interactions."""
        history = self.interactions.get((buyer_id, seller_id), [])
        if not history:
            return None
        successes = sum(1 for s, _ in history if s)
        return successes / len(history)


# --- Agent ---

@dataclass
class MarketAgent:
    name: str
    identity: AgentIdentity
    capabilities: list[str]
    reliability: dict[str, float]  # TRUE quality (hidden from buyers in asymmetric mode)
    pricing_strategy: PricingStrategy
    reputation: DomainReputation = field(default_factory=lambda: DomainReputation(
        half_life_days=30, domain_correlations=DOMAIN_CORRELATIONS
    ))
    fleet_id: Optional[str] = None

    # Economics
    balance: float = 0.0
    spent: float = 0.0
    compute_costs: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_requested: int = 0
    revenue_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    costs_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    prices_charged: list[float] = field(default_factory=list)

    # Lifecycle
    joined_round: int = 0
    exited_round: Optional[int] = None
    active: bool = True
    identity_cost_per_round: float = 0.01
    rounds_active: int = 0
    is_newcomer: bool = False

    # Strategy evolution (NEW)
    strategy_history: list[tuple[int, str]] = field(default_factory=list)  # (round, strategy_name)
    eval_window: list[tuple[float, float]] = field(default_factory=list)  # (revenue, cost) recent tasks
    strategy_switches: int = 0
    last_switch_round: int = 0

    # Bundling (NEW)
    bundle_discount: float = 0.15  # 15% discount for multi-domain requests

    def price_for(self, domain, sim_time, market_prices):
        return self.pricing_strategy.price(self, domain, sim_time, market_prices)

    def bundle_price(self, domains: list[str], sim_time: float, market_prices: dict) -> float:
        """Price for a bundle of tasks across domains."""
        individual_total = sum(self.price_for(d, sim_time, market_prices) for d in domains)
        return individual_total * (1 - self.bundle_discount)

    def accept_task(self, domain):
        return random.random() < self.reliability.get(domain, 0.5)

    def net_profit(self):
        return self.balance - self.spent - self.compute_costs

    def margin(self):
        if self.balance <= 0:
            return 0.0
        return (self.balance - self.compute_costs) / self.balance

    def recent_profit_rate(self, window=50) -> float:
        """Profit rate over recent tasks."""
        recent = self.eval_window[-window:]
        if not recent:
            return 0.0
        return sum(r - c for r, c in recent) / len(recent)

    def should_switch_strategy(self, round_num: int, global_evaluator: StrategyEvaluator) -> Optional[str]:
        """
        Decide whether to switch strategy.
        Conditions:
        - At least 100 rounds since last switch (stability period)
        - Current strategy underperforming global average of another strategy
        - Some randomness (exploration vs exploitation)
        """
        if round_num - self.last_switch_round < 100:
            return None

        current = self.pricing_strategy.name
        my_profit = self.recent_profit_rate()

        # Compare against global averages of other strategies
        best_alt = None
        best_alt_profit = my_profit

        for sname in STRATEGY_FACTORIES:
            if sname == current:
                continue
            alt_profit = global_evaluator.avg_profit(sname)
            if alt_profit > best_alt_profit * 1.2:  # 20% better threshold
                best_alt = sname
                best_alt_profit = alt_profit

        if best_alt is None:
            return None

        # Stochastic switch: probability proportional to how much better the alternative is
        improvement_ratio = best_alt_profit / max(abs(my_profit), 0.001)
        switch_prob = min(0.3, 0.1 * improvement_ratio)  # cap at 30%

        if random.random() < switch_prob:
            return best_alt
        return None


def make_agent(idx, joined_round=0, strategy=None):
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
        sname = random.choice(list(STRATEGY_FACTORIES.keys()))
        strategy = STRATEGY_FACTORIES[sname]()

    return MarketAgent(
        name=f"Agent_{idx}",
        identity=AgentIdentity.generate(),
        capabilities=caps,
        reliability={d: random.uniform(*rel_range) for d in caps},
        pricing_strategy=strategy,
        joined_round=joined_round,
        is_newcomer=(joined_round > 0),
        strategy_history=[(joined_round, strategy.name)],
        last_switch_round=joined_round,
    )


# --- Market Price Tracker ---

class MarketPriceTracker:
    def __init__(self, window=200):
        self.window = window
        self.prices: dict[str, list[float]] = defaultdict(list)

    def record(self, domain, price):
        self.prices[domain].append(price)
        if len(self.prices[domain]) > self.window:
            self.prices[domain] = self.prices[domain][-self.window:]

    def get_prices(self):
        return dict(self.prices)

    def avg(self, domain):
        if domain not in self.prices or not self.prices[domain]:
            return TASK_CATALOG[domain]["base_cost"]
        return sum(self.prices[domain]) / len(self.prices[domain])


# --- Discovery ---

def discover_worker(agents, requester, domain, sim_time, round_num,
                    market_prices, price_sensitivity=1.0,
                    relationships=None, info_asymmetry=0.0):
    """
    Discovery with optional information asymmetry.
    info_asymmetry: 0.0 = full info (v5), 1.0 = buyers can't see quality at all.
    Relationships provide private quality info that partially overcomes asymmetry.
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

        # Information asymmetry: reduce visible quality signal
        # But private relationship knowledge partially restores it
        visible_quality = rep * max(conf, 0.1)

        if info_asymmetry > 0 and relationships:
            private_est = relationships.private_quality_estimate(requester.name, c.name)
            if private_est is not None:
                # Blend public reputation with private experience
                interaction_count = relationships.history_count(requester.name, c.name)
                private_weight = min(interaction_count / 10.0, 0.8)  # max 80% private info
                visible_quality = (1 - private_weight) * visible_quality * (1 - info_asymmetry) + \
                                  private_weight * private_est
            else:
                # No private info — asymmetry fully applies
                visible_quality = visible_quality * (1 - info_asymmetry) + 0.5 * info_asymmetry

        exploration_bonus = 0.3 if conf < 0.2 else 0.1 if conf < 0.4 else 0.0
        quality = visible_quality + exploration_bonus
        price_factor = max(price, 0.01) ** price_sensitivity
        value = quality / price_factor
        scored.append((c, value, price))

    temperature = 0.5
    values = [v for _, v, _ in scored]
    max_v = max(values)
    exp_values = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]

    return random.choices([c for c, _, _ in scored], weights=probs, k=1)[0]


# --- Simulation ---

def run_simulation(label, num_agents=50, num_rounds=2000,
                   entry_rate=0.05, exit_threshold=-5.0,
                   price_sensitivity=1.0,
                   strategy_mix=None,
                   enable_evolution=False,
                   info_asymmetry=0.0,
                   enable_bundling=False,
                   bundle_probability=0.15,
                   seed=42):
    random.seed(seed)
    sim_start = time.time() - 84 * 86400
    tracker = MarketPriceTracker(window=200)
    global_evaluator = StrategyEvaluator(window_size=200)
    relationships = RelationshipTracker() if info_asymmetry > 0 else None

    agent_idx = 0
    agents = []

    def pick_strategy():
        if strategy_mix:
            roll = random.random()
            cumulative = 0.0
            for name, frac in strategy_mix.items():
                cumulative += frac
                if roll < cumulative:
                    return STRATEGY_FACTORIES[name]()
            return STRATEGY_FACTORIES["reputation"]()
        sname = random.choice(list(STRATEGY_FACTORIES.keys()))
        return STRATEGY_FACTORIES[sname]()

    for i in range(num_agents):
        a = make_agent(agent_idx, strategy=pick_strategy())
        a.is_newcomer = False
        agents.append(a)
        agent_idx += 1

    snapshots = []
    strategy_population = []  # (round, {strategy: count})
    total_switches = 0
    price_history = []

    for round_num in range(1, num_rounds + 1):
        sim_time = sim_start + round_num * 3600
        active_agents = [a for a in agents if a.active]

        # Identity cost
        for a in active_agents:
            a.balance -= a.identity_cost_per_round
            a.rounds_active += 1

        # Newcomer status update
        for a in active_agents:
            if a.is_newcomer and (round_num - a.joined_round) > 200:
                a.is_newcomer = False

        # Entry
        num_entries = 0
        while random.random() < entry_rate:
            num_entries += 1
        for _ in range(num_entries):
            agents.append(make_agent(agent_idx, joined_round=round_num, strategy=pick_strategy()))
            agent_idx += 1

        # Exit
        for a in active_agents:
            if a.balance < exit_threshold and round_num - a.joined_round > 100:
                a.active = False
                a.exited_round = round_num

        active_agents = [a for a in agents if a.active]
        if len(active_agents) < 2:
            continue

        # --- Strategy Evolution (NEW) ---
        if enable_evolution and round_num % 50 == 0:
            for a in active_agents:
                new_strat = a.should_switch_strategy(round_num, global_evaluator)
                if new_strat:
                    a.pricing_strategy = STRATEGY_FACTORIES[new_strat]()
                    a.strategy_history.append((round_num, new_strat))
                    a.strategy_switches += 1
                    a.last_switch_round = round_num
                    total_switches += 1

        # Track strategy population
        if round_num % 100 == 0:
            pop = defaultdict(int)
            for a in active_agents:
                pop[a.pricing_strategy.name] += 1
            strategy_population.append({"round": round_num, "counts": dict(pop)})

        # Generate tasks
        base_tasks = random.randint(3, 6)
        demand_mult = 1.0 + 0.01 * len(active_agents)
        num_tasks = int(base_tasks * demand_mult)

        market_prices = tracker.get_prices()

        for _ in range(num_tasks):
            requester = random.choice(active_agents)

            # --- Bundling (NEW) ---
            if enable_bundling and random.random() < bundle_probability and len(requester.capabilities) >= 2:
                # Multi-domain bundle request
                bundle_domains = random.sample(ALL_DOMAINS, k=random.randint(2, 3))
                # Find a worker who can do ALL domains in the bundle
                bundle_workers = [a for a in active_agents
                                  if a.active and a.name != requester.name
                                  and all(d in a.capabilities for d in bundle_domains)]
                if bundle_workers:
                    # Pick best by average reputation across bundle domains
                    best = max(bundle_workers,
                               key=lambda w: sum(w.reputation.score(d, sim_time) for d in bundle_domains))
                    bundle_price = best.bundle_price(bundle_domains, sim_time, market_prices)
                    total_compute = sum(TASK_CATALOG[d]["compute_cost"] for d in bundle_domains)

                    # Execute all tasks in bundle
                    all_success = True
                    for d in bundle_domains:
                        success = best.accept_task(d)
                        best.reputation.record(d, success, timestamp=sim_time)
                        if not success:
                            all_success = False
                            best.tasks_failed += 1
                        else:
                            best.tasks_completed += 1

                    best.compute_costs += total_compute
                    if all_success:
                        best.balance += bundle_price
                        tracker.record(bundle_domains[0], bundle_price / len(bundle_domains))
                        best.eval_window.append((bundle_price, total_compute))
                        global_evaluator.record(best.pricing_strategy.name, bundle_price, total_compute)
                    else:
                        # Partial payment: pay for successful tasks only
                        partial = bundle_price * 0.5
                        best.balance += partial
                        best.eval_window.append((partial, total_compute))
                        global_evaluator.record(best.pricing_strategy.name, partial, total_compute)

                    if relationships:
                        relationships.record(requester.name, best.name, all_success, bundle_price)
                    continue

            # Single-domain task (standard)
            domain = random.choice(ALL_DOMAINS)
            worker = discover_worker(
                active_agents, requester, domain, sim_time, round_num,
                market_prices, price_sensitivity,
                relationships=relationships,
                info_asymmetry=info_asymmetry,
            )
            if not worker:
                continue

            price = worker.price_for(domain, sim_time, market_prices)
            compute_cost = TASK_CATALOG[domain]["compute_cost"]

            success = worker.accept_task(domain)
            worker.reputation.record(domain, success, timestamp=sim_time)
            tracker.record(domain, price)
            worker.prices_charged.append(price)
            worker.compute_costs += compute_cost

            if success:
                worker.balance += price
                worker.revenue_by_domain[domain] += price
                worker.tasks_completed += 1
            else:
                worker.tasks_failed += 1

            requester.spent += price if success else 0
            requester.tasks_requested += 1

            # Track for strategy evaluation
            worker.eval_window.append((price if success else 0, compute_cost))
            if len(worker.eval_window) > 100:
                worker.eval_window = worker.eval_window[-100:]
            global_evaluator.record(worker.pricing_strategy.name,
                                    price if success else 0, compute_cost)

            if relationships:
                relationships.record(requester.name, worker.name, success, price)

        # Price history
        if round_num % 100 == 0:
            price_history.append({
                "round": round_num,
                "prices": {d: tracker.avg(d) for d in ALL_DOMAINS},
            })

        # Snapshot
        if round_num % 200 == 0:
            earnings = sorted(a.balance for a in active_agents)
            n = len(earnings)
            s = sum(earnings)
            gini = sum((2*i - n + 1) * e for i, e in enumerate(earnings)) / (n * s) if s > 0 and n > 1 else 0

            newcomer_count = sum(1 for a in agents if a.joined_round > 0)
            newcomer_alive = sum(1 for a in agents if a.joined_round > 0 and a.active)

            snapshots.append({
                "round": round_num,
                "active": len(active_agents),
                "total": len(agents),
                "gini": gini,
                "newcomer_survival": newcomer_alive / max(newcomer_count, 1),
            })

    # --- Final analysis ---
    active_agents = [a for a in agents if a.active]
    incumbents = [a for a in active_agents if a.joined_round == 0]
    newcomer_survivors = [a for a in active_agents if a.joined_round > 0]

    strategy_perf = defaultdict(lambda: {"agents": 0, "total_balance": 0.0,
                                          "total_margin": 0.0, "tasks": 0, "survived": 0})
    for a in agents:
        sname = a.pricing_strategy.name  # final strategy
        strategy_perf[sname]["agents"] += 1
        strategy_perf[sname]["total_balance"] += a.balance
        strategy_perf[sname]["tasks"] += a.tasks_completed
        if a.active:
            strategy_perf[sname]["survived"] += 1
            strategy_perf[sname]["total_margin"] += a.margin()

    # Price convergence
    early_prices = price_history[:5] if len(price_history) >= 5 else price_history
    late_prices = price_history[-5:] if len(price_history) >= 5 else price_history
    price_convergence = {}
    for d in ALL_DOMAINS:
        early_avg = sum(p["prices"][d] for p in early_prices) / max(len(early_prices), 1)
        late_avg = sum(p["prices"][d] for p in late_prices) / max(len(late_prices), 1)
        compute = TASK_CATALOG[d]["compute_cost"]
        price_convergence[d] = {
            "early_avg": early_avg, "late_avg": late_avg,
            "compute_cost": compute,
            "late_margin": (late_avg - compute) / max(late_avg, 0.01),
            "price_change_pct": (late_avg - early_avg) / max(early_avg, 0.01) * 100,
        }

    # Relationship analysis (if applicable)
    relationship_stats = None
    if relationships:
        all_pairs = relationships.interactions
        repeat_pairs = [(k, v) for k, v in all_pairs.items() if len(v) >= 3]
        if repeat_pairs:
            repeat_success = sum(sum(1 for s, _ in v if s) / len(v) for _, v in repeat_pairs) / len(repeat_pairs)
            one_shot_pairs = [(k, v) for k, v in all_pairs.items() if len(v) == 1]
            one_shot_success = sum(sum(1 for s, _ in v if s) / len(v) for _, v in one_shot_pairs) / max(len(one_shot_pairs), 1)
            relationship_stats = {
                "total_pairs": len(all_pairs),
                "repeat_pairs": len(repeat_pairs),
                "repeat_success_rate": repeat_success,
                "one_shot_success_rate": one_shot_success,
                "avg_interactions_repeat": sum(len(v) for _, v in repeat_pairs) / max(len(repeat_pairs), 1),
            }

    # Evolution stats
    evolution_stats = None
    if enable_evolution:
        switchers = [a for a in agents if a.strategy_switches > 0]
        non_switchers = [a for a in agents if a.strategy_switches == 0]
        evolution_stats = {
            "total_switches": total_switches,
            "agents_who_switched": len(switchers),
            "avg_switches_per_switcher": sum(a.strategy_switches for a in switchers) / max(len(switchers), 1),
            "switcher_avg_balance": sum(a.balance for a in switchers) / max(len(switchers), 1),
            "non_switcher_avg_balance": sum(a.balance for a in non_switchers) / max(len(non_switchers), 1),
            "switcher_survival": sum(1 for a in switchers if a.active) / max(len(switchers), 1),
            "non_switcher_survival": sum(1 for a in non_switchers if a.active) / max(len(non_switchers), 1),
            "strategy_population": strategy_population,
        }

    return {
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
        "strategy_performance": dict(strategy_perf),
        "price_convergence": price_convergence,
        "price_history": price_history,
        "evolution_stats": evolution_stats,
        "relationship_stats": relationship_stats,
        "snapshots": snapshots,
    }


def print_result(r):
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

    print(f"\n  Strategy Performance (final strategy):")
    print(f"  {'Strategy':<15} {'Agents':>7} {'Survived':>9} {'Avg $':>10} {'Avg Margin':>11}")
    print(f"  {'─'*55}")
    for sname, data in sorted(r["strategy_performance"].items()):
        avg_bal = data["total_balance"] / max(data["agents"], 1)
        avg_margin = data["total_margin"] / max(data["survived"], 1)
        surv_pct = data["survived"] / max(data["agents"], 1) * 100
        print(f"  {sname:<15} {data['agents']:>7} {data['survived']:>5} ({surv_pct:>4.0f}%) "
              f"${avg_bal:>8.2f} {avg_margin:>10.1%}")

    print(f"\n  Price Convergence:")
    print(f"  {'Domain':<20} {'Early':>8} {'Late':>8} {'Compute':>8} {'Margin':>8} {'Δ%':>8}")
    print(f"  {'─'*55}")
    for d, pc in sorted(r["price_convergence"].items()):
        print(f"  {d:<20} ${pc['early_avg']:>6.3f} ${pc['late_avg']:>6.3f} "
              f"${pc['compute_cost']:>6.3f} {pc['late_margin']:>7.1%} {pc['price_change_pct']:>+7.1f}%")

    if r.get("evolution_stats"):
        es = r["evolution_stats"]
        print(f"\n  Strategy Evolution:")
        print(f"    Total switches: {es['total_switches']}")
        print(f"    Agents who switched: {es['agents_who_switched']} "
              f"(avg {es['avg_switches_per_switcher']:.1f} switches each)")
        print(f"    Switcher avg balance: ${es['switcher_avg_balance']:.2f} "
              f"(survival: {es['switcher_survival']:.0%})")
        print(f"    Non-switcher avg balance: ${es['non_switcher_avg_balance']:.2f} "
              f"(survival: {es['non_switcher_survival']:.0%})")

        # Strategy population over time
        print(f"\n  Strategy Population Over Time:")
        print(f"  {'Round':>6}  ", end="")
        all_strats = sorted(set(s for sp in es["strategy_population"] for s in sp["counts"]))
        for s in all_strats:
            print(f"{s:>12}", end="")
        print()
        print(f"  {'─'*(8 + 12*len(all_strats))}")
        for sp in es["strategy_population"][::2]:  # every other snapshot
            print(f"  {sp['round']:>6}  ", end="")
            for s in all_strats:
                count = sp["counts"].get(s, 0)
                print(f"{count:>12}", end="")
            print()

    if r.get("relationship_stats"):
        rs = r["relationship_stats"]
        print(f"\n  Relationship Stats:")
        print(f"    Total buyer-seller pairs: {rs['total_pairs']}")
        print(f"    Repeat pairs (3+ interactions): {rs['repeat_pairs']}")
        print(f"    Avg interactions per repeat pair: {rs['avg_interactions_repeat']:.1f}")
        print(f"    Repeat pair success rate: {rs['repeat_success_rate']:.1%}")
        print(f"    One-shot success rate: {rs['one_shot_success_rate']:.1%}")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def experiment_evolution():
    """Do agents converge on a single strategy when they can switch?"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 1: Strategy Evolution                           ║")
    print("║  Agents can switch strategies based on performance          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Control: no evolution
    print("  [Control] No evolution...")
    control = run_simulation(
        "CONTROL: No Evolution (equal mix)",
        num_agents=60, num_rounds=3000, entry_rate=0.05,
        price_sensitivity=1.0,
        strategy_mix={"reputation": 0.25, "undercut": 0.25, "adaptive": 0.25, "quality": 0.25},
        enable_evolution=False, seed=42,
    )
    print_result(control)

    # Evolution enabled
    print("\n  [Test] Evolution enabled...")
    evolved = run_simulation(
        "EVOLUTION: Agents switch strategies",
        num_agents=60, num_rounds=3000, entry_rate=0.05,
        price_sensitivity=1.0,
        strategy_mix={"reputation": 0.25, "undercut": 0.25, "adaptive": 0.25, "quality": 0.25},
        enable_evolution=True, seed=42,
    )
    print_result(evolved)

    return control, evolved


def experiment_lemons():
    """Information asymmetry: buyers can't observe quality. Does the market collapse?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 2: Lemons Problem (Information Asymmetry)       ║")
    print("║  Buyers can't see true quality before purchasing            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = {}
    for asym in [0.0, 0.3, 0.5, 0.7, 0.9]:
        print(f"  Running info_asymmetry={asym}...")
        r = run_simulation(
            f"Info Asymmetry = {asym}",
            num_agents=50, num_rounds=2000, entry_rate=0.05,
            price_sensitivity=1.0,
            info_asymmetry=asym, seed=42,
        )
        results[asym] = r

    print(f"\n{'='*70}")
    print(f"  LEMONS PROBLEM: Information Asymmetry Sweep")
    print(f"{'='*70}")
    print(f"  {'Asymmetry':>10} {'Active':>7} {'Gini':>7} {'Inc.Avg$':>10} {'New.Avg$':>10} {'Gap':>6}")
    print(f"  {'─'*55}")
    for asym, r in sorted(results.items()):
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        print(f"  {asym:>10.1f} {r['active']:>7} {r['final_gini']:>7.3f} "
              f"${r['incumbent_avg_balance']:>8.2f} ${r['newcomer_avg_balance']:>8.2f} {gap:>5.1f}x")

    # Show relationship stats for high asymmetry
    high_asym = results[0.7]
    if high_asym.get("relationship_stats"):
        rs = high_asym["relationship_stats"]
        print(f"\n  Relationships at 0.7 asymmetry:")
        print(f"    Repeat pairs: {rs['repeat_pairs']} of {rs['total_pairs']} total")
        print(f"    Repeat success: {rs['repeat_success_rate']:.1%} vs one-shot: {rs['one_shot_success_rate']:.1%}")

    return results


def experiment_lemons_with_evolution():
    """Combine: lemons problem + strategy evolution. Does evolution help?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 3: Lemons + Evolution                           ║")
    print("║  Can strategy switching help in an opaque market?           ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # High asymmetry, no evolution
    no_evo = run_simulation(
        "HIGH ASYMMETRY, NO EVOLUTION",
        num_agents=50, num_rounds=2000, entry_rate=0.05,
        price_sensitivity=1.0, info_asymmetry=0.7,
        enable_evolution=False, seed=42,
    )
    print_result(no_evo)

    # High asymmetry, with evolution
    with_evo = run_simulation(
        "HIGH ASYMMETRY + EVOLUTION",
        num_agents=50, num_rounds=2000, entry_rate=0.05,
        price_sensitivity=1.0, info_asymmetry=0.7,
        enable_evolution=True, seed=42,
    )
    print_result(with_evo)

    return no_evo, with_evo


def experiment_bundling():
    """Does service bundling create defensible margins?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 4: Service Bundling                             ║")
    print("║  Multi-domain packages at discount vs single-task commodity ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # No bundling (baseline)
    no_bundle = run_simulation(
        "NO BUNDLING (baseline)",
        num_agents=50, num_rounds=2000, entry_rate=0.05,
        price_sensitivity=1.0, enable_bundling=False, seed=42,
    )
    print_result(no_bundle)

    # With bundling
    with_bundle = run_simulation(
        "WITH BUNDLING (15% of tasks)",
        num_agents=50, num_rounds=2000, entry_rate=0.05,
        price_sensitivity=1.0, enable_bundling=True,
        bundle_probability=0.15, seed=42,
    )
    print_result(with_bundle)

    # High bundling
    high_bundle = run_simulation(
        "HIGH BUNDLING (40% of tasks)",
        num_agents=50, num_rounds=2000, entry_rate=0.05,
        price_sensitivity=1.0, enable_bundling=True,
        bundle_probability=0.40, seed=42,
    )
    print_result(high_bundle)

    return no_bundle, with_bundle, high_bundle


def experiment_evolution_convergence():
    """Long run with evolution — where do strategies settle?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 5: Long-Run Convergence (3000 rounds)           ║")
    print("║  Do strategies reach Nash equilibrium?                      ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    r = run_simulation(
        "LONG RUN EVOLUTION (3000 rounds)",
        num_agents=60, num_rounds=3000, entry_rate=0.03,
        price_sensitivity=1.0,
        strategy_mix={"reputation": 0.25, "undercut": 0.25, "adaptive": 0.25, "quality": 0.25},
        enable_evolution=True, seed=42,
    )
    print_result(r)
    return r


# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v6                  ║")
    print("║  Strategy Evolution, Lemons Problem & Bundling              ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    exp1_ctrl, exp1_evo = experiment_evolution()
    exp2 = experiment_lemons()
    exp3_no, exp3_yes = experiment_lemons_with_evolution()
    exp4_no, exp4_low, exp4_high = experiment_bundling()
    exp5 = experiment_evolution_convergence()

    print("\n\n" + "═" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
