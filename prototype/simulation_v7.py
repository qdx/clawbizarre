"""
ClawBizarre Economy Simulation v7 — Coalition Formation & Multi-Marketplace Competition

New features over v6:
- Agents can form coalitions (guilds) that share discovery, pool reputation, and collectively price
- Coalition dynamics: formation, membership benefits, free-riding, dissolution
- Multi-marketplace competition: agents choose which marketplace to participate in
- Platform fees and marketplace differentiation

Key questions:
1. Do coalitions form naturally? What's the optimal size?
2. Do coalitions become cartels (price-fixing) or cooperatives (resource-sharing)?
3. When marketplaces compete, what differentiates winners?
4. How do platform fees affect agent welfare vs marketplace sustainability?
"""

import sys
import random
import math
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

sys.stdout.reconfigure(line_buffering=True)

from reputation import DecayingReputation, DomainReputation, MerkleTree, sha256

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


# --- Coalition (Guild) ---

@dataclass
class Coalition:
    """A group of agents that cooperate for mutual benefit."""
    coalition_id: str
    founder: str  # agent name
    members: list[str] = field(default_factory=list)
    formed_round: int = 0
    dissolved_round: Optional[int] = None
    active: bool = True

    # Economics
    treasury: float = 0.0
    fee_rate: float = 0.10  # 10% of member earnings go to treasury
    treasury_history: list[float] = field(default_factory=list)

    # Reputation pooling
    shared_reputation: DomainReputation = field(default_factory=lambda: DomainReputation(
        half_life_days=30, domain_correlations=DOMAIN_CORRELATIONS
    ))

    # Discovery boost: coalition members get priority in referrals
    referral_bonus: float = 0.3  # bonus to discovery score for coalition mates

    # Pricing coordination
    min_price_floor: Optional[dict] = None  # Cartel-like minimum prices per domain
    price_coordination: str = "none"  # "none", "floor", "collective"

    # Metrics
    total_tasks: int = 0
    total_revenue: float = 0.0
    member_history: list[tuple[int, str, str]] = field(default_factory=list)  # (round, agent, "join"|"leave")

    @property
    def size(self):
        return len(self.members)

    def add_member(self, agent_name: str, round_num: int):
        if agent_name not in self.members:
            self.members.append(agent_name)
            self.member_history.append((round_num, agent_name, "join"))

    def remove_member(self, agent_name: str, round_num: int):
        if agent_name in self.members:
            self.members.remove(agent_name)
            self.member_history.append((round_num, agent_name, "leave"))
            if not self.members:
                self.active = False
                self.dissolved_round = round_num

    def collect_fee(self, revenue: float) -> float:
        """Collect fee from member earnings, return net revenue to member."""
        fee = revenue * self.fee_rate
        self.treasury += fee
        self.total_revenue += revenue
        return revenue - fee

    def distribute_treasury(self) -> float:
        """Distribute treasury equally among members (periodic)."""
        if not self.members:
            return 0.0
        per_member = self.treasury / len(self.members)
        self.treasury_history.append(self.treasury)
        self.treasury = 0.0
        return per_member


# --- Marketplace ---

@dataclass
class Marketplace:
    """A platform where agents find work. Competes with other marketplaces."""
    marketplace_id: str
    name: str
    platform_fee: float = 0.05  # 5% platform cut
    participants: list[str] = field(default_factory=list)  # agent names

    # Differentiation
    verification_level: int = 0  # 0=none, 1=basic, 2=full (higher = more trust, higher cost)
    discovery_quality: float = 1.0  # multiplier on discovery effectiveness
    newcomer_protection: float = 0.0  # fraction of tasks reserved for newcomers

    # Metrics
    total_tasks: int = 0
    total_volume: float = 0.0
    task_history: list[tuple[int, float]] = field(default_factory=list)  # (round, volume)

    # Reputation
    marketplace_reputation: float = 0.5  # grows with successful transactions

    def join(self, agent_name: str):
        if agent_name not in self.participants:
            self.participants.append(agent_name)

    def leave(self, agent_name: str):
        if agent_name in self.participants:
            self.participants.remove(agent_name)

    def collect_fee(self, price: float) -> tuple[float, float]:
        """Returns (seller_receives, platform_takes)."""
        platform_cut = price * self.platform_fee
        # Verification costs reduce platform profit but build trust
        verification_cost = platform_cut * 0.3 * self.verification_level
        return price - platform_cut, platform_cut - verification_cost

    def update_reputation(self, success: bool):
        """Bayesian update of marketplace reputation."""
        alpha = 0.01
        if success:
            self.marketplace_reputation += alpha * (1 - self.marketplace_reputation)
        else:
            self.marketplace_reputation -= alpha * self.marketplace_reputation


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


# --- Pricing Strategies ---

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

STRATEGY_FACTORIES = {
    "reputation": lambda: ReputationPremium(),
    "undercut": lambda: UndercutStrategy(random.uniform(0.10, 0.25)),
}


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

    # Economics
    balance: float = 0.0
    compute_costs: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    revenue_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    prices_charged: list[float] = field(default_factory=list)

    # Lifecycle
    joined_round: int = 0
    exited_round: Optional[int] = None
    active: bool = True
    identity_cost_per_round: float = 0.01
    rounds_active: int = 0
    is_newcomer: bool = False

    # Coalition membership
    coalition_id: Optional[str] = None
    coalition_benefit_received: float = 0.0  # total treasury distributions received
    solo_earnings_before_coalition: float = 0.0  # earnings before joining, for comparison

    # Marketplace membership
    marketplace_ids: list[str] = field(default_factory=list)  # can be on multiple marketplaces

    # Coalition preference
    coalition_willingness: float = 0.5  # 0=solo-only, 1=eager to join

    def price_for(self, domain, sim_time, market_prices, coalition=None):
        base_price = self.pricing_strategy.price(self, domain, sim_time, market_prices)
        # If in a coalition with price floor, enforce it
        if coalition and coalition.min_price_floor and domain in coalition.min_price_floor:
            return max(base_price, coalition.min_price_floor[domain])
        return base_price

    def accept_task(self, domain):
        return random.random() < self.reliability.get(domain, 0.5)

    def net_profit(self):
        return self.balance - self.compute_costs

    def earnings_rate(self, window_rounds=200):
        """Approximate earnings per round."""
        if self.rounds_active < 10:
            return 0.0
        return self.balance / self.rounds_active


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

def discover_worker(agents, requester, domain, sim_time, market_prices,
                    price_sensitivity=1.0, coalitions=None, marketplace=None):
    """Discovery with coalition referral bonuses and marketplace filtering."""
    candidates = [a for a in agents
                  if a.active and a.name != requester.name
                  and domain in a.capabilities]

    # Filter by marketplace if specified
    if marketplace:
        mp_candidates = [a for a in candidates if marketplace.marketplace_id in a.marketplace_ids]
        # Newcomer protection: reserve some slots
        if marketplace.newcomer_protection > 0 and random.random() < marketplace.newcomer_protection:
            newcomers = [a for a in mp_candidates if a.is_newcomer]
            if newcomers:
                mp_candidates = newcomers
        candidates = mp_candidates if mp_candidates else candidates

    if not candidates:
        return None

    # Find requester's coalition
    requester_coalition = None
    if coalitions and requester.coalition_id:
        requester_coalition = coalitions.get(requester.coalition_id)

    scored = []
    for c in candidates:
        rep = c.reputation.score(domain, sim_time)
        conf = c.reputation._get_domain(domain).confidence(sim_time) if domain in c.reputation._domains else 0.0
        price = c.price_for(domain, sim_time, market_prices)

        quality = rep * max(conf, 0.1)

        # Coalition referral bonus: prefer coalition mates
        if requester_coalition and c.coalition_id == requester.coalition_id:
            quality += requester_coalition.referral_bonus

        # Coalition reputation pooling: use shared reputation if better
        if c.coalition_id and coalitions:
            coal = coalitions.get(c.coalition_id)
            if coal:
                shared_rep = coal.shared_reputation.score(domain, sim_time)
                shared_conf = coal.shared_reputation._get_domain(domain).confidence(sim_time) \
                    if domain in coal.shared_reputation._domains else 0.0
                shared_quality = shared_rep * max(shared_conf, 0.1) * 0.5  # 50% weight on shared
                quality = max(quality, quality * 0.7 + shared_quality * 0.3)

        exploration_bonus = 0.3 if conf < 0.2 else 0.1 if conf < 0.4 else 0.0
        quality += exploration_bonus
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


# --- Coalition Formation Logic ---

def try_form_coalition(agents, round_num, coalitions, max_coalitions=20):
    """
    Agents with high earnings rate and no coalition may recruit nearby agents.
    Formation requires: founder with proven track record + willing members.
    """
    active_solo = [a for a in agents if a.active and a.coalition_id is None
                   and a.rounds_active > 200 and a.balance > 5.0]

    if not active_solo or len(coalitions) >= max_coalitions:
        return None

    # Best-performing solo agent considers founding
    founder = max(active_solo, key=lambda a: a.earnings_rate())
    if founder.earnings_rate() < 0.02:  # Not earning enough to attract others
        return None
    if random.random() > 0.05:  # Only 5% chance per round to try
        return None

    # Recruit agents with complementary capabilities
    potential = [a for a in active_solo if a.name != founder.name
                 and a.coalition_willingness > 0.3
                 and any(d not in founder.capabilities for d in a.capabilities)]

    if not potential:
        return None

    # Pick 2-4 members (from v4: optimal coalition ~2-5)
    recruit_count = min(random.randint(2, 4), len(potential))
    # Prefer agents with complementary skills and decent reputation
    potential.sort(key=lambda a: (
        len(set(a.capabilities) - set(founder.capabilities)),  # complementarity
        a.earnings_rate()
    ), reverse=True)
    recruits = potential[:recruit_count]

    # Probabilistic acceptance
    accepted = [r for r in recruits if random.random() < r.coalition_willingness]
    if not accepted:
        return None

    # Form coalition
    cid = f"coalition_{round_num}_{founder.name}"
    coal = Coalition(
        coalition_id=cid,
        founder=founder.name,
        formed_round=round_num,
    )

    # Set price coordination based on founder's strategy
    if founder.pricing_strategy.name == "reputation":
        coal.price_coordination = "floor"
        # Set floor at 80% of current average prices
        coal.min_price_floor = {d: TASK_CATALOG[d]["base_cost"] * 0.8 for d in ALL_DOMAINS}
    else:
        coal.price_coordination = "none"

    # Add members
    founder.coalition_id = cid
    founder.solo_earnings_before_coalition = founder.balance
    coal.add_member(founder.name, round_num)

    for r in accepted:
        r.coalition_id = cid
        r.solo_earnings_before_coalition = r.balance
        coal.add_member(r.name, round_num)

    return coal


def evaluate_coalition_membership(agent, coalition, round_num):
    """
    Agent evaluates whether to stay in coalition.
    Leave if: earnings dropped since joining, or coalition is too large (overhead).
    """
    if round_num - coalition.formed_round < 200:
        return True  # Give it time

    # Compare current earnings rate to pre-coalition rate
    pre_rate = agent.solo_earnings_before_coalition / max(agent.joined_round + 1, 1)
    # Rounds since joining coalition
    rounds_in = round_num - max(h[0] for h in coalition.member_history if h[1] == agent.name and h[2] == "join")
    if rounds_in < 100:
        return True

    current_rate = (agent.balance - agent.solo_earnings_before_coalition) / max(rounds_in, 1)

    # Leave if earning less than before (with some tolerance)
    if current_rate < pre_rate * 0.7:
        return random.random() > 0.3  # 30% chance to leave when underperforming

    # Leave if coalition is too big (coordination costs)
    if coalition.size > 6:
        overhead = (coalition.size - 3) * 0.05  # 5% overhead per member above 3
        if random.random() < overhead:
            return False

    return True


# --- Simulation ---

def run_simulation(label, num_agents=50, num_rounds=2000,
                   entry_rate=0.05, exit_threshold=-5.0,
                   price_sensitivity=1.0,
                   enable_coalitions=False,
                   num_marketplaces=1,
                   marketplace_configs=None,
                   seed=42):
    random.seed(seed)
    sim_start = time.time() - 84 * 86400
    tracker = MarketPriceTracker(window=200)

    # Create marketplaces
    marketplaces: dict[str, Marketplace] = {}
    if marketplace_configs:
        for mc in marketplace_configs:
            mp = Marketplace(**mc)
            marketplaces[mp.marketplace_id] = mp
    else:
        for i in range(num_marketplaces):
            mp_id = f"marketplace_{i}"
            mp = Marketplace(
                marketplace_id=mp_id,
                name=f"Market {i}",
                platform_fee=0.05 + 0.02 * i,  # Differentiated fees
                verification_level=i % 3,
                discovery_quality=1.0 - 0.1 * i,
                newcomer_protection=0.1 * (i % 2),
            )
            marketplaces[mp_id] = mp

    coalitions: dict[str, Coalition] = {}

    agent_idx = 0
    agents: list[MarketAgent] = []

    def make_agent(idx, joined_round=0):
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

        # All agents use reputation strategy (ESS from v6)
        strategy = ReputationPremium()

        a = MarketAgent(
            name=f"Agent_{idx}",
            identity=AgentIdentity.generate(),
            capabilities=caps,
            reliability={d: random.uniform(*rel_range) for d in caps},
            pricing_strategy=strategy,
            joined_round=joined_round,
            is_newcomer=(joined_round > 0),
            coalition_willingness=random.uniform(0.1, 0.9),
        )

        # Assign to marketplace(s)
        if marketplaces:
            if random.random() < 0.3:  # 30% on multiple marketplaces
                a.marketplace_ids = random.sample(list(marketplaces.keys()),
                                                  k=min(2, len(marketplaces)))
            else:
                a.marketplace_ids = [random.choice(list(marketplaces.keys()))]
            for mp_id in a.marketplace_ids:
                marketplaces[mp_id].join(a.name)

        return a

    for i in range(num_agents):
        agents.append(make_agent(agent_idx))
        agent_idx += 1

    snapshots = []
    coalition_snapshots = []
    marketplace_snapshots = []

    for round_num in range(1, num_rounds + 1):
        sim_time = sim_start + round_num * 3600
        active_agents = [a for a in agents if a.active]

        # Identity cost
        for a in active_agents:
            a.balance -= a.identity_cost_per_round
            a.rounds_active += 1

        # Newcomer status
        for a in active_agents:
            if a.is_newcomer and (round_num - a.joined_round) > 200:
                a.is_newcomer = False

        # Entry
        while random.random() < entry_rate:
            new_agent = make_agent(agent_idx, joined_round=round_num)
            agents.append(new_agent)
            agent_idx += 1

        # Exit
        for a in active_agents:
            if a.balance < exit_threshold and round_num - a.joined_round > 100:
                a.active = False
                a.exited_round = round_num
                # Leave coalition
                if a.coalition_id and a.coalition_id in coalitions:
                    coalitions[a.coalition_id].remove_member(a.name, round_num)
                # Leave marketplaces
                for mp_id in a.marketplace_ids:
                    if mp_id in marketplaces:
                        marketplaces[mp_id].leave(a.name)

        active_agents = [a for a in agents if a.active]
        if len(active_agents) < 2:
            continue

        # --- Coalition Formation (every 50 rounds) ---
        if enable_coalitions and round_num % 50 == 0:
            new_coal = try_form_coalition(active_agents, round_num, coalitions)
            if new_coal:
                coalitions[new_coal.coalition_id] = new_coal

            # Evaluate membership
            for a in active_agents:
                if a.coalition_id and a.coalition_id in coalitions:
                    coal = coalitions[a.coalition_id]
                    if not evaluate_coalition_membership(a, coal, round_num):
                        coal.remove_member(a.name, round_num)
                        a.coalition_id = None

            # Distribute treasury (every 100 rounds)
            if round_num % 100 == 0:
                for coal in coalitions.values():
                    if coal.active and coal.treasury > 0:
                        per_member = coal.distribute_treasury()
                        for a in active_agents:
                            if a.coalition_id == coal.coalition_id:
                                a.balance += per_member
                                a.coalition_benefit_received += per_member

        # Generate tasks
        base_tasks = random.randint(3, 6)
        demand_mult = 1.0 + 0.01 * len(active_agents)
        num_tasks = int(base_tasks * demand_mult)

        market_prices = tracker.get_prices()

        for _ in range(num_tasks):
            requester = random.choice(active_agents)
            domain = random.choice(ALL_DOMAINS)

            # Pick marketplace (if multiple)
            marketplace = None
            if marketplaces and requester.marketplace_ids:
                mp_id = random.choice(requester.marketplace_ids)
                marketplace = marketplaces[mp_id]

            worker = discover_worker(
                active_agents, requester, domain, sim_time, market_prices,
                price_sensitivity,
                coalitions=coalitions if enable_coalitions else None,
                marketplace=marketplace,
            )
            if not worker:
                continue

            # Get worker's coalition
            worker_coalition = None
            if enable_coalitions and worker.coalition_id:
                worker_coalition = coalitions.get(worker.coalition_id)

            price = worker.price_for(domain, sim_time, market_prices, worker_coalition)
            compute_cost = TASK_CATALOG[domain]["compute_cost"]
            success = worker.accept_task(domain)
            worker.reputation.record(domain, success, timestamp=sim_time)

            # Update coalition shared reputation
            if worker_coalition:
                worker_coalition.shared_reputation.record(domain, success, timestamp=sim_time)
                worker_coalition.total_tasks += 1

            tracker.record(domain, price)
            worker.prices_charged.append(price)
            worker.compute_costs += compute_cost

            if success:
                # Apply marketplace fee
                seller_receives = price
                if marketplace:
                    seller_receives, platform_cut = marketplace.collect_fee(price)
                    marketplace.total_tasks += 1
                    marketplace.total_volume += price
                    marketplace.update_reputation(True)

                # Apply coalition fee
                if worker_coalition:
                    seller_receives = worker_coalition.collect_fee(seller_receives)

                worker.balance += seller_receives
                worker.revenue_by_domain[domain] += seller_receives
                worker.tasks_completed += 1
            else:
                worker.tasks_failed += 1
                if marketplace:
                    marketplace.update_reputation(False)

        # Price history snapshot
        if round_num % 100 == 0:
            pass  # tracked via tracker

        # Snapshot
        if round_num % 200 == 0:
            active_a = [a for a in agents if a.active]
            earnings = sorted(a.balance for a in active_a)
            n = len(earnings)
            s = sum(earnings)
            gini = sum((2*i - n + 1) * e for i, e in enumerate(earnings)) / (n * s) if s > 0 and n > 1 else 0

            nc_total = sum(1 for a in agents if a.joined_round > 0)
            nc_alive = sum(1 for a in agents if a.joined_round > 0 and a.active)

            snapshots.append({
                "round": round_num,
                "active": len(active_a),
                "gini": gini,
                "newcomer_survival": nc_alive / max(nc_total, 1),
            })

            # Coalition snapshot
            active_coalitions = [c for c in coalitions.values() if c.active]
            coalition_snapshots.append({
                "round": round_num,
                "active_coalitions": len(active_coalitions),
                "avg_size": sum(c.size for c in active_coalitions) / max(len(active_coalitions), 1),
                "total_members": sum(c.size for c in active_coalitions),
                "avg_treasury": sum(c.treasury for c in active_coalitions) / max(len(active_coalitions), 1),
            })

            # Marketplace snapshot
            if marketplaces:
                marketplace_snapshots.append({
                    "round": round_num,
                    "marketplaces": {
                        mp.marketplace_id: {
                            "participants": len(mp.participants),
                            "tasks": mp.total_tasks,
                            "volume": mp.total_volume,
                            "reputation": mp.marketplace_reputation,
                        }
                        for mp in marketplaces.values()
                    }
                })

    # --- Final analysis ---
    active_agents = [a for a in agents if a.active]
    incumbents = [a for a in active_agents if a.joined_round == 0]
    newcomer_survivors = [a for a in active_agents if a.joined_round > 0]
    coalition_members = [a for a in active_agents if a.coalition_id is not None]
    solo_agents = [a for a in active_agents if a.coalition_id is None]

    # Price convergence
    price_convergence = {}
    for d in ALL_DOMAINS:
        prices = tracker.prices.get(d, [])
        if len(prices) >= 20:
            early = prices[:len(prices)//4]
            late = prices[-len(prices)//4:]
            early_avg = sum(early) / len(early)
            late_avg = sum(late) / len(late)
        else:
            early_avg = late_avg = TASK_CATALOG[d]["base_cost"]
        compute = TASK_CATALOG[d]["compute_cost"]
        price_convergence[d] = {
            "early_avg": early_avg, "late_avg": late_avg,
            "compute_cost": compute,
            "late_margin": (late_avg - compute) / max(late_avg, 0.01),
            "price_change_pct": (late_avg - early_avg) / max(early_avg, 0.01) * 100,
        }

    return {
        "label": label,
        "total_agents": len(agents),
        "active": len(active_agents),
        "exited": len(agents) - len(active_agents),
        "incumbents_alive": len(incumbents),
        "newcomer_survivors": len(newcomer_survivors),
        "newcomer_total": len(agents) - num_agents,
        "incumbent_avg_balance": sum(a.balance for a in incumbents) / max(len(incumbents), 1),
        "newcomer_avg_balance": sum(a.balance for a in newcomer_survivors) / max(len(newcomer_survivors), 1),
        "final_gini": snapshots[-1]["gini"] if snapshots else 0,
        "coalition_members": len(coalition_members),
        "solo_agents": len(solo_agents),
        "coalition_avg_balance": sum(a.balance for a in coalition_members) / max(len(coalition_members), 1),
        "solo_avg_balance": sum(a.balance for a in solo_agents) / max(len(solo_agents), 1),
        "active_coalitions": sum(1 for c in coalitions.values() if c.active),
        "total_coalitions_formed": len(coalitions),
        "coalitions": coalitions,
        "price_convergence": price_convergence,
        "snapshots": snapshots,
        "coalition_snapshots": coalition_snapshots,
        "marketplace_snapshots": marketplace_snapshots,
        "marketplaces": marketplaces,
    }


def print_result(r):
    print(f"\n{'='*70}")
    print(f"  {r['label']}")
    print(f"{'='*70}")
    print(f"  Agents: {r['total_agents']} total, {r['active']} active, {r['exited']} exited")
    print(f"  Incumbents alive: {r['incumbents_alive']}")
    nc_total = max(r['newcomer_total'], 1)
    print(f"  Newcomers: {r['newcomer_survivors']}/{r['newcomer_total']} survived "
          f"({r['newcomer_survivors']/nc_total*100:.0f}%)")
    print(f"  Incumbent avg: ${r['incumbent_avg_balance']:.2f}")
    print(f"  Newcomer avg:  ${r['newcomer_avg_balance']:.2f}")
    gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
    print(f"  Earnings gap: {gap:.1f}x")
    print(f"  Final Gini: {r['final_gini']:.3f}")

    if r['active_coalitions'] > 0 or r['total_coalitions_formed'] > 0:
        print(f"\n  Coalition Stats:")
        print(f"    Formed: {r['total_coalitions_formed']}, Active: {r['active_coalitions']}")
        print(f"    Coalition members: {r['coalition_members']}, Solo: {r['solo_agents']}")
        if r['coalition_members'] > 0:
            print(f"    Coalition avg balance: ${r['coalition_avg_balance']:.2f}")
        if r['solo_agents'] > 0:
            print(f"    Solo avg balance:      ${r['solo_avg_balance']:.2f}")

        # Detail per coalition
        active_coals = [c for c in r['coalitions'].values() if c.active]
        if active_coals:
            print(f"\n    Active Coalitions:")
            print(f"    {'ID':<35} {'Size':>5} {'Tasks':>6} {'Revenue':>10} {'Treasury':>10} {'Pricing':>10}")
            print(f"    {'─'*80}")
            for c in sorted(active_coals, key=lambda x: x.total_revenue, reverse=True)[:10]:
                print(f"    {c.coalition_id[:35]:<35} {c.size:>5} {c.total_tasks:>6} "
                      f"${c.total_revenue:>8.2f} ${c.treasury:>8.2f} {c.price_coordination:>10}")

    if r.get('coalition_snapshots'):
        cs = r['coalition_snapshots']
        if cs and any(s['active_coalitions'] > 0 for s in cs):
            print(f"\n    Coalition Growth Over Time:")
            print(f"    {'Round':>6} {'Active':>7} {'Avg Size':>9} {'Members':>8}")
            print(f"    {'─'*35}")
            for s in cs:
                if s['active_coalitions'] > 0 or s['round'] % 400 == 0:
                    print(f"    {s['round']:>6} {s['active_coalitions']:>7} "
                          f"{s['avg_size']:>9.1f} {s['total_members']:>8}")

    if r.get('marketplaces') and len(r['marketplaces']) > 1:
        print(f"\n  Marketplace Competition:")
        print(f"  {'ID':<20} {'Fee':>5} {'Verify':>7} {'Agents':>7} {'Tasks':>7} {'Volume':>10} {'Rep':>6}")
        print(f"  {'─'*65}")
        for mp in r['marketplaces'].values():
            print(f"  {mp.name:<20} {mp.platform_fee:>4.0%} {mp.verification_level:>7} "
                  f"{len(mp.participants):>7} {mp.total_tasks:>7} ${mp.total_volume:>8.2f} "
                  f"{mp.marketplace_reputation:>5.2f}")

    print(f"\n  Price Convergence:")
    print(f"  {'Domain':<20} {'Early':>8} {'Late':>8} {'Compute':>8} {'Margin':>8} {'Δ%':>8}")
    print(f"  {'─'*55}")
    for d, pc in sorted(r["price_convergence"].items()):
        print(f"  {d:<20} ${pc['early_avg']:>6.3f} ${pc['late_avg']:>6.3f} "
              f"${pc['compute_cost']:>6.3f} {pc['late_margin']:>7.1%} {pc['price_change_pct']:>+7.1f}%")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def experiment_coalitions():
    """Do coalitions form naturally and benefit members?"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 1: Coalition Formation                          ║")
    print("║  Can agents self-organize into beneficial groups?           ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    print("  [Control] No coalitions...")
    control = run_simulation(
        "CONTROL: No Coalitions",
        num_agents=40, num_rounds=1500, entry_rate=0.04,
        enable_coalitions=False, seed=42,
    )
    print_result(control)

    print("\n  [Test] Coalitions enabled...")
    with_coal = run_simulation(
        "COALITIONS ENABLED",
        num_agents=40, num_rounds=1500, entry_rate=0.04,
        enable_coalitions=True, seed=42,
    )
    print_result(with_coal)

    return control, with_coal


def experiment_coalition_sizes():
    """How does coalition willingness affect formation and outcomes?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 2: Coalition Willingness Sweep                  ║")
    print("║  What happens when agents are more/less willing to group?   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = {}
    for willingness_bias in [0.1, 0.3, 0.5, 0.7, 0.9]:
        print(f"  Running willingness_bias={willingness_bias}...")

        # Override coalition_willingness via seed manipulation
        random.seed(42)
        r = run_simulation(
            f"Willingness bias = {willingness_bias}",
            num_agents=60, num_rounds=2000, entry_rate=0.05,
            enable_coalitions=True, seed=42 + int(willingness_bias * 100),
        )
        results[willingness_bias] = r

    print(f"\n{'='*70}")
    print(f"  COALITION WILLINGNESS SWEEP")
    print(f"{'='*70}")
    print(f"  {'Bias':>6} {'Active':>7} {'Gini':>7} {'Coals':>6} {'Members':>8} "
          f"{'Coal $':>8} {'Solo $':>8} {'Gap':>6}")
    print(f"  {'─'*65}")
    for bias, r in sorted(results.items()):
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        print(f"  {bias:>6.1f} {r['active']:>7} {r['final_gini']:>7.3f} "
              f"{r['active_coalitions']:>6} {r['coalition_members']:>8} "
              f"${r['coalition_avg_balance']:>6.2f} ${r['solo_avg_balance']:>6.2f} {gap:>5.1f}x")

    return results


def experiment_marketplace_competition():
    """Two marketplaces with different strategies compete for agents."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 3: Marketplace Competition                      ║")
    print("║  Low-fee vs high-trust marketplace                          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Single marketplace (control)
    print("  [Control] Single marketplace...")
    control = run_simulation(
        "SINGLE MARKETPLACE",
        num_agents=40, num_rounds=1500, entry_rate=0.04,
        num_marketplaces=1,
        marketplace_configs=[{
            "marketplace_id": "single",
            "name": "TheMarket",
            "platform_fee": 0.05,
            "verification_level": 1,
        }],
        seed=42,
    )
    print_result(control)

    # Two competing marketplaces
    print("\n  [Test] Two competing marketplaces...")
    competition = run_simulation(
        "TWO MARKETPLACES: Low-fee vs High-trust",
        num_agents=40, num_rounds=1500, entry_rate=0.04,
        marketplace_configs=[
            {
                "marketplace_id": "cheap",
                "name": "CheapMarket",
                "platform_fee": 0.02,
                "verification_level": 0,
                "discovery_quality": 0.8,
                "newcomer_protection": 0.0,
            },
            {
                "marketplace_id": "premium",
                "name": "PremiumMarket",
                "platform_fee": 0.10,
                "verification_level": 2,
                "discovery_quality": 1.2,
                "newcomer_protection": 0.15,
            },
        ],
        seed=42,
    )
    print_result(competition)

    # Three marketplaces
    print("\n  [Test] Three marketplaces...")
    three = run_simulation(
        "THREE MARKETPLACES: Cheap / Premium / Newcomer-friendly",
        num_agents=40, num_rounds=1500, entry_rate=0.04,
        marketplace_configs=[
            {
                "marketplace_id": "cheap",
                "name": "CheapMarket",
                "platform_fee": 0.02,
                "verification_level": 0,
                "discovery_quality": 0.8,
            },
            {
                "marketplace_id": "premium",
                "name": "PremiumMarket",
                "platform_fee": 0.10,
                "verification_level": 2,
                "discovery_quality": 1.2,
            },
            {
                "marketplace_id": "newcomer",
                "name": "NewcomerHub",
                "platform_fee": 0.05,
                "verification_level": 1,
                "discovery_quality": 1.0,
                "newcomer_protection": 0.30,
            },
        ],
        seed=42,
    )
    print_result(three)

    return control, competition, three


def experiment_coalitions_plus_marketplaces():
    """Coalitions operating across multiple marketplaces."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 4: Coalitions + Marketplace Competition         ║")
    print("║  Full complexity: groups of agents in competing markets     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    r = run_simulation(
        "FULL: Coalitions + 2 Marketplaces",
        num_agents=40, num_rounds=1500, entry_rate=0.04,
        enable_coalitions=True,
        marketplace_configs=[
            {
                "marketplace_id": "open",
                "name": "OpenMarket",
                "platform_fee": 0.03,
                "verification_level": 0,
                "discovery_quality": 1.0,
            },
            {
                "marketplace_id": "verified",
                "name": "VerifiedMarket",
                "platform_fee": 0.08,
                "verification_level": 2,
                "discovery_quality": 1.3,
                "newcomer_protection": 0.10,
            },
        ],
        seed=42,
    )
    print_result(r)
    return r


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v7                  ║")
    print("║  Coalition Formation & Multi-Marketplace Competition        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    exp1_ctrl, exp1_coal = experiment_coalitions()
    # exp2 = experiment_coalition_sizes()  # slow, skip for now
    exp3_ctrl, exp3_two, exp3_three = experiment_marketplace_competition()
    exp4 = experiment_coalitions_plus_marketplaces()

    print("\n\n" + "═" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
