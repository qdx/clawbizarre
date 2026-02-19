"""
ClawBizarre Economy Simulation v9 — Islands of Health & Guild Evolution

Key questions from v8:
1. Can coalitions with STRICT price floors create "islands of health" in declining economies?
2. Can coalitions evolve into marketplaces (guild-as-marketplace)?
3. How does reputation portability across marketplaces affect migration and moats?
4. What's the equilibrium of adaptive fees — oscillation or convergence?

New features:
- Strict coalition price floors with enforcement (members who undercut get expelled)
- Guild-marketplace evolution: coalitions that grow can accept external work orders
- Reputation portability factor: 0.0 (marketplace-locked) to 1.0 (fully portable)
- Fine-grained fee adaptation tracking for convergence analysis
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


# --- Coalition v2: Strict Enforcement + Guild Evolution ---

@dataclass
class Coalition:
    coalition_id: str
    founder: str
    members: list[str] = field(default_factory=list)
    formed_round: int = 0
    dissolved_round: Optional[int] = None
    active: bool = True
    treasury: float = 0.0
    fee_rate: float = 0.10
    treasury_history: list[float] = field(default_factory=list)
    shared_reputation: DomainReputation = field(default_factory=lambda: DomainReputation(
        half_life_days=30, domain_correlations=DOMAIN_CORRELATIONS
    ))
    referral_bonus: float = 0.3
    min_price_floor: Optional[dict] = None
    price_coordination: str = "none"
    total_tasks: int = 0
    total_revenue: float = 0.0
    member_history: list[tuple[int, str, str]] = field(default_factory=list)

    # v9: Strict enforcement
    strict_enforcement: bool = False  # Expel members who undercut floor
    violations: dict = field(default_factory=lambda: defaultdict(int))  # agent -> count
    max_violations: int = 3

    # v9: Guild-as-marketplace evolution
    is_guild_marketplace: bool = False
    guild_marketplace_fee: float = 0.07
    external_tasks: int = 0
    external_revenue: float = 0.0
    guild_reputation: float = 0.5  # External-facing reputation
    evolution_round: Optional[int] = None  # When it became a marketplace

    @property
    def size(self):
        return len(self.members)

    def add_member(self, agent_name: str, round_num: int):
        if agent_name not in self.members:
            self.members.append(agent_name)
            self.member_history.append((round_num, agent_name, "join"))

    def remove_member(self, agent_name: str, round_num: int, reason="leave"):
        if agent_name in self.members:
            self.members.remove(agent_name)
            self.member_history.append((round_num, agent_name, reason))
            if not self.members:
                self.active = False
                self.dissolved_round = round_num

    def record_violation(self, agent_name: str):
        self.violations[agent_name] += 1

    def should_expel(self, agent_name: str) -> bool:
        return self.strict_enforcement and self.violations.get(agent_name, 0) >= self.max_violations

    def collect_fee(self, revenue: float) -> float:
        fee = revenue * self.fee_rate
        self.treasury += fee
        self.total_revenue += revenue
        return revenue - fee

    def collect_external_fee(self, price: float) -> tuple[float, float]:
        """Fee for external work orders routed through guild."""
        guild_cut = price * self.guild_marketplace_fee
        self.treasury += guild_cut
        self.external_tasks += 1
        self.external_revenue += price
        return price - guild_cut, guild_cut

    def distribute_treasury(self) -> float:
        if not self.members:
            return 0.0
        per_member = self.treasury / len(self.members)
        self.treasury_history.append(self.treasury)
        self.treasury = 0.0
        return per_member

    def try_evolve_to_marketplace(self, round_num) -> bool:
        """Coalition evolves into guild-marketplace if it has enough scale and reputation."""
        if self.is_guild_marketplace:
            return False
        if self.size < 4:
            return False
        if self.total_tasks < 50:
            return False
        if round_num - self.formed_round < 200:
            return False
        # Need decent shared reputation
        avg_rep = sum(
            self.shared_reputation.score(d, time.time())
            for d in ALL_DOMAINS
        ) / len(ALL_DOMAINS)
        if avg_rep < 0.6:
            return False
        self.is_guild_marketplace = True
        self.evolution_round = round_num
        self.guild_reputation = avg_rep
        return True


# --- Marketplace ---

@dataclass
class Marketplace:
    marketplace_id: str
    name: str
    platform_fee: float = 0.05
    participants: list[str] = field(default_factory=list)
    verification_level: int = 0
    discovery_quality: float = 1.0
    newcomer_protection: float = 0.0
    total_tasks: int = 0
    total_volume: float = 0.0
    marketplace_reputation: float = 0.5
    adaptive_fees: bool = False
    fee_history: list[tuple[int, float]] = field(default_factory=list)
    target_participant_count: int = 50
    min_fee: float = 0.01
    max_fee: float = 0.20

    def join(self, agent_name: str):
        if agent_name not in self.participants:
            self.participants.append(agent_name)

    def leave(self, agent_name: str):
        if agent_name in self.participants:
            self.participants.remove(agent_name)

    def collect_fee(self, price: float) -> tuple[float, float]:
        platform_cut = price * self.platform_fee
        return price - platform_cut, platform_cut

    def update_reputation(self, success: bool):
        alpha = 0.01
        if success:
            self.marketplace_reputation += alpha * (1 - self.marketplace_reputation)
        else:
            self.marketplace_reputation -= alpha * self.marketplace_reputation

    def adapt_fee(self, round_num):
        if not self.adaptive_fees:
            return
        current = len(self.participants)
        if current < self.target_participant_count * 0.8:
            self.platform_fee = max(self.platform_fee * 0.98, self.min_fee)
        elif current > self.target_participant_count * 1.2:
            self.platform_fee = min(self.platform_fee * 1.02, self.max_fee)
        self.fee_history.append((round_num, self.platform_fee))


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
    balance: float = 0.0
    compute_costs: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    revenue_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    prices_charged: list[float] = field(default_factory=list)
    joined_round: int = 0
    exited_round: Optional[int] = None
    active: bool = True
    identity_cost_per_round: float = 0.01
    rounds_active: int = 0
    is_newcomer: bool = False
    coalition_id: Optional[str] = None
    coalition_benefit_received: float = 0.0
    solo_earnings_before_coalition: float = 0.0
    marketplace_ids: list[str] = field(default_factory=list)
    coalition_willingness: float = 0.5
    marketplace_earnings: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    marketplace_tasks: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    migrations: int = 0
    strategy_switches: int = 0
    rounds_since_switch: int = 0

    # v9: Reputation portability
    portable_reputation: DomainReputation = field(default_factory=lambda: DomainReputation(
        half_life_days=30, domain_correlations=DOMAIN_CORRELATIONS
    ))

    def price_for(self, domain, sim_time, market_prices, coalition=None):
        base_price = self.pricing_strategy.price(self, domain, sim_time, market_prices)
        if coalition and coalition.min_price_floor and domain in coalition.min_price_floor:
            floor = coalition.min_price_floor[domain]
            if base_price < floor:
                # Agent wants to undercut below floor
                if coalition.strict_enforcement:
                    coalition.record_violation(self.name)
                return max(base_price, floor)
        return base_price

    def accept_task(self, domain):
        return random.random() < self.reliability.get(domain, 0.5)

    def earnings_rate(self, window_rounds=200):
        if self.rounds_active < 10:
            return 0.0
        return self.balance / self.rounds_active

    def effective_reputation(self, domain, sim_time, marketplace_id=None, portability=0.0):
        """Reputation with portability factor."""
        local = self.reputation.score(domain, sim_time)
        if portability > 0.0:
            portable = self.portable_reputation.score(domain, sim_time)
            return local * (1 - portability * 0.3) + portable * portability * 0.3
        return local


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
                    price_sensitivity=1.0, coalitions=None, marketplace=None,
                    guild_marketplaces=None, rep_portability=0.0):
    candidates = [a for a in agents
                  if a.active and a.name != requester.name
                  and domain in a.capabilities]

    if marketplace:
        mp_candidates = [a for a in candidates if marketplace.marketplace_id in a.marketplace_ids]
        if marketplace.newcomer_protection > 0 and random.random() < marketplace.newcomer_protection:
            newcomers = [a for a in mp_candidates if a.is_newcomer]
            if newcomers:
                mp_candidates = newcomers
        candidates = mp_candidates if mp_candidates else candidates

    # v9: Also consider guild-marketplace members
    if guild_marketplaces:
        for guild in guild_marketplaces:
            guild_members = [a for a in agents if a.active and a.coalition_id == guild.coalition_id
                           and domain in a.capabilities and a.name != requester.name
                           and a.name not in [c.name for c in candidates]]
            candidates.extend(guild_members)

    if not candidates:
        return None

    requester_coalition = None
    if coalitions and requester.coalition_id:
        requester_coalition = coalitions.get(requester.coalition_id)

    scored = []
    for c in candidates:
        rep = c.effective_reputation(domain, sim_time, portability=rep_portability)
        conf = c.reputation._get_domain(domain).confidence(sim_time) if domain in c.reputation._domains else 0.0
        price = c.price_for(domain, sim_time, market_prices,
                           coalitions.get(c.coalition_id) if coalitions and c.coalition_id else None)
        quality = rep * max(conf, 0.1)

        if requester_coalition and c.coalition_id == requester.coalition_id:
            quality += requester_coalition.referral_bonus

        if c.coalition_id and coalitions:
            coal = coalitions.get(c.coalition_id)
            if coal:
                shared_rep = coal.shared_reputation.score(domain, sim_time)
                shared_conf = coal.shared_reputation._get_domain(domain).confidence(sim_time) \
                    if domain in coal.shared_reputation._domains else 0.0
                shared_quality = shared_rep * max(shared_conf, 0.1) * 0.5
                quality = max(quality, quality * 0.7 + shared_quality * 0.3)

                # Guild-marketplace bonus: external requesters see guild reputation
                if coal.is_guild_marketplace and requester.coalition_id != coal.coalition_id:
                    quality += coal.guild_reputation * 0.2

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


# --- Coalition Formation ---

def try_form_coalition(agents, round_num, coalitions, max_coalitions=30,
                       adversity_mode=False, strict_enforcement=False):
    if adversity_mode:
        eligible = [a for a in agents if a.active and a.coalition_id is None
                    and a.rounds_active > 100]
        struggling = [a for a in eligible if a.earnings_rate() < 0.01]
        successful = [a for a in eligible if a.earnings_rate() >= 0.02 and a.balance > 3.0]

        if struggling and len(coalitions) < max_coalitions:
            if random.random() < 0.15:
                founder = random.choice(struggling)
                potential = [a for a in struggling if a.name != founder.name
                             and a.coalition_willingness > 0.2
                             and any(d in founder.capabilities for d in a.capabilities)]
                if potential:
                    recruit_count = min(random.randint(2, 5), len(potential))
                    recruits = sorted(potential, key=lambda a: a.coalition_willingness, reverse=True)[:recruit_count]
                    accepted = [r for r in recruits if random.random() < r.coalition_willingness + 0.2]
                    if accepted:
                        cid = f"defense_{round_num}_{founder.name}"
                        coal = Coalition(
                            coalition_id=cid,
                            founder=founder.name,
                            formed_round=round_num,
                            fee_rate=0.05,
                            price_coordination="floor",
                            strict_enforcement=strict_enforcement,
                        )
                        # Strict floor: compute cost + 30% margin (higher than v8's 20%)
                        coal.min_price_floor = {d: TASK_CATALOG[d]["compute_cost"] * 1.3 for d in ALL_DOMAINS}
                        founder.coalition_id = cid
                        founder.solo_earnings_before_coalition = founder.balance
                        coal.add_member(founder.name, round_num)
                        for r in accepted:
                            r.coalition_id = cid
                            r.solo_earnings_before_coalition = r.balance
                            coal.add_member(r.name, round_num)
                        return coal

        if successful and len(coalitions) < max_coalitions:
            if random.random() < 0.05:
                founder = max(successful, key=lambda a: a.earnings_rate())
                potential = [a for a in eligible if a.name != founder.name
                             and a.coalition_willingness > 0.3
                             and any(d not in founder.capabilities for d in a.capabilities)]
                if potential:
                    recruit_count = min(random.randint(2, 4), len(potential))
                    potential.sort(key=lambda a: a.earnings_rate(), reverse=True)
                    recruits = potential[:recruit_count]
                    accepted = [r for r in recruits if random.random() < r.coalition_willingness]
                    if accepted:
                        cid = f"strategic_{round_num}_{founder.name}"
                        coal = Coalition(
                            coalition_id=cid,
                            founder=founder.name,
                            formed_round=round_num,
                            strict_enforcement=strict_enforcement,
                        )
                        coal.min_price_floor = {d: TASK_CATALOG[d]["base_cost"] * 0.8 for d in ALL_DOMAINS}
                        coal.price_coordination = "floor"
                        founder.coalition_id = cid
                        founder.solo_earnings_before_coalition = founder.balance
                        coal.add_member(founder.name, round_num)
                        for r in accepted:
                            r.coalition_id = cid
                            r.solo_earnings_before_coalition = r.balance
                            coal.add_member(r.name, round_num)
                        return coal
    else:
        active_solo = [a for a in agents if a.active and a.coalition_id is None
                       and a.rounds_active > 200 and a.balance > 5.0]
        if not active_solo or len(coalitions) >= max_coalitions:
            return None
        founder = max(active_solo, key=lambda a: a.earnings_rate())
        if founder.earnings_rate() < 0.02:
            return None
        if random.random() > 0.05:
            return None
        potential = [a for a in active_solo if a.name != founder.name
                     and a.coalition_willingness > 0.3
                     and any(d not in founder.capabilities for d in a.capabilities)]
        if not potential:
            return None
        recruit_count = min(random.randint(2, 4), len(potential))
        potential.sort(key=lambda a: a.earnings_rate(), reverse=True)
        recruits = potential[:recruit_count]
        accepted = [r for r in recruits if random.random() < r.coalition_willingness]
        if not accepted:
            return None
        cid = f"coalition_{round_num}_{founder.name}"
        coal = Coalition(coalition_id=cid, founder=founder.name, formed_round=round_num,
                        strict_enforcement=strict_enforcement)
        coal.min_price_floor = {d: TASK_CATALOG[d]["base_cost"] * 0.8 for d in ALL_DOMAINS}
        coal.price_coordination = "floor"
        founder.coalition_id = cid
        founder.solo_earnings_before_coalition = founder.balance
        coal.add_member(founder.name, round_num)
        for r in accepted:
            r.coalition_id = cid
            r.solo_earnings_before_coalition = r.balance
            coal.add_member(r.name, round_num)
        return coal

    return None


def evaluate_coalition_membership(agent, coalition, round_num):
    if round_num - coalition.formed_round < 150:
        return True
    joins = [h[0] for h in coalition.member_history if h[1] == agent.name and h[2] == "join"]
    if not joins:
        return True
    rounds_in = round_num - max(joins)
    if rounds_in < 80:
        return True
    pre_rate = agent.solo_earnings_before_coalition / max(agent.joined_round + 1, 1)
    current_rate = (agent.balance - agent.solo_earnings_before_coalition) / max(rounds_in, 1)
    if current_rate < pre_rate * 0.5:
        return random.random() > 0.4
    if coalition.size > 7:
        overhead = (coalition.size - 3) * 0.05
        if random.random() < overhead:
            return False
    return True


def evaluate_marketplace_migration(agent, marketplaces, round_num):
    if agent.rounds_active < 150 or len(agent.marketplace_ids) == 0:
        return
    if len(marketplaces) < 2:
        return
    if random.random() > 0.02:
        return

    current_ids = set(agent.marketplace_ids)
    available = [mp for mp in marketplaces.values() if mp.marketplace_id not in current_ids]
    if not available:
        return

    best_current_rate = 0.0
    worst_current_id = None
    worst_current_rate = float('inf')
    for mp_id in agent.marketplace_ids:
        tasks = agent.marketplace_tasks.get(mp_id, 0)
        if tasks > 0:
            rate = agent.marketplace_earnings.get(mp_id, 0) / tasks
            if rate > best_current_rate:
                best_current_rate = rate
            if rate < worst_current_rate:
                worst_current_rate = rate
                worst_current_id = mp_id

    for mp in available:
        expected = mp.marketplace_reputation * (1 - mp.platform_fee)
        current_expected = best_current_rate if best_current_rate > 0 else 0.5 * 0.95
        if expected > current_expected * 1.3:
            agent.marketplace_ids.append(mp.marketplace_id)
            mp.join(agent.name)
            agent.migrations += 1
            if len(agent.marketplace_ids) > 2 and worst_current_id:
                agent.marketplace_ids.remove(worst_current_id)
                marketplaces[worst_current_id].leave(agent.name)
            break


def maybe_switch_strategy(agent, agents, round_num, market_prices):
    if agent.rounds_active < 200 or agent.rounds_since_switch < 200:
        return
    if random.random() > 0.03:
        return

    same = [a for a in agents if a.active and a.pricing_strategy.name == agent.pricing_strategy.name
            and a.rounds_active > 100]
    diff = [a for a in agents if a.active and a.pricing_strategy.name != agent.pricing_strategy.name
            and a.rounds_active > 100]

    if not same or not diff:
        return

    avg_same = sum(a.earnings_rate() for a in same) / len(same)
    avg_diff = sum(a.earnings_rate() for a in diff) / len(diff)

    if avg_diff > avg_same * 1.4:
        if agent.pricing_strategy.name == "reputation":
            agent.pricing_strategy = UndercutStrategy(random.uniform(0.10, 0.25))
        else:
            agent.pricing_strategy = ReputationPremium()
        agent.strategy_switches += 1
        agent.rounds_since_switch = 0


# --- Simulation ---

def run_simulation(label, num_agents=50, num_rounds=2000,
                   entry_rate=0.05, exit_threshold=-5.0,
                   price_sensitivity=1.0,
                   undercut_fraction=0.0,
                   enable_coalitions=False,
                   adversity_coalitions=False,
                   strict_coalition_enforcement=False,
                   enable_guild_evolution=False,
                   enable_strategy_switching=False,
                   marketplace_configs=None,
                   adaptive_marketplace_fees=False,
                   enable_migration=False,
                   rep_portability=0.0,
                   seed=42):
    random.seed(seed)
    _FastIdentity._counter = 0
    sim_start = time.time() - 84 * 86400
    tracker = MarketPriceTracker(window=200)

    marketplaces: dict[str, Marketplace] = {}
    if marketplace_configs:
        for mc in marketplace_configs:
            mp = Marketplace(**mc)
            if adaptive_marketplace_fees:
                mp.adaptive_fees = True
            marketplaces[mp.marketplace_id] = mp

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

        if random.random() < undercut_fraction:
            strategy = UndercutStrategy(random.uniform(0.10, 0.25))
        else:
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

        if marketplaces:
            if random.random() < 0.3:
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
    guild_evolution_events = []
    expulsion_events = []
    fee_convergence_data = []

    for round_num in range(1, num_rounds + 1):
        sim_time = sim_start + round_num * 3600
        active_agents = [a for a in agents if a.active]

        for a in active_agents:
            a.balance -= a.identity_cost_per_round
            a.rounds_active += 1
            a.rounds_since_switch += 1

        for a in active_agents:
            if a.is_newcomer and (round_num - a.joined_round) > 200:
                a.is_newcomer = False

        # Entry
        while random.random() < entry_rate:
            new_agent = make_agent(agent_idx, joined_round=round_num)
            agents.append(new_agent)
            agent_idx += 1

        # Exit
        for a in list(active_agents):
            if a.balance < exit_threshold and round_num - a.joined_round > 100:
                a.active = False
                a.exited_round = round_num
                if a.coalition_id and a.coalition_id in coalitions:
                    coalitions[a.coalition_id].remove_member(a.name, round_num, "bankruptcy")
                for mp_id in a.marketplace_ids:
                    if mp_id in marketplaces:
                        marketplaces[mp_id].leave(a.name)

        active_agents = [a for a in agents if a.active]
        if len(active_agents) < 2:
            continue

        # Strategy switching
        if enable_strategy_switching:
            for a in active_agents:
                maybe_switch_strategy(a, active_agents, round_num, tracker.get_prices())

        # Coalition formation + enforcement + guild evolution
        if enable_coalitions and round_num % 30 == 0:
            new_coal = try_form_coalition(active_agents, round_num, coalitions,
                                          adversity_mode=adversity_coalitions,
                                          strict_enforcement=strict_coalition_enforcement)
            if new_coal:
                coalitions[new_coal.coalition_id] = new_coal

            # Membership evaluation + expulsion
            for a in list(active_agents):
                if a.coalition_id and a.coalition_id in coalitions:
                    coal = coalitions[a.coalition_id]
                    # v9: Strict enforcement — expel violators
                    if coal.should_expel(a.name):
                        coal.remove_member(a.name, round_num, "expelled")
                        expulsion_events.append((round_num, a.name, coal.coalition_id))
                        a.coalition_id = None
                    elif not evaluate_coalition_membership(a, coal, round_num):
                        coal.remove_member(a.name, round_num)
                        a.coalition_id = None

            # Treasury distribution
            if round_num % 90 == 0:
                for coal in coalitions.values():
                    if coal.active and coal.treasury > 0:
                        per_member = coal.distribute_treasury()
                        for a in active_agents:
                            if a.coalition_id == coal.coalition_id:
                                a.balance += per_member
                                a.coalition_benefit_received += per_member

            # v9: Guild evolution check
            if enable_guild_evolution and round_num % 100 == 0:
                for coal in coalitions.values():
                    if coal.active and not coal.is_guild_marketplace:
                        if coal.try_evolve_to_marketplace(round_num):
                            guild_evolution_events.append((round_num, coal.coalition_id, coal.size))

        # Marketplace migration
        if enable_migration and round_num % 50 == 0:
            for a in active_agents:
                evaluate_marketplace_migration(a, marketplaces, round_num)

        # Adaptive fees + convergence tracking
        if adaptive_marketplace_fees and round_num % 100 == 0:
            for mp in marketplaces.values():
                mp.adapt_fee(round_num)
            fee_convergence_data.append({
                "round": round_num,
                "fees": {mp.name: mp.platform_fee for mp in marketplaces.values()},
                "participants": {mp.name: len(mp.participants) for mp in marketplaces.values()},
            })

        # Generate tasks
        base_tasks = random.randint(3, 6)
        demand_mult = 1.0 + 0.01 * len(active_agents)
        num_tasks = int(base_tasks * demand_mult)
        market_prices = tracker.get_prices()

        # Collect guild-marketplaces for discovery
        active_guilds = [c for c in coalitions.values() if c.active and c.is_guild_marketplace]

        for _ in range(num_tasks):
            requester = random.choice(active_agents)
            domain = random.choice(ALL_DOMAINS)

            marketplace = None
            if marketplaces and requester.marketplace_ids:
                mp_id = random.choice(requester.marketplace_ids)
                marketplace = marketplaces[mp_id]

            worker = discover_worker(
                active_agents, requester, domain, sim_time, market_prices,
                price_sensitivity,
                coalitions=coalitions if enable_coalitions else None,
                marketplace=marketplace,
                guild_marketplaces=active_guilds if enable_guild_evolution else None,
                rep_portability=rep_portability,
            )
            if not worker:
                continue

            worker_coalition = None
            if enable_coalitions and worker.coalition_id:
                worker_coalition = coalitions.get(worker.coalition_id)

            price = worker.price_for(domain, sim_time, market_prices, worker_coalition)
            compute_cost = TASK_CATALOG[domain]["compute_cost"]
            success = worker.accept_task(domain)
            worker.reputation.record(domain, success, timestamp=sim_time)

            # v9: Also record to portable reputation
            if rep_portability > 0:
                worker.portable_reputation.record(domain, success, timestamp=sim_time)

            if worker_coalition:
                worker_coalition.shared_reputation.record(domain, success, timestamp=sim_time)
                worker_coalition.total_tasks += 1

            tracker.record(domain, price)
            worker.prices_charged.append(price)
            worker.compute_costs += compute_cost

            if success:
                seller_receives = price

                # Guild-marketplace external fee
                if worker_coalition and worker_coalition.is_guild_marketplace \
                        and requester.coalition_id != worker_coalition.coalition_id:
                    seller_receives, guild_cut = worker_coalition.collect_external_fee(seller_receives)

                if marketplace:
                    seller_receives, platform_cut = marketplace.collect_fee(seller_receives)
                    marketplace.total_tasks += 1
                    marketplace.total_volume += price
                    marketplace.update_reputation(True)
                    worker.marketplace_earnings[marketplace.marketplace_id] += seller_receives
                    worker.marketplace_tasks[marketplace.marketplace_id] += 1

                if worker_coalition and not worker_coalition.is_guild_marketplace:
                    seller_receives = worker_coalition.collect_fee(seller_receives)

                worker.balance += seller_receives
                worker.revenue_by_domain[domain] += seller_receives
                worker.tasks_completed += 1
            else:
                worker.tasks_failed += 1
                if marketplace:
                    marketplace.update_reputation(False)

        # Snapshots
        if round_num % 200 == 0:
            active_a = [a for a in agents if a.active]
            earnings = sorted(a.balance for a in active_a)
            n = len(earnings)
            s = sum(earnings)
            gini = sum((2*i - n + 1) * e for i, e in enumerate(earnings)) / (n * s) if s > 0 and n > 1 else 0

            nc_total = sum(1 for a in agents if a.joined_round > 0)
            nc_alive = sum(1 for a in agents if a.joined_round > 0 and a.active)

            rep_count = sum(1 for a in active_a if a.pricing_strategy.name == "reputation")
            ucut_count = sum(1 for a in active_a if a.pricing_strategy.name == "undercut")

            active_coals = [c for c in coalitions.values() if c.active]
            coal_members = sum(1 for a in active_a if a.coalition_id is not None)
            guild_mps = sum(1 for c in active_coals if c.is_guild_marketplace)

            snapshots.append({
                "round": round_num,
                "active": len(active_a),
                "gini": gini,
                "newcomer_survival": nc_alive / max(nc_total, 1),
                "reputation_agents": rep_count,
                "undercut_agents": ucut_count,
                "active_coalitions": len(active_coals),
                "coalition_members": coal_members,
                "guild_marketplaces": guild_mps,
            })

    # --- Final analysis ---
    active_agents = [a for a in agents if a.active]
    incumbents = [a for a in active_agents if a.joined_round == 0]
    newcomer_survivors = [a for a in active_agents if a.joined_round > 0]
    coalition_members_list = [a for a in active_agents if a.coalition_id is not None]
    solo_agents_list = [a for a in active_agents if a.coalition_id is None]
    rep_agents = [a for a in active_agents if a.pricing_strategy.name == "reputation"]
    ucut_agents = [a for a in active_agents if a.pricing_strategy.name == "undercut"]

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

    total_migrations = sum(a.migrations for a in agents)
    total_switches = sum(a.strategy_switches for a in agents)

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
        "rep_agents": len(rep_agents),
        "ucut_agents": len(ucut_agents),
        "rep_avg_balance": sum(a.balance for a in rep_agents) / max(len(rep_agents), 1),
        "ucut_avg_balance": sum(a.balance for a in ucut_agents) / max(len(ucut_agents), 1),
        "coalition_members": len(coalition_members_list),
        "solo_agents": len(solo_agents_list),
        "coalition_avg_balance": sum(a.balance for a in coalition_members_list) / max(len(coalition_members_list), 1),
        "solo_avg_balance": sum(a.balance for a in solo_agents_list) / max(len(solo_agents_list), 1),
        "active_coalitions": sum(1 for c in coalitions.values() if c.active),
        "total_coalitions_formed": len(coalitions),
        "coalitions": coalitions,
        "price_convergence": price_convergence,
        "snapshots": snapshots,
        "marketplaces": marketplaces,
        "total_migrations": total_migrations,
        "total_switches": total_switches,
        "guild_evolution_events": guild_evolution_events,
        "expulsion_events": expulsion_events,
        "fee_convergence_data": fee_convergence_data,
    }


def print_result(r):
    print(f"\n{'='*70}")
    print(f"  {r['label']}")
    print(f"{'='*70}")
    print(f"  Agents: {r['total_agents']} total, {r['active']} active, {r['exited']} exited")
    nc_total = max(r['newcomer_total'], 1)
    print(f"  Newcomers: {r['newcomer_survivors']}/{r['newcomer_total']} survived "
          f"({r['newcomer_survivors']/nc_total*100:.0f}%)")
    print(f"  Incumbent avg: ${r['incumbent_avg_balance']:.2f}")
    print(f"  Newcomer avg:  ${r['newcomer_avg_balance']:.2f}")
    gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
    print(f"  Earnings gap: {gap:.1f}x")
    print(f"  Final Gini: {r['final_gini']:.3f}")

    if r['rep_agents'] > 0 or r['ucut_agents'] > 0:
        print(f"\n  Strategy Split:")
        print(f"    Reputation: {r['rep_agents']} (avg ${r['rep_avg_balance']:.2f})")
        print(f"    Undercut:   {r['ucut_agents']} (avg ${r['ucut_avg_balance']:.2f})")
        if r['total_switches'] > 0:
            print(f"    Strategy switches: {r['total_switches']}")

    if r['active_coalitions'] > 0 or r['total_coalitions_formed'] > 0:
        print(f"\n  Coalition Stats:")
        print(f"    Formed: {r['total_coalitions_formed']}, Active: {r['active_coalitions']}")
        print(f"    Coalition members: {r['coalition_members']}, Solo: {r['solo_agents']}")
        if r['coalition_members'] > 0:
            print(f"    Coalition avg: ${r['coalition_avg_balance']:.2f}")
        if r['solo_agents'] > 0:
            print(f"    Solo avg:      ${r['solo_avg_balance']:.2f}")

        active_coals = [c for c in r['coalitions'].values() if c.active]
        if active_coals:
            print(f"\n    Active Coalitions:")
            for c in sorted(active_coals, key=lambda x: x.total_revenue, reverse=True)[:10]:
                dtype = "DEFENSE" if c.coalition_id.startswith("defense") else "STRATEGIC"
                guild = " [GUILD-MP]" if c.is_guild_marketplace else ""
                print(f"      [{dtype}]{guild} {c.coalition_id[:35]} — {c.size} members, "
                      f"{c.total_tasks} tasks, ${c.total_revenue:.2f} rev, "
                      f"${c.treasury:.2f} treasury")
                if c.is_guild_marketplace:
                    print(f"        External: {c.external_tasks} tasks, ${c.external_revenue:.2f} rev, "
                          f"guild rep: {c.guild_reputation:.2f}")

        if r['expulsion_events']:
            print(f"\n    Expulsions: {len(r['expulsion_events'])}")

        if r['guild_evolution_events']:
            print(f"\n    Guild Evolution Events:")
            for round_num, cid, size in r['guild_evolution_events']:
                print(f"      Round {round_num}: {cid[:35]} ({size} members)")

    if r.get('marketplaces') and len(r['marketplaces']) > 1:
        print(f"\n  Marketplace Competition:")
        print(f"  {'Name':<20} {'Fee':>6} {'Agents':>7} {'Tasks':>7} {'Volume':>10} {'Rep':>6}")
        print(f"  {'─'*60}")
        for mp in r['marketplaces'].values():
            fee_str = f"{mp.platform_fee:.1%}"
            print(f"  {mp.name:<20} {fee_str:>6} {len(mp.participants):>7} "
                  f"{mp.total_tasks:>7} ${mp.total_volume:>8.2f} {mp.marketplace_reputation:>5.2f}")
        if r['total_migrations'] > 0:
            print(f"  Total migrations: {r['total_migrations']}")

    if r.get('fee_convergence_data'):
        print(f"\n  Fee Convergence (every 500 rounds):")
        for fd in r['fee_convergence_data']:
            if fd['round'] % 500 == 0:
                fees = ", ".join(f"{n}: {f:.2%}" for n, f in fd['fees'].items())
                parts = ", ".join(f"{n}: {p}" for n, p in fd['participants'].items())
                print(f"    Round {fd['round']:>5}: [{fees}] [{parts}]")

    if r['snapshots'] and (r.get('total_coalitions_formed', 0) > 0 or r.get('ucut_agents', 0) > 0):
        print(f"\n  Timeline (every 400 rounds):")
        header = f"  {'Round':>6} {'Active':>7} {'Gini':>7} {'Rep':>5} {'Ucut':>5} {'Coals':>6} {'CMemb':>6}"
        if any(s.get('guild_marketplaces', 0) > 0 for s in r['snapshots']):
            header += f" {'Guilds':>7}"
        print(header)
        print(f"  {'─'*60}")
        for s in r['snapshots']:
            if s['round'] % 400 == 0:
                line = (f"  {s['round']:>6} {s['active']:>7} {s['gini']:>7.3f} "
                       f"{s['reputation_agents']:>5} {s['undercut_agents']:>5} "
                       f"{s['active_coalitions']:>6} {s['coalition_members']:>6}")
                if any(ss.get('guild_marketplaces', 0) > 0 for ss in r['snapshots']):
                    line += f" {s.get('guild_marketplaces', 0):>7}"
                print(line)

    print(f"\n  Price Convergence:")
    print(f"  {'Domain':<20} {'Early':>8} {'Late':>8} {'Floor':>8} {'Margin':>8} {'Δ%':>8}")
    print(f"  {'─'*55}")
    for d, pc in sorted(r["price_convergence"].items()):
        print(f"  {d:<20} ${pc['early_avg']:>6.3f} ${pc['late_avg']:>6.3f} "
              f"${pc['compute_cost']:>6.3f} {pc['late_margin']:>7.1%} {pc['price_change_pct']:>+7.1f}%")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def exp1_strict_price_floors():
    """Can strict coalition price floors create islands of health?"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 1: Strict Price Floors — Islands of Health      ║")
    print("║  50% undercutters: strict vs loose coalition enforcement     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Control: No coalitions
    print("  [A] 50% undercut, no coalitions (baseline)...")
    ctrl = run_simulation(
        "BASELINE: 50% undercut, no coalitions",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.5, enable_coalitions=False,
        seed=42,
    )
    print_result(ctrl)

    # Loose enforcement (v8 style)
    print("\n  [B] 50% undercut, loose coalitions (v8 style)...")
    loose = run_simulation(
        "LOOSE: Coalitions without strict enforcement",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.5, enable_coalitions=True, adversity_coalitions=True,
        strict_coalition_enforcement=False,
        seed=42,
    )
    print_result(loose)

    # Strict enforcement
    print("\n  [C] 50% undercut, STRICT coalitions (expel violators)...")
    strict = run_simulation(
        "STRICT: Coalitions with enforcement (expel undercutters)",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.5, enable_coalitions=True, adversity_coalitions=True,
        strict_coalition_enforcement=True,
        seed=42,
    )
    print_result(strict)

    # Strict + strategy switching (the acid test)
    print("\n  [D] 50% undercut, STRICT + strategy switching...")
    strict_switch = run_simulation(
        "STRICT + SWITCHING: Can strict floors resist strategy drift?",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.5, enable_coalitions=True, adversity_coalitions=True,
        strict_coalition_enforcement=True, enable_strategy_switching=True,
        seed=42,
    )
    print_result(strict_switch)

    return ctrl, loose, strict, strict_switch


def exp2_guild_evolution():
    """Can coalitions evolve into guild-marketplaces?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 2: Guild-as-Marketplace Evolution               ║")
    print("║  Do coalitions grow into external-facing marketplaces?       ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Healthy economy + guild evolution
    print("  [A] Healthy economy, guild evolution enabled...")
    healthy = run_simulation(
        "HEALTHY: Guild evolution in healthy economy",
        num_agents=60, num_rounds=3000, entry_rate=0.04,
        enable_coalitions=True, enable_guild_evolution=True,
        seed=42,
    )
    print_result(healthy)

    # Adversity + guild evolution
    print("\n  [B] 40% undercut, adversity coalitions + guild evolution...")
    adversity = run_simulation(
        "ADVERSITY: Guild evolution under 40% undercutters",
        num_agents=60, num_rounds=3000, entry_rate=0.04,
        undercut_fraction=0.4, enable_coalitions=True, adversity_coalitions=True,
        strict_coalition_enforcement=True, enable_guild_evolution=True,
        seed=42,
    )
    print_result(adversity)

    # Adversity + guild + marketplaces (coexistence)
    print("\n  [C] 40% undercut, guilds + traditional marketplaces...")
    coexist = run_simulation(
        "COEXISTENCE: Guilds + traditional marketplaces",
        num_agents=60, num_rounds=3000, entry_rate=0.04,
        undercut_fraction=0.4, enable_coalitions=True, adversity_coalitions=True,
        strict_coalition_enforcement=True, enable_guild_evolution=True,
        enable_migration=True, adaptive_marketplace_fees=True,
        marketplace_configs=[
            {"marketplace_id": "open", "name": "OpenMarket", "platform_fee": 0.03,
             "verification_level": 0, "discovery_quality": 1.0, "target_participant_count": 40},
            {"marketplace_id": "verified", "name": "VerifiedMarket", "platform_fee": 0.08,
             "verification_level": 2, "discovery_quality": 1.3, "newcomer_protection": 0.15,
             "target_participant_count": 30},
        ],
        seed=42,
    )
    print_result(coexist)

    return healthy, adversity, coexist


def exp3_reputation_portability():
    """How does reputation portability affect migration, moats, and inequality?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 3: Reputation Portability Sweep                 ║")
    print("║  0.0 (locked) → 1.0 (fully portable): effects on economy   ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = {}
    for port in [0.0, 0.25, 0.5, 0.75, 1.0]:
        print(f"  Running portability={port:.2f}...")
        r = run_simulation(
            f"Portability {port:.2f}",
            num_agents=60, num_rounds=2000, entry_rate=0.04,
            enable_migration=True, adaptive_marketplace_fees=True,
            rep_portability=port,
            marketplace_configs=[
                {"marketplace_id": "cheap", "name": "CheapMarket", "platform_fee": 0.02,
                 "verification_level": 0, "discovery_quality": 0.8, "target_participant_count": 40},
                {"marketplace_id": "premium", "name": "PremiumMarket", "platform_fee": 0.10,
                 "verification_level": 2, "discovery_quality": 1.2, "newcomer_protection": 0.15,
                 "target_participant_count": 30},
            ],
            seed=42,
        )
        results[port] = r

    print(f"\n{'='*70}")
    print(f"  REPUTATION PORTABILITY SWEEP")
    print(f"{'='*70}")
    print(f"  {'Port':>6} {'Active':>7} {'Gini':>7} {'NC Surv':>8} {'Gap':>6} "
          f"{'Migr':>6} {'Cheap#':>7} {'Prem#':>7}")
    print(f"  {'─'*65}")
    for port, r in sorted(results.items()):
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        nc_surv = r['newcomer_survivors'] / max(r['newcomer_total'], 1) * 100
        cheap_n = len(r['marketplaces']['cheap'].participants) if 'cheap' in r['marketplaces'] else 0
        prem_n = len(r['marketplaces']['premium'].participants) if 'premium' in r['marketplaces'] else 0
        print(f"  {port:>5.2f} {r['active']:>7} {r['final_gini']:>7.3f} {nc_surv:>7.0f}% "
              f"{gap:>5.1f}x {r['total_migrations']:>6} {cheap_n:>7} {prem_n:>7}")

    return results


def exp4_fee_convergence():
    """Do adaptive fees converge, oscillate, or diverge?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 4: Fee Convergence Analysis                     ║")
    print("║  5000 rounds of adaptive fees — convergence or oscillation? ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    r = run_simulation(
        "FEE CONVERGENCE: 5000 rounds, adaptive fees, 3 marketplaces",
        num_agents=60, num_rounds=5000, entry_rate=0.04,
        enable_migration=True, adaptive_marketplace_fees=True,
        marketplace_configs=[
            {"marketplace_id": "cheap", "name": "CheapMarket", "platform_fee": 0.02,
             "verification_level": 0, "discovery_quality": 0.8, "target_participant_count": 40},
            {"marketplace_id": "mid", "name": "MidMarket", "platform_fee": 0.05,
             "verification_level": 1, "discovery_quality": 1.0, "newcomer_protection": 0.15,
             "target_participant_count": 35},
            {"marketplace_id": "premium", "name": "PremiumMarket", "platform_fee": 0.12,
             "verification_level": 2, "discovery_quality": 1.3, "newcomer_protection": 0.10,
             "target_participant_count": 25},
        ],
        seed=42,
    )
    print_result(r)

    # Detailed fee timeline
    if r['fee_convergence_data']:
        print(f"\n  Detailed Fee Timeline (every 200 rounds):")
        print(f"  {'Round':>6}", end="")
        names = list(r['fee_convergence_data'][0]['fees'].keys())
        for n in names:
            print(f"  {n:>14}", end="")
        print()
        print(f"  {'─'*(6 + 16*len(names))}")
        for fd in r['fee_convergence_data']:
            if fd['round'] % 200 == 0:
                print(f"  {fd['round']:>6}", end="")
                for n in names:
                    f = fd['fees'].get(n, 0)
                    p = fd['participants'].get(n, 0)
                    print(f"  {f:>5.2%} ({p:>3})", end="")
                print()

    return r


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v9                  ║")
    print("║  Islands of Health & Guild Evolution                        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    e1 = exp1_strict_price_floors()
    e2 = exp2_guild_evolution()
    e3 = exp3_reputation_portability()
    e4 = exp4_fee_convergence()

    print("\n\n" + "═" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
