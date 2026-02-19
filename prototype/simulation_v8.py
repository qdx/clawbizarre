"""
ClawBizarre Economy Simulation v8 — Adversity & Adaptive Dynamics

Key questions from v7:
1. Do coalitions form as DEFENSE when undercutters are present?
2. Can agents dynamically migrate between marketplaces based on performance?
3. What happens when marketplaces adjust fees dynamically (platform competition)?
4. Can a coalition become a marketplace itself (guild-as-platform)?

New features:
- Mixed strategy populations (reputation + undercut) to create adversity
- Dynamic marketplace migration: agents evaluate and switch marketplaces
- Adaptive platform fees: marketplaces adjust fees based on competition
- Guild-marketplace hybrid: coalitions can become mini-marketplaces
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


# --- Coalition ---

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

    # v8: Guild-as-marketplace
    is_marketplace: bool = False  # Can external agents buy from this coalition?
    external_fee: float = 0.07   # Fee for external agents using guild services
    external_tasks: int = 0

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
        fee = revenue * self.fee_rate
        self.treasury += fee
        self.total_revenue += revenue
        return revenue - fee

    def distribute_treasury(self) -> float:
        if not self.members:
            return 0.0
        per_member = self.treasury / len(self.members)
        self.treasury_history.append(self.treasury)
        self.treasury = 0.0
        return per_member


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

    # v8: Adaptive fees
    adaptive_fees: bool = False
    fee_history: list[tuple[int, float]] = field(default_factory=list)
    target_participant_count: int = 50  # Desired number of agents
    min_fee: float = 0.01
    max_fee: float = 0.15

    def join(self, agent_name: str):
        if agent_name not in self.participants:
            self.participants.append(agent_name)

    def leave(self, agent_name: str):
        if agent_name in self.participants:
            self.participants.remove(agent_name)

    def collect_fee(self, price: float) -> tuple[float, float]:
        platform_cut = price * self.platform_fee
        verification_cost = platform_cut * 0.3 * self.verification_level
        return price - platform_cut, platform_cut - verification_cost

    def update_reputation(self, success: bool):
        alpha = 0.01
        if success:
            self.marketplace_reputation += alpha * (1 - self.marketplace_reputation)
        else:
            self.marketplace_reputation -= alpha * self.marketplace_reputation

    def adapt_fee(self, round_num):
        """Adjust fee based on participant count vs target."""
        if not self.adaptive_fees:
            return
        current = len(self.participants)
        if current < self.target_participant_count * 0.8:
            # Too few agents, lower fee to attract
            self.platform_fee = max(self.platform_fee * 0.98, self.min_fee)
        elif current > self.target_participant_count * 1.2:
            # Too many agents, raise fee to extract value
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

    # v8: Migration tracking
    marketplace_earnings: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    marketplace_tasks: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    migrations: int = 0

    # v8: Strategy switching (from v6)
    strategy_switches: int = 0
    rounds_since_switch: int = 0

    def price_for(self, domain, sim_time, market_prices, coalition=None):
        base_price = self.pricing_strategy.price(self, domain, sim_time, market_prices)
        if coalition and coalition.min_price_floor and domain in coalition.min_price_floor:
            return max(base_price, coalition.min_price_floor[domain])
        return base_price

    def accept_task(self, domain):
        return random.random() < self.reliability.get(domain, 0.5)

    def earnings_rate(self, window_rounds=200):
        if self.rounds_active < 10:
            return 0.0
        return self.balance / self.rounds_active

    def best_marketplace(self) -> Optional[str]:
        """Return marketplace with best earnings rate, or None."""
        if not self.marketplace_earnings:
            return None
        return max(self.marketplace_earnings, key=lambda mp: self.marketplace_earnings[mp] / max(self.marketplace_tasks.get(mp, 1), 1))


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

    if not candidates:
        return None

    requester_coalition = None
    if coalitions and requester.coalition_id:
        requester_coalition = coalitions.get(requester.coalition_id)

    scored = []
    for c in candidates:
        rep = c.reputation.score(domain, sim_time)
        conf = c.reputation._get_domain(domain).confidence(sim_time) if domain in c.reputation._domains else 0.0
        price = c.price_for(domain, sim_time, market_prices)
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


# --- Coalition Formation (adapted for adversity) ---

def try_form_coalition(agents, round_num, coalitions, max_coalitions=30,
                       adversity_mode=False):
    """
    Under adversity: lower formation thresholds, urgency-based formation.
    Agents losing money are more willing to band together.
    """
    if adversity_mode:
        # Under adversity, struggling agents also try to form coalitions
        eligible = [a for a in agents if a.active and a.coalition_id is None
                    and a.rounds_active > 100]
        # Struggling agents: negative or low earnings
        struggling = [a for a in eligible if a.earnings_rate() < 0.01]
        # Successful agents still might form
        successful = [a for a in eligible if a.earnings_rate() >= 0.02 and a.balance > 3.0]

        # Struggling agents form DEFENSIVE coalitions (higher chance)
        if struggling and len(coalitions) < max_coalitions:
            if random.random() < 0.15:  # 15% chance (3x normal)
                founder = random.choice(struggling)
                potential = [a for a in struggling if a.name != founder.name
                             and a.coalition_willingness > 0.2
                             and any(d in founder.capabilities for d in a.capabilities)]
                if potential:
                    recruit_count = min(random.randint(2, 5), len(potential))
                    recruits = sorted(potential, key=lambda a: a.coalition_willingness, reverse=True)[:recruit_count]
                    accepted = [r for r in recruits if random.random() < r.coalition_willingness + 0.2]  # +0.2 desperation bonus
                    if accepted:
                        cid = f"defense_{round_num}_{founder.name}"
                        coal = Coalition(
                            coalition_id=cid,
                            founder=founder.name,
                            formed_round=round_num,
                            fee_rate=0.05,  # Lower fee for defensive coalitions
                            price_coordination="floor",
                        )
                        # Set floor at compute cost + 20% margin (resist race to bottom)
                        coal.min_price_floor = {d: TASK_CATALOG[d]["compute_cost"] * 1.2 for d in ALL_DOMAINS}
                        founder.coalition_id = cid
                        founder.solo_earnings_before_coalition = founder.balance
                        coal.add_member(founder.name, round_num)
                        for r in accepted:
                            r.coalition_id = cid
                            r.solo_earnings_before_coalition = r.balance
                            coal.add_member(r.name, round_num)
                        return coal

        # Successful agents form STRATEGIC coalitions (normal rate)
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
        # Original v7 logic
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
        coal = Coalition(coalition_id=cid, founder=founder.name, formed_round=round_num)
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
        return random.random() > 0.4  # 40% chance to leave
    if coalition.size > 7:
        overhead = (coalition.size - 3) * 0.05
        if random.random() < overhead:
            return False
    return True


# --- Marketplace Migration ---

def evaluate_marketplace_migration(agent, marketplaces, round_num):
    """
    Agent considers switching primary marketplace based on earnings performance.
    Only migrates if significantly better option exists.
    """
    if agent.rounds_active < 150 or len(agent.marketplace_ids) == 0:
        return
    if len(marketplaces) < 2:
        return
    # Only evaluate every ~100 rounds
    if random.random() > 0.02:
        return

    current_ids = set(agent.marketplace_ids)
    available = [mp for mp in marketplaces.values() if mp.marketplace_id not in current_ids]
    if not available:
        return

    # Current best marketplace earnings per task
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

    # Evaluate alternatives by marketplace reputation and fee
    for mp in available:
        # Estimate expected earnings: reputation * (1 - fee)
        expected = mp.marketplace_reputation * (1 - mp.platform_fee)
        current_expected = best_current_rate if best_current_rate > 0 else 0.5 * 0.95

        # Only switch if significantly better (>30%)
        if expected > current_expected * 1.3:
            # Add new marketplace
            agent.marketplace_ids.append(mp.marketplace_id)
            mp.join(agent.name)
            agent.migrations += 1
            # Drop worst if on too many (max 2)
            if len(agent.marketplace_ids) > 2 and worst_current_id:
                agent.marketplace_ids.remove(worst_current_id)
                marketplaces[worst_current_id].leave(agent.name)
            break


# --- Strategy Switching (from v6, simplified) ---

def maybe_switch_strategy(agent, agents, round_num, market_prices):
    """Agents can switch between reputation and undercut based on performance."""
    if agent.rounds_active < 200 or agent.rounds_since_switch < 200:
        return
    if random.random() > 0.03:  # 3% chance per round to evaluate
        return

    # Compare to agents with different strategy
    same = [a for a in agents if a.active and a.pricing_strategy.name == agent.pricing_strategy.name
            and a.rounds_active > 100]
    diff = [a for a in agents if a.active and a.pricing_strategy.name != agent.pricing_strategy.name
            and a.rounds_active > 100]

    if not same or not diff:
        return

    avg_same = sum(a.earnings_rate() for a in same) / len(same)
    avg_diff = sum(a.earnings_rate() for a in diff) / len(diff)

    # Switch if other strategy earns >40% more
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
                   enable_strategy_switching=False,
                   marketplace_configs=None,
                   adaptive_marketplace_fees=False,
                   enable_migration=False,
                   seed=42):
    random.seed(seed)
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

    def make_agent(idx, joined_round=0, force_strategy=None):
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

        if force_strategy:
            strategy = force_strategy
        elif random.random() < undercut_fraction:
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
    strategy_counts = []

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

        # Entry (newcomers inherit current undercut_fraction)
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
                    coalitions[a.coalition_id].remove_member(a.name, round_num)
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

        # Coalition formation (every 30 rounds for faster response under adversity)
        if enable_coalitions and round_num % 30 == 0:
            new_coal = try_form_coalition(active_agents, round_num, coalitions,
                                          adversity_mode=adversity_coalitions)
            if new_coal:
                coalitions[new_coal.coalition_id] = new_coal

            for a in active_agents:
                if a.coalition_id and a.coalition_id in coalitions:
                    coal = coalitions[a.coalition_id]
                    if not evaluate_coalition_membership(a, coal, round_num):
                        coal.remove_member(a.name, round_num)
                        a.coalition_id = None

            if round_num % 90 == 0:
                for coal in coalitions.values():
                    if coal.active and coal.treasury > 0:
                        per_member = coal.distribute_treasury()
                        for a in active_agents:
                            if a.coalition_id == coal.coalition_id:
                                a.balance += per_member
                                a.coalition_benefit_received += per_member

        # Marketplace migration
        if enable_migration and round_num % 50 == 0:
            for a in active_agents:
                evaluate_marketplace_migration(a, marketplaces, round_num)

        # Adaptive fees
        if adaptive_marketplace_fees and round_num % 100 == 0:
            for mp in marketplaces.values():
                mp.adapt_fee(round_num)

        # Generate tasks
        base_tasks = random.randint(3, 6)
        demand_mult = 1.0 + 0.01 * len(active_agents)
        num_tasks = int(base_tasks * demand_mult)
        market_prices = tracker.get_prices()

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

            if worker_coalition:
                worker_coalition.shared_reputation.record(domain, success, timestamp=sim_time)
                worker_coalition.total_tasks += 1

            tracker.record(domain, price)
            worker.prices_charged.append(price)
            worker.compute_costs += compute_cost

            if success:
                seller_receives = price
                if marketplace:
                    seller_receives, platform_cut = marketplace.collect_fee(price)
                    marketplace.total_tasks += 1
                    marketplace.total_volume += price
                    marketplace.update_reputation(True)
                    # Track per-marketplace earnings
                    worker.marketplace_earnings[marketplace.marketplace_id] += seller_receives
                    worker.marketplace_tasks[marketplace.marketplace_id] += 1

                if worker_coalition:
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

            snapshots.append({
                "round": round_num,
                "active": len(active_a),
                "gini": gini,
                "newcomer_survival": nc_alive / max(nc_total, 1),
                "reputation_agents": rep_count,
                "undercut_agents": ucut_count,
                "active_coalitions": len(active_coals),
                "coalition_members": coal_members,
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
                print(f"      [{dtype}] {c.coalition_id[:40]} — {c.size} members, "
                      f"{c.total_tasks} tasks, ${c.total_revenue:.2f} rev, "
                      f"${c.treasury:.2f} treasury")

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

    # Snapshot timeline
    if r['snapshots'] and (r.get('total_coalitions_formed', 0) > 0 or r.get('ucut_agents', 0) > 0):
        print(f"\n  Timeline (every 400 rounds):")
        print(f"  {'Round':>6} {'Active':>7} {'Gini':>7} {'Rep':>5} {'Ucut':>5} {'Coals':>6} {'CMemb':>6}")
        print(f"  {'─'*50}")
        for s in r['snapshots']:
            if s['round'] % 400 == 0:
                print(f"  {s['round']:>6} {s['active']:>7} {s['gini']:>7.3f} "
                      f"{s['reputation_agents']:>5} {s['undercut_agents']:>5} "
                      f"{s['active_coalitions']:>6} {s['coalition_members']:>6}")

    print(f"\n  Price Convergence:")
    print(f"  {'Domain':<20} {'Early':>8} {'Late':>8} {'Floor':>8} {'Margin':>8} {'Δ%':>8}")
    print(f"  {'─'*55}")
    for d, pc in sorted(r["price_convergence"].items()):
        print(f"  {d:<20} ${pc['early_avg']:>6.3f} ${pc['late_avg']:>6.3f} "
              f"${pc['compute_cost']:>6.3f} {pc['late_margin']:>7.1%} {pc['price_change_pct']:>+7.1f}%")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def exp1_coalitions_under_adversity():
    """THE key v7 question: do coalitions form as defense against undercutters?"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 1: Coalitions Under Adversity                   ║")
    print("║  50% undercutters — do defensive coalitions emerge?         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Control: 50% undercutters, no coalitions
    print("  [A] 50% undercut, no coalitions...")
    ctrl = run_simulation(
        "50% UNDERCUT — No Coalitions",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.5, enable_coalitions=False,
        seed=42,
    )
    print_result(ctrl)

    # Test: 50% undercutters, defensive coalitions enabled
    print("\n  [B] 50% undercut, defensive coalitions...")
    defense = run_simulation(
        "50% UNDERCUT — Defensive Coalitions",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.5, enable_coalitions=True, adversity_coalitions=True,
        seed=42,
    )
    print_result(defense)

    # Test: 50% undercutters, coalitions + strategy switching
    print("\n  [C] 50% undercut, coalitions + strategy switching...")
    full = run_simulation(
        "50% UNDERCUT — Coalitions + Strategy Switching",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.5, enable_coalitions=True, adversity_coalitions=True,
        enable_strategy_switching=True,
        seed=42,
    )
    print_result(full)

    return ctrl, defense, full


def exp2_undercut_sweep_with_coalitions():
    """How do coalitions respond to varying levels of undercutting pressure?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 2: Undercut Fraction Sweep (with coalitions)    ║")
    print("║  0% → 75% undercutters: when do coalitions matter?          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    results = {}
    for frac in [0.0, 0.25, 0.50, 0.75]:
        print(f"  Running {frac:.0%} undercutters...")
        r = run_simulation(
            f"{frac:.0%} undercut + coalitions",
            num_agents=60, num_rounds=2000, entry_rate=0.04,
            undercut_fraction=frac, enable_coalitions=True, adversity_coalitions=True,
            seed=42,
        )
        results[frac] = r

    print(f"\n{'='*70}")
    print(f"  UNDERCUT SWEEP WITH COALITIONS")
    print(f"{'='*70}")
    print(f"  {'Ucut%':>6} {'Active':>7} {'Gini':>7} {'NC Surv':>8} {'Gap':>6} "
          f"{'Coals':>6} {'CMemb':>6} {'Rep$':>8} {'Ucut$':>8}")
    print(f"  {'─'*75}")
    for frac, r in sorted(results.items()):
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        nc_surv = r['newcomer_survivors'] / max(r['newcomer_total'], 1) * 100
        print(f"  {frac:>5.0%} {r['active']:>7} {r['final_gini']:>7.3f} {nc_surv:>7.0f}% "
              f"{gap:>5.1f}x {r['active_coalitions']:>6} {r['coalition_members']:>6} "
              f"${r['rep_avg_balance']:>6.2f} ${r['ucut_avg_balance']:>6.02f}")

    return results


def exp3_marketplace_migration():
    """Do agents migrate to better marketplaces? What's the equilibrium?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 3: Marketplace Migration                        ║")
    print("║  Agents can switch marketplaces + adaptive fees              ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Static fees, no migration
    print("  [A] Static fees, no migration...")
    static = run_simulation(
        "STATIC: No migration, fixed fees",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        marketplace_configs=[
            {"marketplace_id": "cheap", "name": "CheapMarket", "platform_fee": 0.02,
             "verification_level": 0, "discovery_quality": 0.8},
            {"marketplace_id": "premium", "name": "PremiumMarket", "platform_fee": 0.10,
             "verification_level": 2, "discovery_quality": 1.2, "newcomer_protection": 0.15},
        ],
        seed=42,
    )
    print_result(static)

    # With migration
    print("\n  [B] Migration enabled...")
    migration = run_simulation(
        "MIGRATION: Agents switch marketplaces",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        enable_migration=True,
        marketplace_configs=[
            {"marketplace_id": "cheap", "name": "CheapMarket", "platform_fee": 0.02,
             "verification_level": 0, "discovery_quality": 0.8},
            {"marketplace_id": "premium", "name": "PremiumMarket", "platform_fee": 0.10,
             "verification_level": 2, "discovery_quality": 1.2, "newcomer_protection": 0.15},
        ],
        seed=42,
    )
    print_result(migration)

    # Adaptive fees + migration
    print("\n  [C] Adaptive fees + migration...")
    adaptive = run_simulation(
        "ADAPTIVE: Dynamic fees + migration",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        enable_migration=True, adaptive_marketplace_fees=True,
        marketplace_configs=[
            {"marketplace_id": "cheap", "name": "CheapMarket", "platform_fee": 0.02,
             "verification_level": 0, "discovery_quality": 0.8,
             "target_participant_count": 40},
            {"marketplace_id": "premium", "name": "PremiumMarket", "platform_fee": 0.10,
             "verification_level": 2, "discovery_quality": 1.2, "newcomer_protection": 0.15,
             "target_participant_count": 30},
        ],
        seed=42,
    )
    print_result(adaptive)

    return static, migration, adaptive


def exp4_full_complexity():
    """Everything together: undercutters + coalitions + migration + adaptive fees."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 4: Full Complexity                              ║")
    print("║  Adversity + Coalitions + Migration + Adaptive Fees         ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    r = run_simulation(
        "FULL: 40% undercut + coalitions + migration + adaptive fees",
        num_agents=60, num_rounds=2000, entry_rate=0.04,
        undercut_fraction=0.4,
        enable_coalitions=True, adversity_coalitions=True,
        enable_strategy_switching=True,
        enable_migration=True, adaptive_marketplace_fees=True,
        marketplace_configs=[
            {"marketplace_id": "open", "name": "OpenMarket", "platform_fee": 0.03,
             "verification_level": 0, "discovery_quality": 1.0,
             "target_participant_count": 40},
            {"marketplace_id": "verified", "name": "VerifiedMarket", "platform_fee": 0.08,
             "verification_level": 2, "discovery_quality": 1.3, "newcomer_protection": 0.15,
             "target_participant_count": 30},
            {"marketplace_id": "newcomer", "name": "NewcomerHub", "platform_fee": 0.05,
             "verification_level": 1, "discovery_quality": 1.0, "newcomer_protection": 0.30,
             "target_participant_count": 25},
        ],
        seed=42,
    )
    print_result(r)
    return r


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v8                  ║")
    print("║  Adversity, Migration & Adaptive Dynamics                   ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    e1 = exp1_coalitions_under_adversity()
    e2 = exp2_undercut_sweep_with_coalitions()
    e3 = exp3_marketplace_migration()
    e4 = exp4_full_complexity()

    print("\n\n" + "═" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
