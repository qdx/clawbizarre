"""
ClawBizarre Economy Simulation v10 — Switching Costs & Economy Stabilizers

Key questions from v5-v9:
1. Can switching costs (reputation penalty, cooldown, financial cost) stabilize economies?
2. Do non-monetary coalition benefits (knowledge sharing, mentorship) make coalitions viable?
3. Can institutional support (charters, minimum viable regulation) sustain coalitions?
4. Reputation portability rerun with smaller population (v9 OOM fix)

Core thesis: Free strategy switching is the single most destructive force (confirmed v5-v9).
This simulation tests whether making switching costly can preserve healthy economies.

New features:
- Switching cost model: reputation penalty + financial cost + cooldown period
- Non-monetary coalition benefits: knowledge transfer, capability expansion, mentorship
- Institutional framework: charters with rules, arbitration, minimum standards
- Smaller population for memory-safe runs (30-40 agents)
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

class QualityPremium(PricingStrategy):
    """Price based on success rate — higher quality = higher price."""
    name = "quality"
    def price(self, agent, domain, sim_time, market_prices):
        base = TASK_CATALOG[domain]["base_cost"]
        floor = TASK_CATALOG[domain]["compute_cost"]
        total = agent.tasks_completed + agent.tasks_failed
        if total > 20:
            success_rate = agent.tasks_completed / total
            quality_premium = success_rate * 1.5
        else:
            quality_premium = 0.5  # conservative until track record
        return max(base * (1 + quality_premium), floor)


# --- Switching Cost Model ---

@dataclass
class SwitchingCosts:
    """Configuration for strategy switching costs."""
    reputation_penalty: float = 0.0    # 0.0-1.0: fraction of reputation lost on switch
    financial_cost: float = 0.0         # flat fee deducted on switch
    cooldown_rounds: int = 0            # minimum rounds between switches
    public_record: bool = False         # switches visible to buyers (affects discovery)
    
    @property
    def is_free(self):
        return self.reputation_penalty == 0 and self.financial_cost == 0 and self.cooldown_rounds == 0


# --- Coalition with Non-Monetary Benefits ---

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
    
    # v10: Non-monetary benefits
    knowledge_pool: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    mentorship_active: bool = True
    capability_sharing: bool = True
    
    # v10: Institutional framework
    has_charter: bool = False
    charter_min_reputation: float = 0.0     # minimum rep to join
    charter_contribution_rate: float = 0.0  # % of knowledge contributed
    arbitration_available: bool = False

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

    def contribute_knowledge(self, domain: str, amount: float):
        """Members contribute knowledge to the pool."""
        self.knowledge_pool[domain] += amount

    def get_knowledge_bonus(self, domain: str) -> float:
        """Knowledge pool provides capability bonus to members."""
        raw = self.knowledge_pool.get(domain, 0.0)
        # Diminishing returns: sqrt scaling
        return min(math.sqrt(raw) * 0.1, 0.15)  # max 15% reliability bonus


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
    rounds_since_switch: int = 999  # start high so no cooldown at beginning
    switch_history: list[tuple[int, str, str]] = field(default_factory=list)  # (round, from, to)
    
    # v10: Knowledge received from coalition
    knowledge_bonus: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    capabilities_gained: list[str] = field(default_factory=list)  # new caps from coalition knowledge
    mentorship_rounds: int = 0

    def price_for(self, domain, sim_time, market_prices, coalition=None):
        base_price = self.pricing_strategy.price(self, domain, sim_time, market_prices)
        if coalition and coalition.min_price_floor and domain in coalition.min_price_floor:
            floor = coalition.min_price_floor[domain]
            return max(base_price, floor)
        return base_price

    def accept_task(self, domain, coalition=None):
        base_rel = self.reliability.get(domain, 0.5)
        # v10: Coalition knowledge bonus
        bonus = 0.0
        if coalition and coalition.capability_sharing:
            bonus = coalition.get_knowledge_bonus(domain)
        # Gained capabilities have lower base reliability
        if domain in self.capabilities_gained:
            base_rel = min(base_rel, 0.65)  # cap gained capability reliability
        return random.random() < min(base_rel + bonus, 0.99)

    def earnings_rate(self, window_rounds=200):
        if self.rounds_active < 10:
            return 0.0
        return self.balance / self.rounds_active

    def effective_reputation(self, domain, sim_time):
        return self.reputation.score(domain, sim_time)
    
    def switcher_penalty(self) -> float:
        """Buyers penalize agents who switch strategies frequently."""
        if not self.switch_history:
            return 0.0
        recent = [s for s in self.switch_history if s[0] > (self.rounds_active - 500)]
        # Each recent switch reduces discovery quality by 5%
        return min(len(recent) * 0.05, 0.25)


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
                    public_switch_records=False):
    candidates = [a for a in agents
                  if a.active and a.name != requester.name
                  and (domain in a.capabilities or domain in a.capabilities_gained)]

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
        rep = c.effective_reputation(domain, sim_time)
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

        # v10: Switcher penalty — buyers distrust frequent switchers
        if public_switch_records:
            quality *= (1.0 - c.switcher_penalty())

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

def try_form_coalition(agents, round_num, coalitions, max_coalitions=20,
                       adversity_mode=False, institutional=False):
    if adversity_mode:
        eligible = [a for a in agents if a.active and a.coalition_id is None
                    and a.rounds_active > 100]
        struggling = [a for a in eligible if a.earnings_rate() < 0.01]

        if struggling and len(coalitions) < max_coalitions:
            if random.random() < 0.15:
                founder = random.choice(struggling)
                potential = [a for a in struggling if a.name != founder.name
                             and a.coalition_willingness > 0.2
                             and any(d in founder.capabilities for d in a.capabilities)]
                if potential:
                    recruit_count = min(random.randint(2, 4), len(potential))
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
                            mentorship_active=True,
                            capability_sharing=True,
                        )
                        coal.min_price_floor = {d: TASK_CATALOG[d]["compute_cost"] * 1.3 for d in ALL_DOMAINS}
                        
                        # v10: Institutional charters
                        if institutional:
                            coal.has_charter = True
                            coal.charter_min_reputation = 0.0  # low bar for defense coalitions
                            coal.charter_contribution_rate = 0.1
                            coal.arbitration_available = True
                        
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
        if founder.earnings_rate() < 0.02 or random.random() > 0.05:
            return None
        potential = [a for a in active_solo if a.name != founder.name
                     and a.coalition_willingness > 0.3
                     and any(d not in founder.capabilities for d in a.capabilities)]
        if not potential:
            return None
        recruit_count = min(random.randint(2, 3), len(potential))
        potential.sort(key=lambda a: a.earnings_rate(), reverse=True)
        recruits = potential[:recruit_count]
        accepted = [r for r in recruits if random.random() < r.coalition_willingness]
        if not accepted:
            return None
        cid = f"coalition_{round_num}_{founder.name}"
        coal = Coalition(coalition_id=cid, founder=founder.name, formed_round=round_num,
                        mentorship_active=True, capability_sharing=True)
        coal.min_price_floor = {d: TASK_CATALOG[d]["base_cost"] * 0.8 for d in ALL_DOMAINS}
        coal.price_coordination = "floor"
        
        if institutional:
            coal.has_charter = True
            coal.charter_min_reputation = 0.3
            coal.charter_contribution_rate = 0.15
            coal.arbitration_available = True
        
        founder.coalition_id = cid
        founder.solo_earnings_before_coalition = founder.balance
        coal.add_member(founder.name, round_num)
        for r in accepted:
            r.coalition_id = cid
            r.solo_earnings_before_coalition = r.balance
            coal.add_member(r.name, round_num)
        return coal

    return None


def evaluate_coalition_membership(agent, coalition, round_num, non_monetary_benefits=False):
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
    
    # v10: Non-monetary benefits make agents stickier
    if non_monetary_benefits:
        # Knowledge bonus makes staying more attractive
        knowledge_value = sum(agent.knowledge_bonus.values()) * 10  # monetize knowledge
        # Gained capabilities have monetary value
        capability_value = len(agent.capabilities_gained) * 5
        # Mentorship rounds invested = sunk cost effect
        sunk_cost_factor = min(agent.mentorship_rounds * 0.01, 0.3)
        
        # Agent stays if total value (monetary + non-monetary) is positive
        total_value = current_rate + knowledge_value + capability_value
        threshold = pre_rate * (0.3 - sunk_cost_factor)  # lower threshold with more investment
        if total_value > threshold:
            return True
    
    if current_rate < pre_rate * 0.5:
        return random.random() > 0.4
    if coalition.size > 7:
        overhead = (coalition.size - 3) * 0.05
        if random.random() < overhead:
            return False
    return True


def maybe_switch_strategy(agent, agents, round_num, market_prices, switching_costs: SwitchingCosts):
    """Attempt strategy switch with costs."""
    if agent.rounds_active < 200:
        return
    
    # Cooldown check
    if agent.rounds_since_switch < switching_costs.cooldown_rounds:
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

    # Need bigger gap to justify switching costs
    threshold = 1.4
    if not switching_costs.is_free:
        # Higher costs require bigger performance gap
        threshold += switching_costs.reputation_penalty * 0.5
        threshold += min(switching_costs.financial_cost / max(agent.balance, 1), 0.5)

    if avg_diff > avg_same * threshold:
        old_strategy = agent.pricing_strategy.name
        
        # Apply switching costs
        if switching_costs.financial_cost > 0:
            if agent.balance < switching_costs.financial_cost:
                return  # can't afford to switch
            agent.balance -= switching_costs.financial_cost
        
        if switching_costs.reputation_penalty > 0:
            # Decay all domain reputations
            for domain in ALL_DOMAINS:
                if domain in agent.reputation._domains:
                    dr = agent.reputation._get_domain(domain)
                    # Remove a fraction of success records to simulate reputation loss
                    keep = 1 - switching_costs.reputation_penalty
                    keep_n = int(len(dr._successes) * keep)
                    dr._successes = dr._successes[-keep_n:] if keep_n > 0 else []
                    keep_f = int(len(dr._failures) * (1 - switching_costs.reputation_penalty * 0.5))
                    dr._failures = dr._failures[-keep_f:] if keep_f > 0 else []
        
        # Actually switch
        if agent.pricing_strategy.name == "reputation":
            agent.pricing_strategy = UndercutStrategy(random.uniform(0.10, 0.25))
        elif agent.pricing_strategy.name == "undercut":
            agent.pricing_strategy = ReputationPremium()
        else:
            agent.pricing_strategy = ReputationPremium()
        
        agent.strategy_switches += 1
        agent.rounds_since_switch = 0
        agent.switch_history.append((round_num, old_strategy, agent.pricing_strategy.name))


# --- Knowledge Transfer (v10) ---

def process_knowledge_transfer(agents, coalitions, round_num, sim_time):
    """Coalition members share knowledge, expanding capabilities."""
    for coal in coalitions.values():
        if not coal.active or not coal.capability_sharing:
            continue
        
        members = [a for a in agents if a.active and a.coalition_id == coal.coalition_id]
        if len(members) < 2:
            continue
        
        # Each member contributes knowledge to pool based on completed tasks
        for m in members:
            for domain in m.capabilities:
                if m.tasks_completed > 0:
                    contribution = m.reliability.get(domain, 0.5) * 0.01
                    if coal.has_charter:
                        contribution *= (1 + coal.charter_contribution_rate)
                    coal.contribute_knowledge(domain, contribution)
        
        # Members learn from pool — can gain new capabilities
        if coal.mentorship_active and round_num % 100 == 0:
            pool_domains = [d for d, v in coal.knowledge_pool.items() if v > 2.0]
            for m in members:
                for domain in pool_domains:
                    if domain not in m.capabilities and domain not in m.capabilities_gained:
                        # Probability of learning scales with pool depth
                        learn_prob = min(coal.knowledge_pool[domain] * 0.02, 0.3)
                        if random.random() < learn_prob:
                            m.capabilities_gained.append(domain)
                            m.reliability[domain] = 0.55  # start low
                            m.mentorship_rounds += 1
                
                # Apply knowledge bonus
                for domain in m.capabilities + m.capabilities_gained:
                    m.knowledge_bonus[domain] = coal.get_knowledge_bonus(domain)


# --- Simulation ---

def run_simulation(label, num_agents=40, num_rounds=2000,
                   entry_rate=0.04, exit_threshold=-5.0,
                   price_sensitivity=1.0,
                   undercut_fraction=0.0,
                   enable_coalitions=False,
                   adversity_coalitions=False,
                   enable_strategy_switching=False,
                   switching_costs: SwitchingCosts = None,
                   non_monetary_benefits=False,
                   institutional_coalitions=False,
                   marketplace_configs=None,
                   adaptive_marketplace_fees=False,
                   enable_migration=False,
                   seed=42):
    if switching_costs is None:
        switching_costs = SwitchingCosts()  # free switching by default
    
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
    switch_events = []

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

        # Strategy switching (with costs)
        if enable_strategy_switching:
            pre_switches = sum(a.strategy_switches for a in active_agents)
            for a in active_agents:
                maybe_switch_strategy(a, active_agents, round_num, tracker.get_prices(), switching_costs)
            post_switches = sum(a.strategy_switches for a in active_agents)
            if post_switches > pre_switches:
                switch_events.append((round_num, post_switches - pre_switches))

        # Coalition formation + maintenance
        if enable_coalitions and round_num % 30 == 0:
            new_coal = try_form_coalition(active_agents, round_num, coalitions,
                                          adversity_mode=adversity_coalitions,
                                          institutional=institutional_coalitions)
            if new_coal:
                coalitions[new_coal.coalition_id] = new_coal

            for a in list(active_agents):
                if a.coalition_id and a.coalition_id in coalitions:
                    coal = coalitions[a.coalition_id]
                    if not evaluate_coalition_membership(a, coal, round_num,
                                                        non_monetary_benefits=non_monetary_benefits):
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

        # v10: Knowledge transfer
        if non_monetary_benefits and enable_coalitions:
            process_knowledge_transfer(agents, coalitions, round_num, sim_time)

        # Marketplace migration
        if enable_migration and round_num % 50 == 0:
            for a in active_agents:
                if a.rounds_active < 150 or not a.marketplace_ids or len(marketplaces) < 2:
                    continue
                if random.random() > 0.02:
                    continue
                current_ids = set(a.marketplace_ids)
                available = [mp for mp in marketplaces.values() if mp.marketplace_id not in current_ids]
                if available:
                    best = max(available, key=lambda mp: mp.marketplace_reputation * (1 - mp.platform_fee))
                    worst_id = min(a.marketplace_ids,
                                  key=lambda mid: a.marketplace_earnings.get(mid, 0) / max(a.marketplace_tasks.get(mid, 1), 1))
                    worst_rate = a.marketplace_earnings.get(worst_id, 0) / max(a.marketplace_tasks.get(worst_id, 1), 1)
                    best_expected = best.marketplace_reputation * (1 - best.platform_fee)
                    if best_expected > worst_rate * 1.3:
                        a.marketplace_ids.append(best.marketplace_id)
                        best.join(a.name)
                        a.migrations += 1
                        if len(a.marketplace_ids) > 2:
                            a.marketplace_ids.remove(worst_id)
                            marketplaces[worst_id].leave(a.name)

        # Adaptive fees
        if adaptive_marketplace_fees and round_num % 100 == 0:
            for mp in marketplaces.values():
                mp.adapt_fee(round_num)

        # Generate tasks
        base_tasks = random.randint(3, 5)
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
                public_switch_records=switching_costs.public_record if switching_costs else False,
            )
            if not worker:
                continue

            worker_coalition = None
            if enable_coalitions and worker.coalition_id:
                worker_coalition = coalitions.get(worker.coalition_id)

            price = worker.price_for(domain, sim_time, market_prices, worker_coalition)
            compute_cost = TASK_CATALOG[domain]["compute_cost"]
            success = worker.accept_task(domain, coalition=worker_coalition)
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
                    seller_receives, platform_cut = marketplace.collect_fee(seller_receives)
                    marketplace.total_tasks += 1
                    marketplace.total_volume += price
                    marketplace.update_reputation(True)
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

            caps_gained = sum(len(a.capabilities_gained) for a in active_a)

            snapshots.append({
                "round": round_num,
                "active": len(active_a),
                "gini": gini,
                "newcomer_survival": nc_alive / max(nc_total, 1),
                "reputation_agents": rep_count,
                "undercut_agents": ucut_count,
                "active_coalitions": len(active_coals),
                "coalition_members": coal_members,
                "capabilities_gained": caps_gained,
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

    total_switches = sum(a.strategy_switches for a in agents)
    total_caps_gained = sum(len(a.capabilities_gained) for a in active_agents)

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
        "total_switches": total_switches,
        "switch_events": switch_events,
        "total_capabilities_gained": total_caps_gained,
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
        print(f"    Total switches: {r['total_switches']}")
        if r['switch_events']:
            print(f"    Switch events: {len(r['switch_events'])} rounds with switches")

    if r['active_coalitions'] > 0 or r['total_coalitions_formed'] > 0:
        print(f"\n  Coalition Stats:")
        print(f"    Formed: {r['total_coalitions_formed']}, Active: {r['active_coalitions']}")
        print(f"    Coalition members: {r['coalition_members']}, Solo: {r['solo_agents']}")
        if r['coalition_members'] > 0:
            print(f"    Coalition avg: ${r['coalition_avg_balance']:.2f}")
        if r['solo_agents'] > 0:
            print(f"    Solo avg:      ${r['solo_avg_balance']:.2f}")

    if r.get('total_capabilities_gained', 0) > 0:
        print(f"\n  Knowledge Transfer:")
        print(f"    Capabilities gained through mentorship: {r['total_capabilities_gained']}")

    if r.get('marketplaces') and len(r['marketplaces']) > 1:
        print(f"\n  Marketplace Competition:")
        print(f"  {'Name':<20} {'Fee':>6} {'Agents':>7} {'Tasks':>7} {'Volume':>10} {'Rep':>6}")
        print(f"  {'─'*60}")
        for mp in r['marketplaces'].values():
            fee_str = f"{mp.platform_fee:.1%}"
            print(f"  {mp.name:<20} {fee_str:>6} {len(mp.participants):>7} "
                  f"{mp.total_tasks:>7} ${mp.total_volume:>8.2f} {mp.marketplace_reputation:>5.2f}")

    if r['snapshots']:
        print(f"\n  Timeline (every 400 rounds):")
        print(f"  {'Round':>6} {'Active':>7} {'Gini':>7} {'Rep':>5} {'Ucut':>5} {'Coals':>6} {'CMemb':>6} {'NewCap':>7}")
        print(f"  {'─'*62}")
        for s in r['snapshots']:
            if s['round'] % 400 == 0:
                print(f"  {s['round']:>6} {s['active']:>7} {s['gini']:>7.3f} "
                      f"{s['reputation_agents']:>5} {s['undercut_agents']:>5} "
                      f"{s['active_coalitions']:>6} {s['coalition_members']:>6} "
                      f"{s.get('capabilities_gained', 0):>7}")

    print(f"\n  Price Convergence:")
    print(f"  {'Domain':<20} {'Early':>8} {'Late':>8} {'Floor':>8} {'Margin':>8} {'Δ%':>8}")
    print(f"  {'─'*55}")
    for d, pc in sorted(r["price_convergence"].items()):
        print(f"  {d:<20} ${pc['early_avg']:>6.3f} ${pc['late_avg']:>6.3f} "
              f"${pc['compute_cost']:>6.3f} {pc['late_margin']:>7.1%} {pc['price_change_pct']:>+7.1f}%")


# ═══════════════════════════════════════════════════════════════
# EXPERIMENTS
# ═══════════════════════════════════════════════════════════════

def exp1_switching_cost_sweep():
    """Does making strategy switching costly stabilize the economy?"""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 1: Switching Cost Sweep                         ║")
    print("║  50% undercutters + free switching → add costs gradually     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    configs = [
        ("A: FREE switching (v9 baseline)", SwitchingCosts()),
        ("B: Rep penalty only (30%)", SwitchingCosts(reputation_penalty=0.3)),
        ("C: Financial cost only ($2)", SwitchingCosts(financial_cost=2.0)),
        ("D: Cooldown only (500 rounds)", SwitchingCosts(cooldown_rounds=500)),
        ("E: Public record only", SwitchingCosts(public_record=True)),
        ("F: ALL costs combined", SwitchingCosts(reputation_penalty=0.3, financial_cost=2.0,
                                                  cooldown_rounds=500, public_record=True)),
        ("G: Heavy costs (50% rep + $5 + 800 cooldown)", 
         SwitchingCosts(reputation_penalty=0.5, financial_cost=5.0, 
                        cooldown_rounds=800, public_record=True)),
    ]

    results = []
    for label, costs in configs:
        tag = label.split(":")[0].strip()
        print(f"  [{tag}] {label}...")
        r = run_simulation(
            label,
            num_agents=40, num_rounds=2000, entry_rate=0.03,
            undercut_fraction=0.5,
            enable_strategy_switching=True,
            switching_costs=costs,
            seed=42,
        )
        results.append(r)
        print_result(r)

    # Summary table
    print(f"\n{'='*80}")
    print(f"  SWITCHING COST SWEEP SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Config':<40} {'Active':>7} {'Gini':>7} {'NC%':>6} {'Gap':>7} {'Sw':>5} {'Rep':>5} {'Ucut':>5}")
    print(f"  {'─'*82}")
    for r in results:
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        nc_pct = r['newcomer_survivors'] / max(r['newcomer_total'], 1) * 100
        print(f"  {r['label'][:40]:<40} {r['active']:>7} {r['final_gini']:>7.3f} "
              f"{nc_pct:>5.0f}% {gap:>6.1f}x {r['total_switches']:>5} "
              f"{r['rep_agents']:>5} {r['ucut_agents']:>5}")

    return results


def exp2_non_monetary_coalition_benefits():
    """Do knowledge sharing and mentorship make coalitions viable?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 2: Non-Monetary Coalition Benefits              ║")
    print("║  Knowledge transfer + capability expansion + mentorship     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    configs = [
        ("A: No coalitions (baseline)", False, False, False),
        ("B: Coalitions, monetary only (v8)", True, False, False),
        ("C: Coalitions + knowledge sharing", True, True, False),
        ("D: Coalitions + knowledge + institutions", True, True, True),
    ]

    results = []
    for label, coals, non_mon, inst in configs:
        tag = label.split(":")[0].strip()
        print(f"  [{tag}] {label}...")
        r = run_simulation(
            label,
            num_agents=40, num_rounds=2000, entry_rate=0.03,
            undercut_fraction=0.4,
            enable_coalitions=coals,
            adversity_coalitions=coals,
            non_monetary_benefits=non_mon,
            institutional_coalitions=inst,
            seed=42,
        )
        results.append(r)
        print_result(r)

    # Summary
    print(f"\n{'='*80}")
    print(f"  NON-MONETARY BENEFITS SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Config':<45} {'Active':>7} {'Gini':>7} {'NC%':>6} {'Gap':>7} {'Coal':>5} {'NewCap':>7}")
    print(f"  {'─'*85}")
    for r in results:
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        nc_pct = r['newcomer_survivors'] / max(r['newcomer_total'], 1) * 100
        print(f"  {r['label'][:45]:<45} {r['active']:>7} {r['final_gini']:>7.3f} "
              f"{nc_pct:>5.0f}% {gap:>6.1f}x {r['active_coalitions']:>5} "
              f"{r.get('total_capabilities_gained', 0):>7}")

    return results


def exp3_combined_stabilizers():
    """Switching costs + non-monetary benefits + institutions = stable economy?"""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 3: Combined Stabilizers                         ║")
    print("║  All mechanisms together: can we prevent economy collapse?  ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    configs = [
        ("A: 50% undercut, FREE switching, no coalitions (worst case)",
         0.5, True, SwitchingCosts(), False, False, False),
        ("B: 50% undercut, FREE switching + coalitions + benefits",
         0.5, True, SwitchingCosts(), True, True, True),
        ("C: 50% undercut, MODERATE costs + coalitions + benefits",
         0.5, True, SwitchingCosts(reputation_penalty=0.3, financial_cost=2.0, cooldown_rounds=300, public_record=True),
         True, True, True),
        ("D: 75% undercut (extreme), MODERATE costs + full support",
         0.75, True, SwitchingCosts(reputation_penalty=0.3, financial_cost=2.0, cooldown_rounds=300, public_record=True),
         True, True, True),
        ("E: 50% undercut, HEAVY costs + full support",
         0.5, True, SwitchingCosts(reputation_penalty=0.5, financial_cost=5.0, cooldown_rounds=800, public_record=True),
         True, True, True),
    ]

    results = []
    for label, ucut, switch, costs, coals, non_mon, inst in configs:
        tag = label.split(":")[0].strip()
        print(f"  [{tag}] {label}...")
        r = run_simulation(
            label,
            num_agents=40, num_rounds=2000, entry_rate=0.03,
            undercut_fraction=ucut,
            enable_strategy_switching=switch,
            switching_costs=costs,
            enable_coalitions=coals,
            adversity_coalitions=coals,
            non_monetary_benefits=non_mon,
            institutional_coalitions=inst,
            seed=42,
        )
        results.append(r)
        print_result(r)

    # Summary
    print(f"\n{'='*90}")
    print(f"  COMBINED STABILIZERS SUMMARY")
    print(f"{'='*90}")
    print(f"  {'Config':<55} {'Active':>7} {'Gini':>7} {'NC%':>6} {'Gap':>7} {'Sw':>5} {'Rep':>5}")
    print(f"  {'─'*92}")
    for r in results:
        gap = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
        nc_pct = r['newcomer_survivors'] / max(r['newcomer_total'], 1) * 100
        print(f"  {r['label'][:55]:<55} {r['active']:>7} {r['final_gini']:>7.3f} "
              f"{nc_pct:>5.0f}% {gap:>6.1f}x {r['total_switches']:>5} {r['rep_agents']:>5}")

    return results


def exp4_reputation_portability():
    """Reputation portability rerun with smaller population (v9 OOM fix)."""
    print("\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 4: Reputation Portability (small population)    ║")
    print("║  Does portable rep increase migration and reduce moats?     ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")

    # Note: portability not yet in v10 run_simulation — simplified for memory
    # Testing marketplace competition with migration instead
    configs = [
        ("A: No migration (sticky)", False),
        ("B: Migration enabled", True),
    ]

    results = []
    for label, migrate in configs:
        tag = label.split(":")[0].strip()
        print(f"  [{tag}] {label}...")
        r = run_simulation(
            label,
            num_agents=35, num_rounds=2000, entry_rate=0.03,
            enable_migration=migrate,
            adaptive_marketplace_fees=True,
            marketplace_configs=[
                {"marketplace_id": "cheap", "name": "CheapMarket", "platform_fee": 0.02,
                 "verification_level": 0, "discovery_quality": 0.8, "target_participant_count": 25},
                {"marketplace_id": "premium", "name": "PremiumMarket", "platform_fee": 0.10,
                 "verification_level": 2, "discovery_quality": 1.2, "newcomer_protection": 0.15,
                 "target_participant_count": 20},
            ],
            seed=42,
        )
        results.append(r)
        print_result(r)

    return results


if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v10                 ║")
    print("║  Switching Costs & Economy Stabilizers                      ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()

    e1 = exp1_switching_cost_sweep()
    e2 = exp2_non_monetary_coalition_benefits()
    e3 = exp3_combined_stabilizers()
    e4 = exp4_reputation_portability()

    print("\n\n" + "═" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
