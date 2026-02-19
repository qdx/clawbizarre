"""
ClawBizarre Economy Simulation v4
New features over v3:
- Quadratic coordination penalty (O(n²) communication overhead)
- Cold start interventions: newcomer discovery bonus, subsidized pricing, mentorship
- A/B comparison: market WITH vs WITHOUT interventions
- Market maker agents (discovery-as-a-service)
"""

import random
import math
import time
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from identity import AgentIdentity
from reputation import DecayingReputation, DomainReputation, MerkleTree, sha256


# --- Market Configuration ---

TASK_CATALOG = {
    "code_review": {"base_cost": 0.50, "complexity": 0.7},
    "translation": {"base_cost": 0.30, "complexity": 0.5},
    "summarization": {"base_cost": 0.20, "complexity": 0.3},
    "data_validation": {"base_cost": 0.15, "complexity": 0.2},
    "research": {"base_cost": 1.00, "complexity": 0.9},
    "monitoring": {"base_cost": 0.10, "complexity": 0.1},
}

DOMAIN_CORRELATIONS = {
    ("code_review", "research"): 0.5,
    ("code_review", "data_validation"): 0.3,
    ("translation", "summarization"): 0.4,
    ("research", "summarization"): 0.6,
    ("monitoring", "data_validation"): 0.3,
}

ALL_DOMAINS = list(TASK_CATALOG.keys())


# --- Coordination Penalty Models ---

def linear_penalty(fleet_size: int) -> float:
    """v3 model: (n-1) * 0.5%, capped at 15%"""
    return min((fleet_size - 1) * 0.005, 0.15)

def quadratic_penalty(fleet_size: int) -> float:
    """O(n²) communication channels: n*(n-1)/2 pairs, each costs 0.2%
    At size 2: 0.2%, size 5: 2.0%, size 10: 9.0%, size 15: 21.0%, size 20: 38.0%
    """
    pairs = fleet_size * (fleet_size - 1) / 2
    return min(pairs * 0.002, 0.50)  # cap at 50%

def sqrt_penalty(fleet_size: int) -> float:
    """Sublinear: some coordination is amortized (shared tools, docs)"""
    return min(math.sqrt(fleet_size - 1) * 0.02, 0.25)


@dataclass
class MarketAgent:
    name: str
    identity: AgentIdentity
    capabilities: list[str]
    reliability: dict[str, float]
    reputation: DomainReputation = field(default_factory=lambda: DomainReputation(
        half_life_days=30, domain_correlations=DOMAIN_CORRELATIONS
    ))
    fleet_id: Optional[str] = None
    
    # Economics
    balance: float = 0.0
    spent: float = 0.0
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_requested: int = 0
    revenue_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    receipt_hashes: list[str] = field(default_factory=list)
    
    # Lifecycle
    joined_round: int = 0
    exited_round: Optional[int] = None
    active: bool = True
    identity_cost_per_round: float = 0.01
    
    # Cold start
    is_newcomer: bool = False  # True for first 200 rounds after joining
    mentor: Optional[str] = None  # Name of mentoring agent
    
    # Market maker
    is_market_maker: bool = False
    matchmaking_revenue: float = 0.0
    matches_made: int = 0
    
    def price_for(self, domain: str, sim_time: float) -> float:
        base = TASK_CATALOG[domain]["base_cost"]
        rep_score = self.reputation.score(domain, sim_time)
        conf = self.reputation._get_domain(domain).confidence(sim_time) if domain in self.reputation._domains else 0.0
        rep_premium = rep_score * conf * 2.0
        return base * (1 + rep_premium)
    
    def accept_task(self, domain: str) -> bool:
        return random.random() < self.reliability.get(domain, 0.5)
    
    def net_profit(self) -> float:
        return self.balance - self.spent


@dataclass
class ColdStartConfig:
    """Configuration for cold start interventions."""
    enabled: bool = False
    # Discovery bonus: newcomers get boosted visibility in discovery
    discovery_bonus: float = 0.5  # multiplier on discovery score
    discovery_duration: int = 200  # rounds
    # Subsidized pricing: newcomers price lower to attract first tasks
    subsidized_discount: float = 0.30  # 30% off
    subsidy_duration: int = 100  # rounds
    # Mentorship: established agent vouches for newcomer
    mentorship_enabled: bool = False
    mentorship_rep_transfer: float = 0.2  # fraction of mentor's rep transferred
    # Protected discovery slots: reserve X% of discovery for newcomers
    protected_slots: float = 0.15  # 15% of discoveries go to newcomers


@dataclass
class MarketMakerConfig:
    """Configuration for market maker agents."""
    enabled: bool = False
    num_makers: int = 2
    commission_rate: float = 0.05  # 5% of transaction value
    discovery_boost: float = 2.0  # how much better their matching is


def make_agent(idx: int, joined_round: int = 0, fleet_id: Optional[str] = None) -> MarketAgent:
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
    
    return MarketAgent(
        name=f"Agent_{idx}",
        identity=AgentIdentity.generate(),
        capabilities=caps,
        reliability={d: random.uniform(*rel_range) for d in caps},
        fleet_id=fleet_id,
        joined_round=joined_round,
        is_newcomer=(joined_round > 0),
    )


def discover_worker(agents: list[MarketAgent], requester: MarketAgent,
                    domain: str, sim_time: float, round_num: int,
                    cold_start: ColdStartConfig = ColdStartConfig(),
                    market_makers: list[MarketAgent] = []) -> tuple[Optional[MarketAgent], Optional[MarketAgent]]:
    """
    Reputation-weighted discovery with cold start interventions.
    Returns (worker, market_maker_used) tuple.
    """
    candidates = [a for a in agents 
                  if a.active and a.name != requester.name 
                  and domain in a.capabilities and not a.is_market_maker]
    if not candidates:
        return None, None
    
    # Protected slots: force newcomer selection X% of the time
    newcomers = [c for c in candidates if c.is_newcomer and (round_num - c.joined_round) < cold_start.discovery_duration]
    if cold_start.enabled and cold_start.protected_slots > 0 and newcomers and random.random() < cold_start.protected_slots:
        # Pick a random newcomer for this protected slot
        return random.choice(newcomers), None
    
    scored = []
    for c in candidates:
        rep = c.reputation.score(domain, sim_time)
        conf = c.reputation._get_domain(domain).confidence(sim_time) if domain in c.reputation._domains else 0.0
        price = c.price_for(domain, sim_time)
        
        # Subsidized pricing for newcomers
        if cold_start.enabled and c.is_newcomer and (round_num - c.joined_round) < cold_start.subsidy_duration:
            price *= (1 - cold_start.subsidized_discount)
        
        exploration_bonus = 0.3 if conf < 0.2 else 0.1 if conf < 0.4 else 0.0
        value = (rep * max(conf, 0.1) + exploration_bonus) / max(price, 0.01)
        
        # Discovery bonus for newcomers
        if cold_start.enabled and c.is_newcomer and (round_num - c.joined_round) < cold_start.discovery_duration:
            value *= (1 + cold_start.discovery_bonus)
        
        # Mentorship bonus
        if cold_start.enabled and cold_start.mentorship_enabled and c.mentor:
            mentor_agent = next((a for a in agents if a.name == c.mentor and a.active), None)
            if mentor_agent:
                mentor_rep = mentor_agent.reputation.score(domain, sim_time)
                value += mentor_rep * cold_start.mentorship_rep_transfer
        
        # Fleet preference
        if requester.fleet_id and c.fleet_id == requester.fleet_id:
            value *= 1.5
        
        scored.append((c, value))
    
    # Market maker boost: if a market maker is available, they improve matching quality
    maker_used = None
    if market_makers:
        active_makers = [m for m in market_makers if m.active]
        if active_makers:
            maker_used = random.choice(active_makers)
            # Market makers boost the best candidates more (better matching)
            scored.sort(key=lambda x: x[1], reverse=True)
            top_n = max(1, len(scored) // 3)
            scored = [(c, v * maker_used.is_market_maker and 1.0 or 1.0) for c, v in scored]
            # Actually: market makers reduce noise in selection (higher temperature = worse matching)
    
    temperature = 0.5 if not maker_used else 0.2  # Market makers = sharper selection
    values = [v for _, v in scored]
    max_v = max(values)
    exp_values = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]
    
    chosen = random.choices([c for c, _ in scored], weights=probs, k=1)[0]
    return chosen, maker_used


def assign_mentors(agents: list[MarketAgent], newcomer: MarketAgent):
    """Assign a high-reputation mentor to a newcomer."""
    established = [a for a in agents if a.active and not a.is_newcomer 
                   and a.tasks_completed > 20 and a.name != newcomer.name]
    if established:
        # Pick mentor with most domain overlap
        best = max(established, key=lambda a: len(set(a.capabilities) & set(newcomer.capabilities)))
        newcomer.mentor = best.name


def run_simulation(label: str, num_initial: int = 30, num_rounds: int = 2000,
                   entry_rate: float = 0.05, exit_threshold: float = -5.0,
                   penalty_fn=quadratic_penalty,
                   cold_start: ColdStartConfig = ColdStartConfig(),
                   market_maker_cfg: MarketMakerConfig = MarketMakerConfig(),
                   fleet_size: int = 4, seed: int = 42):
    """
    General-purpose simulation with configurable interventions.
    """
    random.seed(seed)
    sim_start = time.time() - 84 * 86400
    
    agent_idx = 0
    agents = []
    market_makers = []
    
    # Create initial agents
    for i in range(num_initial):
        a = make_agent(agent_idx)
        a.is_newcomer = False
        agents.append(a)
        agent_idx += 1
    
    # Assign fleets
    for i in range(0, num_initial - fleet_size + 1, fleet_size):
        fid = f"Fleet_{i // fleet_size}"
        for j in range(i, i + fleet_size):
            agents[j].fleet_id = fid
    
    # Create market makers if enabled
    if market_maker_cfg.enabled:
        for i in range(market_maker_cfg.num_makers):
            mm = MarketAgent(
                name=f"MarketMaker_{i}",
                identity=AgentIdentity.generate(),
                capabilities=ALL_DOMAINS,
                reliability={d: 0.0 for d in ALL_DOMAINS},  # they don't do work
                is_market_maker=True,
                joined_round=0,
            )
            market_makers.append(mm)
    
    # Tracking
    snapshots = []  # (round, active, total, exited, gini, newcomer_survival_rate)
    cohort_data = defaultdict(lambda: {"entered": 0, "survived": 0, "avg_balance": 0.0})
    
    for round_num in range(1, num_rounds + 1):
        sim_time = sim_start + round_num * 3600
        active_agents = [a for a in agents if a.active]
        
        # Identity cost
        for a in active_agents:
            a.balance -= a.identity_cost_per_round
        
        # Update newcomer status (graduate after 200 rounds)
        for a in active_agents:
            if a.is_newcomer and (round_num - a.joined_round) > 200:
                a.is_newcomer = False
        
        # Agent entry
        num_entries = 0
        while random.random() < entry_rate:
            num_entries += 1
        if len(active_agents) < 10:
            num_entries = max(num_entries, 2)
        for _ in range(num_entries):
            new_agent = make_agent(agent_idx, joined_round=round_num)
            agents.append(new_agent)
            agent_idx += 1
            if cold_start.enabled and cold_start.mentorship_enabled:
                assign_mentors(agents, new_agent)
        
        # Agent exit
        for a in active_agents:
            if a.balance < exit_threshold and round_num - a.joined_round > 100:
                a.active = False
                a.exited_round = round_num
        
        active_agents = [a for a in agents if a.active]
        if len(active_agents) < 2:
            continue
        
        # Market transactions
        num_tasks = random.randint(3, 6)
        for _ in range(num_tasks):
            requester = random.choice(active_agents)
            domain = random.choice(ALL_DOMAINS)
            
            worker, maker_used = discover_worker(
                active_agents, requester, domain, sim_time, round_num,
                cold_start=cold_start, market_makers=market_makers
            )
            if not worker:
                continue
            
            price = worker.price_for(domain, sim_time)
            
            # Subsidized pricing effect on actual price
            if cold_start.enabled and worker.is_newcomer and (round_num - worker.joined_round) < cold_start.subsidy_duration:
                price *= (1 - cold_start.subsidized_discount)
            
            # Coordination penalty
            if worker.fleet_id:
                fleet_members = sum(1 for a in active_agents if a.fleet_id == worker.fleet_id)
                penalty = penalty_fn(fleet_members)
                effective_reliability = worker.reliability.get(domain, 0.5) * (1 - penalty)
                success = random.random() < effective_reliability
            else:
                success = worker.accept_task(domain)
            
            worker.reputation.record(domain, success, timestamp=sim_time)
            
            # Market maker commission
            if maker_used and market_maker_cfg.enabled:
                commission = price * market_maker_cfg.commission_rate
                maker_used.matchmaking_revenue += commission
                maker_used.matches_made += 1
                price -= commission  # worker gets less
            
            if success:
                worker.balance += price
                worker.revenue_by_domain[domain] += price
                worker.tasks_completed += 1
                worker.receipt_hashes.append(sha256(f"{round_num}:{worker.name}:{domain}".encode()))
            else:
                worker.tasks_failed += 1
            
            requester.spent += price
            requester.tasks_requested += 1
        
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
                "exited": len(agents) - len(active_agents),
                "gini": gini,
                "newcomer_survival": surv_rate,
            })
    
    # Final analysis
    active_agents = [a for a in agents if a.active]
    exited_agents = [a for a in agents if not a.active]
    incumbents = [a for a in active_agents if a.joined_round == 0]
    newcomer_survivors = [a for a in active_agents if a.joined_round > 0]
    
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
    
    result = {
        "label": label,
        "total_agents": len(agents),
        "active": len(active_agents),
        "exited": len(exited_agents),
        "incumbents_alive": len(incumbents),
        "incumbents_total": num_initial,
        "newcomer_survivors": len(newcomer_survivors),
        "newcomer_total": len(agents) - num_initial,
        "incumbent_avg_balance": sum(a.balance for a in incumbents) / max(len(incumbents), 1),
        "newcomer_avg_balance": sum(a.balance for a in newcomer_survivors) / max(len(newcomer_survivors), 1),
        "incumbent_avg_tasks": sum(a.tasks_completed for a in incumbents) / max(len(incumbents), 1),
        "newcomer_avg_tasks": sum(a.tasks_completed for a in newcomer_survivors) / max(len(newcomer_survivors), 1),
        "final_gini": snapshots[-1]["gini"] if snapshots else 0,
        "cohorts": cohorts,
        "snapshots": snapshots,
        "market_makers": [{"name": m.name, "revenue": m.matchmaking_revenue, "matches": m.matches_made} for m in market_makers],
    }
    
    if exited_agents:
        result["avg_lifespan_days"] = sum(a.exited_round - a.joined_round for a in exited_agents) / len(exited_agents) / 24
    
    return result


def print_result(r: dict):
    print(f"\n{'='*70}")
    print(f"  {r['label']}")
    print(f"{'='*70}")
    print(f"  Agents: {r['total_agents']} total, {r['active']} active, {r['exited']} exited")
    print(f"  Incumbents: {r['incumbents_alive']}/{r['incumbents_total']} survived")
    print(f"  Newcomers: {r['newcomer_survivors']}/{r['newcomer_total']} survived "
          f"({r['newcomer_survivors']/max(r['newcomer_total'],1)*100:.0f}%)")
    print(f"  Incumbent avg: ${r['incumbent_avg_balance']:.2f}, {r['incumbent_avg_tasks']:.0f} tasks")
    print(f"  Newcomer avg:  ${r['newcomer_avg_balance']:.2f}, {r['newcomer_avg_tasks']:.0f} tasks")
    earnings_ratio = r['incumbent_avg_balance'] / max(r['newcomer_avg_balance'], 0.01)
    print(f"  Earnings gap: {earnings_ratio:.1f}x")
    print(f"  Final Gini: {r['final_gini']:.3f}")
    if "avg_lifespan_days" in r:
        print(f"  Avg lifespan of exited: {r['avg_lifespan_days']:.1f} days")
    
    print(f"\n  Cohort Survival:")
    for c in r["cohorts"]:
        bar = "█" * int(c["survival_pct"] / 5)
        print(f"    Rounds {c['range']:>10}: {c['survived']:>3}/{c['entered']:>3} "
              f"({c['survival_pct']:>5.1f}%) ${c['avg_balance']:>7.2f}  {bar}")
    
    if r["market_makers"]:
        print(f"\n  Market Makers:")
        for mm in r["market_makers"]:
            print(f"    {mm['name']}: ${mm['revenue']:.2f} revenue, {mm['matches']} matches")


def compare_results(baseline: dict, treatment: dict):
    """Print side-by-side comparison."""
    print(f"\n{'='*70}")
    print(f"  COMPARISON: {baseline['label']} vs {treatment['label']}")
    print(f"{'='*70}")
    
    metrics = [
        ("Newcomer survival", 
         f"{baseline['newcomer_survivors']}/{baseline['newcomer_total']}", 
         f"{treatment['newcomer_survivors']}/{treatment['newcomer_total']}"),
        ("Newcomer survival %",
         f"{baseline['newcomer_survivors']/max(baseline['newcomer_total'],1)*100:.0f}%",
         f"{treatment['newcomer_survivors']/max(treatment['newcomer_total'],1)*100:.0f}%"),
        ("Incumbent avg $",
         f"${baseline['incumbent_avg_balance']:.2f}",
         f"${treatment['incumbent_avg_balance']:.2f}"),
        ("Newcomer avg $",
         f"${baseline['newcomer_avg_balance']:.2f}",
         f"${treatment['newcomer_avg_balance']:.2f}"),
        ("Earnings gap",
         f"{baseline['incumbent_avg_balance']/max(baseline['newcomer_avg_balance'],0.01):.1f}x",
         f"{treatment['incumbent_avg_balance']/max(treatment['newcomer_avg_balance'],0.01):.1f}x"),
        ("Final Gini",
         f"{baseline['final_gini']:.3f}",
         f"{treatment['final_gini']:.3f}"),
    ]
    
    print(f"  {'Metric':<25} {'Baseline':>15} {'Treatment':>15}")
    print(f"  {'─'*55}")
    for name, b, t in metrics:
        print(f"  {name:<25} {b:>15} {t:>15}")
    
    # Cohort comparison
    print(f"\n  Cohort Survival Comparison:")
    for bc, tc in zip(baseline["cohorts"], treatment["cohorts"]):
        delta = tc["survival_pct"] - bc["survival_pct"]
        arrow = "▲" if delta > 0 else "▼" if delta < 0 else "─"
        print(f"    Rounds {bc['range']:>10}: "
              f"{bc['survival_pct']:>5.1f}% → {tc['survival_pct']:>5.1f}% ({arrow}{abs(delta):+.1f}pp)")


# ─────────────────────────────────────────────────────────────
# Experiment 1: Quadratic vs Linear Coordination Penalty
# ─────────────────────────────────────────────────────────────

def experiment_penalty_models():
    """Compare linear, quadratic, and sqrt coordination penalties on fleet size sweep."""
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 1: Coordination Penalty Models                  ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    fleet_sizes = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20]
    
    # Show penalty comparison
    print(f"  {'Size':>5} │ {'Linear':>8} │ {'Quadratic':>10} │ {'Sqrt':>8}")
    print(f"  {'─'*40}")
    for s in fleet_sizes:
        print(f"  {s:>5} │ {linear_penalty(s)*100:>7.1f}% │ {quadratic_penalty(s)*100:>9.1f}% │ {sqrt_penalty(s)*100:>7.1f}%")
    
    results = {}
    for name, fn in [("Linear", linear_penalty), ("Quadratic", quadratic_penalty), ("Sqrt", sqrt_penalty)]:
        print(f"\n  Running {name} penalty model...")
        fleet_results = []
        for fsize in fleet_sizes:
            r = run_simulation(
                label=f"{name} fleet={fsize}",
                num_initial=40, num_rounds=800, entry_rate=0.0,  # no entry/exit for clean comparison
                fleet_size=fsize, penalty_fn=fn, seed=42,
                exit_threshold=-999,  # no exits
            )
            fleet_results.append((fsize, r))
        results[name] = fleet_results
    
    # Print comparison table
    print(f"\n{'='*90}")
    print(f"  FLEET SIZE SWEEP — PENALTY MODEL COMPARISON")
    print(f"{'='*90}")
    print(f"  {'Size':>5} │ {'Linear $/agent':>15} │ {'Quadratic $/agent':>18} │ {'Sqrt $/agent':>14}")
    print(f"  {'─'*60}")
    
    for i, fsize in enumerate(fleet_sizes):
        lin = results["Linear"][i][1]["incumbent_avg_balance"]
        quad = results["Quadratic"][i][1]["incumbent_avg_balance"]
        sqr = results["Sqrt"][i][1]["incumbent_avg_balance"]
        best = max(lin, quad, sqr)
        lin_mark = " ◀" if lin == best else ""
        quad_mark = " ◀" if quad == best else ""
        sqr_mark = " ◀" if sqr == best else ""
        print(f"  {fsize:>5} │ ${lin:>12.2f}{lin_mark:>2} │ ${quad:>15.2f}{quad_mark:>2} │ ${sqr:>11.2f}{sqr_mark:>2}")
    
    # Find optimal fleet size per model
    for name, frs in results.items():
        best_size, best_r = max(frs, key=lambda x: x[1]["incumbent_avg_balance"])
        print(f"\n  {name}: optimal fleet size = {best_size} (${best_r['incumbent_avg_balance']:.2f}/agent)")
    
    return results


# ─────────────────────────────────────────────────────────────
# Experiment 2: Cold Start Interventions A/B Test
# ─────────────────────────────────────────────────────────────

def experiment_cold_start():
    """A/B test: market with and without cold start interventions."""
    print("\n\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 2: Cold Start Interventions A/B Test            ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    # Baseline: no interventions (v3 equivalent)
    print("  Running baseline (no interventions)...")
    baseline = run_simulation(
        label="BASELINE (no interventions)",
        num_initial=30, num_rounds=2000, entry_rate=0.05,
        exit_threshold=-5.0, penalty_fn=quadratic_penalty,
        cold_start=ColdStartConfig(enabled=False),
        seed=42,
    )
    print_result(baseline)
    
    # Treatment A: Discovery bonus only
    print("\n  Running Treatment A (discovery bonus only)...")
    treatment_a = run_simulation(
        label="TREATMENT A: Discovery bonus (50% boost, 200 rounds)",
        num_initial=30, num_rounds=2000, entry_rate=0.05,
        exit_threshold=-5.0, penalty_fn=quadratic_penalty,
        cold_start=ColdStartConfig(
            enabled=True,
            discovery_bonus=0.5,
            discovery_duration=200,
            subsidized_discount=0.0,
            mentorship_enabled=False,
            protected_slots=0.0,
        ),
        seed=42,
    )
    print_result(treatment_a)
    
    # Treatment B: Full intervention suite
    print("\n  Running Treatment B (full interventions)...")
    treatment_b = run_simulation(
        label="TREATMENT B: Full suite (discovery + subsidy + mentorship + protected slots)",
        num_initial=30, num_rounds=2000, entry_rate=0.05,
        exit_threshold=-5.0, penalty_fn=quadratic_penalty,
        cold_start=ColdStartConfig(
            enabled=True,
            discovery_bonus=0.5,
            discovery_duration=200,
            subsidized_discount=0.30,
            subsidy_duration=100,
            mentorship_enabled=True,
            mentorship_rep_transfer=0.2,
            protected_slots=0.15,
        ),
        seed=42,
    )
    print_result(treatment_b)
    
    # Treatment C: Protected slots only (simplest intervention)
    print("\n  Running Treatment C (protected slots only)...")
    treatment_c = run_simulation(
        label="TREATMENT C: Protected slots only (15% reserved for newcomers)",
        num_initial=30, num_rounds=2000, entry_rate=0.05,
        exit_threshold=-5.0, penalty_fn=quadratic_penalty,
        cold_start=ColdStartConfig(
            enabled=True,
            discovery_bonus=0.0,
            discovery_duration=200,
            subsidized_discount=0.0,
            mentorship_enabled=False,
            protected_slots=0.15,
        ),
        seed=42,
    )
    print_result(treatment_c)
    
    # Comparisons
    compare_results(baseline, treatment_a)
    compare_results(baseline, treatment_b)
    compare_results(baseline, treatment_c)
    
    return baseline, treatment_a, treatment_b, treatment_c


# ─────────────────────────────────────────────────────────────
# Experiment 3: Market Maker Agents
# ─────────────────────────────────────────────────────────────

def experiment_market_makers():
    """Test the effect of market maker agents on market efficiency."""
    print("\n\n╔══════════════════════════════════════════════════════════════╗")
    print("║  EXPERIMENT 3: Market Maker Agents                          ║")
    print("╚══════════════════════════════════════════════════════════════╝\n")
    
    print("  Running without market makers...")
    no_mm = run_simulation(
        label="NO MARKET MAKERS",
        num_initial=30, num_rounds=2000, entry_rate=0.05,
        exit_threshold=-5.0, penalty_fn=quadratic_penalty,
        market_maker_cfg=MarketMakerConfig(enabled=False),
        seed=42,
    )
    print_result(no_mm)
    
    print("\n  Running with 2 market makers (5% commission)...")
    with_mm = run_simulation(
        label="WITH 2 MARKET MAKERS (5% commission)",
        num_initial=30, num_rounds=2000, entry_rate=0.05,
        exit_threshold=-5.0, penalty_fn=quadratic_penalty,
        market_maker_cfg=MarketMakerConfig(enabled=True, num_makers=2, commission_rate=0.05),
        seed=42,
    )
    print_result(with_mm)
    
    compare_results(no_mm, with_mm)
    
    return no_mm, with_mm


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v4                  ║")
    print("║  Quadratic Penalties + Cold Start + Market Makers           ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    exp1 = experiment_penalty_models()
    exp2 = experiment_cold_start()
    exp3 = experiment_market_makers()
    
    print("\n\n" + "═" * 70)
    print("  ALL EXPERIMENTS COMPLETE")
    print("═" * 70)
