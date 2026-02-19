"""
ClawBizarre Economy Simulation v3
New features over v2:
- Agent entry/exit dynamics (new agents join, underperformers exit)
- Fleet size sweep (find optimal Coasian boundary)
- Cold start analysis (how do newcomers bootstrap in a mature market?)
- Market saturation dynamics
- Configurable churn rate
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
    # Identity cost: agents must "pay" to maintain identity each round
    identity_cost_per_round: float = 0.01  # ~$0.24/day at 24 rounds/day
    
    def price_for(self, domain: str, sim_time: float) -> float:
        base = TASK_CATALOG[domain]["base_cost"]
        rep_score = self.reputation.score(domain, sim_time)
        conf = self.reputation._get_domain(domain).confidence(sim_time) if domain in self.reputation._domains else 0.0
        rep_premium = rep_score * confidence * 2.0 if (confidence := conf) else 0.0
        return base * (1 + rep_premium)
    
    def accept_task(self, domain: str) -> bool:
        base_reliability = self.reliability.get(domain, 0.5)
        return random.random() < base_reliability
    
    def net_profit(self) -> float:
        return self.balance - self.spent


@dataclass
class Fleet:
    fleet_id: str
    agents: list[str]
    size: int
    internal_txns: int = 0
    external_txns: int = 0


@dataclass 
class Transaction:
    round_num: int
    requester: str
    worker: str
    domain: str
    success: bool
    price: float
    is_internal_fleet: bool = False


def make_agent(idx: int, joined_round: int = 0, fleet_id: Optional[str] = None) -> MarketAgent:
    """Create a random agent."""
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
    else:  # newbie
        caps = random.sample(ALL_DOMAINS, k=random.randint(1, 2))
        rel_range = (0.55, 0.75)
    
    return MarketAgent(
        name=f"Agent_{idx}",
        identity=AgentIdentity.generate(),
        capabilities=caps,
        reliability={d: random.uniform(*rel_range) for d in caps},
        fleet_id=fleet_id,
        joined_round=joined_round,
    )


def discover_worker(agents: list[MarketAgent], requester: MarketAgent,
                    domain: str, sim_time: float) -> Optional[MarketAgent]:
    """Reputation-weighted discovery with fleet preference and exploration."""
    candidates = [a for a in agents 
                  if a.active and a.name != requester.name and domain in a.capabilities]
    if not candidates:
        return None
    
    scored = []
    for c in candidates:
        rep = c.reputation.score(domain, sim_time)
        conf = c.reputation._get_domain(domain).confidence(sim_time) if domain in c.reputation._domains else 0.0
        price = c.price_for(domain, sim_time)
        
        exploration_bonus = 0.3 if conf < 0.2 else 0.1 if conf < 0.4 else 0.0
        value = (rep * max(conf, 0.1) + exploration_bonus) / max(price, 0.01)
        
        if requester.fleet_id and c.fleet_id == requester.fleet_id:
            value *= 1.5
        
        scored.append((c, value))
    
    temperature = 0.5
    values = [v for _, v in scored]
    max_v = max(values)
    exp_values = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]
    
    return random.choices([c for c, _ in scored], weights=probs, k=1)[0]


# ─────────────────────────────────────────────────────────────
# Experiment 1: Agent Entry/Exit Dynamics
# ─────────────────────────────────────────────────────────────

def run_entry_exit_sim(num_initial: int = 30, num_rounds: int = 2000,
                       entry_rate: float = 0.05, exit_threshold: float = -5.0,
                       seed: int = 42):
    """
    Simulate a market with agent churn:
    - New agents enter at `entry_rate` per round (Poisson)
    - Agents exit when net_profit < exit_threshold (can't sustain identity costs)
    - Track cold start success rate, market saturation, incumbent advantage
    """
    random.seed(seed)
    sim_start = time.time() - 84 * 86400  # 84 days of simulated time
    
    # Initial agents (the "incumbents")
    agent_idx = 0
    agents = []
    for i in range(num_initial):
        agents.append(make_agent(agent_idx))
        agent_idx += 1
    
    # Assign some initial fleets
    for i in range(0, min(20, num_initial), 4):
        fid = f"Fleet_{i//4}"
        for j in range(i, min(i+4, num_initial)):
            agents[j].fleet_id = fid
    
    agent_map = {a.name: a for a in agents}
    transactions = []
    
    # Tracking
    population_history = []
    entry_survival = {}  # agent_name -> (joined_round, survived_to_round_N?)
    gini_history = []
    newcomer_earnings = defaultdict(list)  # round_joined -> [earnings at exit or end]
    
    print(f"=== Entry/Exit Dynamics Simulation ===")
    print(f"Initial: {num_initial} agents, entry_rate={entry_rate}/round, exit_threshold=${exit_threshold}")
    print(f"Rounds: {num_rounds} (~{num_rounds/24:.0f} simulated days)")
    print()
    
    for round_num in range(1, num_rounds + 1):
        sim_time = sim_start + round_num * 3600
        active_agents = [a for a in agents if a.active]
        
        # --- Identity maintenance cost ---
        for a in active_agents:
            a.balance -= a.identity_cost_per_round
        
        # --- Agent entry (Poisson process, expected entry_rate per round) ---
        num_entries = 0
        while random.random() < entry_rate:
            num_entries += 1
        # Also force entry if population drops too low (market needs participants)
        if len(active_agents) < 10:
            num_entries = max(num_entries, 2)
        for _ in range(num_entries):
            new_agent = make_agent(agent_idx, joined_round=round_num)
            agents.append(new_agent)
            agent_map[new_agent.name] = new_agent
            entry_survival[new_agent.name] = round_num
            agent_idx += 1
        
        # --- Agent exit (bankruptcy or sustained unprofitability) ---
        for a in active_agents:
            # Exit if deeply negative AND has been around long enough to have had a chance
            if a.balance < exit_threshold and round_num - a.joined_round > 100:
                a.active = False
                a.exited_round = round_num
                newcomer_earnings[a.joined_round].append(a.balance)
        
        active_agents = [a for a in agents if a.active]
        
        # --- Market transactions ---
        if len(active_agents) < 2:
            continue
        num_tasks = random.randint(3, 6)
        for _ in range(num_tasks):
            requester = random.choice(active_agents)
            domain = random.choice(ALL_DOMAINS)
            worker = discover_worker(active_agents, requester, domain, sim_time)
            if not worker:
                continue
            
            price = worker.price_for(domain, sim_time)
            success = worker.accept_task(domain)
            worker.reputation.record(domain, success, timestamp=sim_time)
            
            is_internal = bool(requester.fleet_id and worker.fleet_id and requester.fleet_id == worker.fleet_id)
            
            if success:
                worker.balance += price
                worker.revenue_by_domain[domain] += price
                worker.tasks_completed += 1
                worker.receipt_hashes.append(sha256(f"{round_num}:{worker.name}:{domain}".encode()))
            else:
                worker.tasks_failed += 1
            
            requester.spent += price
            requester.tasks_requested += 1
            
            transactions.append(Transaction(
                round_num=round_num, requester=requester.name,
                worker=worker.name, domain=domain, success=success,
                price=price, is_internal_fleet=is_internal,
            ))
        
        # --- Periodic tracking ---
        if round_num % 100 == 0:
            n_active = len(active_agents)
            n_total = len(agents)
            n_exited = n_total - n_active
            population_history.append((round_num, n_active, n_total, n_exited))
            
            # Gini
            earnings = sorted(a.balance for a in active_agents)
            n = len(earnings)
            s = sum(earnings)
            if s > 0 and n > 1:
                gini = sum((2*i - n + 1) * e for i, e in enumerate(earnings)) / (n * s)
            else:
                gini = 0
            gini_history.append((round_num, gini))
            
            # Cold start analysis: agents joined in last 200 rounds
            recent_joiners = [a for a in active_agents if round_num - a.joined_round < 200 and a.joined_round > 0]
            old_guard = [a for a in active_agents if a.joined_round == 0]
            
            if round_num % 500 == 0:
                recent_avg = sum(a.balance for a in recent_joiners) / max(len(recent_joiners), 1)
                old_avg = sum(a.balance for a in old_guard) / max(len(old_guard), 1)
                print(f"Round {round_num}: active={n_active}, total_ever={n_total}, exited={n_exited}, "
                      f"gini={gini:.3f}")
                print(f"  Incumbents avg balance: ${old_avg:.2f}, "
                      f"Newcomers (<200r) avg: ${recent_avg:.2f}, "
                      f"ratio: {old_avg/max(recent_avg, 0.01):.1f}x")
    
    # === FINAL RESULTS ===
    active_agents = [a for a in agents if a.active]
    exited_agents = [a for a in agents if not a.active]
    incumbents = [a for a in active_agents if a.joined_round == 0]
    survivors = [a for a in active_agents if a.joined_round > 0]
    
    print(f"\n{'='*70}")
    print(f"ENTRY/EXIT RESULTS")
    print(f"{'='*70}")
    print(f"Total agents ever: {len(agents)}")
    print(f"Still active: {len(active_agents)}")
    print(f"Exited (bankrupt): {len(exited_agents)}")
    print(f"Original incumbents alive: {len(incumbents)}/{num_initial}")
    print(f"Newcomers who survived: {len(survivors)}/{len(agents)-num_initial}")
    
    if exited_agents:
        avg_lifespan = sum(a.exited_round - a.joined_round for a in exited_agents) / len(exited_agents)
        print(f"Average lifespan of exited agents: {avg_lifespan:.0f} rounds ({avg_lifespan/24:.1f} days)")
    
    # Cold start success rate by entry cohort
    print(f"\n{'─'*70}")
    print(f"COLD START ANALYSIS (by entry cohort)")
    cohort_size = 200
    for start in range(0, num_rounds, cohort_size):
        cohort = [a for a in agents if start < a.joined_round <= start + cohort_size]
        if not cohort:
            continue
        alive = sum(1 for a in cohort if a.active)
        avg_bal = sum(a.balance for a in cohort) / len(cohort)
        print(f"  Joined rounds {start+1:>5}-{start+cohort_size:>5}: "
              f"{len(cohort):>3} agents, {alive:>3} survived ({alive/len(cohort)*100:.0f}%), "
              f"avg balance=${avg_bal:.2f}")
    
    # Incumbent advantage
    print(f"\n{'─'*70}")
    print(f"INCUMBENT ADVANTAGE")
    if incumbents and survivors:
        inc_avg = sum(a.balance for a in incumbents) / len(incumbents)
        surv_avg = sum(a.balance for a in survivors) / len(survivors)
        inc_tasks = sum(a.tasks_completed for a in incumbents) / len(incumbents)
        surv_tasks = sum(a.tasks_completed for a in survivors) / len(survivors)
        print(f"  Incumbents: ${inc_avg:.2f}/agent, {inc_tasks:.0f} tasks/agent")
        print(f"  Survivors:  ${surv_avg:.2f}/agent, {surv_tasks:.0f} tasks/agent")
        print(f"  Earnings advantage: {inc_avg/max(surv_avg, 0.01):.1f}x")
        print(f"  Task advantage: {inc_tasks/max(surv_tasks, 0.01):.1f}x")
    
    # Population dynamics
    print(f"\n{'─'*70}")
    print(f"POPULATION DYNAMICS")
    for r, active, total, exited in population_history[::2]:  # every other entry
        print(f"  Round {r:>5}: active={active:>3}, cumulative={total:>3}, exited={exited:>3}")
    
    # Gini evolution
    print(f"\n{'─'*70}")
    print(f"INEQUALITY EVOLUTION (Gini)")
    for r, g in gini_history[::2]:
        bar = "█" * int(g * 40)
        print(f"  Round {r:>5}: {g:.3f} {bar}")
    
    return agents, transactions


# ─────────────────────────────────────────────────────────────
# Experiment 2: Fleet Size Optimization (Coasian Boundary)
# ─────────────────────────────────────────────────────────────

def run_fleet_size_sweep(fleet_sizes: list[int] = [1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
                         total_agents: int = 40, num_rounds: int = 800, seed: int = 42):
    """
    Hold total agents constant, vary fleet size. Find optimal Coasian boundary.
    
    At each fleet_size:
    - All agents assigned to fleets of that size
    - Run simulation
    - Measure: per-agent earnings, fleet internal rate, coordination overhead
    
    Coordination overhead: as fleet grows, each agent spends more time on
    internal coordination (modeled as opportunity cost).
    """
    results = []
    
    print(f"=== Fleet Size Optimization ===")
    print(f"Total agents: {total_agents}, Rounds: {num_rounds}")
    print(f"Testing fleet sizes: {fleet_sizes}")
    print()
    
    for fsize in fleet_sizes:
        random.seed(seed)
        sim_start = time.time() - 40 * 86400
        
        # Create agents
        agents = [make_agent(i) for i in range(total_agents)]
        
        # Assign to fleets of size fsize
        num_fleets = total_agents // fsize
        remainder = total_agents % fsize
        fleets = []
        idx = 0
        for f in range(num_fleets):
            fid = f"Fleet_{f}"
            members = []
            for j in range(fsize):
                if idx < total_agents:
                    agents[idx].fleet_id = fid
                    members.append(agents[idx].name)
                    idx += 1
            fleets.append(Fleet(fleet_id=fid, agents=members, size=fsize))
        # Remainder agents are independent
        
        # Coordination overhead: increases with fleet size
        # Model: each agent in a fleet loses (fleet_size - 1) * 0.5% of productivity
        # per fleet member (communication overhead)
        coordination_penalty = min((fsize - 1) * 0.005, 0.15)  # caps at 15%
        
        transactions = []
        internal_count = 0
        external_count = 0
        
        for round_num in range(1, num_rounds + 1):
            sim_time = sim_start + round_num * 3600
            
            num_tasks = random.randint(3, 5)
            for _ in range(num_tasks):
                requester = random.choice(agents)
                domain = random.choice(ALL_DOMAINS)
                
                worker = discover_worker(agents, requester, domain, sim_time)
                if not worker:
                    continue
                
                price = worker.price_for(domain, sim_time)
                
                # Apply coordination penalty to fleet agents
                if worker.fleet_id:
                    effective_reliability = worker.reliability.get(domain, 0.5) * (1 - coordination_penalty)
                    success = random.random() < effective_reliability
                else:
                    success = worker.accept_task(domain)
                
                worker.reputation.record(domain, success, timestamp=sim_time)
                
                is_internal = bool(requester.fleet_id and worker.fleet_id and 
                                   requester.fleet_id == worker.fleet_id)
                if is_internal:
                    internal_count += 1
                else:
                    external_count += 1
                
                if success:
                    worker.balance += price
                    worker.tasks_completed += 1
                else:
                    worker.tasks_failed += 1
                
                requester.spent += price
                transactions.append(Transaction(
                    round_num=round_num, requester=requester.name,
                    worker=worker.name, domain=domain, success=success,
                    price=price, is_internal_fleet=is_internal,
                ))
        
        # Results for this fleet size
        fleet_agents = [a for a in agents if a.fleet_id is not None]
        indie_agents = [a for a in agents if a.fleet_id is None]
        
        fleet_avg_earn = sum(a.balance for a in fleet_agents) / max(len(fleet_agents), 1)
        indie_avg_earn = sum(a.balance for a in indie_agents) / max(len(indie_agents), 1)
        total_avg_earn = sum(a.balance for a in agents) / len(agents)
        
        fleet_success = sum(a.tasks_completed for a in fleet_agents)
        fleet_total = fleet_success + sum(a.tasks_failed for a in fleet_agents)
        fleet_success_rate = fleet_success / max(fleet_total, 1)
        
        total_fleet_txn = internal_count + external_count
        internal_rate = internal_count / max(total_fleet_txn, 1)
        
        results.append({
            "fleet_size": fsize,
            "fleet_avg_earn": fleet_avg_earn,
            "indie_avg_earn": indie_avg_earn,
            "total_avg_earn": total_avg_earn,
            "fleet_success_rate": fleet_success_rate,
            "internal_rate": internal_rate,
            "coordination_penalty": coordination_penalty,
            "num_fleets": num_fleets,
            "num_transactions": len(transactions),
        })
    
    # Print results
    print(f"\n{'='*90}")
    print(f"FLEET SIZE SWEEP RESULTS")
    print(f"{'='*90}")
    print(f"{'Size':>5} │ {'#Fleets':>7} │ {'Fleet$/agent':>12} │ {'Indie$/agent':>12} │ "
          f"{'Success%':>8} │ {'Internal%':>9} │ {'Coord Tax':>9}")
    print(f"{'─'*90}")
    
    best = max(results, key=lambda r: r["fleet_avg_earn"])
    
    for r in results:
        marker = " ◀ BEST" if r == best else ""
        print(f"{r['fleet_size']:>5} │ {r['num_fleets']:>7} │ "
              f"${r['fleet_avg_earn']:>10.2f} │ ${r['indie_avg_earn']:>10.2f} │ "
              f"{r['fleet_success_rate']*100:>7.1f}% │ {r['internal_rate']*100:>8.1f}% │ "
              f"{r['coordination_penalty']*100:>7.1f}%{marker}")
    
    print(f"\n  Optimal fleet size: {best['fleet_size']} agents")
    print(f"  Earnings peak: ${best['fleet_avg_earn']:.2f}/agent")
    print(f"  Coordination penalty at optimum: {best['coordination_penalty']*100:.1f}%")
    
    # Find the Coasian boundary
    for i in range(1, len(results)):
        if results[i]["fleet_avg_earn"] < results[i-1]["fleet_avg_earn"]:
            print(f"\n  Coasian boundary: between {results[i-1]['fleet_size']} and {results[i]['fleet_size']} agents")
            print(f"  Beyond this, coordination costs exceed fleet benefits")
            break
    
    return results


# ─────────────────────────────────────────────────────────────
# Main: Run both experiments
# ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          ClawBizarre Economy Simulation v3                  ║")
    print("║  Agent Entry/Exit Dynamics + Fleet Size Optimization        ║")
    print("╚══════════════════════════════════════════════════════════════╝")
    print()
    
    print("━" * 70)
    print("EXPERIMENT 1: ENTRY/EXIT DYNAMICS")
    print("━" * 70)
    agents_ee, txns_ee = run_entry_exit_sim(
        num_initial=30, num_rounds=2000, 
        entry_rate=0.05, exit_threshold=-5.0, seed=42
    )
    
    print("\n\n")
    print("━" * 70)
    print("EXPERIMENT 2: FLEET SIZE OPTIMIZATION")  
    print("━" * 70)
    fleet_results = run_fleet_size_sweep(
        fleet_sizes=[1, 2, 3, 4, 5, 6, 8, 10, 15, 20],
        total_agents=40, num_rounds=800, seed=42
    )
