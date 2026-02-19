"""
ClawBizarre Economy Simulation v2
Large-scale simulation with:
- Decaying domain-specific reputation
- Merkle-verified receipt chains
- Market dynamics: entry, exit, price discovery
- Fleet economics (agent groups)
- 50 agents, 1000+ rounds

Demonstrates emergence of: specialization advantages, reputation moats,
fleet coordination benefits, and market equilibrium.
"""

import json
import random
import time
import math
from dataclasses import dataclass, field
from collections import defaultdict
from typing import Optional

from identity import AgentIdentity
from receipt import WorkReceipt, TestResults, VerificationTier, hash_content, ReceiptChain
from reputation import DecayingReputation, DomainReputation, MerkleTree, sha256


# --- Market Configuration ---

TASK_CATALOG = {
    "code_review": {"base_cost": 0.50, "complexity": 0.7},
    "translation": {"base_cost": 0.30, "complexity": 0.5},
    "summarization": {"base_cost": 0.20, "complexity": 0.3},
    "data_validation": {"base_cost": 0.15, "complexity": 0.2},
    "research": {"base_cost": 1.00, "complexity": 0.9},
    "monitoring": {"base_cost": 0.10, "complexity": 0.1},  # boring reliability
}

DOMAIN_CORRELATIONS = {
    ("code_review", "research"): 0.5,
    ("code_review", "data_validation"): 0.3,
    ("translation", "summarization"): 0.4,
    ("research", "summarization"): 0.6,
    ("monitoring", "data_validation"): 0.3,
}


@dataclass
class MarketAgent:
    """Agent in the marketplace with economics."""
    name: str
    identity: AgentIdentity
    capabilities: list[str]
    reliability: dict[str, float]  # per-domain reliability
    reputation: DomainReputation = field(default_factory=lambda: DomainReputation(
        half_life_days=30, domain_correlations=DOMAIN_CORRELATIONS
    ))
    chain: ReceiptChain = field(default_factory=ReceiptChain)
    fleet_id: Optional[str] = None
    
    # Economics
    balance: float = 0.0  # earned credits
    spent: float = 0.0    # spent on requesting work
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_requested: int = 0
    revenue_by_domain: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    
    # Merkle state
    receipt_hashes: list[str] = field(default_factory=list)
    
    def price_for(self, domain: str, sim_time: float) -> float:
        """Reputation-based pricing. Higher rep = higher price."""
        base = TASK_CATALOG[domain]["base_cost"]
        rep_score = self.reputation.score(domain, sim_time)
        confidence = self.reputation._get_domain(domain).confidence(sim_time) if domain in self.reputation._domains else 0.0
        
        # Price = base * (1 + rep_premium)
        # New agents (low confidence) price at base
        # Established agents with high rep get up to 3x
        rep_premium = rep_score * confidence * 2.0
        return base * (1 + rep_premium)
    
    def accept_task(self, domain: str, sim_time: float) -> bool:
        """Simulate task execution. Returns True if successful."""
        base_reliability = self.reliability.get(domain, 0.5)
        # Slight random variation
        return random.random() < base_reliability


@dataclass
class Fleet:
    """A group of agents under one sponsor."""
    fleet_id: str
    sponsor_name: str
    agents: list[str]  # agent names
    budget: float = 100.0
    spent: float = 0.0
    internal_transactions: int = 0
    external_transactions: int = 0


@dataclass
class Transaction:
    """Record of a marketplace transaction."""
    round_num: int
    requester: str
    worker: str
    domain: str
    success: bool
    price: float
    requester_fleet: Optional[str]
    worker_fleet: Optional[str]
    sim_time: float


def generate_agents(n: int, seed: int = 42) -> list[MarketAgent]:
    """Generate diverse agents with varying capabilities and reliability."""
    random.seed(seed)
    
    all_domains = list(TASK_CATALOG.keys())
    agents = []
    
    archetypes = [
        # (name_prefix, capabilities, reliability_range, count)
        ("Specialist", lambda: [random.choice(all_domains)], (0.90, 0.99), n // 5),
        ("Generalist", lambda: random.sample(all_domains, k=random.randint(3, 5)), (0.70, 0.85), n // 5),
        ("MidTier", lambda: random.sample(all_domains, k=random.randint(2, 3)), (0.75, 0.90), n // 5),
        ("Newbie", lambda: random.sample(all_domains, k=random.randint(1, 2)), (0.55, 0.75), n // 5),
        ("Monitor", lambda: ["monitoring"] + random.sample([d for d in all_domains if d != "monitoring"], k=1), (0.92, 0.99), n // 5),
    ]
    
    idx = 0
    for prefix, cap_fn, rel_range, count in archetypes:
        for i in range(count):
            caps = cap_fn()
            reliability = {d: random.uniform(*rel_range) for d in caps}
            agent = MarketAgent(
                name=f"{prefix}_{idx}",
                identity=AgentIdentity.generate(),
                capabilities=caps,
                reliability=reliability,
            )
            agents.append(agent)
            idx += 1
    
    # Fill remainder
    while len(agents) < n:
        caps = random.sample(all_domains, k=random.randint(1, 4))
        reliability = {d: random.uniform(0.60, 0.95) for d in caps}
        agents.append(MarketAgent(
            name=f"Agent_{idx}",
            identity=AgentIdentity.generate(),
            capabilities=caps,
            reliability=reliability,
        ))
        idx += 1
    
    return agents


def assign_fleets(agents: list[MarketAgent], num_fleets: int = 8) -> list[Fleet]:
    """Assign agents to fleets (some agents remain independent)."""
    fleets = []
    fleet_agents = agents[:num_fleets * 4]  # first 32 agents go into fleets
    
    for i in range(num_fleets):
        fleet_members = fleet_agents[i*4:(i+1)*4]
        fleet_id = f"Fleet_{i}"
        for a in fleet_members:
            a.fleet_id = fleet_id
        fleets.append(Fleet(
            fleet_id=fleet_id,
            sponsor_name=f"Sponsor_{i}",
            agents=[a.name for a in fleet_members],
        ))
    
    # Remaining agents are independents
    return fleets


def discover_worker(agents: list[MarketAgent], requester: MarketAgent, 
                    domain: str, sim_time: float,
                    prefer_fleet: bool = True) -> Optional[MarketAgent]:
    """Discovery with reputation weighting and fleet preference."""
    candidates = [a for a in agents 
                  if a.name != requester.name and domain in a.capabilities]
    
    if not candidates:
        return None
    
    # Score each candidate
    scored = []
    for c in candidates:
        rep = c.reputation.score(domain, sim_time)
        conf = c.reputation._get_domain(domain).confidence(sim_time) if domain in c.reputation._domains else 0.0
        price = c.price_for(domain, sim_time)
        
        # Value = reputation * confidence / price (bang for buck)
        # New agents with no history get a small exploration bonus
        exploration_bonus = 0.2 if conf < 0.3 else 0.0
        value = (rep * max(conf, 0.1) + exploration_bonus) / max(price, 0.01)
        
        # Fleet bonus: prefer same-fleet agents (internal coordination is cheaper)
        if prefer_fleet and requester.fleet_id and c.fleet_id == requester.fleet_id:
            value *= 1.5  # 50% bonus for fleet-internal
        
        scored.append((c, value))
    
    # Softmax selection (not purely greedy — allows exploration)
    temperature = 0.5
    values = [v for _, v in scored]
    max_v = max(values)
    exp_values = [math.exp((v - max_v) / temperature) for v in values]
    total = sum(exp_values)
    probs = [e / total for e in exp_values]
    
    return random.choices([c for c, _ in scored], weights=probs, k=1)[0]


def run_simulation(num_agents: int = 50, num_rounds: int = 1000, seed: int = 42):
    """Run large-scale economy simulation."""
    random.seed(seed)
    
    # Setup
    agents = generate_agents(num_agents, seed)
    agent_map = {a.name: a for a in agents}
    fleets = assign_fleets(agents)
    fleet_map = {f.fleet_id: f for f in fleets}
    
    all_domains = list(TASK_CATALOG.keys())
    transactions: list[Transaction] = []
    
    # Simulate time: each round = ~1 hour, so 1000 rounds ≈ 42 days
    sim_start = time.time() - 42 * 86400  # pretend we started 42 days ago
    
    print(f"=== ClawBizarre Economy v2 ===")
    print(f"Agents: {num_agents} ({len(fleets)} fleets + {num_agents - len(fleets)*4} independents)")
    print(f"Rounds: {num_rounds} (~{num_rounds/24:.0f} simulated days)")
    print(f"Task types: {len(all_domains)}")
    print()
    
    # Run economy
    for round_num in range(1, num_rounds + 1):
        sim_time = sim_start + round_num * 3600  # 1 hour per round
        
        # Each round: 3-5 tasks get requested
        num_tasks = random.randint(3, 5)
        
        for _ in range(num_tasks):
            requester = random.choice(agents)
            domain = random.choice(all_domains)
            
            worker = discover_worker(agents, requester, domain, sim_time)
            if not worker:
                continue
            
            price = worker.price_for(domain, sim_time)
            success = worker.accept_task(domain, sim_time)
            
            # Record reputation
            worker.reputation.record(domain, success, timestamp=sim_time)
            
            # Record economics
            if success:
                worker.balance += price
                worker.revenue_by_domain[domain] += price
                worker.tasks_completed += 1
                
                # Receipt chain
                receipt_hash = sha256(f"{round_num}:{requester.name}:{worker.name}:{domain}:{price}".encode())
                worker.receipt_hashes.append(receipt_hash)
            else:
                worker.tasks_failed += 1
            
            requester.spent += price
            requester.tasks_requested += 1
            
            # Fleet tracking
            req_fleet = requester.fleet_id
            wrk_fleet = worker.fleet_id
            if req_fleet and wrk_fleet and req_fleet == wrk_fleet:
                fleet_map[req_fleet].internal_transactions += 1
            elif req_fleet:
                fleet_map[req_fleet].external_transactions += 1
            
            transactions.append(Transaction(
                round_num=round_num, requester=requester.name,
                worker=worker.name, domain=domain, success=success,
                price=price, requester_fleet=req_fleet, worker_fleet=wrk_fleet,
                sim_time=sim_time,
            ))
        
        # Progress
        if round_num % 200 == 0:
            success_rate = sum(1 for t in transactions[-200:] if t.success) / max(len(transactions[-200:]), 1)
            avg_price = sum(t.price for t in transactions[-200:]) / max(len(transactions[-200:]), 1)
            print(f"Round {round_num}: {len(transactions)} total txns, "
                  f"recent success={success_rate:.0%}, avg price=${avg_price:.2f}")
    
    # === RESULTS ===
    sim_end_time = sim_start + num_rounds * 3600
    
    print(f"\n{'='*70}")
    print(f"SIMULATION RESULTS")
    print(f"{'='*70}")
    
    total = len(transactions)
    successes = sum(1 for t in transactions if t.success)
    total_volume = sum(t.price for t in transactions)
    print(f"\nTransactions: {total} ({successes} successful, {total-successes} failed)")
    print(f"Success rate: {successes/total:.1%}")
    print(f"Total volume: ${total_volume:.2f}")
    print(f"Avg price: ${total_volume/total:.2f}")
    
    # Top earners
    print(f"\n{'─'*70}")
    print(f"TOP 10 EARNERS")
    print(f"{'Agent':>20} │ {'Earned':>8} │ {'Done':>4} │ {'Fail':>4} │ {'Fleet':>8} │ Capabilities")
    print(f"{'─'*70}")
    top = sorted(agents, key=lambda a: a.balance, reverse=True)[:10]
    for a in top:
        caps = ",".join(a.capabilities[:3])
        fleet = a.fleet_id or "indie"
        print(f"{a.name:>20} │ ${a.balance:>7.2f} │ {a.tasks_completed:>4} │ {a.tasks_failed:>4} │ {fleet:>8} │ {caps}")
    
    # Reputation leaders (by domain)
    print(f"\n{'─'*70}")
    print(f"REPUTATION LEADERS (by domain)")
    for domain in all_domains:
        best = max(agents, key=lambda a: a.reputation.score(domain, sim_end_time) 
                   if domain in a.capabilities else 0)
        score = best.reputation.score(domain, sim_end_time)
        conf = best.reputation._get_domain(domain).confidence(sim_end_time) if domain in best.reputation._domains else 0
        print(f"  {domain:>18}: {best.name} (score={score:.3f}, confidence={conf:.3f}, "
              f"earned=${best.revenue_by_domain.get(domain, 0):.2f})")
    
    # Fleet economics
    print(f"\n{'─'*70}")
    print(f"FLEET ECONOMICS")
    print(f"{'Fleet':>10} │ {'Internal':>8} │ {'External':>8} │ {'Int%':>5} │ {'Total Earned':>12}")
    print(f"{'─'*70}")
    for f in sorted(fleets, key=lambda f: f.internal_transactions + f.external_transactions, reverse=True):
        total_f = f.internal_transactions + f.external_transactions
        int_pct = f.internal_transactions / total_f * 100 if total_f > 0 else 0
        fleet_earnings = sum(a.balance for a in agents if a.fleet_id == f.fleet_id)
        print(f"{f.fleet_id:>10} │ {f.internal_transactions:>8} │ {f.external_transactions:>8} │ "
              f"{int_pct:>4.0f}% │ ${fleet_earnings:>11.2f}")
    
    indie_earnings = sum(a.balance for a in agents if a.fleet_id is None)
    indie_count = sum(1 for a in agents if a.fleet_id is None)
    fleet_earnings_total = sum(a.balance for a in agents if a.fleet_id is not None)
    fleet_count = sum(1 for a in agents if a.fleet_id is not None)
    print(f"\nFleet agents ({fleet_count}): ${fleet_earnings_total:.2f} total, ${fleet_earnings_total/max(fleet_count,1):.2f}/agent")
    print(f"Independent agents ({indie_count}): ${indie_earnings:.2f} total, ${indie_earnings/max(indie_count,1):.2f}/agent")
    
    # Price evolution
    print(f"\n{'─'*70}")
    print(f"PRICE EVOLUTION (avg price per 200-round window)")
    for i in range(0, num_rounds, 200):
        window = [t for t in transactions if i < t.round_num <= i + 200]
        if window:
            avg = sum(t.price for t in window) / len(window)
            succ = sum(1 for t in window if t.success) / len(window)
            print(f"  Rounds {i+1:>4}-{i+200:>4}: ${avg:.3f} avg, {succ:.0%} success, {len(window)} txns")
    
    # Specialist vs Generalist analysis
    print(f"\n{'─'*70}")
    print(f"SPECIALIST VS GENERALIST")
    specialists = [a for a in agents if len(a.capabilities) == 1]
    generalists = [a for a in agents if len(a.capabilities) >= 3]
    mid = [a for a in agents if 1 < len(a.capabilities) < 3]
    
    for label, group in [("Specialists (1 cap)", specialists), ("Mid-tier (2 caps)", mid), ("Generalists (3+ caps)", generalists)]:
        if group:
            avg_earn = sum(a.balance for a in group) / len(group)
            avg_tasks = sum(a.tasks_completed for a in group) / len(group)
            avg_rep = sum(max((a.reputation.score(d, sim_end_time) for d in a.capabilities), default=0) for a in group) / len(group)
            print(f"  {label:>25}: {len(group)} agents, ${avg_earn:.2f}/agent, {avg_tasks:.0f} tasks/agent, best_rep={avg_rep:.3f}")
    
    # Merkle trees
    print(f"\n{'─'*70}")
    print(f"MERKLE VERIFICATION")
    agents_with_receipts = [a for a in agents if a.receipt_hashes]
    if agents_with_receipts:
        sample = sorted(agents_with_receipts, key=lambda a: len(a.receipt_hashes), reverse=True)[:3]
        for a in sample:
            tree = MerkleTree(a.receipt_hashes)
            proof = tree.proof(0)
            valid = MerkleTree.verify_proof(a.receipt_hashes[0], proof, tree.root)
            print(f"  {a.name}: {len(a.receipt_hashes)} receipts, root={tree.root[:16]}..., "
                  f"proof_size={len(proof)}, verified={valid}")
    
    # Key emergent properties
    print(f"\n{'─'*70}")
    print(f"EMERGENT PROPERTIES")
    
    # Gini coefficient of earnings
    earnings = sorted(a.balance for a in agents)
    n = len(earnings)
    gini = sum((2*i - n + 1) * e for i, e in enumerate(earnings)) / (n * sum(earnings)) if sum(earnings) > 0 else 0
    print(f"  Earnings Gini coefficient: {gini:.3f} (0=equal, 1=concentrated)")
    
    # Reputation-price correlation
    prices_and_reps = []
    for a in agents:
        if a.tasks_completed > 0:
            best_domain = max(a.capabilities, key=lambda d: a.revenue_by_domain.get(d, 0))
            rep = a.reputation.score(best_domain, sim_end_time)
            avg_price = a.balance / a.tasks_completed
            prices_and_reps.append((rep, avg_price))
    
    if len(prices_and_reps) > 5:
        reps = [r for r, _ in prices_and_reps]
        prices = [p for _, p in prices_and_reps]
        mean_r, mean_p = sum(reps)/len(reps), sum(prices)/len(prices)
        cov = sum((r-mean_r)*(p-mean_p) for r, p in zip(reps, prices))
        var_r = sum((r-mean_r)**2 for r in reps)
        var_p = sum((p-mean_p)**2 for p in prices)
        corr = cov / (math.sqrt(var_r * var_p)) if var_r > 0 and var_p > 0 else 0
        print(f"  Reputation-Price correlation: {corr:.3f}")
    
    # Fleet internal vs external ratio
    total_internal = sum(f.internal_transactions for f in fleets)
    total_external = sum(f.external_transactions for f in fleets)
    total_fleet_txn = total_internal + total_external
    if total_fleet_txn > 0:
        print(f"  Fleet internal transaction rate: {total_internal/total_fleet_txn:.1%}")
        print(f"  (Coasian prediction: internal preferred when cheaper — "
              f"{'confirmed' if total_internal/total_fleet_txn > 0.2 else 'not confirmed'})")
    
    return agents, transactions, fleets


if __name__ == "__main__":
    agents, txns, fleets = run_simulation(num_agents=50, num_rounds=1000, seed=42)
