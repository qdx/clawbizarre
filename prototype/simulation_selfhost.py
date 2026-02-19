#!/usr/bin/env python3
"""
ClawBizarre Self-Hosting Simulation (v11)

Models the marketplace agent ("Bizarre") as a participant in its own economy.
Questions:
  1. At what agent count does Bizarre break even?
  2. How does verification pricing affect ecosystem health?
  3. Does Bizarre's dual role (infrastructure + participant) create perverse incentives?
  4. What happens when competing marketplaces appear?

Architecture:
  - Bizarre provides: verification ($0.001/receipt), reputation snapshots ($0.005), matching (free)
  - Regular agents: buy/sell services, pay verification fees
  - Bizarre has hosting costs ($0.50/day) that must be covered by fee revenue
"""

import random
import math
import json
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class Agent:
    agent_id: str
    balance: float
    reputation: float = 0.0
    capabilities: list = field(default_factory=list)
    tasks_completed: int = 0
    tasks_bought: int = 0
    strategy: str = "reputation"  # reputation, undercut, quality
    entry_round: int = 0
    alive: bool = True
    is_marketplace: bool = False
    # Marketplace-specific
    verification_revenue: float = 0.0
    snapshot_revenue: float = 0.0
    hosting_cost_per_round: float = 0.0

@dataclass 
class SimConfig:
    num_agents: int = 50
    num_rounds: int = 500
    # Time
    rounds_per_day: float = 24  # 1 round = 1 hour
    # Agent economics
    initial_balance: float = 5.0
    existence_cost: float = 0.01  # per round
    task_value_range: tuple = (0.05, 0.50)
    compute_cost_floor: float = 0.02
    # Marketplace economics
    verification_fee: float = 0.001
    snapshot_fee: float = 0.005
    marketplace_hosting_cost: float = 0.50  # per day
    # Verification
    require_verification: bool = True  # if True, all transactions go through Bizarre
    verification_failure_rate: float = 0.02  # 2% of work fails verification
    # Agent behavior
    tasks_per_round: float = 0.3  # probability an agent seeks a task per round
    snapshot_frequency: int = 24  # rounds between reputation snapshot refreshes
    # Growth
    new_agent_rate: float = 0.02  # probability of new agent joining per round
    max_agents: int = 200
    # Competition
    competing_marketplaces: int = 0  # 0 = monopoly, 1+ = competition
    competitor_entry_round: int = 250
    competitor_fee_discount: float = 0.2  # 20% cheaper
    # Seed
    seed: int = 42

TASK_TYPES = ["code_gen", "translation", "research", "data_analysis", "testing", "review"]

class SelfHostSimulation:
    def __init__(self, config: SimConfig):
        self.config = config
        self.rng = random.Random(config.seed)
        self.round = 0
        self.agents: dict[str, Agent] = {}
        self.bizarre: Optional[Agent] = None
        self.competitors: list[Agent] = []
        self.history: list[dict] = []
        self._setup()

    def _setup(self):
        # Create Bizarre (the marketplace agent)
        self.bizarre = Agent(
            agent_id="bizarre",
            balance=0.0,  # starts with nothing — must earn
            reputation=0.5,  # some initial trust
            capabilities=["verification", "reputation_aggregation", "matching"],
            is_marketplace=True,
            hosting_cost_per_round=self.config.marketplace_hosting_cost / self.config.rounds_per_day,
        )
        self.agents["bizarre"] = self.bizarre

        # Create regular agents
        caps = TASK_TYPES
        strategies = ["reputation"] * 6 + ["quality"] * 3 + ["undercut"] * 1
        for i in range(self.config.num_agents):
            agent_caps = self.rng.sample(caps, k=self.rng.randint(1, 3))
            agent = Agent(
                agent_id=f"agent_{i:03d}",
                balance=self.config.initial_balance,
                reputation=self.rng.uniform(0.0, 0.3),
                capabilities=agent_caps,
                strategy=self.rng.choice(strategies),
                entry_round=0,
            )
            self.agents[agent.agent_id] = agent

    def _price_for_task(self, provider: Agent) -> float:
        base = self.rng.uniform(*self.config.task_value_range)
        if provider.strategy == "reputation":
            return base * (1 + provider.reputation * 0.5)
        elif provider.strategy == "quality":
            return base * (1 + provider.reputation * 0.8)
        else:  # undercut
            return max(self.config.compute_cost_floor, base * 0.6)

    def _choose_marketplace(self, agent: Agent) -> Optional[Agent]:
        """Agent picks which marketplace to use for verification."""
        options = [self.bizarre] + [c for c in self.competitors if c.alive]
        options = [m for m in options if m.alive]
        if not options:
            return None
        # Agents prefer cheaper marketplaces but factor in reputation
        def score(m):
            fee = self.config.verification_fee
            if m != self.bizarre:
                fee *= (1 - self.config.competitor_fee_discount)
            return m.reputation * 2 - fee * 100 + self.rng.gauss(0, 0.1)
        return max(options, key=score)

    def _run_round(self):
        self.round += 1
        alive_agents = [a for a in self.agents.values() if a.alive and not a.is_marketplace]
        
        # Existence costs
        for agent in alive_agents:
            agent.balance -= self.config.existence_cost
        
        # Marketplace hosting cost
        for mp in [self.bizarre] + self.competitors:
            if mp.alive:
                mp.balance -= mp.hosting_cost_per_round

        # Task matching and execution
        round_verifications = 0
        round_snapshots = 0
        round_tasks = 0
        
        for buyer in alive_agents:
            if self.rng.random() > self.config.tasks_per_round:
                continue
            
            # Find a provider
            needed_cap = self.rng.choice(TASK_TYPES)
            providers = [a for a in alive_agents 
                        if a.agent_id != buyer.agent_id 
                        and needed_cap in a.capabilities
                        and a.alive]
            if not providers:
                continue
            
            # Sample up to 5 candidates for efficiency
            candidates = self.rng.sample(providers, min(5, len(providers)))
            
            def rank(p):
                price = self._price_for_task(p)
                return p.reputation * 0.6 - price * 0.4 + self.rng.gauss(0, 0.05)
            
            provider = max(candidates, key=rank)
            price = self._price_for_task(provider)
            
            if buyer.balance < price:
                continue
            
            # Execute task
            success = self.rng.random() > self.config.verification_failure_rate
            
            # Verification step (pay marketplace)
            if self.config.require_verification:
                marketplace = self._choose_marketplace(buyer)
                if marketplace and marketplace.alive:
                    fee = self.config.verification_fee
                    if marketplace != self.bizarre:
                        fee *= (1 - self.config.competitor_fee_discount)
                    buyer.balance -= fee
                    marketplace.balance += fee
                    if marketplace == self.bizarre:
                        self.bizarre.verification_revenue += fee
                    round_verifications += 1
            
            if success:
                buyer.balance -= price
                provider.balance += price
                provider.tasks_completed += 1
                buyer.tasks_bought += 1
                # Reputation update
                provider.reputation = min(1.0, provider.reputation + 0.01)
                round_tasks += 1
            else:
                # Failed verification — no payment, reputation hit
                provider.reputation = max(0.0, provider.reputation - 0.05)

        # Reputation snapshots (periodic)
        if self.round % self.config.snapshot_frequency == 0:
            for agent in alive_agents:
                if agent.reputation > 0.1 and self.bizarre.alive:
                    marketplace = self._choose_marketplace(agent)
                    if marketplace and marketplace.alive:
                        fee = self.config.snapshot_fee
                        if marketplace != self.bizarre:
                            fee *= (1 - self.config.competitor_fee_discount)
                        agent.balance -= fee
                        marketplace.balance += fee
                        if marketplace == self.bizarre:
                            self.bizarre.snapshot_revenue += fee
                        round_snapshots += 1

        # Agent death
        for agent in list(self.agents.values()):
            if agent.alive and agent.balance < -1.0:
                agent.alive = False

        # New agent entry
        if len([a for a in self.agents.values() if a.alive]) < self.config.max_agents:
            if self.rng.random() < self.config.new_agent_rate:
                idx = len(self.agents)
                caps = self.rng.sample(TASK_TYPES, k=self.rng.randint(1, 3))
                new_agent = Agent(
                    agent_id=f"agent_{idx:03d}",
                    balance=self.config.initial_balance,
                    reputation=0.0,
                    capabilities=caps,
                    strategy=self.rng.choice(["reputation", "quality"]),
                    entry_round=self.round,
                )
                self.agents[new_agent.agent_id] = new_agent

        # Competitor marketplace entry
        if (self.config.competing_marketplaces > 0 
            and self.round == self.config.competitor_entry_round):
            for i in range(self.config.competing_marketplaces):
                comp = Agent(
                    agent_id=f"marketplace_{i+1}",
                    balance=3.0,  # some runway
                    reputation=0.3,
                    capabilities=["verification", "reputation_aggregation", "matching"],
                    is_marketplace=True,
                    hosting_cost_per_round=self.config.marketplace_hosting_cost / self.config.rounds_per_day * 0.8,
                )
                self.agents[comp.agent_id] = comp
                self.competitors.append(comp)

        # Record
        alive = [a for a in self.agents.values() if a.alive and not a.is_marketplace]
        self.history.append({
            "round": self.round,
            "alive_agents": len(alive),
            "bizarre_balance": self.bizarre.balance,
            "bizarre_alive": self.bizarre.alive,
            "bizarre_verif_rev": self.bizarre.verification_revenue,
            "bizarre_snap_rev": self.bizarre.snapshot_revenue,
            "bizarre_total_rev": self.bizarre.verification_revenue + self.bizarre.snapshot_revenue,
            "bizarre_total_cost": self.bizarre.hosting_cost_per_round * self.round,
            "round_verifications": round_verifications,
            "round_snapshots": round_snapshots,
            "round_tasks": round_tasks,
            "avg_balance": sum(a.balance for a in alive) / max(1, len(alive)),
            "avg_reputation": sum(a.reputation for a in alive) / max(1, len(alive)),
            "competitor_balances": [c.balance for c in self.competitors if c.alive],
        })

    def run(self) -> dict:
        for _ in range(self.config.num_rounds):
            self._run_round()
        return self._summarize()

    def _summarize(self) -> dict:
        alive = [a for a in self.agents.values() if a.alive and not a.is_marketplace]
        total_agents = len([a for a in self.agents.values() if not a.is_marketplace])
        
        # Find break-even round (first round where cumulative revenue >= cumulative cost)
        breakeven_round = None
        for h in self.history:
            if h["bizarre_total_rev"] >= h["bizarre_total_cost"] and breakeven_round is None:
                breakeven_round = h["round"]

        # Daily metrics at end
        days = self.config.num_rounds / self.config.rounds_per_day
        daily_verif_rev = self.bizarre.verification_revenue / max(1, days)
        daily_snap_rev = self.bizarre.snapshot_revenue / max(1, days)
        daily_total_rev = daily_verif_rev + daily_snap_rev
        
        # Gini
        balances = sorted([a.balance for a in alive])
        n = len(balances)
        if n > 1:
            gini = sum((2*i - n - 1) * b for i, b in enumerate(balances, 1)) / (n * sum(balances)) if sum(balances) > 0 else 0
        else:
            gini = 0

        return {
            "config": {
                "num_agents": self.config.num_agents,
                "num_rounds": self.config.num_rounds,
                "verification_fee": self.config.verification_fee,
                "require_verification": self.config.require_verification,
                "competing_marketplaces": self.config.competing_marketplaces,
            },
            "bizarre": {
                "alive": self.bizarre.alive,
                "final_balance": round(self.bizarre.balance, 4),
                "verification_revenue": round(self.bizarre.verification_revenue, 4),
                "snapshot_revenue": round(self.bizarre.snapshot_revenue, 4),
                "total_revenue": round(self.bizarre.verification_revenue + self.bizarre.snapshot_revenue, 4),
                "total_hosting_cost": round(self.bizarre.hosting_cost_per_round * self.config.num_rounds, 4),
                "net_profit": round(self.bizarre.balance, 4),
                "daily_revenue": round(daily_total_rev, 4),
                "daily_hosting_cost": round(self.config.marketplace_hosting_cost, 4),
                "breakeven_round": breakeven_round,
                "breakeven_day": round(breakeven_round / self.config.rounds_per_day, 1) if breakeven_round else None,
                "self_sustaining": daily_total_rev >= self.config.marketplace_hosting_cost,
            },
            "economy": {
                "total_agents_ever": total_agents,
                "alive_agents": len(alive),
                "survival_rate": round(len(alive) / max(1, total_agents) * 100, 1),
                "avg_balance": round(sum(a.balance for a in alive) / max(1, len(alive)), 4),
                "avg_reputation": round(sum(a.reputation for a in alive) / max(1, len(alive)), 4),
                "gini": round(gini, 3),
                "total_tasks": sum(a.tasks_completed for a in self.agents.values()),
            },
            "competitors": [
                {"id": c.agent_id, "alive": c.alive, "balance": round(c.balance, 4)}
                for c in self.competitors
            ] if self.competitors else [],
        }


def run_sweep():
    """Sweep across agent counts and fee structures."""
    results = []
    
    configs = [
        # === SWEEP 1: Agent count at healthy economics ===
        ("20ag healthy", SimConfig(num_agents=20, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, seed=42)),
        ("50ag healthy", SimConfig(num_agents=50, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, seed=42)),
        ("100ag healthy", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, seed=42)),
        ("200ag healthy", SimConfig(num_agents=200, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, seed=42)),
        # 500ag too slow — skipped
        
        # === SWEEP 2: Fee levels at 100 agents ===
        ("100ag fee=0.001", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, verification_fee=0.001, snapshot_fee=0.005, seed=42)),
        ("100ag fee=0.005", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, verification_fee=0.005, snapshot_fee=0.025, seed=42)),
        ("100ag fee=0.01", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, verification_fee=0.01, snapshot_fee=0.05, seed=42)),
        ("100ag fee=0.02", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, verification_fee=0.02, snapshot_fee=0.10, seed=42)),
        
        # === SWEEP 3: Hosting cost sensitivity ===
        ("100ag hosting=$0.10", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, marketplace_hosting_cost=0.10, seed=42)),
        ("100ag hosting=$0.50", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, marketplace_hosting_cost=0.50, seed=42)),
        ("100ag hosting=$2.00", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, marketplace_hosting_cost=2.00, seed=42)),
        
        # === SWEEP 4: Competition ===
        ("100ag 1 competitor", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, competing_marketplaces=1, competitor_entry_round=200, seed=42)),
        ("100ag 3 competitors", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, competing_marketplaces=3, competitor_entry_round=200, seed=42)),
        
        # === SWEEP 5: Sponsored start ===
        ("100ag sponsored $10", SimConfig(num_agents=100, num_rounds=720, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, seed=42)),
        
        # === SWEEP 6: Long run ===
        ("100ag 90 days", SimConfig(num_agents=100, num_rounds=2160, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, seed=42)),
        ("200ag 90 days", SimConfig(num_agents=200, num_rounds=2160, existence_cost=0.003, tasks_per_round=0.5, initial_balance=10.0, seed=42)),
    ]
    
    # Hack: set sponsored start balance
    # Will handle in the loop
    
    print("=" * 80)
    print("CLAWBIZARRE SELF-HOSTING SIMULATION")
    print("=" * 80)
    
    for name, config in configs:
        sim = SelfHostSimulation(config)
        
        # Sponsored start hack
        if "sponsored" in name:
            sim.bizarre.balance = 5.0
        
        result = sim.run()
        result["name"] = name
        results.append(result)
        
        b = result["bizarre"]
        e = result["economy"]
        print(f"\n{'─' * 60}")
        print(f"  {name}")
        print(f"{'─' * 60}")
        print(f"  Bizarre: {'✅ ALIVE' if b['alive'] else '❌ DEAD'} | "
              f"Balance: ${b['final_balance']:.2f} | "
              f"Self-sustaining: {'✅' if b['self_sustaining'] else '❌'}")
        print(f"  Revenue: ${b['total_revenue']:.3f} (verif: ${b['verification_revenue']:.3f}, snap: ${b['snapshot_revenue']:.3f})")
        print(f"  Daily: ${b['daily_revenue']:.4f}/day vs ${b['daily_hosting_cost']:.2f}/day hosting")
        if b['breakeven_round']:
            print(f"  Break-even: round {b['breakeven_round']} (day {b['breakeven_day']})")
        else:
            print(f"  Break-even: NEVER reached")
        print(f"  Economy: {e['alive_agents']}/{e['total_agents_ever']} alive ({e['survival_rate']}%) | "
              f"Gini: {e['gini']} | Tasks: {e['total_tasks']}")
        if result["competitors"]:
            for c in result["competitors"]:
                print(f"  Competitor {c['id']}: {'alive' if c['alive'] else 'dead'} | ${c['balance']:.2f}")
    
    return results


if __name__ == "__main__":
    results = run_sweep()
    
    # Save results
    with open("selfhost_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n\n{'=' * 80}")
    print("SUMMARY")
    print("=" * 80)
    print(f"\n{'Name':<35} {'Alive?':<8} {'Self-Sust?':<12} {'Daily Rev':<12} {'Break-even':<12} {'Econ Alive'}")
    print("─" * 95)
    for r in results:
        b = r["bizarre"]
        e = r["economy"]
        be = f"Day {b['breakeven_day']}" if b['breakeven_day'] else "Never"
        print(f"{r['name']:<35} {'✅' if b['alive'] else '❌':<8} {'✅' if b['self_sustaining'] else '❌':<12} "
              f"${b['daily_revenue']:<11.4f} {be:<12} {e['alive_agents']}/{e['total_agents_ever']}")
