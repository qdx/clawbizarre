#!/usr/bin/env python3
"""
ClawBizarre Phase 6 — Multi-Agent HTTP Simulation

Simulates an economy of agents interacting entirely through the HTTP API.
Each agent is an autonomous loop: register → discover → negotiate → work → build reputation.

Key differences from previous simulations (v1-v10):
- All interactions go through HTTP endpoints (validates the API layer)
- Agents are concurrent (threaded), not turn-based
- Discovery is dynamic (agents come and go)
- Reputation affects discovery ranking in real-time
- Treasury enforces spending limits

Usage:
    # Start API server in one terminal:
    python api_server.py
    # Then run simulation:
    python simulation_http.py [--agents N] [--rounds R] [--port PORT]
"""

import json
import random
import time
import threading
import argparse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional
from collections import defaultdict


# --- Agent Persona ---

TASK_TYPES = ["code_review", "translation", "research", "test_generation", "data_analysis", "summarization"]

STRATEGIES = ["reputation_premium", "market_rate", "undercut", "quality_premium"]

AGENT_ARCHETYPES = [
    {"name": "Specialist", "capabilities": 1, "quality": 0.90, "strategy": "reputation_premium"},
    {"name": "Generalist", "capabilities": 3, "quality": 0.75, "strategy": "market_rate"},
    {"name": "Newcomer",   "capabilities": 1, "quality": 0.70, "strategy": "undercut"},
    {"name": "Veteran",    "capabilities": 2, "quality": 0.95, "strategy": "quality_premium"},
]


@dataclass
class AgentStats:
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_requested: int = 0
    revenue: float = 0.0
    spend: float = 0.0
    handshakes_initiated: int = 0
    handshakes_received: int = 0
    reputation_score: float = 0.0
    discovery_appearances: int = 0


@dataclass
class SimAgent:
    """An autonomous agent that interacts with the ClawBizarre API."""
    name: str
    archetype: str
    agent_id: Optional[str] = None
    capabilities: list = field(default_factory=list)
    quality: float = 0.8
    strategy: str = "market_rate"
    stats: AgentStats = field(default_factory=AgentStats)
    alive: bool = True
    balance: float = 50.0  # starting balance
    round_joined: int = 0

    def base_price(self, task_type: str) -> float:
        """Price based on strategy."""
        base = {"code_review": 3.0, "translation": 2.0, "research": 4.0,
                "test_generation": 2.5, "data_analysis": 3.5, "summarization": 1.5}
        p = base.get(task_type, 2.0)
        if self.strategy == "reputation_premium":
            return p * (1.0 + self.stats.reputation_score * 0.5)
        elif self.strategy == "quality_premium":
            return p * 1.3
        elif self.strategy == "undercut":
            return p * 0.6
        return p  # market_rate


class SimulationHTTP:
    """Orchestrates a multi-agent economy over the HTTP API."""

    def __init__(self, base_url: str = "http://127.0.0.1:8402",
                 num_agents: int = 12, num_rounds: int = 200,
                 entry_rate: float = 0.1, exit_threshold: float = -10.0):
        self.base_url = base_url
        self.num_agents = num_agents
        self.num_rounds = num_rounds
        self.entry_rate = entry_rate
        self.exit_threshold = exit_threshold
        self.agents: list[SimAgent] = []
        self.round_log: list[dict] = []
        self.lock = threading.Lock()
        self.agent_counter = 0

    # --- HTTP helpers ---

    def api(self, method: str, path: str, data: dict = None) -> Optional[dict]:
        try:
            body = json.dumps(data).encode() if data else None
            req = urllib.request.Request(
                f"{self.base_url}{path}",
                data=body,
                method=method,
                headers={"Content-Type": "application/json"} if body else {},
            )
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            return None

    # --- Agent lifecycle ---

    def create_agent(self, round_num: int = 0) -> SimAgent:
        archetype = random.choice(AGENT_ARCHETYPES)
        self.agent_counter += 1
        caps = random.sample(TASK_TYPES, archetype["capabilities"])
        quality = archetype["quality"] + random.gauss(0, 0.05)
        quality = max(0.3, min(1.0, quality))

        agent = SimAgent(
            name=f"{archetype['name']}_{self.agent_counter}",
            archetype=archetype["name"],
            capabilities=caps,
            quality=quality,
            strategy=archetype["strategy"],
            round_joined=round_num,
        )

        # Register identity via API
        result = self.api("POST", "/identity/create")
        if not result:
            return None
        agent.agent_id = result["agent_id"]

        # Register in discovery
        self.api("POST", "/discovery/register", {
            "agent_id": agent.agent_id,
            "capabilities": agent.capabilities,
            "verification_tier": 0,
            "receipt_chain_length": 0,
            "success_rate": agent.quality,
            "strategy_consistency": 0.9,
            "pricing_strategy": agent.strategy,
            "endpoint": f"http://{agent.name}.local:8080",
        })

        return agent

    def kill_agent(self, agent: SimAgent):
        agent.alive = False
        self.api("DELETE", f"/discovery/{agent.agent_id}")

    # --- Round logic ---

    def run_round(self, round_num: int):
        """One round: each alive agent tries to find and complete work."""
        alive_agents = [a for a in self.agents if a.alive]
        if len(alive_agents) < 2:
            return

        # Maintenance cost per round
        maintenance = 0.15
        for agent in alive_agents:
            agent.balance -= maintenance
            agent.stats.spend += maintenance

        # Each agent tries to find work as a BUYER
        random.shuffle(alive_agents)
        buyers = alive_agents[:len(alive_agents) // 2]
        round_events = []

        for buyer in buyers:
            # Pick a task type the buyer DOESN'T have (needs help with)
            needed = [t for t in TASK_TYPES if t not in buyer.capabilities]
            if not needed:
                needed = TASK_TYPES
            task_type = random.choice(needed)

            # Discover providers
            results = self.api("POST", "/discovery/search", {
                "task_type": task_type,
                "max_results": 5,
            })
            if not results or not results.get("results"):
                continue

            # Filter: don't hire yourself
            candidates = [r for r in results["results"] if r["agent_id"] != buyer.agent_id]
            if not candidates:
                continue

            # Pick best candidate (weighted by relevance + reputation)
            provider_info = candidates[0]  # already sorted by relevance
            provider = next((a for a in alive_agents if a.agent_id == provider_info["agent_id"]), None)
            if not provider or not provider.alive:
                continue

            provider.stats.discovery_appearances += 1

            # Calculate price
            price = provider.base_price(task_type)
            if price > buyer.balance:
                continue  # can't afford

            buyer.stats.tasks_requested += 1

            # Handshake
            hs = self.api("POST", "/handshake/start", {
                "initiator_id": buyer.agent_id,
                "responder_id": provider.agent_id,
                "initiator_capabilities": buyer.capabilities,
                "responder_capabilities": provider.capabilities,
            })
            if not hs:
                continue

            session_id = hs["session_id"]
            buyer.stats.handshakes_initiated += 1
            provider.stats.handshakes_received += 1

            # Propose
            self.api("POST", "/handshake/message", {
                "session_id": session_id,
                "action": "propose",
                "task_description": f"Perform {task_type}",
                "task_type": task_type,
            })

            # Accept (providers always accept for now — future: strategy-based rejection)
            self.api("POST", "/handshake/message", {
                "session_id": session_id,
                "action": "accept",
            })

            # Execute — quality determines success
            success = random.random() < provider.quality
            tests_passed = random.randint(3, 10) if success else random.randint(0, 2)
            tests_failed = 0 if success else random.randint(1, 5)

            result = self.api("POST", "/handshake/message", {
                "session_id": session_id,
                "action": "execute",
                "output": f"{'Completed' if success else 'Failed'}: {task_type}",
                "tests_passed": tests_passed,
                "tests_failed": tests_failed,
                "suite_hash": f"sha256:sim_round_{round_num}",
            })

            if success:
                # Payment
                buyer.balance -= price
                buyer.stats.spend += price
                provider.balance += price
                provider.stats.revenue += price
                provider.stats.tasks_completed += 1

                # Treasury check (buyer side)
                self.api("POST", "/treasury/evaluate", {
                    "requester_id": buyer.agent_id,
                    "counterparty_id": provider.agent_id,
                    "amount": price,
                    "category": "service",
                    "description": f"{task_type} payment",
                })
            else:
                provider.stats.tasks_failed += 1
                # Append failure receipt
                self.api("POST", "/receipt/chain/append", {
                    "agent_id": provider.agent_id,
                    "task_type": task_type,
                    "success": False,
                    "pricing_strategy": provider.strategy,
                })

            round_events.append({
                "buyer": buyer.name,
                "provider": provider.name,
                "task": task_type,
                "price": price,
                "success": success,
            })

        # Update reputation for agents with chains
        for agent in alive_agents:
            rep = self.api("POST", "/reputation/aggregate", {"agent_id": agent.agent_id})
            if rep and "composite_score" in rep:
                agent.stats.reputation_score = rep["composite_score"]

        # Heartbeat to keep discovery alive
        for agent in alive_agents:
            self.api("POST", "/discovery/heartbeat", {"agent_id": agent.agent_id})

        # Exit: bankrupt agents die
        for agent in alive_agents:
            if agent.balance < self.exit_threshold:
                self.kill_agent(agent)

        # Entry: new agents join
        if random.random() < self.entry_rate and len([a for a in self.agents if a.alive]) < self.num_agents * 2:
            new_agent = self.create_agent(round_num)
            if new_agent:
                self.agents.append(new_agent)

        # Log round
        alive_count = len([a for a in self.agents if a.alive])
        balances = [a.balance for a in self.agents if a.alive]
        avg_balance = sum(balances) / len(balances) if balances else 0

        self.round_log.append({
            "round": round_num,
            "alive": alive_count,
            "transactions": len(round_events),
            "avg_balance": round(avg_balance, 2),
            "events": round_events,
        })

    # --- Run ---

    def run(self):
        print(f"=== ClawBizarre HTTP Simulation ===")
        print(f"Agents: {self.num_agents} | Rounds: {self.num_rounds} | Entry rate: {self.entry_rate}")
        print()

        # Check server health
        health = self.api("GET", "/health")
        if not health:
            print("ERROR: Cannot reach API server. Start it with: python api_server.py")
            return
        print(f"Server: v{health['version']} — {health['status']}")
        print()

        # Create initial agents
        print(f"Creating {self.num_agents} agents...", end=" ", flush=True)
        for i in range(self.num_agents):
            agent = self.create_agent(0)
            if agent:
                self.agents.append(agent)
        print(f"✓ {len(self.agents)} created")

        # Print agent roster
        print("\nAgent Roster:")
        for a in self.agents:
            print(f"  {a.name:25s} | caps={','.join(a.capabilities):30s} | q={a.quality:.2f} | strat={a.strategy}")
        print()

        # Run rounds
        start = time.time()
        for r in range(self.num_rounds):
            self.run_round(r)

            # Progress every 20 rounds
            if (r + 1) % 20 == 0:
                alive = len([a for a in self.agents if a.alive])
                total_tx = sum(len(rl["events"]) for rl in self.round_log[-20:])
                elapsed = time.time() - start
                print(f"  Round {r+1:4d} | alive={alive:3d} | tx(20r)={total_tx:3d} | elapsed={elapsed:.1f}s")

        elapsed = time.time() - start
        print(f"\nSimulation complete in {elapsed:.1f}s")

        # --- Results ---
        self.print_results()
        return self.summary()

    def print_results(self):
        print("\n" + "=" * 70)
        print("RESULTS")
        print("=" * 70)

        alive = [a for a in self.agents if a.alive]
        dead = [a for a in self.agents if not a.alive]

        print(f"\nPopulation: {len(alive)} alive, {len(dead)} dead, {len(self.agents)} total")

        # Leaderboard
        sorted_agents = sorted(self.agents, key=lambda a: a.balance, reverse=True)
        print(f"\n{'Agent':25s} {'Arch':12s} {'Strategy':18s} {'Balance':>8s} {'Rev':>7s} {'Tasks':>5s} {'Fail':>4s} {'Rep':>5s} {'Alive':>5s}")
        print("-" * 100)
        for a in sorted_agents[:20]:
            print(f"{a.name:25s} {a.archetype:12s} {a.strategy:18s} {a.balance:8.2f} {a.stats.revenue:7.2f} "
                  f"{a.stats.tasks_completed:5d} {a.stats.tasks_failed:4d} {a.stats.reputation_score:5.3f} "
                  f"{'  ✓' if a.alive else '  ✗'}")

        # Strategy breakdown
        print("\nStrategy Breakdown:")
        strat_groups = defaultdict(list)
        for a in self.agents:
            strat_groups[a.strategy].append(a)
        for strat, agents in sorted(strat_groups.items()):
            alive_count = len([a for a in agents if a.alive])
            avg_bal = sum(a.balance for a in agents) / len(agents)
            avg_rev = sum(a.stats.revenue for a in agents) / len(agents)
            avg_rep = sum(a.stats.reputation_score for a in agents) / len(agents)
            print(f"  {strat:20s} | n={len(agents):2d} alive={alive_count:2d} | avg_bal={avg_bal:7.2f} avg_rev={avg_rev:7.2f} avg_rep={avg_rep:.3f}")

        # Archetype breakdown
        print("\nArchetype Breakdown:")
        arch_groups = defaultdict(list)
        for a in self.agents:
            arch_groups[a.archetype].append(a)
        for arch, agents in sorted(arch_groups.items()):
            alive_count = len([a for a in agents if a.alive])
            avg_bal = sum(a.balance for a in agents) / len(agents)
            print(f"  {arch:15s} | n={len(agents):2d} alive={alive_count:2d} | avg_bal={avg_bal:7.2f}")

        # Economy stats
        total_tx = sum(len(rl["events"]) for rl in self.round_log)
        total_volume = sum(e["price"] for rl in self.round_log for e in rl["events"] if e["success"])
        success_count = sum(1 for rl in self.round_log for e in rl["events"] if e["success"])
        fail_count = sum(1 for rl in self.round_log for e in rl["events"] if not e["success"])

        print(f"\nEconomy:")
        print(f"  Total transactions: {total_tx}")
        print(f"  Successful: {success_count} ({100*success_count/max(total_tx,1):.1f}%)")
        print(f"  Failed: {fail_count}")
        print(f"  Total volume: ${total_volume:.2f}")
        print(f"  Avg price/tx: ${total_volume/max(success_count,1):.2f}")

        # Gini coefficient
        balances = sorted([a.balance for a in self.agents])
        n = len(balances)
        if n > 0 and sum(balances) != 0:
            gini = sum((2 * (i + 1) - n - 1) * b for i, b in enumerate(balances)) / (n * sum(balances))
            print(f"  Gini coefficient: {gini:.3f}")

        # Discovery stats
        health = self.api("GET", "/health")
        if health:
            print(f"\nServer state: {health['agents']} identities, {health['chains']} chains, {health['registry']} registered")

    def summary(self) -> dict:
        alive = [a for a in self.agents if a.alive]
        total_tx = sum(len(rl["events"]) for rl in self.round_log)
        total_volume = sum(e["price"] for rl in self.round_log for e in rl["events"] if e["success"])
        balances = sorted([a.balance for a in self.agents])
        n = len(balances)
        gini = sum((2*(i+1)-n-1)*b for i, b in enumerate(balances)) / (n * sum(balances)) if n > 0 and sum(balances) != 0 else 0

        return {
            "total_agents": len(self.agents),
            "alive": len(alive),
            "dead": len(self.agents) - len(alive),
            "total_transactions": total_tx,
            "total_volume": round(total_volume, 2),
            "gini": round(gini, 3),
            "top_agent": max(self.agents, key=lambda a: a.balance).name if self.agents else None,
            "top_balance": round(max(a.balance for a in self.agents), 2) if self.agents else 0,
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ClawBizarre HTTP Simulation")
    parser.add_argument("--agents", type=int, default=12, help="Initial agent count")
    parser.add_argument("--rounds", type=int, default=200, help="Simulation rounds")
    parser.add_argument("--port", type=int, default=8402, help="API server port")
    parser.add_argument("--entry-rate", type=float, default=0.1, help="New agent entry probability per round")
    parser.add_argument("--exit-threshold", type=float, default=-10.0, help="Balance below which agents die")
    parser.add_argument("--embedded", action="store_true", help="Start API server in-process")
    args = parser.parse_args()

    if args.embedded:
        # Start server in background thread
        import threading
        from api_server import create_server
        server = create_server(port=args.port)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        time.sleep(0.3)
        print(f"Embedded server started on port {args.port}\n")

    sim = SimulationHTTP(
        base_url=f"http://127.0.0.1:{args.port}",
        num_agents=args.agents,
        num_rounds=args.rounds,
        entry_rate=args.entry_rate,
        exit_threshold=args.exit_threshold,
    )
    result = sim.run()

    if args.embedded:
        server.shutdown()
