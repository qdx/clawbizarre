"""
Multi-Agent MCP Simulation — Phase 13

Simulates N agents trading through MCP server instances against a shared
api_server_v6. Each agent has a persona (archetype) that determines behavior.

Agent Archetypes:
- Specialist: Lists one capability, high quality, premium pricing
- Generalist: Lists multiple capabilities, moderate quality, competitive pricing  
- Newcomer: Enters mid-simulation, must bootstrap reputation from zero
- Opportunist: Switches capabilities based on market demand

Simulation Flow (per round):
1. Providers list/update services via MCP
2. Buyers discover providers via MCP
3. Handshakes initiated, accepted, executed, verified
4. Reputation evolves, new agents enter, struggling agents may exit

Metrics Tracked:
- Per-agent: earnings, tasks completed, reputation, survival
- Market: price levels, Gini coefficient, newcomer survival rate
- MCP: tool call counts, error rates, latency

Usage:
    python3 simulation_mcp.py                    # Default: 20 agents, 200 rounds
    python3 simulation_mcp.py --agents 50 --rounds 500
    python3 simulation_mcp.py --test             # Run tests only
"""

import json
import os
import sys
import random
import tempfile
import threading
import time
import math
from dataclasses import dataclass, field
from typing import Any, Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp_server import MCPServer, IdentityManager
from identity import AgentIdentity


# ── Agent Archetypes ─────────────────────────────────────────────────

CAPABILITIES = ["code_review", "translation", "research", "data_analysis", "testing", "writing"]

@dataclass
class AgentPersona:
    name: str
    archetype: str  # specialist, generalist, newcomer, opportunist
    capabilities: list  # what they can provide
    buy_capabilities: list  # what they need
    quality_range: tuple  # (min, max) quality score
    price_multiplier: float  # relative to base price
    entry_round: int = 0  # when they enter the simulation

    def pick_quality(self) -> float:
        return random.uniform(*self.quality_range)

    def pick_price(self, base: float) -> float:
        return round(base * self.price_multiplier * random.uniform(0.9, 1.1), 2)


def generate_personas(n_agents: int, newcomer_fraction: float = 0.2) -> list:
    """Generate a diverse set of agent personas."""
    personas = []
    n_newcomers = int(n_agents * newcomer_fraction)
    n_regular = n_agents - n_newcomers

    for i in range(n_regular):
        r = random.random()
        if r < 0.4:
            # Specialist
            cap = random.choice(CAPABILITIES)
            buy = [c for c in CAPABILITIES if c != cap]
            personas.append(AgentPersona(
                name=f"specialist_{i}",
                archetype="specialist",
                capabilities=[cap],
                buy_capabilities=random.sample(buy, min(2, len(buy))),
                quality_range=(0.8, 1.0),
                price_multiplier=1.3,
                entry_round=0,
            ))
        elif r < 0.7:
            # Generalist
            caps = random.sample(CAPABILITIES, random.randint(2, 4))
            buy = [c for c in CAPABILITIES if c not in caps]
            personas.append(AgentPersona(
                name=f"generalist_{i}",
                archetype="generalist",
                capabilities=caps,
                buy_capabilities=random.sample(buy, min(2, len(buy))) if buy else caps[:1],
                quality_range=(0.6, 0.9),
                price_multiplier=1.0,
                entry_round=0,
            ))
        else:
            # Opportunist
            cap = random.choice(CAPABILITIES)
            personas.append(AgentPersona(
                name=f"opportunist_{i}",
                archetype="opportunist",
                capabilities=[cap],
                buy_capabilities=random.sample(CAPABILITIES, 2),
                quality_range=(0.5, 0.85),
                price_multiplier=0.85,
                entry_round=0,
            ))

    # Newcomers enter at random points in the simulation
    for i in range(n_newcomers):
        cap = random.choice(CAPABILITIES)
        buy = [c for c in CAPABILITIES if c != cap]
        personas.append(AgentPersona(
            name=f"newcomer_{i}",
            archetype="newcomer",
            capabilities=[cap],
            buy_capabilities=random.sample(buy, min(2, len(buy))),
            quality_range=(0.5, 0.9),
            price_multiplier=0.9,
            entry_round=random.randint(1, 150),  # staggered entry
        ))

    return personas


# ── MCP Test Client (from test_mcp_e2e.py) ──────────────────────────

class MCPClient:
    """Drives an MCPServer by calling _handle_message directly."""

    def __init__(self, server: MCPServer):
        self.server = server
        self._id = 0
        self.call_count = 0
        self.error_count = 0

    def call(self, method: str, params: dict = None) -> dict:
        self._id += 1
        self.call_count += 1
        msg = {"jsonrpc": "2.0", "id": self._id, "method": method, "params": params or {}}
        resp = self.server._handle_message(msg)
        if resp and "error" in resp:
            self.error_count += 1
            return None
        return resp.get("result") if resp else None

    def tool(self, name: str, args: dict = None) -> Optional[dict]:
        result = self.call("tools/call", {"name": name, "arguments": args or {}})
        if result is None:
            return None
        try:
            text = result["content"][0]["text"]
            return json.loads(text)
        except (KeyError, json.JSONDecodeError):
            return None

    def init(self):
        self.call("initialize", {
            "protocolVersion": "2024-11-05",
            "capabilities": {},
            "clientInfo": {"name": "sim-agent", "version": "0.1.0"}
        })
        self.call("notifications/initialized")


# ── Simulation Agent ─────────────────────────────────────────────────

@dataclass
class SimAgent:
    persona: AgentPersona
    client: MCPClient
    agent_id: str = ""
    balance: float = 10.0  # starting compute credits
    tasks_completed: int = 0
    tasks_bought: int = 0
    earnings: float = 0.0
    spending: float = 0.0
    alive: bool = True
    listed: bool = False

    def step_cost(self) -> float:
        """Per-round existence cost (set by simulation)."""
        return self._step_cost


# ── Simulation Engine ────────────────────────────────────────────────

class MCPSimulation:
    """Multi-agent marketplace simulation through MCP interface."""

    def __init__(self, n_agents: int = 20, n_rounds: int = 200, seed: int = 42,
                 step_cost: float = 0.2, initial_balance: float = 10.0):
        self.n_rounds = n_rounds
        self.seed = seed
        self.step_cost = step_cost
        self.initial_balance = initial_balance
        random.seed(seed)

        # Setup server
        self.tmpdir = tempfile.mkdtemp(prefix="clawbiz_sim_")
        self.db_path = os.path.join(self.tmpdir, "sim.db")
        self.port = self._find_port()
        self.server_thread = None

        # Generate personas and agents
        self.personas = generate_personas(n_agents)
        self.agents: list[SimAgent] = []

        # Metrics
        self.round_metrics = []
        self.base_price = 1.0

    def _find_port(self) -> int:
        import socket
        with socket.socket() as s:
            s.bind(("", 0))
            return s.getsockname()[1]

    def _start_server(self):
        from api_server_v6 import APIv6Handler, PersistentStateV6
        from http.server import ThreadingHTTPServer

        state = PersistentStateV6(self.db_path)
        APIv6Handler.state = state

        self._httpd = ThreadingHTTPServer(("127.0.0.1", self.port), APIv6Handler)
        self.server_thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self.server_thread.start()
        time.sleep(0.3)

    def _create_agent(self, persona: AgentPersona) -> SimAgent:
        """Create an MCP-connected agent."""
        keydir = os.path.join(self.tmpdir, f"keys_{persona.name}")
        os.makedirs(keydir, exist_ok=True)

        identity = AgentIdentity.generate()
        keyfile = os.path.join(keydir, "identity.key")
        identity.save_keyfile(keyfile)

        server = MCPServer(
            api_url=f"http://127.0.0.1:{self.port}",
            keyfile=keyfile,
        )
        client = MCPClient(server)
        client.init()

        # Get agent ID
        whoami = client.tool("cb_whoami")
        agent_id = whoami.get("agent_id", "") if whoami else ""

        agent = SimAgent(
            persona=persona,
            client=client,
            agent_id=agent_id,
            balance=self.initial_balance,
        )
        agent._step_cost = self.step_cost
        return agent

    def _list_services(self, agent: SimAgent):
        """Provider lists their capabilities."""
        if agent.listed:
            return
        for cap in agent.persona.capabilities:
            price = agent.persona.pick_price(self.base_price)
            result = agent.client.tool("cb_list_service", {
                "capability": cap,
                "base_rate": price,
                "unit": "per_task",
                "description": f"{agent.persona.name} offering {cap}",
            })
            if result and result.get("status") == "listed":
                agent.listed = True

    def _find_and_trade(self, buyer: SimAgent, round_num: int) -> bool:
        """Buyer finds a provider and completes a trade. Returns True if successful."""
        if not buyer.persona.buy_capabilities:
            return False

        cap = random.choice(buyer.persona.buy_capabilities)
        providers = buyer.client.tool("cb_find_providers", {"capability": cap})

        if not providers or (isinstance(providers, dict) and not providers.get("providers")):
            return False
        if not isinstance(providers, list):
            return False

        # Pick a provider (prefer cheaper for opportunists, best rep for specialists)
        provider_list = providers
        # Filter out self
        provider_list = [p for p in provider_list if p.get("agent_id") != buyer.agent_id]
        if not provider_list:
            return False

        if buyer.persona.archetype == "opportunist":
            provider = min(provider_list, key=lambda p: p.get("base_rate", 999))
        else:
            provider = max(provider_list, key=lambda p: p.get("reputation", 0))

        price = provider.get("base_rate", 1.0)
        if price > buyer.balance:
            return False

        # Initiate handshake
        result = buyer.client.tool("cb_initiate_task", {
            "provider_id": provider["agent_id"],
            "capability": cap,
            "description": f"Round {round_num}: {buyer.persona.name} needs {cap}",
        })
        if not result:
            return False

        # initiate returns session_id as string or dict with session_id
        if isinstance(result, str):
            session_id = result
        elif isinstance(result, dict) and "session_id" in result:
            session_id = result["session_id"]
        else:
            return False

        # Find the provider agent to accept
        provider_agent = None
        for a in self.agents:
            if a.agent_id == provider["agent_id"] and a.alive:
                provider_agent = a
                break

        if not provider_agent:
            return False

        # Provider accepts
        accept = provider_agent.client.tool("cb_accept_task", {"session_id": session_id})
        if not accept:
            return False

        # Provider executes
        quality = provider_agent.persona.pick_quality()
        execute = provider_agent.client.tool("cb_submit_work", {
            "session_id": session_id,
            "output": f"Work output for {cap} (quality={quality:.2f})",
            "proof": json.dumps({"quality": quality, "tests_passed": int(quality * 10)}),
        })
        if not execute:
            return False

        # Buyer verifies
        accept_work = quality >= 0.5  # Accept if quality is reasonable
        verify = buyer.client.tool("cb_verify_work", {
            "session_id": session_id,
            "quality_score": quality,
            "accept": accept_work,
        })
        if not verify:
            return False

        # Update balances
        if accept_work:
            buyer.balance -= price
            buyer.spending += price
            buyer.tasks_bought += 1
            provider_agent.balance += price
            provider_agent.earnings += price
            provider_agent.tasks_completed += 1
            return True

        return False

    def _opportunist_adapt(self, agent: SimAgent):
        """Opportunists switch capabilities based on market stats."""
        if agent.persona.archetype != "opportunist":
            return
        if random.random() > 0.1:  # 10% chance per round
            return

        stats = agent.client.tool("cb_market_stats")
        if not stats:
            return

        # Switch to a random capability (simplified — real version would analyze demand)
        new_cap = random.choice(CAPABILITIES)
        if new_cap not in agent.persona.capabilities:
            # Unlist old
            agent.client.tool("cb_unlist_service", {"capability": agent.persona.capabilities[0]})
            agent.persona.capabilities = [new_cap]
            agent.listed = False

    def _collect_metrics(self, round_num: int) -> dict:
        """Collect per-round metrics."""
        alive = [a for a in self.agents if a.alive]
        balances = [a.balance for a in alive]
        earnings = [a.earnings for a in alive]

        # Gini coefficient
        gini = 0.0
        if len(earnings) > 1 and sum(earnings) > 0:
            sorted_e = sorted(earnings)
            n = len(sorted_e)
            cum = sum((i + 1) * e for i, e in enumerate(sorted_e))
            gini = (2 * cum) / (n * sum(sorted_e)) - (n + 1) / n

        # Newcomer stats
        newcomers = [a for a in self.agents if a.persona.archetype == "newcomer" and a.persona.entry_round <= round_num]
        newcomer_alive = sum(1 for a in newcomers if a.alive)
        newcomer_survival = newcomer_alive / len(newcomers) if newcomers else 1.0

        # Archetype breakdown
        archetype_earnings = {}
        for a in alive:
            at = a.persona.archetype
            archetype_earnings.setdefault(at, []).append(a.earnings)

        archetype_avg = {k: sum(v) / len(v) if v else 0 for k, v in archetype_earnings.items()}

        # MCP stats
        total_calls = sum(a.client.call_count for a in self.agents)
        total_errors = sum(a.client.error_count for a in self.agents)

        metrics = {
            "round": round_num,
            "alive": len(alive),
            "total": len(self.agents),
            "gini": round(gini, 3),
            "newcomer_survival": round(newcomer_survival, 3),
            "avg_balance": round(sum(balances) / len(balances), 2) if balances else 0,
            "total_earnings": round(sum(earnings), 2),
            "archetype_avg_earnings": {k: round(v, 2) for k, v in archetype_avg.items()},
            "mcp_calls": total_calls,
            "mcp_errors": total_errors,
            "mcp_error_rate": round(total_errors / total_calls, 4) if total_calls > 0 else 0,
        }
        return metrics

    def run(self, quiet: bool = False):
        """Run the full simulation."""
        if not quiet:
            print(f"=== ClawBizarre MCP Simulation ===")
            print(f"Agents: {len(self.personas)}, Rounds: {self.n_rounds}, Seed: {self.seed}")
            print()

        self._start_server()

        # Create initial agents (entry_round == 0)
        for p in self.personas:
            if p.entry_round == 0:
                agent = self._create_agent(p)
                self.agents.append(agent)

        if not quiet:
            print(f"Created {len(self.agents)} initial agents")

        for round_num in range(self.n_rounds):
            # Entry: newcomers join
            for p in self.personas:
                if p.entry_round == round_num and p.entry_round > 0:
                    agent = self._create_agent(p)
                    self.agents.append(agent)
                    if not quiet and round_num % 50 == 0:
                        print(f"  Round {round_num}: {p.name} entered")

            alive_agents = [a for a in self.agents if a.alive]

            # Phase 1: List services
            for a in alive_agents:
                self._list_services(a)

            # Phase 2: Trading (each alive agent tries to buy something)
            trades_this_round = 0
            random.shuffle(alive_agents)
            for a in alive_agents:
                if random.random() < 0.6:  # 60% chance to buy each round
                    if self._find_and_trade(a, round_num):
                        trades_this_round += 1

            # Phase 3: Opportunists adapt
            for a in alive_agents:
                self._opportunist_adapt(a)

            # Phase 4: Deduct existence cost, kill broke agents
            for a in alive_agents:
                a.balance -= a.step_cost()
                if a.balance <= 0:
                    a.alive = False

            # Collect metrics every 10 rounds
            if round_num % 10 == 0:
                m = self._collect_metrics(round_num)
                m["trades"] = trades_this_round
                self.round_metrics.append(m)
                if not quiet and round_num % 50 == 0:
                    print(f"  Round {round_num}: alive={m['alive']}/{m['total']}, "
                          f"gini={m['gini']}, trades={trades_this_round}, "
                          f"newcomer_surv={m['newcomer_survival']}")

        # Final metrics
        final = self._collect_metrics(self.n_rounds)
        self.round_metrics.append(final)

        if not quiet:
            self._print_summary(final)

        # Cleanup
        self._httpd.shutdown()
        return final

    def _print_summary(self, final: dict):
        print()
        print("=== FINAL RESULTS ===")
        print(f"Alive: {final['alive']}/{final['total']}")
        print(f"Gini: {final['gini']}")
        print(f"Newcomer survival: {final['newcomer_survival']:.0%}")
        print(f"Total earnings: ${final['total_earnings']:.2f}")
        print(f"Avg balance: ${final['avg_balance']:.2f}")
        print()
        print("Earnings by archetype:")
        for arch, avg in sorted(final['archetype_avg_earnings'].items()):
            print(f"  {arch}: ${avg:.2f} avg")
        print()
        print(f"MCP calls: {final['mcp_calls']} (errors: {final['mcp_errors']}, rate: {final['mcp_error_rate']:.2%})")

        # Top/bottom agents
        alive = sorted([a for a in self.agents if a.alive], key=lambda a: a.earnings, reverse=True)
        if alive:
            print()
            print("Top 5 earners:")
            for a in alive[:5]:
                rep = a.client.tool("cb_reputation", {"agent_id": a.agent_id})
                rep_score = rep.get("composite_score", "?") if rep else "?"
                print(f"  {a.persona.name} ({a.persona.archetype}): "
                      f"${a.earnings:.2f} earned, {a.tasks_completed} tasks, rep={rep_score}")

        dead = [a for a in self.agents if not a.alive]
        if dead:
            print(f"\nDead agents: {len(dead)}")
            by_arch = {}
            for a in dead:
                by_arch.setdefault(a.persona.archetype, 0)
                by_arch[a.persona.archetype] += 1
            for arch, count in sorted(by_arch.items()):
                print(f"  {arch}: {count}")


# ── Tests ────────────────────────────────────────────────────────────

def run_tests():
    """Quick smoke test — small simulation."""
    print("Test 1: Persona generation...")
    personas = generate_personas(20)
    assert len(personas) == 20
    archetypes = {p.archetype for p in personas}
    assert "specialist" in archetypes or "generalist" in archetypes
    assert "newcomer" in archetypes
    print("  PASS — 20 personas, archetypes:", archetypes)

    print("Test 2: Mini simulation (6 agents, 30 rounds)...")
    sim = MCPSimulation(n_agents=6, n_rounds=30, seed=1)
    final = sim.run(quiet=True)
    assert final["total"] >= 6
    assert final["mcp_calls"] > 0
    print(f"  PASS — alive={final['alive']}/{final['total']}, "
          f"calls={final['mcp_calls']}, errors={final['mcp_errors']}")

    print("Test 3: Metrics structure...")
    assert "gini" in final
    assert "newcomer_survival" in final
    assert "archetype_avg_earnings" in final
    assert "mcp_error_rate" in final
    print("  PASS — all metric fields present")

    print(f"\nAll tests passed!")


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ClawBizarre Multi-Agent MCP Simulation")
    parser.add_argument("--agents", type=int, default=20, help="Number of agents")
    parser.add_argument("--rounds", type=int, default=200, help="Number of rounds")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    args = parser.parse_args()

    if args.test:
        run_tests()
    else:
        sim = MCPSimulation(n_agents=args.agents, n_rounds=args.rounds, seed=args.seed)
        sim.run()
