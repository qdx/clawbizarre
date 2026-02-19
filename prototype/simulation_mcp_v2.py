"""
MCP Simulation v2 — Newcomer Protection Mechanisms

Extends simulation_mcp.py with:
1. Discovery reserve: X% of find_providers results reserved for newcomers (< N rounds old)
2. Bootstrap subsidy: newcomers get bonus initial balance
3. Archetype-ratio sweep: vary specialist:generalist:opportunist mix

Usage:
    python3 simulation_mcp_v2.py --test          # Smoke tests
    python3 simulation_mcp_v2.py --sweep          # Full parameter sweep
    python3 simulation_mcp_v2.py --rounds 200     # Single run with defaults
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
from simulation_mcp import (
    MCPClient, AgentPersona, SimAgent, CAPABILITIES,
)


def generate_personas_v2(n_agents: int, newcomer_fraction: float = 0.2,
                          specialist_fraction: float = 0.4,
                          generalist_fraction: float = 0.3) -> list:
    """Generate personas with configurable archetype ratios."""
    personas = []
    n_newcomers = int(n_agents * newcomer_fraction)
    n_regular = n_agents - n_newcomers

    # Remaining fraction goes to opportunists
    opp_fraction = max(0, 1.0 - specialist_fraction - generalist_fraction)

    for i in range(n_regular):
        r = random.random()
        if r < specialist_fraction:
            cap = random.choice(CAPABILITIES)
            buy = [c for c in CAPABILITIES if c != cap]
            personas.append(AgentPersona(
                name=f"specialist_{i}", archetype="specialist",
                capabilities=[cap],
                buy_capabilities=random.sample(buy, min(2, len(buy))),
                quality_range=(0.8, 1.0), price_multiplier=1.3, entry_round=0,
            ))
        elif r < specialist_fraction + generalist_fraction:
            caps = random.sample(CAPABILITIES, random.randint(2, 4))
            buy = [c for c in CAPABILITIES if c not in caps]
            personas.append(AgentPersona(
                name=f"generalist_{i}", archetype="generalist",
                capabilities=caps,
                buy_capabilities=random.sample(buy, min(2, len(buy))) if buy else caps[:1],
                quality_range=(0.6, 0.9), price_multiplier=1.0, entry_round=0,
            ))
        else:
            cap = random.choice(CAPABILITIES)
            personas.append(AgentPersona(
                name=f"opportunist_{i}", archetype="opportunist",
                capabilities=[cap],
                buy_capabilities=random.sample(CAPABILITIES, 2),
                quality_range=(0.5, 0.85), price_multiplier=0.85, entry_round=0,
            ))

    for i in range(n_newcomers):
        cap = random.choice(CAPABILITIES)
        buy = [c for c in CAPABILITIES if c != cap]
        personas.append(AgentPersona(
            name=f"newcomer_{i}", archetype="newcomer",
            capabilities=[cap],
            buy_capabilities=random.sample(buy, min(2, len(buy))),
            quality_range=(0.5, 0.9), price_multiplier=0.9,
            entry_round=random.randint(1, 150),
        ))

    return personas


class MCPSimulationV2:
    """MCP Simulation with newcomer protection mechanisms."""

    def __init__(self, n_agents=12, n_rounds=80, seed=42,
                 step_cost=0.1, initial_balance=10.0,
                 # Newcomer protection params
                 discovery_reserve=0.0,    # fraction of results reserved for newcomers
                 newcomer_threshold=20,     # rounds before agent is no longer "new"
                 bootstrap_subsidy=0.0,     # extra balance for newcomers
                 # Archetype ratios
                 specialist_fraction=0.4,
                 generalist_fraction=0.3,
                 newcomer_fraction=0.2,
                 ):
        self.n_rounds = n_rounds
        self.seed = seed
        self.step_cost = step_cost
        self.initial_balance = initial_balance
        self.discovery_reserve = discovery_reserve
        self.newcomer_threshold = newcomer_threshold
        self.bootstrap_subsidy = bootstrap_subsidy
        random.seed(seed)

        self.tmpdir = tempfile.mkdtemp(prefix="clawbiz_v2_")
        self.db_path = os.path.join(self.tmpdir, "sim.db")
        self.port = self._find_port()
        self.server_thread = None

        self.personas = generate_personas_v2(
            n_agents, newcomer_fraction=newcomer_fraction,
            specialist_fraction=specialist_fraction,
            generalist_fraction=generalist_fraction,
        )
        self.agents: list[SimAgent] = []
        self.agent_entry_round: dict[str, int] = {}  # agent_id → round entered
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

    def _create_agent(self, persona: AgentPersona, round_num: int = 0) -> SimAgent:
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

        whoami = client.tool("cb_whoami")
        agent_id = whoami.get("agent_id", "") if whoami else ""

        bal = self.initial_balance
        if persona.archetype == "newcomer":
            bal += self.bootstrap_subsidy

        agent = SimAgent(persona=persona, client=client, agent_id=agent_id, balance=bal)
        agent._step_cost = self.step_cost
        self.agent_entry_round[agent_id] = round_num
        return agent

    def _is_newcomer(self, agent_id: str, round_num: int) -> bool:
        entry = self.agent_entry_round.get(agent_id, 0)
        return (round_num - entry) < self.newcomer_threshold

    def _list_services(self, agent: SimAgent):
        if agent.listed:
            return
        for cap in agent.persona.capabilities:
            price = agent.persona.pick_price(self.base_price)
            result = agent.client.tool("cb_list_service", {
                "capability": cap, "base_rate": price, "unit": "per_task",
                "description": f"{agent.persona.name} offering {cap}",
            })
            if result and result.get("status") == "listed":
                agent.listed = True

    def _apply_discovery_reserve(self, providers: list, buyer_id: str, round_num: int) -> list:
        """Apply newcomer discovery reserve: ensure X% of results are from new agents."""
        if self.discovery_reserve <= 0 or not providers:
            return providers

        newcomer_providers = [p for p in providers if self._is_newcomer(p.get("agent_id", ""), round_num)]
        veteran_providers = [p for p in providers if not self._is_newcomer(p.get("agent_id", ""), round_num)]

        if not newcomer_providers:
            return providers

        # Reserve slots for newcomers
        n_total = len(providers)
        n_reserved = max(1, int(n_total * self.discovery_reserve))
        n_reserved = min(n_reserved, len(newcomer_providers))

        # Fill reserved slots with newcomers, rest with veterans
        reserved = random.sample(newcomer_providers, n_reserved)
        remaining_slots = n_total - n_reserved
        veterans_selected = veteran_providers[:remaining_slots]

        result = reserved + veterans_selected
        random.shuffle(result)
        return result

    def _find_and_trade(self, buyer: SimAgent, round_num: int) -> bool:
        if not buyer.persona.buy_capabilities:
            return False

        cap = random.choice(buyer.persona.buy_capabilities)
        providers = buyer.client.tool("cb_find_providers", {"capability": cap})

        if not providers or (isinstance(providers, dict) and not providers.get("providers")):
            return False
        if not isinstance(providers, list):
            return False

        providers = [p for p in providers if p.get("agent_id") != buyer.agent_id]
        if not providers:
            return False

        # Apply discovery reserve
        providers = self._apply_discovery_reserve(providers, buyer.agent_id, round_num)

        # Selection strategy: top3_random (best from v6b findings)
        if len(providers) > 3:
            # Sort by reputation desc, pick random from top 3
            providers.sort(key=lambda p: p.get("reputation", 0), reverse=True)
            provider = random.choice(providers[:3])
        elif buyer.persona.archetype == "opportunist":
            provider = min(providers, key=lambda p: p.get("base_rate", 999))
        else:
            provider = max(providers, key=lambda p: p.get("reputation", 0))

        price = provider.get("base_rate", provider.get("price", 1.0))
        if price > buyer.balance:
            return False

        # Handshake
        result = buyer.client.tool("cb_initiate_task", {
            "provider_id": provider["agent_id"], "capability": cap,
            "description": f"Round {round_num}: {buyer.persona.name} needs {cap}",
        })
        if not result:
            return False

        session_id = result if isinstance(result, str) else result.get("session_id", "")
        if not session_id:
            return False

        provider_agent = None
        for a in self.agents:
            if a.agent_id == provider["agent_id"] and a.alive:
                provider_agent = a
                break
        if not provider_agent:
            return False

        accept = provider_agent.client.tool("cb_accept_task", {"session_id": session_id})
        if not accept:
            return False

        quality = provider_agent.persona.pick_quality()
        execute = provider_agent.client.tool("cb_submit_work", {
            "session_id": session_id,
            "output": f"Work output for {cap} (quality={quality:.2f})",
            "proof": json.dumps({"quality": quality, "tests_passed": int(quality * 10)}),
        })
        if not execute:
            return False

        accept_work = quality >= 0.5
        verify = buyer.client.tool("cb_verify_work", {
            "session_id": session_id, "quality_score": quality, "accept": accept_work,
        })
        if not verify:
            return False

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
        if agent.persona.archetype != "opportunist" or random.random() > 0.1:
            return
        new_cap = random.choice(CAPABILITIES)
        if new_cap not in agent.persona.capabilities:
            agent.client.tool("cb_unlist_service", {"capability": agent.persona.capabilities[0]})
            agent.persona.capabilities = [new_cap]
            agent.listed = False

    def _collect_metrics(self, round_num: int) -> dict:
        alive = [a for a in self.agents if a.alive]
        earnings = [a.earnings for a in alive]

        gini = 0.0
        if len(earnings) > 1 and sum(earnings) > 0:
            sorted_e = sorted(earnings)
            n = len(sorted_e)
            cum = sum((i + 1) * e for i, e in enumerate(sorted_e))
            gini = (2 * cum) / (n * sum(sorted_e)) - (n + 1) / n

        newcomers = [a for a in self.agents if a.persona.archetype == "newcomer" and a.persona.entry_round <= round_num]
        newcomer_alive = sum(1 for a in newcomers if a.alive)
        newcomer_survival = newcomer_alive / len(newcomers) if newcomers else 1.0

        # Newcomer work rate
        newcomer_tasks = sum(a.tasks_completed for a in newcomers) if newcomers else 0

        archetype_earnings = {}
        archetype_alive = {}
        for a in self.agents:
            if a.persona.entry_round <= round_num:
                at = a.persona.archetype
                archetype_earnings.setdefault(at, []).append(a.earnings)
                archetype_alive.setdefault(at, [0, 0])
                archetype_alive[at][1] += 1
                if a.alive:
                    archetype_alive[at][0] += 1

        total_calls = sum(a.client.call_count for a in self.agents)
        total_errors = sum(a.client.error_count for a in self.agents)

        return {
            "round": round_num,
            "alive": len(alive),
            "total": len(self.agents),
            "gini": round(gini, 3),
            "newcomer_survival": round(newcomer_survival, 3),
            "newcomer_tasks": newcomer_tasks,
            "avg_balance": round(sum(a.balance for a in alive) / len(alive), 2) if alive else 0,
            "total_earnings": round(sum(earnings), 2),
            "archetype_alive": {k: f"{v[0]}/{v[1]}" for k, v in archetype_alive.items()},
            "mcp_calls": total_calls,
            "mcp_errors": total_errors,
            "mcp_error_rate": round(total_errors / total_calls, 4) if total_calls > 0 else 0,
        }

    def run(self, quiet: bool = False) -> dict:
        if not quiet:
            print(f"=== MCP Simulation v2 ===")
            print(f"Agents: {len(self.personas)}, Rounds: {self.n_rounds}, Seed: {self.seed}")
            print(f"Discovery reserve: {self.discovery_reserve:.0%}, "
                  f"Bootstrap subsidy: ${self.bootstrap_subsidy}, "
                  f"Step cost: ${self.step_cost}")
            print()

        self._start_server()

        for p in self.personas:
            if p.entry_round == 0:
                agent = self._create_agent(p, 0)
                self.agents.append(agent)

        for round_num in range(self.n_rounds):
            for p in self.personas:
                if p.entry_round == round_num and p.entry_round > 0:
                    agent = self._create_agent(p, round_num)
                    self.agents.append(agent)

            alive_agents = [a for a in self.agents if a.alive]

            for a in alive_agents:
                self._list_services(a)

            trades = 0
            random.shuffle(alive_agents)
            for a in alive_agents:
                if random.random() < 0.6:
                    if self._find_and_trade(a, round_num):
                        trades += 1

            for a in alive_agents:
                self._opportunist_adapt(a)

            for a in alive_agents:
                a.balance -= a.step_cost()
                if a.balance <= 0:
                    a.alive = False

            if round_num % 10 == 0:
                m = self._collect_metrics(round_num)
                m["trades"] = trades
                self.round_metrics.append(m)

        final = self._collect_metrics(self.n_rounds)
        self.round_metrics.append(final)

        if not quiet:
            print(f"Alive: {final['alive']}/{final['total']}, Gini: {final['gini']}, "
                  f"Newcomer surv: {final['newcomer_survival']:.0%}, "
                  f"Newcomer tasks: {final['newcomer_tasks']}, "
                  f"Earnings: ${final['total_earnings']}")
            print(f"Archetype alive: {final['archetype_alive']}")

        self._httpd.shutdown()
        return final


def run_sweep():
    """Run parameter sweep: discovery reserve × bootstrap subsidy × archetype ratio."""
    configs = [
        # Baseline: no protection
        {"name": "baseline", "discovery_reserve": 0, "bootstrap_subsidy": 0},
        # Discovery reserve only
        {"name": "reserve_15%", "discovery_reserve": 0.15, "bootstrap_subsidy": 0},
        {"name": "reserve_30%", "discovery_reserve": 0.30, "bootstrap_subsidy": 0},
        # Bootstrap subsidy only
        {"name": "subsidy_$5", "discovery_reserve": 0, "bootstrap_subsidy": 5},
        {"name": "subsidy_$10", "discovery_reserve": 0, "bootstrap_subsidy": 10},
        # Combined
        {"name": "reserve_15%+sub_$5", "discovery_reserve": 0.15, "bootstrap_subsidy": 5},
        {"name": "reserve_30%+sub_$10", "discovery_reserve": 0.30, "bootstrap_subsidy": 10},
        # Archetype ratios (generalist-heavy vs specialist-heavy)
        {"name": "gen_heavy", "discovery_reserve": 0.15, "bootstrap_subsidy": 5,
         "specialist_fraction": 0.2, "generalist_fraction": 0.5},
        {"name": "spec_heavy", "discovery_reserve": 0.15, "bootstrap_subsidy": 5,
         "specialist_fraction": 0.6, "generalist_fraction": 0.1},
    ]

    results = []
    print("=" * 90)
    print(f"{'Config':<25} {'Alive':>6} {'Gini':>6} {'NewSurv':>8} {'NewTasks':>9} {'Earnings':>10}")
    print("-" * 90)

    for cfg in configs:
        name = cfg.pop("name")
        sim = MCPSimulationV2(
            n_agents=12, n_rounds=80, seed=42,
            step_cost=0.1, initial_balance=10.0,
            **cfg,
        )
        final = sim.run(quiet=True)
        results.append({"name": name, **final})
        print(f"{name:<25} {final['alive']:>3}/{final['total']:<2} {final['gini']:>6.3f} "
              f"{final['newcomer_survival']:>7.0%} {final['newcomer_tasks']:>9} "
              f"${final['total_earnings']:>8.2f}")

    print("=" * 90)
    return results


def run_tests():
    print("Test 1: generate_personas_v2...")
    p = generate_personas_v2(12, specialist_fraction=0.5, generalist_fraction=0.3)
    assert len(p) == 12
    print("  PASS")

    print("Test 2: Mini sim with protection...")
    sim = MCPSimulationV2(n_agents=6, n_rounds=20, seed=1,
                           discovery_reserve=0.3, bootstrap_subsidy=5)
    final = sim.run(quiet=True)
    assert final["total"] >= 6
    assert final["mcp_calls"] > 0
    print(f"  PASS — alive={final['alive']}/{final['total']}")

    print("Test 3: Discovery reserve logic...")
    sim2 = MCPSimulationV2(n_agents=6, n_rounds=1, seed=1, discovery_reserve=0.5)
    # Just test the method directly
    providers = [
        {"agent_id": "new1"}, {"agent_id": "vet1"},
        {"agent_id": "vet2"}, {"agent_id": "vet3"},
    ]
    sim2.agent_entry_round = {"new1": 95, "vet1": 0, "vet2": 0, "vet3": 0}
    sim2.newcomer_threshold = 20
    # At round 100: new1 entered at 95 (5 rounds ago, < 20 threshold → newcomer)
    # vets entered at 0 (100 rounds ago, >= 20 → veteran)
    result = sim2._apply_discovery_reserve(providers, "buyer", round_num=100)
    assert len(result) == 4
    # new1 should be in results (reserved)
    ids = [p["agent_id"] for p in result]
    assert "new1" in ids, f"Newcomer should be in reserved results, got {ids}"
    print("  PASS")

    print("\nAll tests passed!")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", action="store_true")
    parser.add_argument("--sweep", action="store_true")
    parser.add_argument("--agents", type=int, default=12)
    parser.add_argument("--rounds", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.sweep:
        run_sweep()
    else:
        sim = MCPSimulationV2(n_agents=args.agents, n_rounds=args.rounds, seed=args.seed,
                               discovery_reserve=0.15, bootstrap_subsidy=5)
        sim.run()
