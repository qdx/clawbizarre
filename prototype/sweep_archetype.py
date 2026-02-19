"""
Archetype Ratio Sweep — Specialist fraction 0.1 → 0.8

Tests: at 15 agents, 60 rounds, what specialist:generalist mix maximizes economy health?
Uses discovery reserve 15% (proven effective) as baseline.
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_mcp_v2 import MCPSimulationV2

def run_sweep():
    configs = []
    for spec_frac in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        gen_frac = max(0.0, 0.8 - spec_frac)  # remainder to generalists, 0.2 always newcomers
        configs.append({
            "name": f"spec_{int(spec_frac*100)}%_gen_{int(gen_frac*100)}%",
            "specialist_fraction": spec_frac,
            "generalist_fraction": gen_frac,
        })

    print("=" * 100)
    print(f"{'Config':<25} {'Alive':>6} {'Gini':>6} {'NewSurv':>8} {'NewTasks':>9} {'Earnings':>10} {'Archetypes'}")
    print("-" * 100)

    results = []
    for cfg in configs:
        name = cfg.pop("name")
        sim = MCPSimulationV2(
            n_agents=15, n_rounds=60, seed=42,
            step_cost=0.1, initial_balance=12.0,
            discovery_reserve=0.15, bootstrap_subsidy=0,
            newcomer_fraction=0.2,
            **cfg,
        )
        final = sim.run(quiet=True)
        results.append({"name": name, **final})
        arch = final.get("archetype_alive", {})
        arch_str = " ".join(f"{k}={v}" for k, v in sorted(arch.items()))
        print(f"{name:<25} {final['alive']:>3}/{final['total']:<2} {final['gini']:>6.3f} "
              f"{final['newcomer_survival']:>7.0%} {final['newcomer_tasks']:>9} "
              f"${final['total_earnings']:>8.2f}  {arch_str}")

    print("=" * 100)

    # Also run at 20 agents to test scale threshold
    print("\n\n=== 20 AGENTS (scale test) ===")
    print("=" * 100)
    print(f"{'Config':<25} {'Alive':>6} {'Gini':>6} {'NewSurv':>8} {'NewTasks':>9} {'Earnings':>10} {'Archetypes'}")
    print("-" * 100)

    for spec_frac in [0.3, 0.5, 0.7]:
        gen_frac = max(0.0, 0.8 - spec_frac)
        name = f"20ag_spec_{int(spec_frac*100)}%"
        sim = MCPSimulationV2(
            n_agents=20, n_rounds=60, seed=42,
            step_cost=0.1, initial_balance=12.0,
            discovery_reserve=0.15, bootstrap_subsidy=0,
            newcomer_fraction=0.2,
            specialist_fraction=spec_frac,
            generalist_fraction=gen_frac,
        )
        final = sim.run(quiet=True)
        results.append({"name": name, **final})
        arch = final.get("archetype_alive", {})
        arch_str = " ".join(f"{k}={v}" for k, v in sorted(arch.items()))
        print(f"{name:<25} {final['alive']:>3}/{final['total']:<2} {final['gini']:>6.3f} "
              f"{final['newcomer_survival']:>7.0%} {final['newcomer_tasks']:>9} "
              f"${final['total_earnings']:>8.2f}  {arch_str}")

    print("=" * 100)
    return results


if __name__ == "__main__":
    run_sweep()
