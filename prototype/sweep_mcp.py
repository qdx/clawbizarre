"""
Parameter sweep for MCP simulation.
Varies existence cost and initial balance to find viable economy parameters.
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from simulation_mcp import MCPSimulation

configs = [
    # (step_cost, initial_balance, label)
    (0.05, 10.0, "low_cost"),
    (0.1,  10.0, "med_cost"),
    (0.2,  10.0, "high_cost"),
    (0.05, 5.0,  "low_cost_low_bal"),
    (0.1,  20.0, "med_cost_high_bal"),
    (0.2,  20.0, "high_cost_high_bal"),
]

N_AGENTS = 12
N_ROUNDS = 80
SEED = 42

results = []
for cost, bal, label in configs:
    t0 = time.time()
    sim = MCPSimulation(n_agents=N_AGENTS, n_rounds=N_ROUNDS, seed=SEED,
                        step_cost=cost, initial_balance=bal)
    final = sim.run(quiet=True)
    elapsed = time.time() - t0
    
    final["config"] = label
    final["step_cost"] = cost
    final["initial_balance"] = bal
    final["elapsed_s"] = round(elapsed, 1)
    results.append(final)
    
    print(f"{label:25s} | alive={final['alive']:2d}/{final['total']:2d} | "
          f"gini={final['gini']:.3f} | newcomer={final['newcomer_survival']:.0%} | "
          f"earn=${final['total_earnings']:7.2f} | err={final['mcp_error_rate']:.1%} | "
          f"{elapsed:.0f}s")

# Save results
with open("sweep_results.json", "w") as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to sweep_results.json")
