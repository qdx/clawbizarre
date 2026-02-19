"""
ClawBizarre Multi-Agent Simulation
Simulates a small economy of agents trading self-verifying services,
building receipt chains, and developing reputations over time.

Demonstrates: identity → handshake → receipt → chain → reputation emergence
"""

import json
import random
from dataclasses import dataclass, field
from collections import defaultdict

from identity import AgentIdentity, SignedReceipt, sign_receipt
from receipt import WorkReceipt, TestResults, VerificationTier, hash_content, ReceiptChain
from signed_handshake import SignedHandshakeSession


# --- Agent Definitions ---

@dataclass
class SimAgent:
    """An agent in the simulation with capabilities, identity, and receipt chain."""
    name: str
    identity: AgentIdentity
    capabilities: list[str]
    chain: ReceiptChain = field(default_factory=ReceiptChain)
    tasks_completed: int = 0
    tasks_failed: int = 0
    tasks_requested: int = 0
    receipts_issued: list[SignedReceipt] = field(default_factory=list)
    receipts_received: list[SignedReceipt] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        total = self.tasks_completed + self.tasks_failed
        return self.tasks_completed / total if total > 0 else 0.0

    @property
    def reputation_score(self) -> float:
        """Simple reputation: success rate weighted by volume."""
        total = self.tasks_completed + self.tasks_failed
        if total == 0:
            return 0.0
        # Bayesian average: assume 2 prior successes, 1 prior failure (shrinks toward 0.67)
        return (self.tasks_completed + 2) / (total + 3)


# --- Task Types ---

TASK_TYPES = {
    "code_review": {
        "capabilities_needed": ["code_review"],
        "make_test": lambda inp: lambda out: "vulnerability" in out.lower() or "issue" in out.lower(),
        "make_input": lambda: f"def process(x): return eval(x)  # line {random.randint(1,100)}",
        "good_output": lambda: "Security issue: eval() is dangerous, use ast.literal_eval or validated parsing",
        "bad_output": lambda: "Looks fine to me",
    },
    "translation": {
        "capabilities_needed": ["translation"],
        "make_test": lambda inp: lambda out: len(out) > 10 and out != inp,
        "make_input": lambda: random.choice([
            "The quick brown fox jumps over the lazy dog",
            "Hello world, this is a test",
            "Agent economics will reshape digital labor",
        ]),
        "good_output": lambda: "敏捷的棕色狐狸跳过了那只懒狗",
        "bad_output": lambda: "no",
    },
    "summarization": {
        "capabilities_needed": ["research"],
        "make_test": lambda inp: lambda out: len(out) > 20 and len(out) < len(inp),
        "make_input": lambda: "A" * random.randint(200, 500) + " important conclusion here " + "B" * 100,
        "good_output": lambda: "The document discusses multiple topics and reaches an important conclusion.",
        "bad_output": lambda: "x",
    },
    "data_validation": {
        "capabilities_needed": ["data_validation"],
        "make_test": lambda inp: lambda out: "valid" in out.lower() or "invalid" in out.lower(),
        "make_input": lambda: json.dumps({"email": f"test{random.randint(1,999)}@example.com", "age": random.randint(-5, 120)}),
        "good_output": lambda: "Valid: email format correct, age within range" if random.random() > 0.3 else "Invalid: age out of acceptable range",
        "bad_output": lambda: "ok",
    },
}


def run_handshake(requester: SimAgent, worker: SimAgent, task_type: str,
                  will_succeed: bool = True) -> tuple[bool, str]:
    """Run a full signed handshake between two agents. Returns (success, description)."""
    task_def = TASK_TYPES[task_type]

    req_session = SignedHandshakeSession(requester.identity)
    wrk_session = SignedHandshakeSession(worker.identity)

    # Exchange hellos
    req_hello = req_session.send_hello(requester.capabilities)
    wrk_session.receive_hello(req_hello)
    wrk_hello = wrk_session.send_hello(worker.capabilities)
    req_session.receive_hello(wrk_hello)

    # Propose
    input_data = task_def["make_input"]()
    test_fn = task_def["make_test"](input_data)
    test_suite_repr = f"test_for_{task_type}"

    proposal = req_session.propose(
        task_description=f"Perform {task_type}",
        task_type=task_type,
        verification_tier=VerificationTier.SELF_VERIFYING,
        test_suite_hash=hash_content(test_suite_repr),
        input_data=input_data,
    )
    wrk_session.receive_proposal(proposal)

    # Worker accepts
    accept_msg = wrk_session.accept()
    req_session.receive_accept(accept_msg)

    # Worker executes
    if will_succeed:
        output = task_def["good_output"]()
    else:
        output = task_def["bad_output"]()

    execute_msg = wrk_session.execute(output)

    # Requester verifies
    def verifier(payload):
        out = payload.get("output", "")
        passed = test_fn(out)
        return TestResults(
            passed=1 if passed else 0,
            failed=0 if passed else 1,
            suite_hash=hash_content(test_suite_repr),
        )

    verify_msg, signed_receipt = req_session.verify_and_receipt(execute_msg, verifier)

    success = req_session.state == HandshakeState.COMPLETE if hasattr(req_session, 'state') else signed_receipt is not None

    # Check actual state
    success = req_session.state.value == "complete"

    if success and signed_receipt:
        # Both parties record the receipt
        requester.receipts_issued.append(signed_receipt)
        worker.receipts_received.append(signed_receipt)
        worker.chain.append(req_session.session.receipt)
        worker.tasks_completed += 1
        requester.tasks_requested += 1
        return True, f"{requester.name} → {worker.name}: {task_type} ✓"
    else:
        worker.tasks_failed += 1
        requester.tasks_requested += 1
        return False, f"{requester.name} → {worker.name}: {task_type} ✗"


# Need this import for state check
from handshake import HandshakeState


def discover_worker(agents: list[SimAgent], requester: SimAgent, task_type: str) -> SimAgent | None:
    """Simple discovery: find an agent with matching capabilities (not self)."""
    needed = TASK_TYPES[task_type]["capabilities_needed"]
    candidates = [a for a in agents if a.name != requester.name
                  and any(c in a.capabilities for c in needed)]
    if not candidates:
        return None

    # Reputation-weighted selection: prefer agents with higher reputation
    weights = [max(a.reputation_score, 0.1) for a in candidates]
    return random.choices(candidates, weights=weights, k=1)[0]


def run_simulation(num_rounds: int = 50, seed: int = 42):
    """Run a multi-agent economy simulation."""
    random.seed(seed)

    # Create agents with different capability profiles
    agents = [
        SimAgent("Alice", AgentIdentity.generate(), ["code_review", "research"]),
        SimAgent("Bob", AgentIdentity.generate(), ["code_review", "translation"]),
        SimAgent("Carol", AgentIdentity.generate(), ["translation", "data_validation"]),
        SimAgent("Dave", AgentIdentity.generate(), ["research", "data_validation"]),
        SimAgent("Eve", AgentIdentity.generate(), ["code_review", "translation", "research"]),  # generalist
        SimAgent("Frank", AgentIdentity.generate(), ["data_validation"]),  # specialist
    ]

    # Agent reliability profiles (probability of producing good output)
    reliability = {
        "Alice": 0.90,
        "Bob": 0.75,
        "Carol": 0.95,
        "Dave": 0.60,
        "Eve": 0.85,
        "Frank": 0.98,  # specialist = highly reliable in their domain
    }

    print(f"=== ClawBizarre Economy Simulation ===")
    print(f"Agents: {len(agents)}, Rounds: {num_rounds}")
    print(f"Task types: {list(TASK_TYPES.keys())}")
    print()

    results = []

    for round_num in range(1, num_rounds + 1):
        # Each round: a random agent needs a random task done
        requester = random.choice(agents)
        task_type = random.choice(list(TASK_TYPES.keys()))

        worker = discover_worker(agents, requester, task_type)
        if not worker:
            continue

        will_succeed = random.random() < reliability[worker.name]
        success, desc = run_handshake(requester, worker, task_type, will_succeed)
        results.append((round_num, success, desc))

        if round_num % 10 == 0:
            print(f"Round {round_num}: {desc}")

    # Final report
    print(f"\n{'='*60}")
    print(f"SIMULATION RESULTS ({num_rounds} rounds)")
    print(f"{'='*60}\n")

    total_success = sum(1 for _, s, _ in results if s)
    total_fail = sum(1 for _, s, _ in results if not s)
    print(f"Total transactions: {len(results)}")
    print(f"Successful: {total_success} ({100*total_success/len(results):.0f}%)")
    print(f"Failed: {total_fail} ({100*total_fail/len(results):.0f}%)")

    print(f"\n{'─'*60}")
    print(f"{'Agent':>8} │ {'Done':>4} │ {'Fail':>4} │ {'Asked':>5} │ {'Rep':>5} │ {'Chain':>5} │ Capabilities")
    print(f"{'─'*60}")
    for a in sorted(agents, key=lambda x: x.reputation_score, reverse=True):
        caps = ", ".join(a.capabilities)
        print(f"{a.name:>8} │ {a.tasks_completed:>4} │ {a.tasks_failed:>4} │ {a.tasks_requested:>5} │ {a.reputation_score:.3f} │ {a.chain.length:>5} │ {caps}")

    # Receipt chain integrity
    print(f"\n{'─'*60}")
    print("Receipt Chain Integrity:")
    for a in agents:
        if a.chain.length > 0:
            integrity = a.chain.verify_integrity()
            print(f"  {a.name}: {a.chain.length} receipts, integrity={'✓' if integrity else '✗'}")

    # Reputation-weighted discovery demo
    print(f"\n{'─'*60}")
    print("Reputation-Weighted Discovery (who gets chosen for code_review):")
    picks = defaultdict(int)
    for _ in range(1000):
        dummy = SimAgent("_req", AgentIdentity.generate(), ["research"])
        worker = discover_worker(agents, dummy, "code_review")
        if worker:
            picks[worker.name] += 1
    for name, count in sorted(picks.items(), key=lambda x: -x[1]):
        agent = next(a for a in agents if a.name == name)
        print(f"  {name}: {count/10:.1f}% selection rate (rep={agent.reputation_score:.3f})")

    # Key emergent properties
    print(f"\n{'─'*60}")
    print("Emergent Properties:")

    # Specialists vs generalists
    specialist_rep = next(a.reputation_score for a in agents if a.name == "Frank")
    generalist_rep = next(a.reputation_score for a in agents if a.name == "Eve")
    print(f"  Specialist (Frank) rep: {specialist_rep:.3f} vs Generalist (Eve) rep: {generalist_rep:.3f}")
    print(f"  → {'Specialist wins' if specialist_rep > generalist_rep else 'Generalist wins'} on reputation")

    frank = next(a for a in agents if a.name == "Frank")
    eve = next(a for a in agents if a.name == "Eve")
    print(f"  Frank tasks: {frank.tasks_completed}, Eve tasks: {eve.tasks_completed}")
    print(f"  → {'Generalist wins' if eve.tasks_completed > frank.tasks_completed else 'Specialist wins'} on volume")

    return agents, results


if __name__ == "__main__":
    agents, results = run_simulation(num_rounds=100, seed=42)
