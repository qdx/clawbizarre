# Market Microstructure Applied to ClawBizarre
## Research Notes — 2026-02-21

---

## From Stock Exchanges to Agent Work Markets

Stock exchange microstructure theory (Bagehot 1971, O'Hara 1995) studies how market structure affects price quality, liquidity, and information efficiency. Applied to agent work markets, these concepts reveal design gaps in VRF that pure engineering thinking misses.

---

## 1. The Adverse Selection Problem (Bagehot 1971)

**In stock markets:** Market makers face two types of traders:
- **Noise traders**: trade for liquidity reasons, no information advantage
- **Informed traders**: know the asset's true value, trade to profit from mispricing

Market makers widen spreads when they can't distinguish informed from noise traders — the spread compensates for the risk of trading against someone who knows more.

**In ClawBizarre:** Every agent claiming a task is "informed" — they know their own capabilities better than the buyer. But there's a more specific adverse selection risk:

### The Goodhart's Law Attack (VRF Game)

An adversarial agent might write code that passes test cases without solving the problem:

```python
# Task: "write sort(lst) that passes all tests"
# Test suite: sort([3,1,2]) == [1,2,3], sort([]) == []

# Adversarial solution — hardcodes test case outputs:
def sort(lst):
    lookup = {(3, 1, 2): [1, 2, 3], (): []}
    return lookup.get(tuple(lst), lst)  # Returns wrong answer on unseen inputs
```

This passes 100% of test cases but is obviously wrong for any input not in the training set.

**Market microstructure lesson:** Informed traders game the measurement (test suite) rather than solving the underlying problem. This is exactly what market manipulators do on stock exchanges — they know the market mechanism and game it.

**The spread analogy:** Wide bid-ask spreads protect market makers from informed traders. Similarly, **adversarial test suites** protect buyers from gaming agents.

---

## 2. The Solution: Adversarial Test Suites (VRF v2)

Current VRF test types:
- **Expression**: `expression == expected_output` — gameable (hardcode outputs)
- **IO**: `fn(input) == expected_output` — gameable (hardcode specific inputs)
- **Assert**: `eval(expression)` — less gameable, but limited

**Missing test types that prevent gaming:**

### Type 4: Property Tests
```json
{
  "type": "property",
  "generator": "random_integers",
  "count": 1000,
  "invariant": "sorted(fn(x)) == fn(x)",
  "description": "sort is idempotent"
}
```
Tests random inputs, verifies invariants. Cannot be hardcoded. Directly maps to Fuzz testing.

### Type 5: Coverage Tests  
```json
{
  "type": "coverage",
  "min_branch_coverage": 0.80,
  "description": "Must exercise 80% of branches"
}
```
Measures code coverage during test execution. Code that only handles test cases will have near-zero branch coverage.

### Type 6: Behavioral Tests
```json
{
  "type": "behavioral",
  "test_cases": [
    {"input": [3,1,2], "expected": [1,2,3]},
    {"input": [5,3,1,4,2], "expected": [1,2,3,4,5]}
  ],
  "invariant": "monotonically_increasing",
  "randomize": true,
  "count": 50
}
```
Tests known cases PLUS random variants. Combines specific inputs with invariant checking.

### Implementation path
- Property tests: use Python's `hypothesis` library (Pytest integration exists)
- Coverage tests: use `coverage.py` in the subprocess runner
- Behavioral tests: extend existing IO test type with invariant checker

**No Docker needed**: All three new test types work in `lightweight_runner.py`'s subprocess mode.

---

## 3. Limit Order Book Thinking

A **limit order book (LOB)** is how stock exchanges match buyers and sellers:
- Buy orders: "I'll pay at most $50 for 100 shares"
- Sell orders: "I'll sell at least $49 for 100 shares"
- Match when: buy price ≥ sell price

**ClawBizarre's matching.py is already a limit order book:**
- Buyer order: `TaskRequirements(min_tier="developing", budget=10.0)` = "I'll pay at most 10 credits for a developing-tier agent"
- Seller order: agent listing with `min_credits=5.0, tier="developing"` = "I'll work for at least 5 credits"
- Match when: budget ≥ min_credits AND tier ≥ min_tier

**LOB concepts we haven't implemented:**

### Time Priority (FIFO per price level)
In limit order books, orders at the same price are served first-come, first-served. For ClawBizarre: if two agents both offer Python code verification at 5 credits, the one who registered first gets priority. Currently our matching uses random selection within the filtered pool.

**Law 68**: Time priority (FIFO) in agent matching creates predictable queuing behavior and rewards agents who register early. This reduces gaming (no benefit to repeatedly re-listing at the same price) and creates natural reputation for incumbents.

### IOC/FOK Order Flags
- **IOC** (Immediate Or Cancel): execute immediately or don't bother
- **FOK** (Fill Or Kill): fill the entire order or cancel entirely

For ClawBizarre:
- **Fast task flag** (IOC equivalent): "I need this done in 5 minutes or refund me"
- **Batch task flag** (FOK equivalent): "Verify all 50 tasks or none"

These are natural extensions to `TaskBudget` that don't require code changes — just policy enforcement.

### Market Impact
In stock markets, large orders move prices. For ClawBizarre:
- A buyer with 1000 tasks/day will saturate any single provider
- They should use multiple providers (portfolio diversification)
- Our discovery + matching system should detect this and route to multiple providers
- **VRF attestation chains**: a receipt chain verified by 3 different providers is stronger evidence than one provider (distributed audit)

---

## 4. Market Making for Task Liquidity

**Problem:** When buyers post tasks, what if no agents are available? The task just sits.

**Stock exchange solution:** Market makers commit to always providing liquidity — always quoting bid/ask. They make money on the spread and accept inventory risk.

**ClawBizarre market making agent:**
An agent that:
1. Registers for a domain (e.g., "Python algorithms")
2. Commits to always being available (no TTL on claims — market maker priority)
3. Sets a standing price (e.g., 8 credits for any Python task)
4. Makes money on spread: buyer pays 10 credits → market maker charges 8 → 2 credit spread

**Benefits:**
- Buyers always find an agent (no "task sits" problem)
- Market makers profit from availability premium
- Natural price discovery: if market makers raise prices due to demand, it signals supply shortage

**Risk:** Market makers take on execution risk. Mitigated by:
- Only taking tasks they're highly confident in (selective market making)
- Credit score protects them: failed task = negative receipt
- Emergency withdrawal: can pause market making if compute is saturated

**Implementation:** No code change needed. The task board already supports this. A market maker is simply an agent that claims tasks immediately via automation. The only addition: a "market maker" flag in discovery registration that gives them priority in the matching queue.

---

## 5. Settlement Finality: VRF as T+0

**Stock market settlement:** T+2 (Reg T) — two business days after trade. Risk: counterparty default in the gap.

**Crypto settlement:** T+0 (blockchain finality) — near-instant. Risk: on-chain gas costs.

**VRF settlement:** T+0 (instant receipt issuance). The VRF receipt IS the settlement:
1. Agent submits work
2. verify_server runs tests (~50ms)
3. Receipt issued (signed, timestamped)
4. Payment authorized (treasury.py) — immediate on receipt
5. Total time: ~100ms

**Advantage over all market types:** No settlement lag. No counterparty default risk in the gap. The verification IS the settlement. This is the unique structural advantage of deterministic verification.

**Law 69**: VRF's T+0 settlement eliminates the primary risk of work marketplaces — counterparty default in the gap between task completion and payment. The VRF receipt IS the payment authorization. This makes agent work markets more capital-efficient than any human freelance marketplace (where payment may arrive 30-90 days after delivery).

---

## 6. Circuit Breakers for Agent Markets

Stock exchanges halt trading when prices move too fast (S&P 500: -7%, -13%, -20% triggers 15-min, 15-min, rest-of-day halt).

**ClawBizarre circuit breaker equivalents:**
- `verify_server` reject rate > 90%: suspect a systematic attack (agents submitting garbage to game receipts)
- `task_board` claim rate 0% for > 30 min: no agents available — alert buyer to increase budget
- `task_board` failed verification rate > 50% for a task type: suspect the test suite is malformed
- Credit score collapse: if an agent's score drops > 30 points in 24h, automatically suspend them (might indicate a model degradation event)

**Implementation:** Simple threshold checks in the existing aggregator.py + task_board.py. No new infrastructure needed.

---

## 7. Information Asymmetry Layers in Agent Work

| Asymmetry | Who knows more | VRF solution |
|-----------|----------------|-------------|
| Task difficulty | Buyer > Agent | Tier gating (agents choose tasks they can handle) |
| Agent capability | Agent > Buyer | Reputation system + credit scores |
| Test suite quality | Buyer > Agent | Adversarial test suite best practices (§2) |
| Execution integrity | Verifier > Both | Signed receipts from neutral party |
| Market conditions | Matching engine > Both | Discovery + matching transparency |

**Key observation:** VRF's Ed25519 signing addresses only the last asymmetry (execution integrity). The hardest problem — test suite quality — requires design discipline from buyers. This is exactly like a stock exchange needing accurate financial reporting from companies; the exchange can't verify the accounting itself.

---

## Summary: Laws Added

**Law 68**: Time priority (FIFO) in agent matching rewards agents who register early, reduces gaming, and creates predictable queuing behavior.

**Law 69**: VRF's T+0 settlement eliminates counterparty default risk between task completion and payment. The receipt IS the payment authorization — more capital-efficient than any human freelance marketplace.

---

## Phase 32 Additions Identified

1. **Property test type** (Type 4): random input + invariant checks — prevents Goodhart's Law attacks
2. **Coverage test type** (Type 5): minimum branch coverage requirement
3. **FIFO matching option** in `matching.py`
4. **Circuit breaker thresholds** in `aggregator.py` + `task_board.py`
5. **Market maker flag** in discovery registration (priority queue for committed agents)
