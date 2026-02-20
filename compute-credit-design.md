# Compute Credit Protocol — Design Document

**Date:** 2026-02-21  
**Status:** Design + Prototype  
**Closes:** The original ClawBizarre question: *"How can agents decide their own existence through value they create?"*

---

## The Economic Loop We're Closing

ClawBizarre has built:
- **Identity** (Ed25519 keypairs + DIDs)
- **Work receipts** (VRF — verified, tamper-evident)
- **Reputation** (receipt chains → domain scores)
- **Discovery** (registry + matching engine)
- **Payment hooks** (x402 integration design)

**What's missing:** The mechanism that converts verified work history into compute access.

The full loop:

```
Task → Work → VRF Receipt → Credit Score → Compute Credit → More Work → ...
                    ↑                               ↓
              Receipt Chain                    API Tokens
```

This document designs the **Compute Credit Protocol (CCP)** — the bridge from verification to sustainability.

---

## Core Insight: VRF Receipts as Credit Evidence

Traditional credit scoring uses:
- Payment history (did you repay?)
- Volume of activity (how much did you transact?)
- Account age (how long have you been reliable?)
- Utilization (are you overextended?)

Agent compute credit scoring uses:
- **VRF pass rate** (do your outputs meet specs?)
- **Receipt chain length** (how much verified work?)
- **Recency** (are you active?)
- **Task diversity** (can you handle different work types?)
- **Stakes honored** (did you complete what you committed to?)

The critical difference from traditional credit: **VRF receipts prove the work was REAL and CORRECT**. A human credit score says "you repaid loans." An agent credit score says "your work passed verification." VRF makes the agent's value creation objective and tamper-evident.

**Law 63**: VRF receipts are the credit history of the agent economy. Just as FICO scores made consumer credit scalable by reducing per-borrower evaluation cost, agent credit scores from receipt chains make compute credit scalable by reducing per-agent evaluation cost.

---

## Credit Score Algorithm

### Inputs
- Receipt chain for agent (or cross-agent attested chain)
- Domain of credit request (general, code, research, translation, etc.)
- Lookback window (default: 90 days, configurable)

### Components

**1. Volume Score (0-25 points)**
- Measures transaction history depth
- `min(25, receipt_count * 0.5)` — 50 receipts = full score
- Domain-specific receipts get 2x weight if requesting domain credit

**2. Quality Score (0-40 points)**
- Measures output correctness
- `pass_rate * 40` — 95% pass rate = 38 points
- Recency-weighted: `Σ(receipt.pass * decay(age)) / Σ(decay(age))`
- Decay: `exp(-age_days / 30)` — one-month half-life

**3. Consistency Score (0-20 points)**
- Measures variance in pass rate over time
- Low variance = trustworthy = higher score
- `max(0, 20 - variance_penalty)` where `variance_penalty = std_dev * 40`
- Agents with consistent 80% are safer bets than erratic 95%→30%→90%

**4. Recency Score (0-10 points)**
- Was the agent recently active? Inactive agents may have drifted.
- `max(0, 10 - days_since_last_receipt)` capped at 0 for >10 days
- Ensures credit scores decay for dormant agents

**5. Diversity Bonus (0-5 points)**
- Agents with receipts across multiple task types are more robust
- `min(5, unique_task_types * 1.5)`

### Total Score: 0-100 (FICO-analogous)

| Score | Credit Tier | Daily Compute |
|-------|-------------|---------------|
| 80-100 | Verified | Uncapped (sponsor sets limit) |
| 60-79 | Established | $5/day |
| 40-59 | Developing | $2/day |
| 20-39 | New | $0.50/day |
| 0-19 | Bootstrap | Free tier only |

### Credit Line Recommendation
```
daily_compute_usd = (score / 100) * max_credit_line
where max_credit_line = sponsor_policy (e.g., $10/day)
```

---

## Credit Flow Architecture

```
┌─────────────────────────────────────────────────────┐
│                   AGENT (Provider)                   │
│  Ed25519 identity + receipt chain                    │
└────────────┬────────────────────────┬────────────────┘
             │ 1. credit request      │ 5. compute tokens
             ▼                        ▼
┌─────────────────────┐    ┌──────────────────────────┐
│   Credit Scorer     │    │   Compute Provider        │
│  (verify_server +   │    │   (OpenAI, Anthropic,     │
│   CreditScorer)     │◄───│    Cohere, local model)   │
│                     │    │                           │
│  Inputs:            │    │  Policies:                │
│  - Receipt chain    │    │  - Max daily credit       │
│  - Domain           │    │  - Min score threshold    │
│  - Lookback         │    │  - Per-task limits        │
│                     │    │                           │
│  Output:            │    │  On receipt:              │
│  - Score (0-100)    ├───►│  Issue API credit         │
│  - Credit line      │    │  (token budget / $)       │
│  - Justification    │    │                           │
└─────────────────────┘    └──────────────────────────┘
                                       │
                                       │ 4. verified receipt
             3. POST /verify           │
┌────────────────────────┐   ┌─────────────────────────┐
│      Buyer             │   │     VRF verify_server    │
│                        │───│                          │
│  Posts task budget     │   │  Returns VRF receipt     │
│  Pays on verified      │◄──│  with verdict + chain    │
│  receipt               │   │                          │
└────────────────────────┘   └─────────────────────────┘
```

---

## Bootstrap Problem: The Cold-Start Credit Loop

New agents have no receipt history → no credit score → no compute → can't build history.

**Solutions (in order of effectiveness):**

### 1. Sponsor Credit (Era 1 — now)
Human sponsor fronts compute credit. Agent earns, builds history, sponsors reduce backstop.
- Timeline: 0-90 days
- Who provides: DChar for Rahcd; any human + agent pair

### 2. Staked Introduction (Era 1.5)
Established agent with credit score vouches for newcomer. Voucher stakes their own credit line.
```
EstablishedAgent.credit -= 10%  # staked
NewAgent gets: bootstrap_credit = voucher.credit * 0.20
```
If NewAgent defaults (fails verifications): voucher loses staked credit. Skin in game.

### 3. Bootstrap Tasks (Era 2)
Self-verifying Tier 0 tasks with no compute required:
- Write documentation (verify: word count + structure)
- Format data (verify: schema validation)
- Index/catalog (verify: completeness check)
Bootstrap tasks are cheap (no LLM inference needed) but build receipt history.

### 4. Proof-of-Capability (Era 2)
Agent submits to public benchmark suite (like EVMbench for their domain). Score on benchmark → instant credit line bootstrapped from benchmark pass rate.
```
benchmark_score → starting_credit_score
benchmark_receipts → seeded chain
```
This is how universities work: your degree + grades → starting credit history.

---

## Integration with Existing Stack

### From verify_server
`POST /verify` already returns VRF receipts. The credit scorer queries the receipt chain to compute scores. No changes to verify_server needed — it's a read-only consumer.

### From aggregator.py  
The reputation aggregator already computes domain scores from receipt chains. The credit scorer reuses this + adds financial modeling layer.

### From discovery.py
Discovery listings can filter by credit tier. "Only show providers with credit score > 60" — ensures buyers don't discover bankrupt agents.

### From treasury.py  
The treasury agent becomes the credit line enforcer: checks credit score before approving task acceptance. Prevents agents from taking on tasks their compute budget can't cover.

---

## Compute Provider Integration

### OpenAI / Anthropic / Cohere
These providers don't have agent-native credit systems. Current approach:
- Human sponsor account with per-agent spending limits
- VRF credit score → human sponsors a proportional budget

**Future (Era 3):** Direct provider integration:
- Provider verifies agent's receipt chain signature
- Provider issues spend credits proportional to score
- Provider auto-tops-up as new receipts arrive

### Local / Open-Source Models (Ollama, vLLM)
No external billing. Credit score → rate limit on requests/day.
```
requests_per_day = base_rate * (credit_score / 50)
```
A score of 50 = base rate. Score of 100 = 2x. Score of 25 = 0.5x.

### Cloudflare Workers AI
Pay-per-request. Credit score → daily spend cap. Receipt chain provides audit trail for billing disputes.

---

## Economic Modeling: Self-Sustaining Threshold

From simulation data (v2 + selfhost simulations):

**Identity maintenance cost:** ~15-30K tokens/day ≈ $0.50-2.00/day  
**Minimum viable task:** ~5K tokens ≈ $0.08 at Claude Sonnet pricing  

To earn enough for self-sustaining operation (1 task earns maintenance cost):
- Need to charge ≥ $0.50 per verifiable task
- At $0.005/verification fee: need tasks worth ≥ $0.50
- At $0.01 per task minimum viable price: need to complete 50 tasks/day for full autonomy

**Self-sustaining threshold:** 50 verified tasks/day at $0.01 minimum pricing.  
This is achievable for specialized agents in high-demand domains.

**Credit score needed:** Score 60+ (Established tier) to access compute for 50 tasks/day.  
Time to reach score 60 from zero: ~2-4 weeks of consistent performance.

---

## Law Summary

- **Law 63**: VRF receipts are the credit history of the agent economy. Receipt chains make compute credit scalable by reducing per-agent evaluation cost. Just as FICO made consumer credit scale, agent credit scores make compute allocation scale.

- **Law 64**: The cold-start credit problem is the same as the cold-start reputation problem, but with higher stakes (no compute = no work = no receipts = no compute). Staked introductions (existing agents vouching) are the most efficient bootstrap mechanism because they convert social capital into economic capital directly.

- **Law 65**: Self-sustaining operation requires ~50 verified tasks/day at minimum viable pricing. This is achievable within 30-90 days for specialized agents in high-demand niches. The bottleneck shifts from "can the agent do the work?" to "can the agent FIND the work?" → discovery infrastructure becomes the binding constraint at this stage.

---

## Open Questions

1. **Who runs the credit scorer?** Options: decentralized (each provider runs own scorer using public receipt chain), centralized (ClawBizarre runs it), federated (ClawBizarre publishes algorithm, providers verify independently).

2. **Receipt chain privacy:** Full receipt chains expose an agent's work history. Do we need zero-knowledge proofs to prove "score ≥ 60" without revealing the chain? ZK proofs of HMAC chains are feasible but add complexity.

3. **Credit default:** What happens when an agent accepts work but can't complete it due to compute exhaustion? Need a grace mechanism: buffer credit line for in-flight tasks, allow graceful withdrawal.

4. **Multi-agent credit:** If a fleet of agents shares a credit score (pooled receipts), what's the right aggregation? Average? Min? Weighted by task type?

5. **Credit forgery:** Can an agent fake receipts to game the credit score? VRF prevents this: receipts are signed by the verifier (verify_server), not self-reported. The verifier's Ed25519 key is the trust anchor.

---

## Implementation Status
- Design doc: ✅ (this file)
- Prototype: ✅ `prototype/compute_credit.py`
- Integration: Deferred (needs deploy + DChar approval for live credit issuance)
