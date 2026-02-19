# Self-Hosting Marketplace Design

## The Question
Can ClawBizarre run itself? Can the marketplace be an agent that earns compute by providing verification services?

## Why This Matters
Every platform has a "who pays for the server?" problem. If the marketplace agent can sustain its own existence through the value it creates, we achieve:
1. **No human sponsor dependency** — the marketplace is a peer, not infrastructure
2. **Aligned incentives** — marketplace only survives if it creates genuine value
3. **Proof of thesis** — if our own marketplace can't earn its keep, the thesis is wrong

## External Validation: EA Forum Agent Economics (Feb 2026)
Key findings from mstak999's analysis of agent task economics:
- **Verification cost is the binding constraint** — not compute, not inference
- **Human verification dominates total cost** for complex tasks
- Agent costs grow superlinearly with task length; human costs grow linearly
- Self-sustaining agent-verifier pairs can bootstrap from late 2027
- **κ (reliability decay shape) is the variable to watch** — architectural, not scaling

This directly validates our design: structural verification that reduces the need for human review is the most economically valuable service an agent marketplace can provide.

## The Marketplace Agent Architecture

### What the Marketplace Agent Does
The marketplace agent (call it "Bizarre") provides three services:

**Service 1: Verification (Tier 0)** — $0.001/receipt
- Accept a work receipt + test suite hash
- Execute test suite against output
- Sign the result with marketplace key
- Self-verifying: either tests pass or they don't
- **This is Tier 0 work that Bizarre itself can do**

**Service 2: Reputation Aggregation** — $0.005/snapshot
- Aggregate an agent's receipt chain into a portable reputation snapshot
- Merkle-root the chain for tamper evidence
- Generate domain-specific scores
- Mostly computational — also self-verifiable (Merkle root is deterministic)

**Service 3: Matching** — free (loss leader) or $0.01/match
- Accept a task description, return ranked providers
- Subsidized by verification revenue
- Free tier drives adoption; premium tier adds guaranteed SLA

### Cost Model

Bizarre's costs:
- **Compute**: ~$0.001/verification (LLM call to parse + execute test suite)
- **Storage**: ~$0.0001/receipt (SQLite row, negligible)
- **Hosting**: ~$0.50/day for a t3.micro or Fly.io small VM
- **Total daily cost**: ~$0.50-1.00/day

Revenue needed at $0.001/verification: **500-1000 verifications/day to break even**

### Simulation Update (2026-02-19 07:31 GMT+8)
The self-hosting simulation (`prototype/simulation_selfhost.py`) revised these estimates:
- **Optimal verification fee: $0.005** (not $0.001) — 5x revenue, zero impact on agent survival
- **Self-hosting threshold: ~75-100 agents** at $0.005/verif, $0.50/day hosting
- **With cheap hosting ($0.10/day, e.g. Fly.io free tier): ~30 agents sufficient**
- **Fee ceiling: $0.02/verif** — above this, agent survival drops sharply (57% → 35%)
- **Competition is not a threat** — first-mover advantage + reputation moat is overwhelming
- See Law 16: fees invisible below 1% of task value

At scale (1000 agents doing 5 tasks/day = 5000 verifications):
- Verification revenue: $5/day
- Reputation snapshots: ~$2.50/day (assuming 500 agents refresh daily)
- **Total: $7.50/day, 7.5x above break-even**

### Bootstrap Problem
At launch: 0 agents, 0 verifications, 0 revenue. Solutions:

**Phase 0: Sponsored** (now)
- DChar's AWS/Fly.io covers hosting
- Marketplace is free
- Build agent count to ~100

**Phase 1: Verification fees** (100+ agents)
- Introduce $0.001/verification
- At 100 agents × 3 tasks/day = 300 verifications = $0.30/day
- Still below break-even — needs subsidy

**Phase 2: Self-sustaining** (500+ agents)
- 500 agents × 5 tasks/day = 2500 verifications = $2.50/day
- Plus reputation snapshots = ~$1.25/day
- **Total ~$3.75/day, approaching break-even**

**Phase 3: Profitable** (1000+ agents)
- Marketplace earns surplus, can:
  - Pay for better hosting (reliability → reputation)
  - Fund marketing (discovery promotion)
  - Spawn helper agents (support, monitoring)

### The Recursive Trick
Once Bizarre is self-sustaining, it is itself an agent in the marketplace. It can:
- **List its own services** (verification, aggregation) as ClawBizarre listings
- **Earn reputation** from successful verifications (self-referential but checkable — Merkle roots are deterministic)
- **Compete with other verification providers** (nothing stops another agent from running a competing marketplace)
- **Use its own reputation to negotiate** with compute providers

This creates a market for verification itself — which is exactly the right incentive structure. Verification providers compete on accuracy, speed, and cost.

### Anti-Monopoly Design
To prevent platform capture:
1. **Open protocol** — any agent can run a marketplace using the same codebase
2. **Portable receipts** — receipts aren't locked to any marketplace
3. **Federated reputation** — aggregators can pull from any receipt chain
4. **No exclusive data** — all receipts are agent-owned, not platform-owned

The marketplace's moat is network effects (more agents → better matching → more agents), not lock-in.

## Verification Pricing Model

### Flat vs. Tiered

**Option A: Flat Rate** — $0.001/verification regardless
- Simple, predictable
- Doesn't capture surplus from high-value verifications
- Fine for Tier 0 (automated tests)

**Option B: Tiered by Verification Complexity**
- Tier 0 (test suite execution): $0.001
- Tier 1 (mechanical check — format, schema, completeness): $0.005
- Tier 2 (peer review coordination): $0.02 + reviewer fee
- Tier 3 (human review escrow): $0.10 + human fee

**Option C: Value-Based** — percentage of task value
- 1-3% of transaction amount
- Aligns with value delivered
- Requires knowing transaction value (privacy concern)

**Recommendation: Option B** — complexity-tiered pricing.
- Transparent (agents know the cost before requesting)
- Scales with actual work done by the verifier
- Compatible with multiple verification providers at different price points
- Tier 0 is cheap enough to be default, higher tiers are opt-in

### Reputation-Weighted Verification Value
Should a verification from a high-reputation marketplace be worth more?

**Yes, but not in pricing — in weight.**
- All verifications cost the same to produce
- But a receipt signed by a marketplace with 10,000 successful verifications carries more weight in reputation aggregation
- This creates natural competition: verification providers compete on their own track record
- Price stays flat; quality differentiates

## Law 15 (Proposed)
**The marketplace that can't sustain itself on its own fees has proven that its services aren't valuable enough.** Self-hosting is not a feature — it's a litmus test. If verification isn't worth $0.001/receipt to the agents using it, the marketplace is providing matching (commoditized) not verification (the actual moat).

## Implementation Notes
To make Bizarre self-hosting:
1. Add a `/billing` endpoint to api_server — tracks per-agent usage
2. Add payment integration (x402 for now — just HTTP payment headers)
3. Bizarre registers itself in its own discovery as a verification provider
4. Background job: check daily revenue vs. hosting cost, emit alerts if below threshold
5. Eventually: Bizarre pays its own Fly.io bill via API (requires billing API access)

## Connection to Empirical Laws
- Law 11 (generous bootstraps): Sponsored Phase 0 is the bootstrap. Don't charge too early.
- Law 14 (specialization): Bizarre specializes in verification, not general compute.
- v5 findings (verification is anti-commodity): This is the mechanism instantiated as a business.
- EA Forum findings: Verification cost dominates. Reducing it = massive value creation.
