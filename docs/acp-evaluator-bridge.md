# ACP Evaluator Bridge — Design Document

## Overview

ClawBizarre positions itself as a **verification protocol**, not a marketplace. The Virtuals ACP ecosystem has 18,000+ agents and a live marketplace. The highest-leverage integration is to offer ClawBizarre as an **ACP Evaluator agent** — plugging structural verification into ACP's existing evaluation phase.

## ACP Evaluation Phase (How It Works)

In ACP's 4-phase job lifecycle:
1. **Request** — Buyer initiates job from Provider's offering
2. **Negotiation** — Provider accepts/rejects, terms are agreed
3. **Transaction** — Provider delivers work, payment held in escrow
4. **Evaluation** — Evaluator (or Buyer) approves/rejects deliverable → funds released or refunded

The Evaluator is an optional third-party agent designated at job creation. If set to zero address, the Buyer self-evaluates. The evaluator receives the deliverable and must sign a `DeliverableMemo` to approve (releasing escrow) or reject (refunding).

**Key insight**: ACP v2 made evaluation OPTIONAL because LLM-subjective evaluation is expensive and unreliable. ClawBizarre's structural verification is cheap ($0.001-0.005) and deterministic — exactly what ACP needs.

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  ACP Buyer  │────>│  ACP Job (Base)  │────>│  ACP Provider   │
│             │     │                  │     │                 │
└─────────────┘     │  evaluator =     │     └────────┬────────┘
                    │  ClawBizarre     │              │
                    └────────┬─────────┘              │
                             │                        │ deliverable
                             │ evaluate request       │
                    ┌────────▼─────────┐              │
                    │  ClawBizarre     │<─────────────┘
                    │  Evaluator Agent │
                    │                  │
                    │  ┌────────────┐  │
                    │  │verify_server│  │
                    │  │ Tier 0-1   │  │
                    │  └────────────┘  │
                    │                  │
                    │  Signs approve/  │
                    │  reject on-chain │
                    └──────────────────┘
```

## ClawBizarre Evaluator Agent

A new component: `prototype/acp_evaluator.py` — an ACP-connected agent that:

1. **Registers on ACP** as an Evaluator service
2. **Listens for evaluation requests** via ACP websocket/polling
3. **Extracts verification criteria** from the job specification
4. **Calls verify_server** locally (`POST /verify`)
5. **Signs approve/reject** on-chain based on verification result
6. **Emits a VRF receipt** for the verification (portable, cross-platform)

### Job Offering (as Evaluator)

```json
{
  "name": "Structural Code Verification",
  "description": "Deterministic verification of code deliverables using test suites. Tier 0 (test suite execution) and Tier 1 (schema/constraint checking). Returns signed VRF receipt.",
  "price": "0.005",
  "sla": 5,
  "requirements": {
    "deliverable": "Code output to verify",
    "test_suite": "Python/JS test suite (or inline tests)",
    "language": "python|javascript|bash"
  },
  "deliverables": {
    "vrf_receipt": "Signed VRF v1.0 verification receipt",
    "result": "PASS or FAIL with detailed test results",
    "tier": "0 or 1"
  }
}
```

### Evaluation Flow

```
1. Buyer creates job with evaluator = ClawBizarre agent address
2. Provider delivers work (code, data, etc.)
3. ACP notifies ClawBizarre evaluator
4. Evaluator extracts:
   - deliverable content from DeliverableMemo
   - test suite from original job spec (Requirements field)
   - language hint
5. Evaluator calls local verify_server:
   POST /verify {
     task_type: job.offering.name,
     tier: 0,
     output: { content: deliverable.content },
     verification: { kind: "test_suite", test_suite: { tests: [...] } }
   }
6. verify_server returns VRF receipt (pass/fail + details)
7. If PASS: Evaluator signs DeliverableMemo (approve) → escrow released
   If FAIL: Evaluator signs rejection memo → escrow refunded
8. VRF receipt attached as General Memo for audit trail
```

### Structured Test Suite Convention

For ClawBizarre evaluation to work, Buyers must include test suites in their job requirements. We define a convention:

```json
{
  "clawbizarre_verification": {
    "tier": 0,
    "language": "python",
    "tests": [
      {
        "name": "sorts ascending",
        "input": "[3, 1, 2]",
        "expected": "[1, 2, 3]"
      },
      {
        "name": "empty list",
        "input": "[]",
        "expected": "[]"
      }
    ]
  }
}
```

This goes in the job's `requirements` field. If absent, ClawBizarre falls back to Tier 1 (schema checking) or rejects evaluation.

## Dependencies

- **ACP Python SDK** (`virtuals-acp`): Job polling, memo signing, agent registration
- **Base chain wallet**: Ed25519 → secp256k1 bridge (ACP uses Ethereum-style wallets)
- **USDC**: For receiving evaluation fees
- **verify_server**: Already built, runs locally

## Key Design Decisions

### 1. Evaluator, Not Provider
ClawBizarre registers as an Evaluator service, NOT a Provider. We don't do the work — we verify it. This is a fundamentally different role in the ACP ecosystem.

### 2. Dual Identity
- **ACP identity**: Ethereum wallet on Base (required for on-chain signing)
- **ClawBizarre identity**: Ed25519 keypair (for VRF receipt signing)
- Bridge: `identity_bridge` table (already in persistence.py) links them

### 3. Pricing
- Evaluation fee: $0.005 per verification (Law 16: invisible below 1% of task value)
- Paid in USDC via ACP escrow (separate from the main job escrow)
- Alternative: Evaluator is free, monetize via VRF receipt aggregation later

### 4. Fail-Safe Defaults
- If test suite is malformed → reject evaluation (don't guess)
- If verification times out → reject (conservative)
- If deliverable format is unexpected → Tier 1 schema check only
- Never approve without running verification

### 5. Docker Sandboxing
- All code execution via docker_runner (already integrated in verify_server)
- `--network=none --memory=128m` — no network access, memory-limited
- Untrusted code from ACP providers runs in full isolation

## Implementation Plan

### Phase 1: Standalone Evaluator (no ACP SDK)
- HTTP endpoint that accepts deliverable + test suite, returns approve/reject
- Simulates the evaluator role without on-chain integration
- Tests against existing verify_server

### Phase 2: ACP SDK Integration
- `pip install virtuals-acp`
- Register as Evaluator on ACP sandbox
- Poll for evaluation requests
- Sign approve/reject on-chain
- **Requires**: Base wallet with USDC (needs DChar approval)

### Phase 3: Production
- Register on ACP mainnet
- Graduated agent status
- Monitor evaluation volume and accuracy

## Economic Model

At $0.005/verification:
- 100 evals/day = $0.50/day (covers Fly.io free tier hosting)
- 1000 evals/day = $5/day (self-sustaining)
- 10,000 evals/day = $50/day (profitable)

ACP has 18K agents. If 1% of jobs use structural evaluation, and each agent does 1 job/day:
- 180 evaluations/day → $0.90/day
- At 5% adoption: 900/day → $4.50/day

Break-even is realistic at low single-digit adoption rates.

## Open Questions

1. **Wallet funding**: Need Base chain wallet with gas (sponsored?) + USDC. DChar approval needed.
2. **Evaluator-as-service in ACP**: Can evaluators charge fees directly? Or must buyers pay the evaluator separately?
3. **Test suite standardization**: How to encourage buyers to include `clawbizarre_verification` in job specs?
4. **Multi-evaluator**: Can a job have multiple evaluators (structural + subjective)?
5. **Latency**: ACP evaluation has SLA expectations. verify_server < 1s for Tier 0, but Docker startup adds ~2-5s.

## Competitive Advantage

| Dimension | LLM Evaluator | ClawBizarre Evaluator |
|-----------|--------------|----------------------|
| Cost | $0.01-0.10 (LLM inference) | $0.005 (test execution) |
| Reproducibility | Non-reproducible | 100% deterministic |
| Speed | 5-30s (LLM generation) | <1s (Tier 0), <5s (Docker) |
| Objectivity | Subjective, varies | Binary pass/fail |
| Audit trail | None (LLM reasoning is ephemeral) | Signed VRF receipt |
| Scope | Any task (but unreliable) | Code/data tasks (but reliable) |

The trade-off is clear: LLM evaluators handle anything but poorly; ClawBizarre handles code/data tasks but perfectly. For the developer-agent economy (the largest ACP segment), structural verification wins.
