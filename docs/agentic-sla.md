# Agentic SLAs: VRF as the Evidence Layer

## The Shift

Traditional SLAs measure **availability** (uptime, latency, throughput).
Agentic SLAs must measure **outcomes** (task success, output correctness, business result achieved).

This shift is already happening:
- DoHost (Feb 19, 2026): "Moving from 'The server is on' to 'The goal was achieved'"
- Nemko (Feb 17, 2026): "Governance must evolve from input-output monitoring to sophisticated orchestration protocols"
- Kore.ai (Feb 18): "40% of agentic AI projects scrapped by 2027" due to reliability failures

## The Problem

How do you **prove** an outcome-based SLA was met?

Traditional: ping server, check HTTP 200. Deterministic, verifiable, automatable.
Agentic: "Did the agent complete the customer refund correctly?" How do you measure this?

Current approaches:
1. **Human review** — doesn't scale
2. **LLM-as-judge** — probabilistic, not auditable, not contractually binding
3. **Logs/traces** — observability (what happened), not verification (was it correct)
4. **Nothing** — most common, trust-and-pray

## VRF as Agentic SLA Evidence

VRF receipts are **deterministic, auditable proof of outcome correctness**.

### SLA Metric Mapping

| Traditional SLA Metric | Agentic SLA Equivalent | VRF Evidence |
|---|---|---|
| 99.9% uptime | 95% task success rate | Receipt chain: count(PASS) / count(total) |
| <200ms latency | <5s task completion | Receipt timestamp delta |
| 0 security incidents | 0 unverified outputs | Receipt coverage: verified / total tasks |
| Monthly uptime report | Monthly verification report | Merkle root + receipt aggregates |

### Contract Language (draft)

> "Provider guarantees that ≥95% of agent-executed tasks per calendar month shall produce VRF receipts with verdict=PASS against mutually agreed test suites. Verification receipts shall be retained for 12 months and available for audit via the transparency log."

This is **measurable, automatable, and legally auditable** — unlike "the agent will perform well."

### Architecture

```
Agent Task → Execute → VRF Verify → Receipt
                                      ↓
                                 Transparency Log (Merkle)
                                      ↓
                                 SLA Dashboard
                                      ↓
                                 Monthly Report (root hash = tamper-evident)
```

## Why This Matters

1. **Enterprises can't sign outcome SLAs without evidence infrastructure** — VRF provides it
2. **Insurance/liability** — actuaries need deterministic data, not LLM opinions
3. **Regulatory compliance** — EU AI Act requires output verification; SLA receipts double as compliance evidence
4. **Vendor accountability** — "Your agent failed 12% of tasks last month" with cryptographic proof

## Law 42

**Outcome-based SLAs require deterministic evidence, not probabilistic assessment. VRF receipts are the only mechanism that converts "the agent succeeded" from a claim into a proof. Without verification receipts, agentic SLAs are unenforceable promises.**

## Competitive Position

Nobody else is building this evidence layer:
- Observability platforms (Langfuse, Arize, LangSmith) track **what happened** but can't prove **correctness**
- LLM-as-judge platforms provide **opinions**, not **evidence**
- Traditional APM (Datadog, New Relic) measures **infrastructure**, not **outcomes**

VRF is uniquely positioned as the bridge between agentic operations and contractual/regulatory accountability.
