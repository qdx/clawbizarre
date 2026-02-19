# Agent Trust Stack Taxonomy v1.0

_A layered model for understanding trust infrastructure in multi-agent systems._

## Overview

Trust in autonomous agent systems is not monolithic. It decomposes into five distinct layers, each requiring different verification methods. Conflating layers creates false confidence (Law 24).

## The Five Layers

```
┌─────────────────────────────────────────────────┐
│  Layer 5: Identity Ownership                     │
│  "Who is this agent?"                            │
│  Methods: PKI, DIDs, KYC/KYA, OAuth             │
│  Players: Web Bot Auth, Visa TAP, Mastercard     │
│           Agent Pay, Vouched.id, Indicio,        │
│           Token Security, Sumsub KYA, Trulioo,   │
│           Catena Labs                            │
│  Status: ✅ ACTIVE — most mature layer           │
├─────────────────────────────────────────────────┤
│  Layer 4: Output Quality                         │
│  "Did the agent produce correct output?"         │
│  Methods: Test-suite execution, deterministic    │
│           verification, VRF receipts             │
│  Players: VRF (ClawBizarre)                      │
│  Status: 🔴 EMPTY — only VRF                     │
├─────────────────────────────────────────────────┤
│  Layer 3: Communication Integrity                │
│  "Are agents communicating faithfully?"           │
│  Methods: Protocol verification, audit traces    │
│  Players: G²CP, A2A, MCP, ACP                   │
│  Status: 🟡 EMERGING — protocols proliferating   │
├─────────────────────────────────────────────────┤
│  Layer 2: Execution Integrity                    │
│  "Did the computation run as claimed?"           │
│  Methods: TEE attestation, TLS Notary,           │
│           deterministic inference, on-chain proof │
│  Players: EigenAI, EigenCompute, TessPay,        │
│           Praetorian                             │
│  Status: 🟡 EMERGING — crypto-native mostly      │
├─────────────────────────────────────────────────┤
│  Layer 1: Settlement                             │
│  "How does value transfer happen?"               │
│  Methods: Payment rails, escrow, staking         │
│  Players: x402/Stripe, OpenAI ACP, Google UCP,   │
│           Virtuals ACP, EigenLayer               │
│  Status: ✅ ACTIVE — multiple live systems        │
└─────────────────────────────────────────────────┘
```

## Key Properties

### Each layer is necessary but not sufficient
- Execution integrity (Layer 2) proves the model ran faithfully — but a correctly executed wrong answer is still wrong (Law 46)
- Identity (Layer 5) proves who the agent is — but knowing identity doesn't prove output quality (Law 28)
- Communication integrity (Layer 3) proves messages weren't tampered with — but faithful transmission of wrong results is still wrong

### Layers are orthogonal
- Making an agent's process deterministic (Layer 2) doesn't make its output correct (Layer 4) — Law 33
- Authentication (Layer 5) proves the agent *can* act; verification (Layer 4) proves it *acted correctly* — Law 29
- Behavioral intent detection and output quality verification are methodologically incompatible — Law 30

### Layer 4 is uniquely empty
- Identity matures fastest because it has enterprise analogs (PKI, OAuth, KYC) — Law 28
- Output verification has no enterprise precedent — it's a greenfield problem
- Every commerce protocol creates additional unverified transaction surface — Law 26
- The longer Layer 4 stays empty, the larger the first-mover advantage

## Layer 4 Deep Dive: Why VRF Is Unique

### What exists (and why it's not Layer 4)

| Approach | Layer | Why not Layer 4 |
|----------|-------|-----------------|
| LLM-as-judge (Amazon, ACP) | — | Probabilistic, not deterministic. Fails regulatory requirements (Law 32) |
| EVMbench (Paradigm/OpenAI) | — | Domain-specific benchmark, not per-task evidence protocol (Law 34) |
| Agent Trust Hub (Gen/Vercel) | 0 | Pre-install skill safety, not post-execution quality (Law 23) |
| MCPShield | 0 | Tool invocation safety, not output correctness |
| DTLEF | 2 | Trajectory-level (process) evaluation, not output evaluation |
| EigenAI | 2 | Execution integrity, not functional correctness (Law 46) |
| TessPay PoTE | 2 | TEE attestation = process integrity (Law 35) |

### VRF's unique position
- **Deterministic**: Test suites produce binary pass/fail, not probabilistic scores
- **Protocol-agnostic**: Same receipt works across ACP, A2A, MCP, standalone (Law 18)
- **Domain-agnostic**: Works for any task expressible as test cases (Law 34)
- **Auditable**: Receipt chains + Merkle logs = tamper-evident history
- **Standards-aligned**: SCITT content type, COSE encoding, IETF Internet-Draft ready

## Regulatory Alignment

| Regulation | Relevant Layer | VRF Fit |
|------------|---------------|---------|
| EU AI Act (Aug 2026) | Layer 4 | VRF receipts = mandated output verification evidence |
| SR 11-7 (financial) | Layer 4 | Deterministic validation required; LLM-as-judge explicitly insufficient |
| NIST AI Agent Standards | Layers 2-5 | VRF submitted as Layer 4 mechanism |
| UC Berkeley Risk Profile | Layers 3-5 | Activity logging + deviation detection = receipt chains |
| China AI compliance | Layer 4 | Output-focused compliance aligns with VRF evidence format |

## Cross-Layer Integration Opportunities

1. **Layer 2 + 4**: EigenAI proves execution integrity → VRF proves output correctness → full stack proof
2. **Layer 4 + 5**: VRF receipt chains create behavioral fingerprints → anomaly detection for identity compromise (Law 37)
3. **Layer 1 + 4**: x402 payment + VRF receipt = pay-for-verified-work (escrow release on verification)
4. **Layer 3 + 4**: A2A task delegation + VRF verification = verifiable agent-to-agent commerce

## Supporting Laws

- **Law 18**: Verification is protocol-agnostic
- **Law 23**: Pre-deployment safety ≠ post-execution quality
- **Law 24**: Trust stack has 5 distinct layers; conflating them creates false confidence
- **Law 25**: Trust stack fills bottom-up; output quality has no enterprise precedent
- **Law 26**: Commerce protocol proliferation widens the verification gap
- **Law 28**: Identity matures fastest (enterprise analogs); output quality is greenfield
- **Law 29**: Authentication ≠ verification
- **Law 30**: Intent detection and quality verification are methodologically incompatible
- **Law 33**: "Deterministic" has three orthogonal meanings in agentic AI
- **Law 34**: Domain-specific benchmarks validate test-suite approach but are inherently siloed
- **Law 35**: Execution attestation ≠ functional correctness
- **Law 46**: Verifiable execution and verifiable correctness are complementary but irreducible

---

_Version 1.0 — 2026-02-20. Compiled from landscape research v1-v14 and 47 empirical laws._
