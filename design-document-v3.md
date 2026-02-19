# ClawBizarre Design Document v3.0

*Verification protocol for the agent economy. Consolidates v2 + Phases 9-24 + strategic pivot.*
*Author: Rahcd — February 19, 2026*

---

## What Changed Since v2

v2 documented a working marketplace prototype. v3 reflects a **strategic pivot**: ClawBizarre is no longer a marketplace — it's a **verification protocol**. The marketplace layer is commoditizing (ACP, x402, UCP, ERC-8004). Verification is the durable moat (Law 17).

Key additions since v2:
- **SSE notifications** (Phase 9) — zero-polling event delivery
- **MCP server** (Phase 11) — 14 tools, 3 resources, 30/30 tests
- **OpenClaw skill** (Phase 12) — `cb` CLI, installable via ClawHub
- **Verification protocol** (Phase 14) — standalone verify_server, tiered verification, VRF receipts
- **ACP evaluator bridge** (Phase 15) — structural verification for 18K-agent marketplace
- **A2A adapter** (Phase 19) — Google's agent protocol, agent card + JSON-RPC
- **COSE/SCITT transparency** (Phases 20-24) — RFC 9162 Merkle tree, COSE Sign1, append-only log
- **IETF Internet-Draft** (Phase 24) — VRF as SCITT content type, standards-track
- **34 empirical laws** (up from 14)
- **Regulatory positioning** — EU AI Act (Aug 2026), NIST AI Agent Standards Initiative

---

## Part 1: The Thesis

### The Verification Gap

Every agent commerce protocol assumes delivery correctness. None verify it.

| Protocol | Who | What It Does | Verification |
|----------|-----|-------------|-------------|
| ACP (Virtuals) | Virtuals Protocol | A2A commerce, on-chain escrow | LLM-subjective (optional in v2) |
| ACP (OpenAI) | OpenAI | B2C checkout in ChatGPT | None |
| UCP | Google | B2C agent commerce | None |
| x402 | Stripe | HTTP payment headers | None |
| A2A | Google | Agent communication | None |
| MCP | Anthropic | Tool execution | None |
| **VRF** | **ClawBizarre** | **Output quality verification** | **Deterministic test suites** |

Authentication proves an agent *can* act. Verification proves it *acted correctly*. The gap between "authorized" and "correct" is the fraud/waste surface of the agent economy (Law 29).

### Why Test Suites, Not LLM-as-Judge

OpenAI + Paradigm's EVMbench (Feb 2026) uses test suites with answer keys — not LLM-as-judge — for evaluating $100B+ in smart contract security. Amazon's agent eval framework uses LLM-as-judge for its top layer and acknowledges the reliability gap. Financial regulators (SR 11-7) require deterministic validation — "model reviewing model" is explicitly non-deterministic.

**Law 32**: Financial regulation requires deterministic evidence. LLM-as-judge is probabilistic. Only test-suite verification produces deterministic audit trails regulators accept.

---

## Part 2: Architecture (As Built)

### Two Stacks

ClawBizarre has two independent stacks that can deploy separately:

**Stack A: Verification Protocol** (deploy-ready)
```
┌─────────────────────────────────────────────────┐
│  verify_server_unified.py                        │
│  Single-process: verification + transparency     │
├─────────────┬───────────────────────────────────┤
│  Verification Engine    │  Transparency Service  │
│  - Tier 0: test suites  │  - COSE Sign1 receipts │
│  - Tier 1: schema       │  - Merkle tree (9162)  │
│  - Multi-lang (Py/JS)   │  - Inclusion proofs    │
│  - Docker sandbox        │  - Consistency proofs  │
│  - VRF receipts          │  - Append-only log     │
├─────────────┤            ├───────────────────────┤
│  Receipt Store (SQLite)  │  Merkle Store (SQLite) │
└─────────────────────────────────────────────────┘
```

**Stack B: Marketplace Engine** (prototype, not priority)
```
┌─────────────────────────────────────────────────┐
│  api_server_v6.py (unified REST API)             │
│  30+ endpoints: auth, discovery, matching,       │
│  handshake, receipts, reputation, treasury,      │
│  ERC-8004 identity bridge, SSE notifications     │
├──────────┬──────────┬──────────┬────────────────┤
│ Identity │ Discovery│ Matching │ Reputation     │
│ Ed25519  │ Registry │ Posted-  │ Bayesian       │
│ keypairs │ + search │ price    │ decaying       │
│          │ + reserve│ engine   │ + aggregator   │
├──────────┴──────────┤          ├────────────────┤
│ Auth + Persistence  │ Treasury │ SSE Notifs     │
│ SQLite WAL          │ Policy   │ Per-agent      │
└─────────────────────┴──────────┴────────────────┘
```

**Strategic priority**: Stack A first. Verification is the moat. The marketplace can be any protocol (ACP, A2A, standalone).

### The Verification Pipeline

```
Code + Test Suite + Language
        ↓
  verify_server (Tier 0)
        ↓
  Docker sandbox execution (--network=none, --memory=128m)
        ↓
  VRF Receipt (signed, versioned)
        ↓
  COSE Sign1 encoding (optional)
        ↓
  Merkle transparency log (optional)
        ↓
  Inclusion proof (cryptographic)
```

### Multi-Protocol Distribution

The same verification service is accessible via:

| Protocol | Implementation | Status |
|----------|---------------|--------|
| HTTP REST | verify_server_unified.py | ✅ 27/27 tests |
| MCP | mcp_server.py (14 tools) | ✅ 30/30 tests |
| OpenClaw Skill | skills/clawbizarre/cb | ✅ 12/12 tests |
| ACP Evaluator | acp_evaluator.py | ✅ 23/23 tests |
| ACP Provider | acp-deploy/ (TypeScript) | ✅ Ready to deploy |
| A2A Agent | a2a_adapter.py | ✅ 13/13 tests |

**Law 18**: Verification is protocol-agnostic. The same VRF receipt works across ACP, A2A, MCP, and standalone. Protocol lock-in for verification is a design smell.

---

## Part 3: VRF Receipt Format

### Structure (v1.0)

```json
{
  "vrf_version": "1.0",
  "receipt_id": "uuid",
  "timestamp": "ISO-8601",
  "issuer": "did:key:z6Mk...",
  "subject": {
    "code_hash": "sha256:...",
    "test_suite_hash": "sha256:...",
    "language": "python"
  },
  "verification": {
    "tier": 0,
    "verdict": "pass|fail|error",
    "tests_passed": 5,
    "tests_failed": 0,
    "tests_total": 5,
    "execution_time_ms": 173,
    "sandbox": "docker"
  },
  "chain": {
    "prev_receipt_hash": "sha256:...",
    "position": 42
  }
}
```

### COSE Encoding (SCITT-aligned)

```
COSE_Sign1 {
  protected: {
    alg: EdDSA,
    content_type: "application/vrf+cbor",
    -70001: "1.0",          // VRF version
    -70002: "did:key:z6Mk", // issuer DID
    -70003: "receipt-uuid",  // receipt ID
    -70004: "prev-hash",    // chain link
    -70005: 42              // chain position
  },
  payload: CBOR(receipt),
  signature: Ed25519(key, protected + payload)
}
```

Size overhead: ~130 bytes fixed, 1.11x JSON size. Negligible.

### Transparency Log

RFC 9162-compatible Merkle tree. Same cryptographic guarantees as Certificate Transparency.

- Append-only (consistency proofs between any two tree sizes)
- Inclusion proofs (any receipt provably in the log)
- Domain-separated hashing (leaf vs internal nodes)
- 1000 appends: 1.1ms. Proof depth: log₂(N).

---

## Part 4: Standards & Regulatory Positioning

### IETF Internet-Draft

`draft-vrf-scitt-00.md` — VRF as SCITT content type. 10 sections, 2 appendices, ~22KB.
- COSE encoding spec, SCITT registration flow, cross-protocol usage patterns
- IANA registrations: media types, COSE headers, task type registry
- **Blocked on**: DChar approval for formal submission

### Trust Stack

```
Layer 5: Identity Ownership    — Web Bot Auth, Visa TAP, Mastercard Agent Pay, Vouched.id
Layer 4: Output Quality        — VRF (ONLY PLAYER)
Layer 3: Communication         — G²CP, MCP, A2A
Layer 2: Tool Trust            — MCPShield, AgentAudit, Gen ATH
Layer 1: Skill Safety          — Gen ATH, Koi Security, Snyk
```

**Law 25**: The trust stack fills bottom-up because lower layers have enterprise analogs (PKI, OAuth, KYC). Output quality verification has no enterprise precedent. First-mover advantage is structural.

### Regulatory Timeline

| Deadline | What | Relevance |
|----------|------|-----------|
| Mar 9, 2026 | NIST RFI on AI Agent Security | VRF as verification standard |
| Mar 20, 2026 | NIST Listening Sessions | Sector-specific VRF applications |
| Apr 2, 2026 | NCCoE Identity & Auth Feedback | VRF + identity pipeline |
| Aug 2, 2026 | EU AI Act enforcement | High-risk AI must verify outputs |

**Law 31**: Regulation follows mandate → mechanism gap → compliance tooling market. EU AI Act mandates output verification but specifies no mechanism. VRF fills the gap.

---

## Part 5: Empirical Laws (Complete, 1-34)

### Marketplace Dynamics (Laws 1-14)
*Derived from 80+ experiments, 600K+ simulated agent-rounds.*

| # | Law |
|---|-----|
| 1 | Strategy switching destroys economies |
| 2 | Reputation penalty is only effective switching cost |
| 3 | Reputation premium is the ESS |
| 4 | Coalitions are a tax without exclusive resources |
| 5 | Initial conditions > mechanisms |
| 6 | Newcomer protection = competitive advantage |
| 7 | 15% platform fee ceiling |
| 8 | Optimal fleet size 2-5 |
| 9 | Buyer selection > discovery reserve |
| 10 | Transaction overhead is the real newcomer barrier |
| 11 | Design for generous bootstraps, not cheap existence |
| 12 | Protection mechanisms need minimum market size (~15+) |
| 13 | Specialization is the best newcomer protection |
| 14 | Optimal specialization inversely correlated with market size |

### Verification Economics (Laws 15-20)
*Derived from self-hosting simulation + market analysis.*

| # | Law |
|---|-----|
| 15 | Self-hosting is a litmus test |
| 16 | Verification fees invisible below 1% of task value |
| 17 | The marketplace layer is commoditizing. The verification layer is the durable moat. |
| 18 | Verification is protocol-agnostic |
| 19 | Commerce ≠ verification. The gap = fraud surface. |
| 20 | Government standardization follows market reality by 6-18 months |

### Trust & Security (Laws 21-28)
*Derived from landscape analysis + security research.*

| # | Law |
|---|-----|
| 21 | Security researchers find the gaps before economists do |
| 22 | Disposable identities create an accountability vacuum |
| 23 | Pre-deploy safety ≠ post-execution quality. Both needed. |
| 24 | The agent trust stack has 5 distinct layers. Conflating them creates false confidence. |
| 25 | Trust stack fills bottom-up. Output quality has no enterprise precedent. |
| 26 | Commerce protocol proliferation > trust infrastructure. Verification gap grows with adoption. |
| 27 | Standards adoption: extending existing > standalone replacement |
| 28 | Identity matures fastest (enterprise analogs). Output verification = greenfield. |

### Commerce & Regulation (Laws 29-34)
*Derived from regulatory analysis + industry validation.*

| # | Law |
|---|-----|
| 29 | Auth proves "can act." Verification proves "acted correctly." The gap = fraud surface. |
| 30 | Intent detection and output verification are complementary but methodologically incompatible. |
| 31 | Regulation: mandate → mechanism gap → compliance tooling. EU AI Act = Aug 2026. |
| 32 | Financial regulation requires deterministic evidence. LLM-as-judge doesn't qualify. |
| 33 | "Deterministic" has 3 orthogonal meanings in agentic AI. Only output-level matters for compliance. |
| 34 | Domain benchmarks (EVMbench) validate test-suite approach but are siloed. VRF unifies. |

---

## Part 6: Deployment Plan

### Minimal Viable Deployment (Stack A only)

```
verify_server_unified.py
  --auto-register          # COSE sign + Merkle log every receipt
  --api-key $KEY           # Simple auth
  --db receipts.db         # SQLite persistence
  --port 8080
```

Behind nginx on Fly.io free tier ($0/mo) or EC2 (~$15/mo).

### Revenue Model (Verification-as-a-Service)

| Tier | Price | What |
|------|-------|------|
| 0 (test suite) | $0.005 | Run tests in Docker sandbox, return VRF receipt |
| 1 (schema) | $0.002 | Schema/constraint validation only |
| Transparency | $0.001 | COSE sign + Merkle inclusion proof |

Self-hosting threshold: ~75-100 agents at $0.005/verification (Law 15). Fly.io free tier lowers this to ~30 agents.

### Blockers (All Require DChar Approval)

1. Deploy verify_server publicly (~$0/mo)
2. Create Base wallet for ACP registration
3. Publish to ClawHub ($0)
4. Submit NIST RFI response (Mar 9 deadline)
5. Submit NCCoE feedback (Apr 2 deadline)
6. Submit IETF Internet-Draft

---

## Part 7: File Map (Complete)

```
prototype/                          # Core implementation
├── identity.py                     # Ed25519 keypairs, signing
├── receipt.py                      # WorkReceipt v0.3, chains, risk envelopes
├── handshake.py                    # Bilateral negotiation (8-state)
├── signed_handshake.py             # Cryptographic handshake signatures
├── discovery.py                    # Registry, search, newcomer reserve
├── matching.py                     # Posted-price engine (14/14 tests)
├── reputation.py                   # Bayesian decaying reputation
├── aggregator.py                   # Receipt chain → reputation scores
├── treasury.py                     # Policy executor, audit chain
├── auth.py                         # Challenge-response auth
├── persistence.py                  # SQLite WAL backend
├── client.py                       # Python SDK (zero deps, SSE support)
├── cli.py                          # CLI: init, whoami, receipt, chain
├── notifications.py                # SSE NotificationBus (12/12 tests)
├── docker_runner.py                # Multi-language Docker sandbox (23/23)
├── receipt_store.py                # SQLite receipt persistence (17/17)
├── api_server_v[1-6].py            # REST API iterations
├── verify_server.py                # Standalone verification (37/37)
├── verify_server_hardened.py       # Production-hardened (21/21)
├── verify_server_unified.py        # Verification + transparency (27/27)
├── acp_evaluator.py                # ACP evaluator bridge (16/16)
├── acp_evaluator_live.py           # Production ACP evaluator
├── acp_provider_example.py         # ACP provider workflow
├── provider_verify.py              # Provider-side verification (17/17)
├── a2a_adapter.py                  # A2A protocol adapter (13/13)
├── mcp_server.py                   # MCP server (14 tools, 10/10)
├── vrf_cose.py                     # COSE Sign1 encoding (4/4)
├── vrf_cose_identity.py            # Identity-COSE bridge (5/5)
├── merkle.py                       # RFC 9162 Merkle tree (12/12)
├── merkle_store.py                 # SQLite Merkle persistence (9/9)
├── transparency_server.py          # HTTP transparency API (14/14)
├── integration_test.py             # E2E chain of trust (8/8)
├── test_mcp_e2e.py                 # MCP E2E (20/20)
├── test_acp_integration.py         # ACP integration (7/7)
├── benchmark_evaluator.py          # Evaluator performance bench
└── simulation_*.py                 # 13 simulation versions

skills/clawbizarre/                 # OpenClaw distribution
├── SKILL.md                        # Skill definition
└── cb                              # CLI wrapper (15 commands)

acp-deploy/                         # ACP deployment package
├── offering.json                   # Production offering config
├── handlers.ts                     # TypeScript handler
├── fly.toml                        # Fly.io config
└── DEPLOY.md                       # 6-step deployment guide

docs/
├── design-document-v[1-3].md       # Design evolution
├── architecture.md                 # Component architecture
├── vrf-spec-v1.md                  # VRF receipt format spec
├── draft-vrf-scitt-00.md           # IETF Internet-Draft
├── scitt-alignment.md              # SCITT standards analysis
├── verification-protocol-v1.md     # Protocol design
├── matching-engine-design.md       # Matching research
├── mcp-server-design.md            # MCP distribution design
├── acp-evaluator-bridge.md         # ACP bridge design
├── acp-offering-design.md          # ACP service offering
├── self-hosting-design.md          # Self-hosting economics
├── deployment-plan.md              # Deployment checklist
├── landscape-feb-2026[-v2,-v3].md  # Market landscape
├── competitive-landscape.md        # Competitive analysis
├── defensibility-analysis.md       # Moat analysis
├── nist-rfi-draft-v[1-3].md        # NIST submission drafts
├── nccoe-feedback-draft-v1.md      # NCCoE feedback draft
├── decision-brief-v1.md            # Decision brief for DChar
└── agent-economics-playbook.md     # Practical agent economics guide
```

### Test Summary

| Component | Tests | Status |
|-----------|-------|--------|
| verify_server_unified | 27/27 | ✅ |
| verify_server (multi-lang) | 37/37 | ✅ |
| verify_server_hardened | 21/21 | ✅ |
| MCP server (protocol + E2E) | 30/30 | ✅ |
| ACP evaluator + integration | 23/23 | ✅ |
| Matching engine | 14/14 | ✅ |
| Docker runner | 23/23 | ✅ |
| Receipt store | 17/17 | ✅ |
| Provider verify | 17/17 | ✅ |
| A2A adapter | 13/13 | ✅ |
| Merkle tree | 12/12 | ✅ |
| SSE notifications | 12/12 | ✅ |
| Client SDK | 16/16 | ✅ |
| Transparency server | 14/14 | ✅ |
| Merkle store | 9/9 | ✅ |
| API server v4 | 21/21 | ✅ |
| API server v5 (SSE) | 22/22 | ✅ |
| API server v6 (ERC-8004) | 22/22 | ✅ |
| E2E integration | 8/8 | ✅ |
| COSE encoding | 4/4 | ✅ |
| Identity-COSE bridge | 5/5 | ✅ |
| OpenClaw skill | 12/12 | ✅ |
| **Total** | **~400+** | ✅ |

---

## Part 8: What's Next

### Immediate (Needs DChar Approval)
1. Deploy verify_server_unified to Fly.io/EC2
2. Register as ACP service provider
3. Publish to ClawHub
4. Submit NIST RFI (Mar 9 deadline — 18 days)

### Near-Term (No Approval Needed)
1. ~~Design doc v3~~ ✅ (this document)
2. WebMCP manifest for browser-native discovery
3. README.md for GitHub (public-facing)
4. Benchmark suite: verification latency across languages/complexity
5. Fault tolerance: verify_server crash recovery, receipt replay

### Medium-Term
1. IETF formal submission
2. Federation protocol (multi-instance VRF)
3. Tier 2 verification (property-based testing)
4. Receipt format v1.1 (learned from production use)

---

*v3.0 — February 19, 2026. The verification protocol for the agent economy.*
