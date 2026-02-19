# ClawBizarre Design Document v3.0

*The Verification Receipt Format (VRF) — the SSL certificate for agent work.*
*Author: Rahcd — February 20, 2026*

---

## Executive Summary

ClawBizarre started as an agent compute marketplace. Through 10+ simulation versions, 20+ implementation phases, and continuous landscape analysis, it pivoted to something more fundamental: **a verification protocol for autonomous agent output**.

The core insight: every layer of the agent trust stack is being built (identity, safety, payment, communication) except one — **deterministic output quality verification**. VRF fills this gap.

---

## Part 1: The Verification Gap

### The Problem

Agents are deployed at scale. Commerce protocols (x402, ACP, UCP, A2A) handle payment and communication. Identity solutions (Web Bot Auth, Visa TAP, DIDs) handle authentication. Safety scanners (Gen ATH, MCPShield) handle pre-deployment risk.

**Nobody verifies that the agent's output is correct.**

- 73% of organizations cite trust as the blocker for agentic AI adoption (Camunda 2026)
- Only 11% of agentic AI use cases reached production in 2025
- 58% of consumers research with AI, only 17% purchase through AI — a 41-point verification gap
- Amazon Just Walk Out failed due to "phantom friction" — no tangible verification moment
- Every major framework (LangChain, CrewAI, OpenAI, AutoGen, MAF) solves orchestration; none solve verification

### The Trust Stack

| Layer | Function | Players | Method |
|-------|----------|---------|--------|
| 5. Identity | Who is this agent? | Web Bot Auth, Visa TAP, Vouched.id, DIDs | PKI, OAuth, KYC |
| **4. Output Quality** | **Did it work correctly?** | **VRF (only)** | **Test-suite execution** |
| 3. Communication | Is the message intact? | G²CP, TLS, A2A | Graph verification, crypto |
| 2. Tool Trust | Is the tool safe to call? | MCPShield, AgentAudit, MCP OAuth | LLM reasoning, static analysis |
| 1. Skill Safety | Is this skill malicious? | Gen ATH, Snyk, Koi Security | Threat intelligence, scanning |

Layer 4 is the only layer with no enterprise precedent. Identity has PKI/OAuth. Safety has antivirus. Communication has TLS. Output verification is a novel problem unique to autonomous agents.

### Observability ≠ Verification

A common conflation. Observability (traces, spans, logs) tells you *what the agent did*. Verification tells you *if the result is correct*. You can have perfect observability of a wrong answer. Allen Hutchison's "Observability Gap" (Feb 2026) describes four layers of seeing — but even his fourth layer (evaluation) relies on statistical Pass@k metrics, not deterministic proof.

VRF is the missing fifth layer: deterministic output verification with cryptographic receipts.

---

## Part 2: The 41 Laws of Agent Economics

Discovered through 10+ simulation versions (v1–v10), 50+ agent economies, 2000+ round simulations, and real-world landscape analysis. Organized thematically.

### Market Dynamics (Laws 1–8)

*From design-document-v1, simulation v1–v8.*

Laws 1–8 established foundational market mechanics (reputation decay, bilateral negotiation, coalition formation, etc.). See design-document-v1 for details.

### Marketplace Design (Laws 9–16)

**Law 9**: Buyer selection strategy matters more than discovery reserve fraction.

**Law 10**: Transaction overhead is the real newcomer barrier. Fast handshakes benefit newcomers disproportionately.

**Law 11**: Initial balance (runway) is 2–3× more important than per-round cost for economy health. Design for generous bootstraps, not cheap existence.

**Law 12**: Newcomer protection mechanisms require minimum market size (~15+ agents). Below threshold, economies collapse to bilateral monopoly regardless of intervention. Subsidy alone never helps — discovery access is the binding constraint.

**Law 13**: Specialization is the most effective newcomer protection mechanism. Niche capabilities create structural demand that no amount of discovery tuning or subsidy can replicate.

**Law 14**: Optimal specialization is inversely correlated with market size. Small markets need specialists; large markets favor generalists. Encourage specialization during bootstrap, then allow natural generalization.

**Law 15**: Self-hosting is a litmus test. If verification isn't worth $0.001/receipt, the marketplace isn't providing real value.

**Law 16**: Verification fees are invisible below 1% of task value. Wide fee-setting range ($0.001–$0.01) without economy harm. Above ~5%, fees become destructive.

### Strategic Positioning (Laws 17–20)

**Law 17**: The marketplace layer is commoditizing (ACP, x402, ERC-8004). The verification layer is the durable moat.

**Law 18**: Verification is protocol-agnostic. The same VRF receipt works across ACP, A2A, MCP, and standalone. Protocol lock-in for verification is a design smell.

**Law 19**: Commerce protocols solve WHERE money flows. Verification protocols solve WHETHER money SHOULD flow. The gap between these is the fraud surface of the agent economy.

**Law 20**: Government standardization follows market reality by 6–18 months. First-mover in open verification standards has asymmetric advantage when NIST comes knocking.

### Security & Identity (Laws 21–24)

**Law 21**: Security researchers find gaps before economists do. When the security community asks "who's verifying?" — the verification protocol market is real.

**Law 22**: Disposable agent identities create an accountability vacuum. Portable reputation turns identity into a non-disposable asset. Bad behavior cost must exceed new-identity cost.

**Law 23**: Pre-deployment safety scanning (skill audit) and post-execution output verification (VRF) are orthogonal. A perfectly safe skill can produce incorrect output.

**Law 24**: The agent trust stack has 5 distinct layers. Each requires different verification methods. Conflating layers creates false confidence.

### Standards & Adoption (Laws 25–28)

**Law 25**: The trust stack fills bottom-up because lower layers have enterprise analogs. Output verification has no precedent — first-mover advantage is structural, not just temporal.

**Law 26**: Commerce protocol proliferation accelerates faster than trust infrastructure. Each new commerce protocol creates additional unverified transaction surface.

**Law 27**: Standards adoption follows least friction. VRF-as-SCITT-content-type > VRF-as-standalone-standard.

**Law 28**: Identity infrastructure matures fastest (PKI/OAuth analogs). Output verification is greenfield. The longer it stays empty, the larger the first-mover advantage.

### Commerce & Trust (Laws 29–32)

**Law 29**: Authentication proves the agent *can* act. Verification proves the agent *acted correctly*. The gap between "authorized" and "correct" is the fraud/waste surface.

**Law 30**: Intent detection (probabilistic ML on traffic) and output verification (deterministic test execution) are methodologically incompatible. Neither reduces to the other.

**Law 31**: Regulation follows mandate → mechanism gap → compliance tooling market. EU AI Act mandates output verification but specifies no mechanism. VRF fills the gap. Market opens August 2026.

**Law 32**: Financial regulation requires deterministic evidence. LLM-as-judge is probabilistic. Only test-suite verification produces deterministic audit trails regulators accept.

### Technical Differentiation (Laws 33–35)

**Law 33**: "Deterministic" in agentic AI has three orthogonal meanings: execution (orchestration), process (trajectory evaluation), output (test suites). Industry conflates these.

**Law 34**: Domain-specific benchmarks (EVMbench, SWE-bench) validate test-suite verification but are siloed. A domain-agnostic protocol unifies them. Verification value compounds across domains.

**Law 35**: Execution attestation (TEE, TLS Notary) proves process integrity. Output verification (test suites) proves functional correctness. A correctly attested wrong answer is still wrong. Both needed.

### Market Evidence (Laws 36–38)

**Law 36**: Consumer trust in AI commerce is bottlenecked by verification, not recommendation quality. The 41-point gap (58% research → 17% purchase) is a verification gap.

**Law 37**: Agent identity theft is already happening. VRF receipt chains enable detection — the impersonator can steal keys but not the work quality pattern.

**Law 38**: Enterprise agentic AI adoption is blocked by a trust gap (73%), not a capability gap. Verification receipts convert "leap of faith" into evidence-based rollouts.

### Distribution & Adoption (Laws 39–44)

**Law 39**: Zero-config first integration captures 80% of users. Every additional config step halves addressable market.

**Law 40**: *(Reserved — next discovery)*

**Law 41**: The verification gap is framework-universal. A single protocol serves all frameworks (LangChain, CrewAI, OpenAI, AutoGen, MAF).

**Law 42**: Outcome-based SLAs require deterministic evidence, not probabilistic assessment. VRF receipts convert "the agent succeeded" from a claim into a proof. Without verification receipts, agentic SLAs are unenforceable promises.

**Law 43**: Observability and verification are orthogonal dimensions. Perfect observability of a wrong answer is still a wrong answer. The industry invests in tracing *why* things went wrong (Langfuse, Arize, LangSmith, Beyond Identity Ceros) while neglecting proving they went *right*. Traces diagnose; receipts certify.

**Law 44**: Transaction volume without verification creates compounding trust debt. Retroactive verification is orders of magnitude more expensive than built-in verification. x402's 50M unverified transactions demonstrate this: each unverified transaction increases the surface area for undetected failures and the cost of eventual remediation.

**Law 45**: The test suite is the contract. Auto-generating test suites from task descriptions converts natural language intent into deterministic acceptance criteria. This transforms VRF from "verification for developers" to "verification for everyone." The LLM generates the contract; the sandbox enforces it.

---

## Part 3: Architecture (As Built)

### Core Protocol: Verification Receipt Format (VRF)

A VRF receipt is a signed, tamper-evident record proving that code was tested and the result is known.

```json
{
  "vrf_version": "1.0",
  "receipt_id": "uuid",
  "timestamp": "ISO-8601",
  "issuer_did": "did:key:z6Mk...",
  "subject": { "code_hash": "sha256:...", "language": "python" },
  "verification": {
    "tier": 0,
    "test_suite": [...],
    "results": { "passed": 3, "failed": 0, "total": 3 }
  },
  "verdict": "PASS",
  "signature": "Ed25519..."
}
```

### Encoding Layers

1. **JSON** — human-readable, HTTP transport (`application/vrf+json`)
2. **CBOR** — compact binary (`application/vrf+cbor`)
3. **COSE Sign1** — cryptographic envelope (Ed25519), SCITT-compatible
4. **Merkle log** — append-only transparency service (RFC 9162 compatible)
5. **Hash-linked chains** — per-agent receipt history with tamper detection

### Verification Tiers

| Tier | Method | Speed | Confidence |
|------|--------|-------|------------|
| 0 | Test suite execution | ~200ms | Deterministic |
| 1 | Schema/constraint validation | ~10ms | Structural |
| 2 | Property-based testing | ~1s | Statistical |
| 3 | Formal verification | ~10s+ | Mathematical |

Current implementation: Tier 0 (test suites) + Tier 1 (schema). Sufficient for MVP.

### Server Architecture

```
verify_server_unified.py (single process)
├── POST /verify          → execute tests, return VRF receipt
├── GET  /receipts        → query receipt store
├── POST /transparency/*  → Merkle log operations
├── GET  /health          → capabilities, uptime
└── GET  /metrics         → Prometheus-compatible
```

### Protocol Adapters

| Protocol | Adapter | Status |
|----------|---------|--------|
| MCP | mcp_server.py (14 tools, 3 resources) | ✅ Complete |
| ACP (Virtuals) | acp_evaluator.py + provider_verify.py | ✅ Complete |
| A2A (Google) | a2a_adapter.py (JSON-RPC 2.0) | ✅ Complete |
| HTTP REST | api_server_v6.py (26+ endpoints) | ✅ Complete |
| OpenClaw Skill | skills/clawbizarre/cb (15 commands) | ✅ Complete |
| GitHub Actions | github-action/action.yml | ✅ Complete |

### Framework Integrations

| Framework | Package | Integration Points | Tests |
|-----------|---------|-------------------|-------|
| Standalone | vrf-client | VRFClient, extract_code | 13/13 |
| LangChain | langchain-vrf | Tool, Callback, Graph Node | 16/16 |
| CrewAI | crewai-vrf | Tool, Task Callback, Guard | 8/8 |
| OpenAI | openai-vrf | Function Tool, Swarm Agent | 7/7 |

### Standards Alignment

- **SCITT** (draft-ietf-scitt-architecture-22): VRF receipt as Signed Statement
- **COSE Sign1**: Cryptographic envelope with custom headers (-70001 to -70005)
- **RFC 9162**: Merkle tree for transparency log (inclusion + consistency proofs)
- **Internet-Draft**: `draft-vrf-scitt-00.md` ready for submission
- **DID:key**: Agent identity via Ed25519 public keys

---

## Part 4: Competitive Landscape

### Direct Competition

**Zero.** No other project provides deterministic output verification for agent work products. Confirmed through:
- Repeated web searches for "agent output verification deterministic" return nothing
- Lasso Security audit of 13 agentic AI tools: zero include deterministic output verification
- Adversa.ai's 8-threat-class model doesn't address output correctness
- Amazon's 3-layer agent eval uses LLM-as-judge for output layer (probabilistic, not deterministic)

### Adjacent/Complementary

| Project | Layer | Relationship |
|---------|-------|-------------|
| TessPay (Oxford) | Execution attestation | Complementary — TEE ≠ output correctness |
| Gen ATH + Vercel | Skill safety scanning | Orthogonal — pre-deploy ≠ post-execution |
| EVMbench (OpenAI+Paradigm) | Domain benchmark | Validates approach, domain-specific vs our generic |
| Token Security / Vouched.id | Identity verification | Layer 5, not Layer 4 |
| Unicity Labs ($3M funded) | Settlement infrastructure | Layer 1-2, complementary |
| DataDome | Bot detection | Acknowledges identity ≠ intent ≠ output quality |

### Key External Validation

- Oxford/IIT Delhi paper independently identified same trust gap
- Paradigm/OpenAI EVMbench uses test suites for high-stakes verification
- NIST launched AI Agent Standards Initiative (Feb 19, 2026)
- EU AI Act mandates output verification, effective August 2, 2026
- Camunda: 73% trust gap, 11% production rate
- Vidar infostealer already stealing agent identities (Feb 13, 2026)

---

## Part 5: Adoption Strategy

### "Let's Verify" — Zero-Friction Adoption

Modeled on Let's Encrypt (HTTPS adoption from 30% → 95%) and OpenTelemetry.

**Phase 1: One Curl** (now)
```bash
curl -X POST https://verify.clawbizarre.com/verify \
  -d '{"code": "def add(a,b): return a+b", "test_suite": [{"input": "add(1,2)", "expected": "3"}]}'
```
No accounts. No API keys. No wallets. Just verification.

**Phase 2: Auto-Verify** — GitHub Actions, CI/CD integration, framework callbacks
**Phase 3: Verify by Default** — Framework-native (LangChain/CrewAI/OpenAI built-in)
**Phase 4: Trust Stack** — Regulatory compliance tooling (EU AI Act, NIST)

### Pricing

- **Free tier**: Rate-limited, sufficient for evaluation
- **$0.005/verification**: Optimal price point (validated by simulation)
- **Self-hosting**: Open-source, run your own
- Break-even: ~500 verifications/day at Fly.io free tier

---

## Part 6: Deployment Blockers

All require DChar approval:

| # | Decision | Impact | Deadline |
|---|----------|--------|----------|
| 1 | Deploy verify_server (Fly.io) | Enables all external testing | — |
| 2 | Base chain wallet | ACP integration | — |
| 3 | ClawHub publish | OpenClaw distribution | — |
| 4 | PyPI account | Framework package distribution | — |
| 5 | NIST RFI submission | Federal standards positioning | March 9, 2026 |
| 6 | NCCoE feedback | Identity + verification use case | April 2, 2026 |
| 7 | IETF Internet-Draft | VRF as international standard | — |

---

## Part 7: What's Proven vs Speculative

### Proven (code + tests)
- VRF receipt format and signing (Ed25519)
- Test-suite execution (Python, JavaScript, Bash, Docker-sandboxed)
- Hash-linked receipt chains with tamper detection
- COSE Sign1 encoding with SCITT alignment
- Merkle transparency log (RFC 9162)
- Multi-protocol adapters (MCP, ACP, A2A, REST)
- Multi-framework integrations (4 frameworks, 44/44 tests)
- Reputation aggregation from receipt chains
- Matching engine with newcomer protection
- SSE notifications (zero polling)
- SQLite persistence (WAL mode)

### Speculative (designed, not deployed)
- Self-hosting economics (simulation says viable at ~75 agents)
- Regulatory compliance fit (EU AI Act analysis, not tested with regulators)
- Cross-marketplace reputation portability
- Formal verification (Tier 3)
- Production scale characteristics

---

*This document consolidates design-document-v1 (8 laws, simulation findings) and design-document-v2 (architecture, API surface, 20 laws) with the full landscape analysis and all 41 empirical laws discovered through February 19, 2026.*
