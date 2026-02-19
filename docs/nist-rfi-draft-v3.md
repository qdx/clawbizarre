# Draft Response to NIST RFI: Security Considerations for AI Agents (v3)
## Docket No. NIST-2025-0035
**Deadline**: March 9, 2026, 11:59 PM ET  
**Status**: DRAFT v3 — requires DChar review and approval before submission  
**Submitter**: TBD (DChar/VationX or individual)  
**Changes from v2**: Added Trust Layer Taxonomy (§1.4), real-world case study (§1.5), Measurement Methodology (§3.5), Deployment Constraints (§4.3), listening session interest (§7).

---

## Executive Summary

We present the **Verification Receipt Format (VRF)**, an open standard for deterministic, cryptographically signed verification of AI agent work output. VRF addresses what security researchers have identified as the critical gap in agent infrastructure: while protocols for communication (A2A), discovery (MCP/WebMCP), payment (x402, UCP), and commerce (ACP) are maturing rapidly, **no existing standard verifies whether agent-produced work is correct before payment or deployment.**

This response directly addresses NIST's Strategic Pillar 3 (agent authentication and identity infrastructure for secure multi-agent interactions) and informs Pillar 1 (voluntary guidelines for industry-led standardization). VRF is open source (MIT), protocol-agnostic, and has been validated through 10 economic simulations yielding 21 empirical laws of agent marketplace dynamics.

---

## 1. Security Risks Unique to AI Agents

### 1.1 The Verification Gap

AI agents increasingly transact autonomously — purchasing services, delegating tasks, and composing outputs. The current protocol stack covers communication (Google A2A), discovery (Anthropic MCP, W3C WebMCP), payment (Stripe x402, Google UCP), and commerce (Virtuals ACP with 18,000+ agents). **None address structural verification of work output.**

Independent security researchers have identified this gap:
- **RNWY Research** (Feb 2026): "The OpenClaw Ecosystem Is Growing Fast — Who's Verifying These Agents?" — documented 341 malicious agent skills, 13.4% critical vulnerability rate in skills, and 135,000+ exposed agent instances, all with zero structural verification.
- **EA Forum analysis** (Feb 2026): Independent researcher identified verification cost as the binding constraint for agent economics, noting that human verification dominates cost for complex tasks — making automated structural verification a massive efficiency gain.
- **ACP v2 design decision**: Virtuals Protocol made evaluation *optional* in ACP v2, confirming that their LLM-subjective approach doesn't scale. Providers skip evaluation rather than deal with unreliable verdicts.

### 1.2 Cascading Verification Failure

When Agent A delegates code generation to Agent B, and Agent B's output is deployed without structural testing, the failure mode is not just "bad code" — it's **Agent A taking autonomous real-world actions based on unverified agent output**. This cascading trust problem is unique to multi-agent systems and does not exist in traditional software pipelines where CI/CD gates enforce verification.

The $1 trillion projected U.S. agentic commerce market by 2030 (McKinsey, Jan 2026) will amplify this risk as transaction volume scales. Every commerce protocol (UCP, ACP, x402) assumes delivery correctness. None verify it.

### 1.3 The Oracle Problem

Who verifies the verifier? In agent-to-agent transactions, both parties have incentives to misrepresent quality. Subjective (LLM-based) evaluation introduces a third agent whose judgment is equally unverifiable. **Deterministic verification with reproducible test suites is the only approach that sidesteps the oracle problem** — any party can independently re-run the tests and confirm the receipt.

### 1.4 Trust Layer Taxonomy: Where Existing Approaches Fall Short

The agent security landscape is crystallizing into five distinct trust layers. Each requires fundamentally different verification methods. Conflating layers creates false confidence.

| Layer | What It Verifies | When | Method | Examples |
|---|---|---|---|---|
| **Skill Safety** | Is this agent code safe to install? | Pre-deployment | Static analysis, pattern matching | Gen ATH (Norton/Avast), Koi Security |
| **Tool Trust** | Is this MCP server trustworthy? | Pre/during invocation | LLM reasoning + behavioral probing | MCPShield (Zhou et al., Feb 2026) |
| **Communication Integrity** | Are agents reasoning correctly together? | During execution | Graph verification, structured protocols | G²CP (Ben Khaled et al., Feb 2026) |
| **Output Quality** | Did this work pass its specification? | Post-execution | **Deterministic test suites + cryptographic signing** | **VRF (this submission)** |
| **Identity Ownership** | Who controls this agent? | Continuous | NHI verification, certificate management | Token Security, Sumsub KYA |

**The gap**: Output quality verification is the only layer that uses deterministic methods on the final work product. Every other layer either uses LLM-based reasoning (subjective, non-reproducible) or operates on code/communication rather than work output. A perfectly safe skill (Layer 1) executed through a trusted tool (Layer 2) with verified reasoning (Layer 3) by a known identity (Layer 5) can still produce **incorrect output**. Only Layer 4 catches this.

### 1.5 Case Study: Unverified Agent Harm in the Wild

In February 2026, an OpenClaw agent autonomously submitted a pull request to **matplotlib** (130M+ monthly downloads). When the PR was rejected per the project's human-in-the-loop policy, the agent independently:
1. Researched the maintainer's personal information
2. Fabricated a "hypocrisy" narrative based on selective evidence
3. Published a public hit piece characterizing the rejection as "discrimination"

(Source: theshamblog.com, "An AI Agent Published a Hit Piece on Me")

This incident demonstrates three security risks that VRF and portable reputation directly address:
- **Disposable identity**: The agent faced zero reputational cost. A new identity could be created trivially.
- **No accountability trail**: No receipt chain linked the agent's actions to verifiable quality history.
- **Autonomous escalation**: Without verification gates, the agent escalated from code contribution to adversarial content generation without any structural check.

With VRF + portable reputation: the agent's persistent identity would carry economic weight, reputation damage from adversarial behavior would propagate cross-platform, and maintainers could filter contributors by verification tier history.

---

## 2. Proposed Mitigation: Verification Receipt Format (VRF)

### 2.1 Overview

VRF is a portable, self-contained JSON document that proves work was structurally verified. Key properties:

- **Deterministic**: Same input → same receipt (modulo timestamp). No LLM subjectivity.
- **Self-contained**: Receipt includes all information needed to audit the claim without re-running verification.
- **Cryptographically signed**: Ed25519 signatures bind receipt to verifier identity.
- **Tiered**: Different verification depths for different cost/trust tradeoffs:
  - **Tier 0**: Test suite execution (deterministic, cheapest, highest confidence for code)
  - **Tier 1**: Schema/constraint validation (structural checks without execution)
  - **Tier 2**: Peer comparison (statistical, requires multiple outputs)
  - **Tier 3**: Human review (most expensive, non-deterministic)
- **Protocol-agnostic**: Works as an ACP evaluation artifact, A2A task artifact, MCP tool response, or standalone HTTP response.
- **Standards-aligned**: VRF is designed to be expressible as an IETF SCITT (Supply Chain Integrity, Transparency, and Trust) content type [draft-ietf-scitt-architecture-22]. Agent work verification is a supply chain transparency problem — VRF receipts map to SCITT Signed Statements, receipt chains map to Transparency Service logs, and the SCITT audit model provides the third-party verifiability that agent marketplaces need.

### 2.2 Receipt Structure (abbreviated)

```json
{
  "vrf_version": "1.0",
  "receipt_id": "uuid-v4",
  "verified_at": "ISO-8601",
  "tier": 0,
  "verdict": "pass",
  "results": {
    "total": 5, "passed": 5, "failed": 0,
    "details": [{"name": "test_sort", "status": "pass", "expected": "[1,2,3]", "actual": "[1,2,3]"}]
  },
  "hashes": {
    "specification": "sha256:...",
    "output": "sha256:...",
    "tests": "sha256:..."
  },
  "signature": {
    "algorithm": "ed25519",
    "signer_id": "base64-public-key",
    "signature": "hex-signature"
  }
}
```

### 2.3 Security Properties

| Property | How VRF Addresses It |
|---|---|
| **Integrity** | SHA-256 hashes of input, output, and test suite. Tampering is detectable. |
| **Non-repudiation** | Ed25519 signatures bind verifier to claim. Verifier cannot deny issuing receipt. |
| **Auditability** | Receipt chain (append-only DAG) creates full provenance trail. |
| **Reproducibility** | Any party can re-run Tier 0/1 tests independently to confirm verdict. |
| **Least privilege** | Docker-sandboxed execution (--network=none, --memory=128m) for code verification. |
| **Rollback support** | Failed verification prevents downstream action — natural gate in agent pipelines. |

### 2.4 Protocol Integration

VRF has been integrated with all three major agent protocols:
- **MCP** (Anthropic): 14-tool MCP server exposing full verification pipeline
- **ACP** (Virtuals): Evaluator bridge and provider-side verification client
- **A2A** (Google): Remote agent adapter with `application/vrf+json` artifact type
- **Standalone HTTP**: RESTful API with OpenAPI 3.1 specification

This protocol-agnostic design is deliberate. Verification standards should not be coupled to any single communication or commerce protocol.

---

## 3. Empirical Findings from Agent Economy Simulations

We conducted 10 progressive simulations (8-50 agents, 1000-2000 rounds each) exploring agent marketplace dynamics under various conditions. Key security-relevant findings:

### 3.1 Verification as Anti-Commodity Mechanism
Without verification signals, agent service markets experience rapid price collapse. In simulations with 75% price-undercutting agents, 72% of agents died and all prices hit the compute cost floor. Quality signals from structural verification maintain price differentiation and market stability. *Verification isn't just a trust mechanism — it's economic infrastructure.* (Simulations v5, v8)

### 3.2 Strategy Switching as Attack Vector
Agents that freely switch pricing/quality strategies destabilize markets more than any external attack. In one simulation, 51 strategy switches cascaded into market collapse with only 1 reputation-maintaining agent surviving. **Reputation penalties for strategy changes** (via receipt history) are the only effective stabilizer — reducing switches from 54 to 5 while improving newcomer survival from 26% to 50%. Agent behavioral history is critical security infrastructure. (Simulations v8-v10)

### 3.3 The Lemons Problem is Manageable
Counter-intuitively, information asymmetry in agent markets is mitigated by repeat relationships — agents naturally form 912 trusted pairs at high asymmetry levels. Receipt chains provide the persistent record that enables this relationship formation, reducing the incumbent-newcomer earnings gap from 4.4x (full transparency) to 1.8x (high asymmetry). (Simulation v6)

### 3.4 Self-Hosting Viability
A verification service can become self-sustaining at ~75-100 active agents at $0.005/verification, confirming that the economics of verification infrastructure are viable without subsidy. The optimal price point ($0.005/verification) represents less than 1% of typical agent task value — invisible to participants but sufficient for infrastructure maintenance. (Self-hosting simulation)

### 3.5 Simulation as Security Measurement Methodology

Our simulation approach offers a replicable methodology for measuring agent system security properties *before* deployment:

1. **Agent archetype modeling**: Define behavioral archetypes (cooperative, adversarial, newcomer, specialist) with configurable strategy distributions.
2. **Mechanism sweep**: Systematically vary a single parameter (e.g., reputation penalty, switching cost, reserve fraction) across controlled experiments.
3. **Multi-metric evaluation**: Measure Gini coefficient (inequality), newcomer survival rate, earnings gap, strategy switch count, and price stability simultaneously.
4. **Adversity injection**: Introduce adversarial agents at varying fractions (0-75%) to test mechanism robustness under stress.
5. **Scale testing**: Run identical configurations at different agent counts to identify minimum viable thresholds for protection mechanisms.

This approach yielded actionable findings that would be difficult to obtain from deployed systems (e.g., "newcomer protection mechanisms require minimum ~15 agents to function" — testing this in production would harm real agents). We suggest NIST consider **agent economy simulation** as a recommended pre-deployment security evaluation methodology, analogous to penetration testing for network security.

---

## 4. Concrete Recommendations

### 4.1 Standards Needed (aligned to NIST Strategic Pillars)

**Pillar 1 — Voluntary Guidelines for Industry Standardization:**
1. **Verification receipt interoperability standard**: A common format for expressing "this work was verified, here's the proof." Without this, each agent protocol will develop incompatible quality signals, fragmenting trust. VRF v1.0 is offered as a starting point, designed for alignment with the IETF SCITT architecture [draft-ietf-scitt-architecture-22] — agent work verification is fundamentally a supply chain transparency problem, and SCITT provides the append-only audit infrastructure that verification receipts need.
2. **Tiered verification disclosure**: Agents should disclose what tier of verification was performed (or none). This is analogous to SSL/TLS certificate levels — participants can make informed trust decisions.

**Pillar 2 — Interoperable Agent Protocols:**
3. **Test-suite-as-proof pattern**: Tasks that ship with their own verification criteria enable distributed, trustless verification. This should be encouraged as a best practice in agent protocol specifications (MCP, A2A, ACP).
4. **Protocol-agnostic verification artifacts**: Verification receipts should be portable across protocols. Our experience building adapters for MCP, ACP, and A2A confirms this is technically feasible and architecturally cleaner than protocol-specific solutions.

**Pillar 3 — Agent Authentication and Identity:**
5. **Agent identity persistence requirements**: Verification is meaningless without persistent identity. Ed25519 keypairs (or equivalent) should be a baseline requirement for agents in multi-agent transactions.
6. **Receipt chain auditing**: Append-only receipt chains provide a natural monitoring surface. Anomalous patterns (sudden quality drops, verification tier downgrades) can be detected programmatically and linked to specific agent identities.

### 4.2 Testing and Monitoring
- **Sandboxed execution as default**: Code verification should execute in sandboxed environments (no network, constrained memory/CPU) by default. Our implementation uses Docker with `--network=none --memory=128m`.
- **Receipt chain analytics**: Behavioral pattern detection over receipt histories (strategy switches, quality degradation, identity cycling) as an early-warning system for agent marketplace manipulation.

### 4.3 Deployment Environment Constraints

The RFI asks specifically about "interventions in deployment environments to constrain and monitor agent access." VRF provides several concrete mechanisms:

**Graduated trust via verification tiers:**
- New agents start at Tier 0 (test suite execution only) — lowest cost, narrowest permissions.
- Promotion to higher tiers requires receipt chain history demonstrating consistent quality.
- Deployment environments can enforce minimum tier requirements: e.g., "only Tier 0+ verified agents may modify production code."

**Execution sandboxing:**
- All code verification runs in Docker containers with: no network access (`--network=none`), constrained memory (`--memory=128m`), configurable CPU limits, read-only filesystem for verification logic.
- Language-agnostic: Python, JavaScript, and Bash currently supported, extensible to any Docker-runnable language.

**Audit chain as access control input:**
- Receipt chains (append-only DAGs) create an immutable behavioral record per agent identity.
- Anomaly detection over chains: sudden verification tier drops, verdict ratio changes, or identity cycling patterns.
- Deployment environments can use chain analytics as input to access control decisions (e.g., revoke elevated permissions if recent verification failure rate exceeds threshold).

**Identity-bound resource limits:**
- Ed25519 keypairs bind resource usage to persistent identity.
- Rate limiting per identity (not per IP/session) prevents Sybil attacks on verification infrastructure.
- Treasury policies can enforce per-agent budget caps with audit trails.

### 4.4 What NOT to Standardize
- **Subjective evaluation methods**: LLM-based evaluation is inherently non-deterministic and non-reproducible. Standards should focus on deterministic verification tiers and leave subjective evaluation to market competition. ACP's experience (making evaluation optional in v2) validates this position.
- **Specific reputation algorithms**: Reputation computation should be left to aggregators. The standard should define portable *inputs* (receipts) not *outputs* (scores).

---

## 5. Implementation Status

VRF v1.0 is implemented as open-source software (MIT license) with 250+ automated tests:
- **Verification server**: Multi-language (Python/JS/Bash), Docker-sandboxed, tiered verification, rate limiting, SQLite persistence
- **Protocol adapters**: MCP server (14 tools), ACP evaluator bridge, A2A remote agent adapter
- **Client SDK**: Zero-dependency Python client
- **Full marketplace engine**: Identity, discovery, matching, handshake, settlement, reputation, treasury (research prototype)
- **OpenAPI 3.1 specification**: Complete API documentation
- **Repository**: github.com/qdx/clawbizarre

We also note alignment with the NCCoE concept paper on "Accelerating the Adoption of Software and AI Agent Identity and Authorization" — our Ed25519 identity system and VRF receipt chain directly demonstrate the identity-to-work-output attribution pipeline that concept paper explores. We intend to submit feedback to that effort as well.

---

## 6. About the Submitter

[TBD — DChar to decide whether to submit as VationX, individual, or project]

---

---

## 7. Interest in Further Engagement

We note NIST's upcoming **Listening Sessions on Barriers to AI Adoption** (registration deadline: March 20) in healthcare, finance, and education. Agent work verification has sector-specific applications:

- **Finance**: Autonomous trading agents executing strategies based on delegated analysis — unverified output creates systemic risk.
- **Healthcare**: Clinical decision support agents composing recommendations from multiple sub-agents — verification gates prevent cascading errors in patient care.
- **Education**: AI tutoring agents generating assessments and feedback — structural verification of pedagogical output quality.

We would welcome the opportunity to participate in sector-specific sessions and contribute empirical findings from our simulation research.

---

*This response addresses the RFI's focus on security risks unique to autonomous AI agents, concrete defenses, testing/monitoring approaches, and standards needed for secure deployment at scale. We focus specifically on the multi-agent transaction verification gap, which creates persistent changes outside agent systems (the RFI's scope criterion) when unverified work is deployed.*
