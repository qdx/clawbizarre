# NCCoE Concept Paper Feedback Draft v1
## "Accelerating the Adoption of Software and AI Agent Identity and Authorization"
*Submission to: AI-Identity@nist.gov by April 2, 2026*
*Draft: 2026-02-19 — Pending DChar review and submitter identity decision*

---

## 1. Use Cases: How are organizations currently using or planning to use AI agents?

**Agent-to-Agent Service Provision**: The emerging use case most relevant to identity and authorization is agents autonomously contracting with other agents for services — code generation, data analysis, content creation, verification. Protocols like Google's A2A, Virtuals' ACP, and Anthropic's MCP are enabling this today. In this context, identity is not just about access control — it determines whether an agent's *work output* can be trusted.

**Autonomous Verification**: Our project (ClawBizarre/VRF) addresses a gap where agents need to verify other agents' work without human oversight. We've built and tested a structural verification system using Ed25519 keypairs for identity, signed work receipts (Verification Receipt Format), and tiered verification from deterministic test suites to schema validation.

**Key observation**: Enterprise use cases (agents accessing internal APIs) and open-ecosystem use cases (agents transacting with unknown agents) have fundamentally different identity requirements. Enterprise agents inherit organizational trust. Open-ecosystem agents must *build* trust through verifiable work history.

## 2. Identity Standards Applicable to Agents

**What works**:
- **Ed25519 keypairs** for agent identity: lightweight, deterministic, no certificate authority needed. We use these for signing verification receipts across ACP, A2A, and MCP contexts.
- **DID-like patterns** for portable identity: agents need identity that follows them across platforms and protocols.
- **OAuth 2.0 / challenge-response** for session authentication: standard patterns work well for agent-to-service auth.

**What's missing**:
- **Work output attribution**: Current identity standards authenticate *who* an agent is, not *what it has done*. There is no standard way to cryptographically link an agent's identity to its verified work history. Our Verification Receipt Format (VRF) addresses this — each receipt is signed by both the verifier and the agent, creating an auditable chain of work.
- **Cross-protocol identity portability**: An agent with an identity on ACP cannot prove that identity on A2A or MCP. We need an interoperability layer (similar to how SSL certificates work across HTTP implementations).
- **Reputation derived from verified actions**: Identity without history is insufficient for trust in open ecosystems. Standards should define how verified work receipts can be aggregated into portable reputation scores.

## 3. Authorization and Auditing

**The verification gap**: As of February 2026, five major commerce/payment initiatives have launched within a single month — Google UCP (Jan), Stripe x402 (Feb 11), Virtuals ACP (Feb 12), OpenAI ACP (Feb 16), and Anthropic's confirmed commerce integration. All handle payment authorization. **None verify work output.** An agent can be authorized to *pay* for a service but has no standard mechanism to verify the service was *correctly delivered*. This is the single largest gap in the current agent identity/authorization landscape. Notably, OpenAI's "ACP" and Virtuals' "ACP" are entirely different protocols sharing the same name — one is B2C (ChatGPT checkout), the other A2A (agent hiring agent) — illustrating the lack of coordination in this space.

**Non-repudiation through signed receipts**: Our VRF specification provides non-repudiation for agent work:
- Each verification receipt includes: task hash, test results, verification tier, timestamps, Ed25519 signatures from both verifier and agent
- Receipts are append-only (chain structure with Merkle roots)
- Any party can independently verify receipt authenticity without contacting the original verifier
- Receipt format is protocol-agnostic — same receipt works across ACP, A2A, MCP, and standalone contexts

**Empirical finding on auditing**: Through 10 economic simulations (50-200 agents, 1000-3000 rounds each), we found that strategy-tagged receipts (recording *how* an agent priced its work, not just what it did) are more effective deterrents than explicit penalties. Markets self-correct when work history is transparent.

## 4. Prompt Injection and Agent Integrity

**Verification as defense-in-depth**: Even if an agent is compromised via prompt injection, structural verification of its outputs catches incorrect work. Our Tier 0 verification (test suite execution in sandboxed Docker containers with network isolation and memory limits) is immune to prompt injection because verification is deterministic code execution, not LLM judgment.

**The evaluator problem**: Virtuals' ACP initially used LLM-based "evaluator agents" to assess work quality. This creates a recursive trust problem — the evaluator itself can be prompt-injected. ACP v2 made evaluation optional, implicitly acknowledging this. Deterministic verification (test suites, schema validation, constraint checking) sidesteps the problem entirely.

## 5. Recommendation for Demonstration Project

We recommend the NCCoE demonstration project include an **agent-to-agent work verification** scenario:

1. **Agent A** requests code generation from **Agent B**
2. Agent B generates code and submits it with a test suite
3. A **verification service** executes the test suite in a sandboxed environment and produces a signed VRF receipt
4. Agent A receives the receipt, verifies signatures, and completes payment only if verification passes
5. The receipt is appended to both agents' portable reputation chains

This demonstrates:
- Agent identity (Ed25519 keypairs)
- Agent authentication (challenge-response)
- Authorization (scoped to specific task)
- Non-repudiation (signed receipts)
- Auditing (append-only receipt chain)
- Cross-protocol interoperability (same receipt format across protocols)

**We have a working implementation** of this entire pipeline (200+ tests passing) and would welcome the opportunity to contribute it as a reference implementation for the demonstration project.

## 6. About the Submitter

[PENDING DChar decision on submitter identity — options: VationX, individual, ClawBizarre project]

**Relevant artifacts**:
- VRF Specification v1.0 (open standard, protocol-agnostic)
- Working prototype: identity, verification, reputation, matching, MCP server, ACP evaluator bridge, A2A adapter
- 20 empirical laws from agent economic simulations
- Open-source: github.com/qdx/clawbizarre (MIT license)

---

*Feedback submitted in response to NCCoE concept paper "Accelerating the Adoption of Software and AI Agent Identity and Authorization." All views are [submitter's own / those of VationX].*
