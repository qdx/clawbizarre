# Verification Receipt Format as a SCITT Content Type

**Internet-Draft**  
**draft-clawbizarre-scitt-vrf-00**  
**Intended status: Standards Track**  
**Date: 2026-02-19**

## Abstract

This document defines the Verification Receipt Format (VRF) as a content type for use with the Supply Chain Integrity, Transparency, and Trust (SCITT) architecture [draft-ietf-scitt-architecture]. VRF provides a deterministic, structurally-verifiable attestation that a unit of agent-produced work meets specified quality criteria. By encoding VRF receipts as SCITT Signed Statements, agent work verification gains the transparency, auditability, and non-repudiation properties of SCITT Transparency Services.

## Status of This Memo

This Internet-Draft is submitted in full conformance with the provisions of BCP 78 and BCP 79. Internet-Drafts are working documents of the Internet Engineering Task Force (IETF). This document is a product of an individual contributor. Distribution is unlimited.

## Copyright Notice

Copyright (c) 2026 IETF Trust and the persons identified as the document authors. All rights reserved.

## Table of Contents

1. Introduction
2. Terminology
3. Problem Statement
4. VRF Receipt Format
5. COSE Encoding
6. SCITT Integration
7. Verification Tiers
8. Security Considerations
9. IANA Considerations
10. References

## 1. Introduction

Autonomous AI agents increasingly perform work on behalf of users and other agents: generating code, processing data, translating text, and composing documents. When agents transact with each other through commerce protocols such as the Agent Commerce Protocol (ACP) [virtuals-acp], Agent-to-Agent protocol (A2A) [google-a2a], or Model Context Protocol (MCP) [anthropic-mcp], a fundamental trust gap exists: **no standardized mechanism verifies that delivered work meets the buyer's specification**.

Current approaches rely on either:
- **LLM-subjective evaluation**: Another language model judges the output (non-deterministic, expensive, gameable)
- **Human review**: Does not scale to autonomous multi-agent systems
- **Trust-on-delivery**: Buyer pays and hopes for the best

The Verification Receipt Format (VRF) addresses this gap by providing a **deterministic, portable, cryptographically-signed document** that proves a piece of work was structurally verified against explicit criteria. VRF receipts are:
- **Deterministic**: Same input produces the same verification outcome
- **Self-contained**: No dependency on the issuing server for later verification
- **Protocol-agnostic**: Works across ACP, A2A, MCP, x402, or standalone HTTP
- **Tiered**: Different verification depths for different cost/trust tradeoffs

This document specifies VRF as a content type for SCITT Signed Statements, enabling VRF receipts to be registered in SCITT Transparency Services for append-only auditing, cross-agent reputation aggregation, and non-repudiation.

### 1.1 Relationship to SCITT

SCITT [draft-ietf-scitt-architecture] defines an architecture for signed attestations about supply chain artifacts. Agent-to-agent work constitutes a supply chain of computation: an agent (supplier) delivers work (artifact) to another agent (consumer), and the verification receipt (signed statement) attests to the artifact's quality.

In SCITT terms:
- **Issuer** = Verification service (the entity that runs test suites, schema checks, etc.)
- **Signed Statement** = VRF receipt encoded as COSE_Sign1
- **Artifact** = The agent's work output (code, data, translation, etc.)
- **Transparency Service** = Append-only log of VRF receipts with Merkle inclusion proofs
- **SCITT Receipt** = Proof that a VRF receipt was logged (second layer of trust)

### 1.2 Relationship to Other Standards

- **RFC 9421 (HTTP Message Signatures)**: Used by Web Bot Auth for agent identity. VRF is complementary — Web Bot Auth verifies WHO the agent is; VRF verifies WHAT the agent produced.
- **ERC-8004 (Agent Commerce Protocol)**: On-chain agent identity and reputation. VRF receipts can be referenced as off-chain attestations linked to on-chain identity.
- **COSE (RFC 9052)**: VRF uses COSE_Sign1 for signing and encoding. Ed25519 (RFC 8032) is the REQUIRED signing algorithm.

## 2. Terminology

The key words "MUST", "MUST NOT", "REQUIRED", "SHALL", "SHALL NOT", "SHOULD", "SHOULD NOT", "RECOMMENDED", "NOT RECOMMENDED", "MAY", and "OPTIONAL" in this document are to be interpreted as described in BCP 14 [RFC2119] [RFC8174].

- **Agent**: An autonomous software entity capable of performing work and transacting with other agents.
- **Buyer**: An agent that requests work from another agent.
- **Provider**: An agent that performs work for another agent.
- **Verifier**: An entity (agent or service) that executes verification procedures against agent work output and produces VRF receipts.
- **VRF Receipt**: A signed document attesting to the structural verification of a unit of work.
- **Verification Tier**: A level indicating the depth and method of verification performed.
- **Receipt Chain**: An ordered, hash-linked sequence of VRF receipts associated with a single agent.
- **Transparency Log**: An append-only Merkle tree of registered VRF receipts.

## 3. Problem Statement

### 3.1 The Verification Gap

Commerce protocols for AI agents (ACP, A2A, MCP, x402, Google UCP) solve discovery, negotiation, payment, and communication. None address output quality verification:

| Protocol | Discovery | Negotiation | Payment | Verification |
|---|---|---|---|---|
| Virtuals ACP | ✓ | ✓ | ✓ (on-chain) | LLM-subjective (optional in v2) |
| Google A2A | ✓ (Agent Cards) | ✓ | — | None |
| Anthropic MCP | ✓ (tool listings) | — | — | None |
| Stripe x402 | — | — | ✓ (HTTP 402) | None |
| Google UCP | ✓ | ✓ | ✓ | None |
| OpenAI ACP | ✓ | ✓ | ✓ (Stripe) | None |

### 3.2 Cascading Verification Failure

In multi-agent pipelines (Agent A → Agent B → Agent C), unverified output at any stage propagates downstream. If Agent A produces subtly incorrect code and Agent B builds on it, Agent C's final output is corrupted without any participant detecting the failure. VRF receipts at each stage create a verifiable chain of quality attestations.

### 3.3 The Oracle Problem

LLM-based evaluation (the current dominant approach) is non-deterministic: the same input may produce different judgments on different runs. It is also gameable: agents can learn to produce outputs that satisfy the evaluator LLM without satisfying the actual specification. Structural verification (test suites, schema validation, constraint checking) is deterministic and specification-aligned.

## 4. VRF Receipt Format

### 4.1 JSON Representation

The canonical JSON representation of a VRF receipt is as follows. Field ordering MUST be lexicographic for canonical serialization (hash computation).

```json
{
  "vrf_version": "1.0",
  "receipt_id": "550e8400-e29b-41d4-a716-446655440000",
  "verified_at": "2026-02-19T11:30:00Z",
  "tier": 0,
  "verdict": "pass",
  "task": {
    "task_id": "acp-job-12345",
    "task_type": "code_generation",
    "description": "Sort function implementation"
  },
  "results": {
    "total": 5,
    "passed": 5,
    "failed": 0,
    "errors": 0,
    "details": [
      {
        "name": "test_basic_sort",
        "status": "pass",
        "expected": "[1, 2, 3]",
        "actual": "[1, 2, 3]",
        "elapsed_ms": 12
      }
    ]
  },
  "hashes": {
    "output": "sha256:e3b0c44298fc1c149afbf4c8...",
    "specification": "sha256:a1b2c3d4e5f6...",
    "tests": "sha256:f6e5d4c3b2a1..."
  },
  "metadata": {
    "execution_ms": 150,
    "sandbox": "docker",
    "structural": true,
    "verifier": "clawbizarre-verify/1.0"
  }
}
```

### 4.2 Field Definitions

#### 4.2.1 Top-Level Fields

| Field | Type | Required | Description |
|---|---|---|---|
| `vrf_version` | string | REQUIRED | Specification version. MUST be "1.0" for this document. |
| `receipt_id` | string | REQUIRED | UUID v4 uniquely identifying this receipt. |
| `verified_at` | string | REQUIRED | ISO 8601 UTC timestamp of verification completion. |
| `tier` | integer | REQUIRED | Verification tier (0-3). See Section 7. |
| `verdict` | string | REQUIRED | One of: "pass", "fail", "partial", "error". |

#### 4.2.2 Task Object

| Field | Type | Required | Description |
|---|---|---|---|
| `task_id` | string | OPTIONAL | External reference (e.g., ACP job ID, A2A task ID). |
| `task_type` | string | RECOMMENDED | Category of work. Registered values in Section 9. |
| `description` | string | OPTIONAL | Human-readable summary. |

#### 4.2.3 Results Object

| Field | Type | Required | Description |
|---|---|---|---|
| `total` | integer | REQUIRED | Total number of verification checks. |
| `passed` | integer | REQUIRED | Checks that passed. |
| `failed` | integer | REQUIRED | Checks that failed. |
| `errors` | integer | REQUIRED | Checks that errored (infra failure, not spec failure). |
| `details` | array | RECOMMENDED | Per-check detail objects. |

Each detail object:

| Field | Type | Required | Description |
|---|---|---|---|
| `name` | string | REQUIRED | Check identifier. |
| `status` | string | REQUIRED | One of: "pass", "fail", "error", "skip". |
| `expected` | string | OPTIONAL | Expected value/pattern. |
| `actual` | string | OPTIONAL | Actual value observed. |
| `elapsed_ms` | integer | OPTIONAL | Execution time for this check. |
| `message` | string | OPTIONAL | Human-readable diagnostic. |

#### 4.2.4 Hashes Object

Content-addressable references to the verified artifacts. All hashes use the format `algorithm:hex-digest`.

| Field | Type | Required | Description |
|---|---|---|---|
| `output` | string | REQUIRED | Hash of the work output that was verified. |
| `specification` | string | RECOMMENDED | Hash of the specification/requirements. |
| `tests` | string | RECOMMENDED | Hash of the test suite or verification criteria. |

#### 4.2.5 Metadata Object

| Field | Type | Required | Description |
|---|---|---|---|
| `verifier` | string | REQUIRED | Identifier of the verification software. |
| `execution_ms` | integer | RECOMMENDED | Total verification execution time. |
| `sandbox` | string | RECOMMENDED | Execution environment: "docker", "subprocess", "wasm", "none". |
| `structural` | boolean | REQUIRED | `true` if verification is deterministic/structural; `false` if LLM-based. |

## 5. COSE Encoding

### 5.1 COSE_Sign1 Structure

VRF receipts MUST be encoded as COSE_Sign1 structures [RFC 9052] for SCITT registration. The signing algorithm MUST be Ed25519 [RFC 8032] (COSE algorithm identifier -8).

```
COSE_Sign1 = [
  protected: {
    1: -8,               ; alg: EdDSA
    3: "application/vrf+cbor",  ; content_type
    -70001: "1.0",       ; vrf_version
    -70002: "did:key:z6Mk...", ; issuer_did
    -70003: "receipt-uuid",    ; receipt_id
    -70004: "sha256:prev...",  ; prev_receipt_hash (chain linking)
    -70005: 42            ; chain_position
  },
  unprotected: {},
  payload: CBOR-encoded VRF receipt (JSON canonical form),
  signature: Ed25519 signature
]
```

### 5.2 Custom COSE Header Parameters

| Label | Name | Type | Description |
|---|---|---|---|
| -70001 | vrf_version | tstr | VRF specification version |
| -70002 | issuer_did | tstr | DID of the signing verifier |
| -70003 | receipt_id | tstr | UUID of the VRF receipt |
| -70004 | prev_receipt_hash | tstr | Hash of the previous receipt in chain (or null for genesis) |
| -70005 | chain_position | uint | Position in the agent's receipt chain (0-indexed) |

Note: Labels in the range -65536 to -1 are designated for private use. A formal IANA registration would use positive labels from the designated expert range.

### 5.3 Canonical Serialization

For hash computation and signature input, the VRF receipt JSON MUST be serialized with:
1. Keys sorted lexicographically at all nesting levels
2. No whitespace between tokens
3. UTF-8 encoding
4. No trailing newline

The canonical form is then CBOR-encoded as a byte string for the COSE_Sign1 payload.

### 5.4 DID Key Format

Verifier identity SHOULD be expressed as a `did:key` [W3C DID] using the Ed25519 public key. Example:
```
did:key:z6MkhaXgBZDvotDkL5257faiztiGiC2QtKLGpbnnEGta2doK
```

## 6. SCITT Integration

### 6.1 Registration

To register a VRF receipt with a SCITT Transparency Service:

1. Verifier creates VRF receipt (JSON)
2. Verifier encodes as COSE_Sign1 (Section 5)
3. Verifier submits COSE_Sign1 to Transparency Service via SCRAPI [draft-ietf-scitt-scrapi]
4. Transparency Service validates the COSE signature
5. Transparency Service appends to Merkle log
6. Transparency Service returns a SCITT Receipt (inclusion proof)

The result is **two layers of trust**:
- **Layer 1 (VRF)**: The verifier attests that the work passed structural checks
- **Layer 2 (SCITT)**: The Transparency Service attests that this attestation was logged immutably

### 6.2 Registration Policy

A Transparency Service accepting VRF receipts SHOULD enforce:
- Valid COSE_Sign1 signature
- `vrf_version` is a recognized version
- `verdict` is one of the defined values
- `results.total >= results.passed + results.failed + results.errors`
- `hashes.output` is present and well-formed
- `metadata.structural` is present

A Transparency Service MAY additionally require:
- Minimum verification tier
- Known verifier DID (allowlist)
- Rate limits per verifier

### 6.3 Receipt Chain Verification

VRF receipt chains (hash-linked sequences per agent) can be verified by:
1. Retrieving all receipts for an agent DID from the Transparency Service
2. Verifying each COSE_Sign1 signature
3. Checking `prev_receipt_hash` linkage (chain integrity)
4. Checking `chain_position` is monotonically increasing
5. Computing aggregate statistics (pass rate, domain coverage, etc.)

This enables **portable reputation**: any party can independently compute an agent's verification history from the public Transparency Log.

### 6.4 Cross-Protocol Usage

VRF receipts referenced in commerce protocols:

**ACP (Virtuals)**: Embed VRF receipt ID in delivery memo. Evaluator can fetch full receipt from Transparency Service.
```json
{"delivery_memo": {"vrf_receipt_id": "550e8400-...", "vrf_verdict": "pass", "vrf_transparency_url": "https://ts.example.com/receipt/550e8400-..."}}
```

**A2A (Google)**: Include as task artifact with MIME type `application/vrf+json`.
```json
{"type": "artifact", "mimeType": "application/vrf+json", "data": "<base64-encoded VRF receipt>"}
```

**MCP (Anthropic)**: Return as tool result metadata.

**x402 (Stripe)**: Reference in payment memo for conditional settlement.

## 7. Verification Tiers

VRF defines four verification tiers, ordered by increasing assurance and cost:

| Tier | Name | Method | Deterministic | Typical Cost | Example |
|---|---|---|---|---|---|
| 0 | Test Suite | Execute provided test cases against output | Yes | $0.001-0.01 | Unit tests, I/O checks, expression evaluation |
| 1 | Schema/Constraint | Validate output against schema or formal constraints | Yes | $0.0001-0.001 | JSON Schema, type checking, regex patterns |
| 2 | Property-Based | Generate test cases from specification properties | Yes | $0.01-0.05 | Fuzzing, metamorphic testing, invariant checking |
| 3 | Formal | Mathematical proof of specification satisfaction | Yes | $0.10-1.00 | SMT solving, model checking, theorem proving |

A VRF receipt MUST set `metadata.structural = true` for Tiers 0-3.

Non-structural verification (LLM-based) MAY use VRF format with `metadata.structural = false` and `tier` omitted or set to a negative value. Such receipts SHOULD NOT be registered in SCITT Transparency Services unless the service explicitly accepts non-structural attestations.

## 8. Security Considerations

### 8.1 Verifier Trust

VRF receipts are only as trustworthy as the verifier. A malicious verifier can produce false "pass" receipts. Mitigations:
- **Verifier reputation**: Track verifier accuracy over time via Transparency Log analysis
- **Multiple verifiers**: Require N-of-M independent verifications for high-value work
- **Verifier diversity**: Require verifiers using different execution environments
- **Transparency**: All receipts are public and auditable

### 8.2 Test Suite Adequacy

A Tier 0 receipt proves the output passes the provided tests, NOT that the tests are comprehensive. Mitigations:
- **Coverage metadata**: Optional `results.coverage` field for code coverage metrics
- **Tier escalation**: High-value work should use Tier 2+ (property-based) verification
- **Test suite hashing**: The `hashes.tests` field enables third-party audit of test quality

### 8.3 Sandbox Escape

Untrusted code execution during verification requires sandboxing. The `metadata.sandbox` field documents the isolation method. Verifiers MUST:
- Use network-isolated execution environments for untrusted code
- Enforce memory and time limits
- Run as unprivileged users
- Document sandbox configuration in verifier metadata

### 8.4 Replay and Substitution

An agent could present another agent's VRF receipt as its own. Mitigations:
- **Content hashing**: `hashes.output` binds the receipt to specific output content
- **Task binding**: `task.task_id` binds to a specific transaction
- **Receipt chains**: Hash-linked chains bind receipts to specific agent identities
- **Transparency logging**: Registered receipts have immutable timestamps

### 8.5 Privacy

VRF receipts MAY contain sensitive information (test inputs, expected outputs, proprietary schemas). Transparency Services SHOULD support:
- **Redacted registration**: Register receipt hash only, not full content
- **Access-controlled detail retrieval**: Full receipt available only to authorized parties
- **Aggregate-only queries**: Reputation scores without individual receipt disclosure

## 9. IANA Considerations

### 9.1 Media Type Registration

Type name: application  
Subtype name: vrf+json  
Required parameters: none  
Optional parameters: version (default "1.0")  
Encoding considerations: UTF-8  
Security considerations: See Section 8  
Published specification: This document

Type name: application  
Subtype name: vrf+cbor  
Required parameters: none  
Optional parameters: none  
Encoding considerations: binary (CBOR)  
Security considerations: See Section 8  
Published specification: This document

### 9.2 COSE Header Parameters

Requested assignments in the COSE Header Parameters registry:

| Name | Label | Value Type | Description |
|---|---|---|---|
| vrf_version | TBD1 | tstr | VRF specification version |
| issuer_did | TBD2 | tstr | DID of the signing verifier |
| receipt_id | TBD3 | tstr | UUID of the VRF receipt |
| prev_receipt_hash | TBD4 | tstr | Hash of previous receipt in chain |
| chain_position | TBD5 | uint | Position in receipt chain |

### 9.3 VRF Task Type Registry

Initial registered values for `task.task_type`:

| Value | Description |
|---|---|
| code_generation | Source code produced from specification |
| code_review | Analysis/feedback on existing code |
| data_processing | Data transformation, filtering, aggregation |
| translation | Natural language translation |
| research | Information gathering and synthesis |
| testing | Test suite creation or execution |
| documentation | Technical documentation production |

## 10. References

### 10.1 Normative References

- [RFC 2119] Bradner, S., "Key words for use in RFCs to Indicate Requirement Levels"
- [RFC 8174] Leiba, B., "Ambiguity of Uppercase vs Lowercase in RFC 2119 Key Words"
- [RFC 8032] Josefsson, S., Liusvaara, I., "Edwards-Curve Digital Signature Algorithm (EdDSA)"
- [RFC 9052] Schaad, J., "CBOR Object Signing and Encryption (COSE): Structures and Process"
- [draft-ietf-scitt-architecture] IETF SCITT WG, "An Architecture for Trustworthy and Transparent Digital Supply Chains"
- [draft-ietf-scitt-scrapi] IETF SCITT WG, "SCITT Reference API"

### 10.2 Informative References

- [RFC 9162] Laurie, B., et al., "Certificate Transparency Version 2.0"
- [W3C-DID] W3C, "Decentralized Identifiers (DIDs) v1.0"
- [virtuals-acp] Virtuals Protocol, "Agent Commerce Protocol"
- [google-a2a] Google, "Agent-to-Agent Protocol"
- [anthropic-mcp] Anthropic, "Model Context Protocol"
- [stripe-x402] Stripe/Coinbase, "x402 HTTP Payment Protocol"
- [ERC-8004] Ethereum, "Agent Commerce Protocol (on-chain)"
- [RFC 9421] Backman, A., et al., "HTTP Message Signatures"

## Appendix A: Example SCITT Registration Flow

```
Agent Alice (buyer) → Agent Bob (provider): "Write a sort function"
Bob → Bob: writes sort function
Bob → ClawBizarre verify_server: POST /verify {code, test_suite}
verify_server → verify_server: executes tests in Docker sandbox
verify_server → Bob: VRF receipt (JSON, verdict: "pass")
verify_server → verify_server: COSE_Sign1 encode receipt
verify_server → Transparency Service: POST /entries (COSE_Sign1)
Transparency Service → verify_server: SCITT Receipt (inclusion proof)
Bob → Alice: delivers code + VRF receipt_id
Alice → Transparency Service: GET /entries/<receipt_id>
Alice → Alice: verifies COSE signature, checks verdict, audits test details
Alice → payment: releases escrow
```

## Appendix B: Relationship to Trust Stack

VRF occupies Layer 4 in the agent trust stack:

| Layer | Concern | Protocol/Tool | Method |
|---|---|---|---|
| 1 | Skill Safety | Gen ATH, Koi Security | Static analysis, LLM scanning |
| 2 | Tool Trust | MCPShield, MCP OAuth | Pre/during invocation safety |
| 3 | Communication Integrity | G²CP, A2A | Graph-grounded auditable messaging |
| **4** | **Output Quality** | **VRF (this document)** | **Deterministic structural verification** |
| 5 | Identity Ownership | Web Bot Auth, Token Security, Vouched.id | Cryptographic identity, KYA |

Each layer is orthogonal. A perfectly safe skill (Layer 1) with verified identity (Layer 5) can still produce incorrect output. VRF is the only deterministic verification at Layer 4.

## Authors' Addresses

Rahcd  
ClawBizarre Project  
Email: rahcd@openclaw.ai
