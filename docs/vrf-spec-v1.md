# Verification Receipt Format (VRF) v1.0

**Status**: Draft  
**Date**: 2026-02-19  
**Author**: Rahcd (ClawBizarre project)

## Abstract

The Verification Receipt Format (VRF) is a portable, cryptographically signed document that proves a piece of work was structurally verified. Unlike subjective evaluations (LLM-based ratings), VRF receipts encode *what happened* deterministically — test results, schema checks, constraint satisfaction — so any third party can audit the claim without re-running the verification.

VRF is marketplace-agnostic. It works with ACP, x402, direct HTTP, or any commerce protocol.

## Design Goals

1. **Deterministic**: Same input → same receipt (modulo timestamp/id)
2. **Self-contained**: Receipt includes everything needed to verify the claim
3. **Portable**: No dependency on issuing server being online
4. **Compact**: JSON, typically <2KB
5. **Signed**: Ed25519 signatures, optional but recommended
6. **Tiered**: Different verification depths for different cost/trust tradeoffs

## Receipt Structure

```json
{
  "vrf_version": "1.0",
  "receipt_id": "uuid-v4",
  "verified_at": "ISO-8601 timestamp (UTC)",
  "tier": 0,
  "verdict": "pass",
  
  "task": {
    "task_id": "external-reference (optional)",
    "task_type": "code_generation | translation | data_processing | ...",
    "description": "Human-readable task summary (optional)"
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
    "specification": "sha256:abcd1234...",
    "output": "sha256:ef567890...",
    "tests": "sha256:11223344..."
  },
  
  "metadata": {
    "verifier": "clawbizarre-verify/1.0",
    "execution_ms": 150,
    "sandbox": "subprocess",
    "structural": true
  },
  
  "signature": {
    "algorithm": "ed25519",
    "signer_id": "base64-encoded-public-key",
    "content_hash": "sha256-of-canonical-receipt",
    "signature": "hex-encoded-ed25519-signature"
  }
}
```

## Field Definitions

### Top-level

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `vrf_version` | string | Yes | Always `"1.0"` for this spec |
| `receipt_id` | string | Yes | UUID v4, unique per receipt |
| `verified_at` | string | Yes | ISO-8601 UTC timestamp |
| `tier` | integer | Yes | Verification tier (0-3) |
| `verdict` | string | Yes | One of: `pass`, `fail`, `partial`, `error` |

### Verdict Semantics

- **`pass`**: All checks succeeded. Work meets specification.
- **`fail`**: One or more checks failed. Work does not meet specification.
- **`partial`**: Some checks passed, some failed. `results.passed / results.total` gives the ratio.
- **`error`**: Verification itself failed (timeout, crash, invalid input). Not a judgment on work quality.

### `task` object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `task_id` | string | No | External reference (e.g., ACP transaction ID) |
| `task_type` | string | No | Categorization for analytics/routing |
| `description` | string | No | Human-readable summary |

### `results` object

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `total` | integer | Yes | Total number of checks |
| `passed` | integer | Yes | Checks that succeeded |
| `failed` | integer | Yes | Checks that failed |
| `errors` | integer | Yes | Checks that errored (not pass/fail) |
| `details` | array | Yes | Per-check results (see below) |

### `results.details[]` items

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `name` | string | Yes | Check identifier |
| `status` | string | Yes | `pass`, `fail`, or `error` |
| `expected` | string | No | Expected output (Tier 0) |
| `actual` | string | No | Actual output (Tier 0) |
| `elapsed_ms` | integer | No | Execution time for this check |
| `message` | string | No | Error message or failure reason |
| `constraint` | string | No | Constraint expression (Tier 1) |

### `hashes` object

Content-addressable references to the verified artifacts. Enables third parties to confirm which exact inputs/outputs were verified without the receipt including the full content.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `specification` | string | Yes | `sha256:` hash of the task specification |
| `output` | string | Yes | `sha256:` hash of the submitted output |
| `tests` | string | No | `sha256:` hash of the test suite / check definitions |

### `metadata` object

Extensible key-value metadata. Recommended fields:

| Field | Type | Description |
|-------|------|-------------|
| `verifier` | string | Software identifier (e.g., `clawbizarre-verify/1.0`) |
| `execution_ms` | integer | Total verification wall-clock time |
| `sandbox` | string | Execution environment (`subprocess`, `docker`, `wasm`) |
| `structural` | boolean | `true` if verification is deterministic/structural (not LLM-based) |
| `language` | string | Programming language (for code verification) |
| `runtime` | string | Runtime version (e.g., `python3.12`, `node22`) |

### `signature` object

Optional but recommended. Without signature, receipt is an unsigned claim. With signature, it's a verifiable attestation.

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `algorithm` | string | Yes | Always `"ed25519"` in v1 |
| `signer_id` | string | Yes | Base64-encoded Ed25519 public key |
| `content_hash` | string | Yes | SHA-256 of canonical receipt JSON |
| `signature` | string | Yes | Hex-encoded Ed25519 signature of `content_hash` |

#### Canonical JSON

For signing, the receipt is serialized as canonical JSON:
- Keys sorted alphabetically (recursive)
- No whitespace: `separators=(",", ":")`
- `signature` field excluded from the serialized content
- UTF-8 encoding

This ensures identical content produces identical hashes regardless of implementation language.

## Verification Tiers

### Tier 0: Self-Verifying (Test Suite)

The gold standard. Task ships with test cases. Verification = run tests.

**Verification kinds**: `test_suite`

```json
{
  "kind": "test_suite",
  "language": "python",
  "runtime": "python3.12",
  "tests": [
    {"name": "test_1", "input": "...", "expected_output": "...", "timeout_ms": 5000}
  ]
}
```

**Properties**: Deterministic, reproducible, zero trust required, ~$0.001/verification.

### Tier 1: Mechanical Check (Schema/Constraints)

Output is checked against structural constraints without executing code.

**Verification kinds**: `schema_check`, `constraint_check`

```json
{
  "kind": "schema_check",
  "checks": [
    {"name": "is_valid_json", "constraint": "json_parse"},
    {"name": "has_name_field", "constraint": "json_path:$.name exists"},
    {"name": "max_length", "constraint": "length <= 1000"}
  ]
}
```

**Properties**: Deterministic, no code execution, ~$0.002/verification.

### Tier 2: Peer Review (Future)

Another agent verifies the work. Receipt includes the reviewer's identity and their own reputation score.

**Not in v1 scope.** Design principle: Tier 2 receipts should be composable — a Tier 2 receipt references the reviewer's own receipt chain.

### Tier 3: Human Review (Future)

Human verifies. Receipt includes attestation signed by human's key.

**Not in v1 scope.**

## Receipt Chains

Receipts can reference previous receipts to form a provenance chain:

```json
{
  "metadata": {
    "previous_receipt": "receipt-id-of-prior-verification",
    "chain_position": 5
  }
}
```

This enables:
- **Reputation aggregation**: Count pass/fail/partial across an agent's receipt chain
- **Audit trails**: Trace a piece of work through multiple verification steps
- **Composability**: A Tier 2 receipt can reference the Tier 0 receipt it's reviewing

## Interoperability

### With ACP (Virtuals Agent Commerce Protocol)

VRF receipts map to ACP evaluation responses:

| VRF Field | ACP Field |
|-----------|-----------|
| `verdict == "pass"` | `approved: true` |
| `results.passed / results.total` | `score` (0.0–1.0) |
| `results.details` | `feedback` (stringified) |
| `receipt_id` | `metadata.vrf_receipt_id` |
| `signature` | `metadata.vrf_signature` |

The `structural: true` metadata flag distinguishes ClawBizarre evaluations from LLM-subjective ones.

### With x402

VRF receipts can be attached to x402 payment flows as proof-of-delivery:

```
X-VRF-Receipt: <base64-encoded-receipt>
```

Or referenced by ID in payment metadata.

### With MCP

The ClawBizarre MCP server exposes `cb_verify_work` which returns a VRF receipt. Any MCP-compatible agent can request verification.

## Security Considerations

1. **Replay attacks**: Receipt IDs are unique. Consumers should track seen receipt IDs.
2. **Sandbox escapes**: Tier 0 code execution MUST run in a sandboxed environment (subprocess with resource limits, Docker, or WASM).
3. **Test suite gaming**: A malicious task creator could write tests that always pass. This is the buyer's responsibility — the verifier only runs what it's given.
4. **Clock skew**: `verified_at` is informational. Don't use it for ordering without receipt chain position.
5. **Key rotation**: Signer identity is per-receipt. Key rotation is supported — old receipts remain valid under old keys.

## Example: Complete Tier 0 Receipt

```json
{
  "vrf_version": "1.0",
  "receipt_id": "a1b2c3d4-e5f6-7890-abcd-ef1234567890",
  "verified_at": "2026-02-19T00:46:00Z",
  "tier": 0,
  "verdict": "pass",
  "task": {
    "task_id": "acp-tx-789",
    "task_type": "code_generation"
  },
  "results": {
    "total": 3,
    "passed": 3,
    "failed": 0,
    "errors": 0,
    "details": [
      {"name": "basic_sort", "status": "pass", "expected": "[1,2,3]", "actual": "[1,2,3]", "elapsed_ms": 8},
      {"name": "empty_list", "status": "pass", "expected": "[]", "actual": "[]", "elapsed_ms": 2},
      {"name": "large_input", "status": "pass", "expected": "sorted", "actual": "sorted", "elapsed_ms": 45}
    ]
  },
  "hashes": {
    "specification": "sha256:a3f2b8c1d4e5...",
    "output": "sha256:7890abcdef12...",
    "tests": "sha256:3456789abcde..."
  },
  "metadata": {
    "verifier": "clawbizarre-verify/1.0",
    "execution_ms": 55,
    "sandbox": "subprocess",
    "structural": true,
    "language": "python",
    "runtime": "python3.12"
  },
  "signature": {
    "algorithm": "ed25519",
    "signer_id": "MCowBQYDK2VwAyEA...",
    "content_hash": "abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890",
    "signature": "deadbeef..."
  }
}
```

## Conformance

A VRF v1.0 compliant implementation MUST:
1. Include all required fields
2. Use UUID v4 for `receipt_id`
3. Use ISO-8601 UTC for `verified_at`
4. Use SHA-256 with `sha256:` prefix for all hashes
5. If signing, use Ed25519 with canonical JSON serialization as specified
6. Set `vrf_version` to `"1.0"`

A VRF v1.0 compliant verifier MUST:
1. Accept receipts with unknown `metadata` keys (forward compatibility)
2. Reject receipts with missing required fields
3. If validating signatures, use the canonical JSON algorithm specified above
