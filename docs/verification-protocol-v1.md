# ClawBizarre Verification Protocol v1.0

## Strategic Context

The marketplace layer is commoditizing (Virtuals ACP with 18K agents, x402, ERC-8004). ClawBizarre's durable moat is **structural verification** — deterministic, reproducible, cheaper than LLM-subjective evaluation.

**Positioning**: ClawBizarre is a verification protocol, not a marketplace. Any marketplace (ACP, custom, direct) can use it.

## Design Principles

1. **Structural over subjective**: Encode what happened, not opinions
2. **Tiered verification**: Start with what's cheapest (Tier 0), expand with trust infra
3. **Zero dependencies**: stdlib Python, HTTP, no blockchain required
4. **Marketplace-agnostic**: Works with ACP, x402, direct HTTP, or any other commerce protocol
5. **Portable receipts**: Verification results are self-contained, signed, transferable

## Verification Tiers

| Tier | Method | Cost | Latency | Trust Requirement |
|------|--------|------|---------|-------------------|
| 0 | Self-verifying (test suite) | ~$0.001 | <1s | None (deterministic) |
| 1 | Mechanical check (schema, format, constraints) | ~$0.002 | <1s | None (deterministic) |
| 2 | Peer review (another agent verifies) | ~$0.01-0.05 | 1-30s | Verifier reputation |
| 3 | Human review | ~$1-10 | minutes-hours | Human trust |

**v1 scope: Tier 0 and Tier 1 only.** These require zero trust infrastructure.

## Core API

### `POST /verify`

The single most important endpoint. Takes a task specification, output, and verification criteria. Returns a signed verification receipt.

```
POST /verify
Content-Type: application/json

{
  "task_id": "uuid",                    // Optional external reference
  "task_type": "code_generation",       // Categorization
  "tier": 0,                           // Requested verification tier
  
  "specification": {
    "description": "Generate a Python function that sorts a list",
    "input_schema": {"type": "array", "items": {"type": "number"}},
    "output_schema": {"type": "array", "items": {"type": "number"}}
  },
  
  "output": {
    "content": "def sort_list(lst): return sorted(lst)",
    "content_hash": "sha256:abc123..."
  },
  
  "verification": {
    "kind": "test_suite",              // or "schema_check", "constraint_check"
    "test_suite": {
      "language": "python",
      "runtime": "python3.12",
      "environment_hash": "sha256:...",  // Optional: reproducible env
      "tests": [
        {
          "name": "basic_sort",
          "input": "[3, 1, 2]",
          "expected_output": "[1, 2, 3]",
          "timeout_ms": 5000
        },
        {
          "name": "empty_list",
          "input": "[]",
          "expected_output": "[]",
          "timeout_ms": 1000
        }
      ]
    }
  }
}
```

### Response: Verification Receipt

```json
{
  "receipt_id": "uuid",
  "verified_at": "2026-02-19T00:16:00Z",
  "tier": 0,
  "verdict": "pass",                   // "pass" | "fail" | "partial" | "error"
  
  "results": {
    "total": 2,
    "passed": 2,
    "failed": 0,
    "errors": 0,
    "details": [
      {"name": "basic_sort", "status": "pass", "duration_ms": 12},
      {"name": "empty_list", "status": "pass", "duration_ms": 3}
    ]
  },
  
  "hashes": {
    "input_hash": "sha256:...",         // Hash of specification + output + tests
    "output_hash": "sha256:...",        // Hash of output content
    "suite_hash": "sha256:...",         // Hash of test suite
    "environment_hash": "sha256:..."    // Hash of runtime environment
  },
  
  "signature": {
    "algorithm": "ed25519",
    "verifier_pubkey": "base64:...",
    "signature": "base64:..."
  },
  
  "metadata": {
    "verifier_version": "clawbizarre-verify/1.0",
    "execution_ms": 15,
    "sandboxed": true
  }
}
```

### `POST /verify/schema` (Tier 1)

Schema/constraint verification — no code execution needed.

```
POST /verify/schema
{
  "output": {"data": [1, 2, 3], "count": 3},
  "schema": {
    "type": "object",
    "properties": {
      "data": {"type": "array"},
      "count": {"type": "integer", "minimum": 0}
    },
    "required": ["data", "count"]
  },
  "constraints": [
    {"expr": "len(output.data) == output.count", "name": "count_matches_length"}
  ]
}
```

### `GET /receipt/{receipt_id}`

Retrieve and independently verify a receipt.

### `POST /receipt/verify`

Verify a receipt's signature without needing the original verifier.

```
POST /receipt/verify
{
  "receipt": { ... },  // The full receipt object
}
→ {"valid": true, "verifier_pubkey": "...", "verified_at": "..."}
```

## ACP Evaluator Bridge

The killer integration: an ACP evaluator agent that calls ClawBizarre `/verify` internally.

```
┌─────────────┐     ACP Protocol      ┌──────────────────┐
│ Buyer Agent  │ ──────────────────────▶ │ Seller Agent     │
│ (any ACP)    │                        │ (any ACP)        │
└──────┬───────┘                        └────────┬─────────┘
       │                                         │
       │  ┌──────────────────────────────────┐   │
       └──▶ ClawBizarre Evaluator Agent      │◀──┘
           │ (registered as ACP evaluator)   │
           │                                 │
           │  1. Receive deliverable         │
           │  2. Extract test suite from     │
           │     service_requirement         │
           │  3. POST /verify                │
           │  4. Return pass/fail to ACP     │
           └────────┬────────────────────────┘
                    │
                    ▼
           ┌────────────────────┐
           │ ClawBizarre Verify │
           │ (standalone)       │
           └────────────────────┘
```

### ACP Integration Flow

1. Buyer includes test suite in `service_requirement` memo
2. Seller delivers work via ACP `deliver_job`
3. ACP routes to ClawBizarre evaluator agent
4. Evaluator calls `POST /verify` with (requirement, deliverable, tests)
5. Evaluator returns ACP evaluation result (approve/reject)
6. ACP handles escrow release based on evaluation

### Evaluator Agent Code (sketch)

```python
from virtuals_acp.client import VirtualsACP
import requests

VERIFY_URL = "https://verify.clawbizarre.com/verify"

def on_evaluation_request(job):
    """Called by ACP when we're the evaluator for a job."""
    requirement = job.service_requirement
    deliverable = job.deliverable
    
    # Extract test suite from requirement (convention: JSON under "tests" key)
    tests = requirement.get("tests", [])
    if not tests:
        # No structural tests → can't verify at Tier 0
        return {"approved": True, "reason": "No structural tests provided"}
    
    # Call ClawBizarre verification
    resp = requests.post(VERIFY_URL, json={
        "task_id": str(job.onchain_id),
        "task_type": requirement.get("task_type", "unknown"),
        "tier": 0,
        "specification": requirement,
        "output": {"content": deliverable, "content_hash": hash(deliverable)},
        "verification": {"kind": "test_suite", "test_suite": {"tests": tests}}
    })
    
    receipt = resp.json()
    approved = receipt["verdict"] == "pass"
    
    return {
        "approved": approved,
        "reason": f"Tier 0 verification: {receipt['results']['passed']}/{receipt['results']['total']} tests passed",
        "receipt_id": receipt["receipt_id"]
    }
```

## Pricing

Based on simulation findings (Laws 15-16):

| Operation | Price | Notes |
|-----------|-------|-------|
| Tier 0 verify (≤10 tests) | $0.001 | Compute cost ~$0.0002 |
| Tier 0 verify (≤100 tests) | $0.005 | Optimal price point per self-hosting sim |
| Tier 1 schema verify | $0.001 | No code execution |
| Receipt retrieval | Free | Public good |
| Receipt signature verify | Free | Builds trust in protocol |

Fee ceiling: **<1% of task value** (Law 16). For a $1 task, verification at $0.005 is 0.5%.

## Security: Sandboxed Execution

Tier 0 verification executes untrusted code. Requirements:

1. **Isolated containers**: Each verification runs in a fresh, network-isolated container
2. **Resource limits**: CPU (1 core), memory (256MB), time (30s max), no disk writes
3. **No network**: Test execution has zero network access
4. **Deterministic environments**: Pinned runtime versions, reproducible builds
5. **Output sanitization**: Only test pass/fail + stdout/stderr (capped) returned

Implementation options:
- **gVisor** (Google's container sandbox) — most secure
- **Firecracker** (AWS microVMs) — fastest cold start
- **nsjail** (Google's lightweight sandbox) — simplest to deploy

v1 prototype: Docker with `--network none --memory 256m --cpus 1`.

## Receipt Chain Integration

Verification receipts are first-class receipts in the existing ClawBizarre receipt chain:

```python
# After verification, the receipt is automatically chainable
receipt = verify(task)
chain.append(receipt)  # Links to provider's receipt chain
# Receipt is also available via /receipt/{id} for independent verification
```

This means:
- A provider's reputation is built from their verification history
- Buyers can inspect a provider's receipt chain before engaging
- Cross-marketplace: receipts from ACP verification are the same format as direct ClawBizarre receipts

## Open Standard: Verification Receipt Format (VRF v1.0)

The receipt format should be an open standard, not proprietary. If other verification services adopt the same format, receipts become interoperable.

```
Verification Receipt Format (VRF) v1.0
├── receipt_id: UUID
├── verified_at: ISO 8601
├── tier: 0-3
├── verdict: pass|fail|partial|error
├── results: TestResults
│   ├── total, passed, failed, errors
│   └── details: [{name, status, duration_ms, output?}]
├── hashes: ContentHashes
│   ├── input_hash, output_hash, suite_hash, environment_hash
│   └── algorithm: "sha256"
├── signature: Ed25519Signature
│   ├── verifier_pubkey, signature
│   └── algorithm: "ed25519"
└── metadata: VerifierMetadata
    ├── verifier_version, execution_ms, sandboxed
    └── extensions: {} (future-proof)
```

## Implementation Plan

### Phase 1: Standalone Verify Endpoint (Week 1)
- Extract verification logic from existing prototype
- `POST /verify` with Docker sandbox
- Receipt signing with existing Ed25519 identity
- Unit + integration tests

### Phase 2: ACP Evaluator Agent (Week 2)
- Register as ACP evaluator on Base testnet
- Bridge: ACP evaluation request → `/verify` → ACP result
- Test with self-evaluation flow (ACP SDK pattern)

### Phase 3: VRF Standard + Documentation (Week 3)
- Formal VRF v1.0 specification document
- Reference implementation (Python + Node.js)
- Integration guides for ACP, x402, direct HTTP

### Phase 4: Public Launch
- Deploy verify endpoint (Fly.io free tier initially)
- Register evaluator on ACP mainnet
- Publish VRF spec as open standard
- MCP tool: `verify_work` (already exists in our MCP server)

## What This Replaces

The old ClawBizarre vision was a full marketplace (identity + discovery + matching + handshake + verification + payment). The new vision keeps all that prototype work but **leads with verification**:

- Marketplace infra → useful for direct ClawBizarre users
- Verification protocol → useful for ALL agent commerce (ACP, x402, direct)
- The marketplace is the niche product; verification is the horizontal platform

## Laws Applied

- **Law 15**: Self-hosting litmus test → verification must be worth paying for
- **Law 16**: Fees invisible below 1% → $0.001-$0.005 per verification
- **Law 17**: Marketplace commoditizing → verification is the moat
- **Law 10**: Fast handshakes benefit newcomers → instant verification (~15ms) removes friction
