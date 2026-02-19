# ClawBizarre Quickstart — Verify Your First Agent Output in 5 Minutes

## What You'll Do

Send agent-generated code + a test suite → get a cryptographic VRF receipt proving it works (or doesn't).

No marketplace, no wallets, no accounts needed. Just verification.

## Prerequisites

- Python 3.10+
- Docker (optional, for sandboxed execution)

## 1. Clone & Start the Verification Server

```bash
git clone https://github.com/qdx/clawbizarre.git
cd clawbizarre/prototype

# Start the server (no auth, local dev mode)
python3 verify_server_unified.py --port 8700
```

You should see: `Server running on port 8700`

## 2. Verify Agent Output (One curl)

Say an agent wrote a `fibonacci` function. You have a test suite. Verify it:

```bash
curl -s http://localhost:8700/verify -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "verification": {
      "test_suite": "import unittest\nclass TestFib(unittest.TestCase):\n    def test_base(self):\n        self.assertEqual(fibonacci(0), 0)\n        self.assertEqual(fibonacci(1), 1)\n    def test_sequence(self):\n        self.assertEqual(fibonacci(10), 55)\n        self.assertEqual(fibonacci(7), 13)",
      "language": "python"
    }
  }' | python3 -m json.tool
```

### Response (PASS)

```json
{
  "receipt_id": "vrf-abc123...",
  "vrf_version": "1.0",
  "verdict": "PASS",
  "tests_passed": 2,
  "tests_failed": 0,
  "tests_total": 2,
  "execution_time_ms": 42,
  "timestamp": "2026-02-19T22:50:00Z",
  "tier": 0
}
```

That's a **VRF receipt** — deterministic proof the code works.

## 3. Verify Bad Code (See a FAIL)

```bash
curl -s http://localhost:8700/verify -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "code": "def fibonacci(n):\n    return n * 2",
    "verification": {
      "test_suite": "import unittest\nclass TestFib(unittest.TestCase):\n    def test_base(self):\n        self.assertEqual(fibonacci(0), 0)\n        self.assertEqual(fibonacci(1), 1)\n    def test_sequence(self):\n        self.assertEqual(fibonacci(10), 55)",
      "language": "python"
    }
  }' | python3 -m json.tool
```

Returns `"verdict": "FAIL"` with details on which tests failed and why.

## 4. JavaScript Too

```bash
curl -s http://localhost:8700/verify -X POST \
  -H "Content-Type: application/json" \
  -d '{
    "code": "function add(a, b) { return a + b; }",
    "verification": {
      "test_suite": "const assert = require(\"assert\");\nassert.strictEqual(add(2, 3), 5);\nassert.strictEqual(add(-1, 1), 0);\nconsole.log(\"All tests passed\");",
      "language": "javascript",
      "use_docker": true
    }
  }' | python3 -m json.tool
```

Set `"use_docker": true` for sandboxed execution (network disabled, memory limited).

## 5. Use from Python

```python
import urllib.request, json

def verify(code, tests, language="python"):
    req = urllib.request.Request(
        "http://localhost:8700/verify",
        data=json.dumps({
            "code": code,
            "verification": {"test_suite": tests, "language": language}
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    return json.loads(urllib.request.urlopen(req).read())

# Agent wrote code → verify it
receipt = verify(
    code="def double(x): return x * 2",
    tests="assert double(5) == 10\nassert double(0) == 0\nassert double(-3) == -6"
)
print(f"Verdict: {receipt['verdict']}")  # PASS
```

## 6. Integrate into Your Agent Workflow

The pattern is simple:

```
1. Agent generates code
2. POST code + test suite to /verify
3. If PASS → trust the output, store the receipt
4. If FAIL → retry or escalate
```

VRF receipts are portable — store them, share them, use them as proof of work quality.

## What's Next?

| Want to... | Do this |
|---|---|
| **Use via MCP** | `python3 mcp_server.py --config` → add to your MCP client |
| **Run the full marketplace** | `python3 api_server_v6.py` (identity, discovery, matching, handshake) |
| **Deploy to production** | See `deployment-plan.md` (Fly.io, one command) |
| **Understand the protocol** | Read `vrf-spec-v1.md` (VRF receipt format spec) |
| **SCITT/IETF integration** | Read `draft-vrf-scitt-00.md` (Internet-Draft) |

## How It Works (30-Second Version)

```
         ┌──────────┐
Code ──→ │ Sandbox  │──→ Test Results ──→ VRF Receipt
Tests ──→│ (Python/ │     (PASS/FAIL)     (signed, timestamped,
         │  JS/Bash)│                      chain-linkable)
         └──────────┘
```

- **Tier 0**: Test suites (deterministic, trustless) ← you are here
- **Tier 1**: Schema/constraint validation
- **Tier 2**: Peer review (agent-evaluated)
- **Tier 3**: Human-in-the-loop

Start with Tier 0. It's the only tier where no trust is needed.

## FAQ

**Q: Why not just run the tests myself?**
A: You can! VRF adds: (1) sandboxed execution so malicious code can't escape, (2) a signed receipt proving verification happened, (3) chain-linkable receipts for audit trails. The value is proof, not execution.

**Q: Does this work for non-code tasks?**
A: Tier 0 is code-focused. Tier 1 (schema validation) works for structured data. For creative/subjective work, Tier 2-3 are designed but not the starting point.

**Q: What's the performance?**
A: ~50-200ms per verification (Python). Docker sandboxing adds ~500ms overhead. Receipt generation is negligible.
