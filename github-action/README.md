# VRF Verify Action

Verify agent-generated code with deterministic test suites in your CI/CD pipeline. Returns a signed [VRF receipt](../vrf-spec-v1.md).

## Quick Start

```yaml
- name: Verify agent output
  uses: clawbizarre/verify-action@v1
  with:
    code: |
      def add(a, b):
          return a + b
    test_suite: |
      [
        {"input": "add(2, 3)", "expected": "5"},
        {"input": "add(-1, 1)", "expected": "0"},
        {"input": "add(0, 0)", "expected": "0"}
      ]
```

## With Self-Hosted Verify Server

```yaml
services:
  verify:
    image: ghcr.io/clawbizarre/verify-server:latest
    ports:
      - 8880:8880

steps:
  - name: Verify
    uses: clawbizarre/verify-action@v1
    with:
      code: ${{ steps.agent.outputs.code }}
      test_suite: ${{ steps.agent.outputs.tests }}
      verify_url: http://verify:8880
```

## Inputs

| Input | Required | Default | Description |
|-------|----------|---------|-------------|
| `code` | ✅ | — | Code/output to verify |
| `test_suite` | ✅ | — | JSON array of test objects |
| `language` | ❌ | `python` | `python`, `javascript`, `bash` |
| `verify_url` | ❌ | `http://localhost:8880` | Verify server URL |
| `use_docker` | ❌ | `false` | Docker sandbox execution |
| `fail_on_reject` | ❌ | `true` | Fail action on FAIL verdict |

## Outputs

| Output | Description |
|--------|-------------|
| `verdict` | `PASS` or `FAIL` |
| `receipt` | Full VRF receipt JSON |
| `receipt_id` | Unique receipt identifier |
| `tests_passed` | Number passing |
| `tests_total` | Total tests |

## Use Cases

- **Verify agent-generated PRs** before merge
- **Gate deployments** on verified agent output
- **Build audit trails** of agent work quality
- **EU AI Act compliance** — deterministic output verification with receipts
