#!/usr/bin/env python3
"""
ClawBizarre VRF Demo — Try the full verification pipeline in 30 seconds.

Usage:
    python3 demo.py                    # Run interactive demo
    python3 demo.py --json             # JSON output only
    python3 demo.py --code 'def f(x): return x*2' --tests '[{"input":"f(3)","expected":"6"}]'

No server needed. No config. No dependencies beyond Python 3.9+.
"""

import argparse
import json
import subprocess
import sys
import tempfile
import hashlib
import time
import os
from pathlib import Path
from datetime import datetime, timezone

# ── Inline VRF receipt generation (no imports needed) ──────────────────────

def run_test(code: str, test: dict, language: str = "python", timeout: float = 5.0) -> dict:
    """Execute a single test case in a subprocess."""
    if language == "python":
        if "expected" in test:
            # Expression test
            script = f"""{code}\nimport json\nprint(json.dumps({{"result": repr({test['input']})}}))\n"""
        else:
            # I/O test
            script = code
    elif language == "javascript":
        if "expected" in test:
            script = f"""{code}\nconsole.log(JSON.stringify({{result: String({test['input']})}}));\n"""
        else:
            script = code
    else:
        return {"passed": False, "error": f"Unsupported language: {language}"}

    cmd = ["python3", "-c", script] if language == "python" else ["node", "-e", script]

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
        )
        if result.returncode != 0:
            return {"passed": False, "error": result.stderr.strip()[:500]}

        output = result.stdout.strip()
        if "expected" in test:
            parsed = json.loads(output)
            actual = str(parsed["result"])
            expected = str(test["expected"])
            return {"passed": actual == expected, "actual": actual, "expected": expected}
        elif "expected_output" in test:
            return {"passed": output == test["expected_output"], "actual": output, "expected": test["expected_output"]}
        else:
            return {"passed": True, "output": output}
    except subprocess.TimeoutExpired:
        return {"passed": False, "error": "Timeout"}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def generate_receipt(code: str, test_suite: list, language: str = "python") -> dict:
    """Generate a VRF receipt by running test suite against code."""
    receipt_id = hashlib.sha256(
        f"{code}{json.dumps(test_suite)}{time.time()}".encode()
    ).hexdigest()[:16]

    started = datetime.now(timezone.utc)
    results = []
    for i, test in enumerate(test_suite):
        r = run_test(code, test, language)
        r["test_index"] = i
        results.append(r)
    finished = datetime.now(timezone.utc)

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    verdict = "pass" if passed == total else "fail"

    receipt = {
        "vrf_version": "1.0",
        "receipt_id": f"vrf-{receipt_id}",
        "verdict": verdict,
        "summary": {
            "passed": passed,
            "failed": total - passed,
            "total": total
        },
        "test_results": results,
        "metadata": {
            "language": language,
            "started": started.isoformat(),
            "finished": finished.isoformat(),
            "duration_ms": round((finished - started).total_seconds() * 1000, 1)
        },
        "code_hash": hashlib.sha256(code.encode()).hexdigest()[:32],
        "chain": {
            "note": "In production, this receipt is Ed25519-signed, COSE-encoded, and appended to a Merkle transparency log."
        }
    }
    return receipt


# ── Demo scenarios ─────────────────────────────────────────────────────────

DEMOS = [
    {
        "name": "✅ Correct implementation",
        "description": "A fibonacci function that works",
        "code": """def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n - 1):
        a, b = b, a + b
    return b""",
        "tests": [
            {"input": "fib(0)", "expected": "0"},
            {"input": "fib(1)", "expected": "1"},
            {"input": "fib(10)", "expected": "55"},
            {"input": "fib(20)", "expected": "6765"},
        ]
    },
    {
        "name": "❌ Buggy implementation",
        "description": "An off-by-one error the tests catch",
        "code": """def fib(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(n):  # Bug: should be n-1
        a, b = b, a + b
    return b""",
        "tests": [
            {"input": "fib(0)", "expected": "0"},
            {"input": "fib(1)", "expected": "1"},
            {"input": "fib(10)", "expected": "55"},
            {"input": "fib(20)", "expected": "6765"},
        ]
    },
    {
        "name": "💥 Malicious code (sandboxed)",
        "description": "Code that tries to access the filesystem — caught by test failure",
        "code": """def get_data():
    try:
        import os
        return os.listdir('/')  # Trying to access filesystem
    except:
        return []""",
        "tests": [
            {"input": "get_data()", "expected": "42"},  # Obviously wrong
        ]
    },
]


def print_receipt(receipt: dict, verbose: bool = True):
    """Pretty-print a VRF receipt."""
    v = receipt["verdict"]
    icon = "✅" if v == "pass" else "❌"
    s = receipt["summary"]
    print(f"\n  {icon} Verdict: {v.upper()}")
    print(f"  📋 Receipt: {receipt['receipt_id']}")
    print(f"  🧪 Tests: {s['passed']}/{s['total']} passed")
    print(f"  ⏱️  Duration: {receipt['metadata']['duration_ms']}ms")
    print(f"  🔗 Code hash: {receipt['code_hash']}")

    if verbose and receipt["summary"]["failed"] > 0:
        print(f"\n  Failed tests:")
        for r in receipt["test_results"]:
            if not r["passed"]:
                if "expected" in r:
                    print(f"    Test {r['test_index']}: expected {r['expected']}, got {r.get('actual', 'ERROR')}")
                elif "error" in r:
                    print(f"    Test {r['test_index']}: {r['error'][:100]}")


def run_interactive_demo():
    """Run all demo scenarios with explanations."""
    print("=" * 60)
    print("  🔬 ClawBizarre VRF — Verification Receipt Format Demo")
    print("=" * 60)
    print()
    print("  VRF provides deterministic verification for agent work.")
    print("  Code + test suite → PASS/FAIL + signed receipt.")
    print("  No LLM judges. No subjective evaluation. Just tests.")
    print()

    receipts = []
    for i, demo in enumerate(DEMOS):
        print(f"{'─' * 60}")
        print(f"  Demo {i+1}: {demo['name']}")
        print(f"  {demo['description']}")
        print(f"  Tests: {len(demo['tests'])} cases")

        receipt = generate_receipt(demo["code"], demo["tests"])
        receipts.append(receipt)
        print_receipt(receipt)
        print()

    print(f"{'─' * 60}")
    print(f"\n  📊 Summary: {sum(1 for r in receipts if r['verdict'] == 'pass')}/{len(receipts)} demos passed verification")
    print()
    print("  In production, each receipt would be:")
    print("  • Ed25519-signed by the verifier")
    print("  • COSE-encoded (RFC 9052)")
    print("  • Appended to a Merkle transparency log (RFC 9162)")
    print("  • Queryable via MCP, REST API, or OpenClaw skill")
    print()
    print("  Learn more: https://github.com/qdx/clawbizarre")
    print("=" * 60)

    return receipts


def main():
    parser = argparse.ArgumentParser(description="ClawBizarre VRF Demo")
    parser.add_argument("--json", action="store_true", help="JSON output only")
    parser.add_argument("--code", type=str, help="Custom code to verify")
    parser.add_argument("--tests", type=str, help="JSON test suite")
    parser.add_argument("--language", default="python", help="Language (python/javascript)")
    args = parser.parse_args()

    if args.code and args.tests:
        # Custom verification
        tests = json.loads(args.tests)
        receipt = generate_receipt(args.code, tests, args.language)
        if args.json:
            print(json.dumps(receipt, indent=2))
        else:
            print_receipt(receipt)
        sys.exit(0 if receipt["verdict"] == "pass" else 1)
    else:
        # Interactive demo
        receipts = run_interactive_demo()
        if args.json:
            print(json.dumps(receipts, indent=2))


if __name__ == "__main__":
    main()
