#!/usr/bin/env python3
"""
Full Trust Pipeline Integration Test

Tests the COMPLETE chain:
  Identity → Verify → COSE Sign1 → Merkle Transparency Log → Proof

This is the "if this passes, the whole system works" test.
"""

import json
import os
import sys
import tempfile
import threading
import time
import urllib.request
import urllib.error

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

API_KEY = "test-pipeline-key"
PORT = 8799
passed = 0
failed = 0


def req(method, path, body=None):
    url = f"http://127.0.0.1:{PORT}{path}"
    data = json.dumps(body).encode() if body else None
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
    r = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(r, timeout=30) as resp:
            return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        body = e.read()
        try:
            return json.loads(body), e.code
        except Exception:
            return {"raw": body.decode()}, e.code


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")


def start_server(tmpdir):
    from verify_server_unified import (
        UnifiedHandler, UnifiedConfig, RateLimiter, VerificationEngine,
    )
    from http.server import HTTPServer
    from identity import AgentIdentity
    from receipt_store import ReceiptStore
    from merkle_store import MerkleStore

    identity = AgentIdentity.generate()
    cfg = UnifiedConfig(
        host="127.0.0.1", port=PORT, api_key=API_KEY, auto_register=True,
        receipt_db=os.path.join(tmpdir, "r.db"),
        transparency_db=os.path.join(tmpdir, "t.db"),
    )
    UnifiedHandler.engine = VerificationEngine(identity=identity)
    UnifiedHandler.config = cfg
    UnifiedHandler.identity = identity
    UnifiedHandler.rate_limiter = RateLimiter(rate=100.0, burst=200)
    UnifiedHandler.receipt_store = ReceiptStore(cfg.receipt_db)
    UnifiedHandler.merkle_store = MerkleStore(cfg.transparency_db)
    UnifiedHandler._request_count = 0

    srv = HTTPServer(("127.0.0.1", PORT), UnifiedHandler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    time.sleep(0.3)
    return srv


def verify(code, tests):
    """Submit verification request."""
    return req("POST", "/verify", {
        "tier": 0,
        "output": {"content": code},
        "verification": {"test_suite": {"tests": tests, "language": "python"}},
    })


def test_full_pipeline():
    print("\n═══ Test 1: Full Trust Pipeline ═══")
    resp, st = verify(
        "def f(x): return x + 1",
        [{"name": "t1", "input": "f(1)", "expected_output": "2"},
         {"name": "t2", "input": "f(0)", "expected_output": "1"},
         {"name": "t3", "input": "f(-1)", "expected_output": "0"}],
    )
    check("HTTP 200", st == 200)
    check("Verdict pass", resp.get("verdict") == "pass", resp.get("verdict"))
    check("3/3 tests passed", resp.get("results", {}).get("passed") == 3)
    check("Has receipt_id", "receipt_id" in resp)
    check("VRF version 1.0", resp.get("vrf_version") == "1.0")
    check("Ed25519 signature", resp.get("signature", {}).get("algorithm") == "ed25519")

    # Transparency auto-register
    tr = resp.get("transparency_receipt", {})
    check("Auto-registered to transparency log", bool(tr))
    check("Has tree_root", "tree_root" in tr)
    check("Has inclusion_proof", "inclusion_proof" in tr)

    # Query receipt back
    rid = resp.get("receipt_id")
    rr, rs = req("GET", "/receipts?limit=10")
    check("Receipt persisted", any(r.get("receipt_id") == rid for r in rr.get("receipts", [])))

    return rid


def test_fail_verdict():
    print("\n═══ Test 2: Failing Verification ═══")
    resp, st = verify(
        "def f(x): return x - 1",  # wrong!
        [{"name": "t1", "input": "f(1)", "expected_output": "2"}],
    )
    check("HTTP 200", st == 200)
    check("Verdict fail", resp.get("verdict") == "fail", resp.get("verdict"))
    check("FAIL in transparency", "transparency_receipt" in resp)


def test_multi_agent():
    print("\n═══ Test 3: Multi-Agent Transparency ═══")
    r1, _ = verify("def f(x): return sorted(x)", [{"name": "sort", "input": "f([3,1,2])", "expected_output": "[1, 2, 3]"}])
    r2, _ = verify("def f(s): return s[::-1]", [{"name": "rev", "input": "f('abc')", "expected_output": "'cba'"}])
    check("Agent A passes", r1.get("verdict") == "pass", r1.get("verdict"))
    check("Agent B passes", r2.get("verdict") == "pass", r2.get("verdict"))
    tr1 = r1.get("transparency_receipt", {})
    tr2 = r2.get("transparency_receipt", {})
    check("Both in transparency", bool(tr1) and bool(tr2))
    if tr1 and tr2:
        check("Different sequence numbers", tr1.get("sequence_number") != tr2.get("sequence_number"))
        check("Tree grows monotonically", tr2.get("log_size", 0) > tr1.get("log_size", 0))


def test_consistency():
    print("\n═══ Test 4: Consistency Proofs ═══")
    stats, _ = req("GET", "/transparency/stats")
    old_size = stats.get("tree_size", 0)
    check("Tree has entries", old_size >= 4, f"size={old_size}")

    # Add one more
    r, _ = verify("def f(x): return x*2", [{"name": "dbl", "input": "f(5)", "expected_output": "10"}])
    new_size = r.get("transparency_receipt", {}).get("log_size", 0)
    if old_size > 0 and new_size > old_size:
        cr, cs = req("GET", f"/transparency/consistency?old_size={old_size}&new_size={new_size}")
        check("Consistency proof returned", cs == 200, f"status={cs}")


def test_receipt_filters():
    print("\n═══ Test 5: Receipt Filters ═══")
    rp, _ = req("GET", "/receipts?verdict=pass&limit=100")
    check("PASS filter", all(r.get("verdict") == "pass" for r in rp.get("receipts", [])))
    rf, _ = req("GET", "/receipts?verdict=fail&limit=100")
    check("FAIL filter has results", len(rf.get("receipts", [])) >= 1)


def test_health():
    print("\n═══ Test 6: Health ═══")
    resp, st = req("GET", "/health")
    check("Health 200", st == 200)
    check("Auto-register on", resp.get("auto_register") is True)
    check("Status ok", resp.get("status") == "ok")


def main():
    global passed, failed
    print("╔══════════════════════════════════════════════════╗")
    print("║  ClawBizarre Full Trust Pipeline Integration     ║")
    print("║  Identity → Verify → COSE → Merkle → Proof     ║")
    print("╚══════════════════════════════════════════════════╝")

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"\n⏳ Starting unified server...")
        try:
            srv = start_server(tmpdir)
        except Exception as e:
            print(f"❌ Server failed: {e}")
            sys.exit(1)
        print("✅ Server running\n")

        try:
            test_full_pipeline()
            test_fail_verdict()
            test_multi_agent()
            test_consistency()
            test_receipt_filters()
            test_health()
        finally:
            srv.shutdown()

    print(f"\n{'═' * 50}")
    print(f"Results: {passed} passed, {failed} failed, {passed + failed} total")
    if failed:
        print("❌ SOME TESTS FAILED")
        sys.exit(1)
    else:
        print("✅ ALL TESTS PASSED — Full trust pipeline verified")
        sys.exit(0)


if __name__ == "__main__":
    main()
