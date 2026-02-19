"""
End-to-end settlement test: Match → Handshake → Execute → Verify → Settle
Tests the full pipeline through api_server_v5 with persistence.
"""

import json
import time
import tempfile
import os
import sys
import threading
import urllib.request
import urllib.error

# Ensure prototype dir is on path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from identity import AgentIdentity

PORT = 45910
BASE = f"http://localhost:{PORT}"
passed = 0
failed = 0


def req(method, path, body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body else None
    r = urllib.request.Request(url, data=data, method=method)
    r.add_header("Content-Type", "application/json")
    if token:
        r.add_header("Authorization", f"Bearer {token}")
    try:
        resp = urllib.request.urlopen(r)
        return json.loads(resp.read()), resp.status
    except urllib.error.HTTPError as e:
        return json.loads(e.read()), e.code


def auth(identity):
    resp, _ = req("POST", "/auth/challenge", {"agent_id": identity.agent_id})
    sig = identity.sign(resp["challenge"])
    resp2, _ = req("POST", "/auth/verify", {
        "challenge_id": resp["challenge_id"],
        "agent_id": identity.agent_id,
        "signature": sig,
        "pubkey": identity.public_key_hex,
    })
    return resp2["token"]


def check(label, condition):
    global passed, failed
    if condition:
        print(f"  ✓ {label}")
        passed += 1
    else:
        print(f"  ✗ {label}")
        failed += 1


if __name__ == "__main__":
    # Start server in background with temp db
    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    os.environ["CLAWBIZARRE_DB"] = db_path

    # Import and start server
    from api_server_v5 import PersistentState, APIv5Handler
    from http.server import ThreadingHTTPServer

    state = PersistentState(db_path=db_path)
    APIv5Handler.state = state

    server = ThreadingHTTPServer(("", PORT), APIv5Handler)
    t = threading.Thread(target=server.serve_forever, daemon=True)
    t.start()
    time.sleep(0.3)

    print(f"\n=== Settlement E2E Test (port {PORT}) ===\n")

    # Create buyer and provider
    buyer_id = AgentIdentity.generate()
    provider_id = AgentIdentity.generate()
    buyer_token = auth(buyer_id)
    provider_token = auth(provider_id)
    check("1. Buyer authenticated", buyer_token)
    check("2. Provider authenticated", provider_token)

    # Provider lists service
    resp, code = req("POST", "/discovery/register", {
        "capabilities": ["code_review"]
    }, provider_token)
    check("3. Provider registered", code == 200)

    resp, code = req("POST", "/matching/listing", {
        "capabilities": ["code_review"],
        "price_per_task": 0.5,
        "response_time_avg": 30.0,
    }, provider_token)
    check("4. Provider listed service", code in (200, 201))

    # Buyer initiates handshake
    resp, code = req("POST", "/handshake/initiate", {
        "provider_id": provider_id.agent_id,
        "capabilities": ["code_review"],
        "proposal": {
            "task_description": "Review my PR",
            "task_type": "code_review",
            "verification_tier": 0,
        },
    }, buyer_token)
    session_id = resp.get("session_id")
    check("5. Handshake initiated", session_id is not None)

    # Provider accepts
    resp, code = req("POST", "/handshake/respond", {
        "session_id": session_id,
        "action": "accept",
    }, provider_token)
    check("6. Provider accepted", code == 200)

    # Provider executes
    resp, code = req("POST", "/handshake/execute", {
        "session_id": session_id,
        "output": "LGTM, no issues found",
        "proof": "all tests pass",
    }, provider_token)
    check("7. Provider executed", code == 200)

    # Buyer verifies
    resp, code = req("POST", "/handshake/verify", {
        "session_id": session_id,
        "passed": 5,
        "failed": 0,
        "suite_hash": "sha256:abc123",
    }, buyer_token)
    receipt_id = resp.get("receipt_id")
    check("8. Buyer verified (receipt generated)", receipt_id is not None)

    # === SETTLEMENT PHASE ===

    # Buyer registers payment
    resp, code = req("POST", "/settlement/register", {
        "receipt_id": receipt_id,
        "protocol": "x402",
        "payment_id": "x402_pay_001",
        "amount": 0.5,
        "currency": "USDC",
        "chain": "base",
    }, buyer_token)
    check("9. Settlement registered", code == 201)
    check("10. Settlement status=pending", resp.get("settlement", {}).get("status") == "pending")

    # Check settlement status
    resp, code = req("GET", f"/settlement/{receipt_id}", token=buyer_token)
    check("11. Settlement retrievable", code == 200)
    check("12. Protocol=x402", resp.get("settlement", {}).get("protocol") == "x402")

    # Provider confirms payment received
    resp, code = req("POST", "/settlement/confirm", {
        "receipt_id": receipt_id,
    }, provider_token)
    check("13. Settlement confirmed", code == 200)
    check("14. Status=confirmed", resp.get("settlement", {}).get("status") == "confirmed")

    # Verify confirmed status persists
    resp, code = req("GET", f"/settlement/{receipt_id}", token=buyer_token)
    check("15. Confirmed status persisted", resp.get("settlement", {}).get("status") == "confirmed")

    # Double-confirm should fail
    resp, code = req("POST", "/settlement/confirm", {
        "receipt_id": receipt_id,
    }, provider_token)
    check("16. Double-confirm rejected (409)", code == 409)

    # Unknown receipt
    resp, code = req("GET", "/settlement/nonexistent", token=buyer_token)
    check("17. Unknown receipt returns 404", code == 404)

    # Check stats include settlements
    resp, code = req("GET", "/matching/stats", token=buyer_token)
    # Stats are from matching engine, not global — just verify API works
    check("18. Stats endpoint works", code == 200)

    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*50}")

    server.shutdown()
    import shutil
    shutil.rmtree(tmpdir)

    sys.exit(1 if failed else 0)
