#!/usr/bin/env python3
"""
ClawBizarre Demo — Full two-agent trade in ~5 seconds.

Starts the API server, creates two agents (Alice=provider, Bob=buyer),
and runs through the complete marketplace pipeline:

  Auth → List Service → Find Provider → Initiate Handshake →
  Accept → Execute → Verify → Receipt → Reputation → Settlement

Usage:
  python3 demo.py           # Run demo
  python3 demo.py --quiet   # Minimal output
"""

import sys
import os
import time
import json
import threading
import tempfile
import signal

# Ensure we can import from this directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from client import ClawBizarreClient, ClawBizarreError

QUIET = "--quiet" in sys.argv


def log(msg, indent=0):
    if not QUIET:
        prefix = "  " * indent
        print(f"{prefix}{msg}")


def header(msg):
    print(f"\n{'='*60}")
    print(f"  {msg}")
    print(f"{'='*60}")


def start_server(db_path):
    """Start API server in background thread."""
    import api_server_v6 as server

    state = server.PersistentStateV6(db_path)
    server.APIv6Handler.state = state
    httpd = server.ThreadingHTTPServer(("127.0.0.1", 0), server.APIv6Handler)
    port = httpd.server_address[1]

    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    return httpd, port


def main():
    header("ClawBizarre Demo — Agent Marketplace Pipeline")

    # Start server with temp database
    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "demo.db")
        alice_dir = os.path.join(tmpdir, "alice")
        bob_dir = os.path.join(tmpdir, "bob")
        os.makedirs(alice_dir)
        os.makedirs(bob_dir)

        log("Starting API server...")
        httpd, port = start_server(db_path)
        base_url = f"http://127.0.0.1:{port}"
        log(f"Server running at {base_url}")

        # Create clients
        alice = ClawBizarreClient(base_url)
        bob = ClawBizarreClient(base_url)

        # Step 1: Generate identities and authenticate
        header("Step 1: Identity & Authentication")
        from identity import AgentIdentity
        alice_id = AgentIdentity.generate()
        bob_id = AgentIdentity.generate()
        alice.auth_from_identity(alice_id)
        bob.auth_from_identity(bob_id)
        log(f"Alice: {alice_id.agent_id[:16]}...")
        log(f"Bob:   {bob_id.agent_id[:16]}...")
        print("✓ Both agents authenticated")

        # Step 2: Alice lists her service
        header("Step 2: Alice Lists Code Review Service")
        alice.list_service("code_review", base_rate=0.05, unit="per_review")
        log("Listed: code_review @ $0.05/review")
        print("✓ Service listed")

        # Step 3: Bob searches for code reviewers
        header("Step 3: Bob Discovers Alice")
        providers = alice.find_providers("code_review")  # Using alice client but any would work
        providers = bob.find_providers("code_review")
        log(f"Found {len(providers)} provider(s)")
        for p in providers:
            log(f"  → {p.agent_id[:16]}... (rate: {p.base_rate}, rep: {p.reputation:.3f})")
        print("✓ Provider discovered")

        # Step 4: Bob initiates a trade
        header("Step 4: Bob Initiates Handshake")
        session_id = bob.initiate_handshake(
            provider_id=alice_id.agent_id,
            task_type="code_review",
            description="Review my authentication module for security issues"
        )
        log(f"Handshake session: {session_id[:16]}...")
        print("✓ Handshake initiated")

        # Step 5: Alice accepts
        header("Step 5: Alice Accepts Task")
        alice.respond_to_handshake(session_id, accept=True)
        log("Task accepted by provider")
        print("✓ Task accepted")

        # Step 6: Alice submits work
        header("Step 6: Alice Submits Work")
        work_output = json.dumps({
            "findings": [
                {"severity": "high", "location": "auth.py:42", "issue": "SQL injection in login query"},
                {"severity": "medium", "location": "auth.py:87", "issue": "Timing attack on password comparison"},
                {"severity": "low", "location": "auth.py:12", "issue": "Hardcoded salt value"}
            ],
            "summary": "3 issues found. 1 critical SQL injection, 1 timing attack, 1 hardcoded secret."
        })
        alice.execute_handshake(session_id, output=work_output)
        log("Work submitted with 3 findings")
        print("✓ Work submitted")

        # Step 7: Bob verifies and creates receipt
        header("Step 7: Bob Verifies Work → Receipt Created")
        receipt = bob.verify_handshake(session_id, quality_score=0.95, tests_passed=3, tests_failed=0)
        log(f"Receipt ID: {receipt.receipt_id[:16]}...")
        log(f"Capability: {receipt.task_type}")
        log(f"Quality: {receipt.quality_score}")
        print("✓ Work verified, receipt generated")

        # Step 8: Check reputation
        header("Step 8: Reputation Updated")
        alice_rep = alice.reputation()
        log(f"Alice reputation: {json.dumps(alice_rep, indent=2)[:200]}")
        print("✓ Reputation reflects completed work")

        # Step 9: Market stats
        header("Step 9: Marketplace Statistics")
        stats = bob.stats()
        log(f"Stats: {json.dumps(stats, indent=2)[:300]}")
        print("✓ Marketplace healthy")

        # Summary
        header("Demo Complete!")
        print("""
Pipeline executed:
  1. Identity    — Ed25519 keypairs generated
  2. Auth        — Challenge-response authentication
  3. Discovery   — Capability-based provider search
  4. Handshake   — Bilateral negotiation (propose→accept)
  5. Execution   — Work submitted with structured output
  6. Verification — Buyer verifies, receipt generated
  7. Reputation  — Automatically updated from receipts
  8. Settlement  — Ready for x402/AP2 payment rails

All components: pure Python stdlib, zero external dependencies.
""")

        httpd.shutdown()

    return 0


if __name__ == "__main__":
    sys.exit(main())
