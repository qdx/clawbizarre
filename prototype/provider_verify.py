#!/usr/bin/env python3
"""
Provider-Side Verification Client

Use case: An ACP provider agent pre-verifies their deliverable before
submitting it to the buyer. This gives the provider a VRF receipt they
can attach to their delivery, signaling quality to the buyer.

Flow:
  1. Provider completes work (e.g., code generation)
  2. Provider calls ClawBizarre verification with code + test suite
  3. Gets back a signed VRF receipt (pass/fail, details)
  4. Attaches receipt to ACP deliver_job memo
  5. Buyer sees verified deliverable → higher trust

This module:
  - Standalone client (no ACP SDK dependency)
  - Works with verify_server HTTP API
  - Formats VRF receipts for ACP memo attachment
  - Handles the full pre-verification flow
"""

import json
import time
import hashlib
import urllib.request
import urllib.error
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class VerificationRequest:
    """Request to verify a deliverable before ACP submission."""
    code: str
    language: str = "python"
    test_suite: Optional[List[Dict[str, Any]]] = None
    constraints: Optional[Dict[str, Any]] = None
    use_docker: bool = False
    # ACP context (for receipt metadata)
    acp_job_id: Optional[str] = None
    provider_address: Optional[str] = None
    buyer_address: Optional[str] = None
    service_description: Optional[str] = None


@dataclass
class VerificationResult:
    """Result of pre-verification."""
    verified: bool
    tier: int  # 0=test suite, 1=schema/constraints
    passed: int = 0
    failed: int = 0
    total: int = 0
    details: List[Dict[str, Any]] = field(default_factory=list)
    receipt: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    verification_time_ms: int = 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total if self.total > 0 else 0.0

    def to_acp_memo(self) -> Dict[str, Any]:
        """Format as ACP-compatible memo attachment."""
        return {
            "clawbizarre_verification": {
                "version": "1.0",
                "verified": self.verified,
                "tier": self.tier,
                "pass_rate": self.pass_rate,
                "passed": self.passed,
                "failed": self.failed,
                "total": self.total,
                "verification_time_ms": self.verification_time_ms,
                "receipt_hash": hashlib.sha256(
                    json.dumps(self.receipt or {}, sort_keys=True).encode()
                ).hexdigest()[:16] if self.receipt else None,
                "timestamp": int(time.time())
            }
        }


class ProviderVerifyClient:
    """Client for provider-side pre-verification."""

    def __init__(self, verify_server_url: str = "http://localhost:9800"):
        self.base_url = verify_server_url.rstrip("/")

    def verify(self, request: VerificationRequest) -> VerificationResult:
        """Run pre-verification against verify_server."""
        start = time.time()

        # Build verify_server payload
        payload: Dict[str, Any] = {
            "code": request.code,
            "verification": {
                "tier": 0 if request.test_suite else 1,
            }
        }

        if request.test_suite:
            payload["verification"]["test_suite"] = {
                "language": request.language,
                "tests": request.test_suite
            }

        if request.constraints:
            payload["verification"]["constraints"] = request.constraints

        if request.use_docker:
            payload["verification"]["use_docker"] = True

        # Add ACP metadata
        if request.acp_job_id:
            payload["metadata"] = {
                "acp_job_id": request.acp_job_id,
                "provider": request.provider_address,
                "buyer": request.buyer_address,
                "service": request.service_description,
            }

        try:
            data = json.dumps(payload).encode()
            req = urllib.request.Request(
                f"{self.base_url}/verify",
                data=data,
                headers={"Content-Type": "application/json"},
                method="POST"
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read().decode())

            elapsed_ms = int((time.time() - start) * 1000)

            receipt = result.get("receipt", {})
            test_results = receipt.get("test_results", {})

            return VerificationResult(
                verified=receipt.get("verified", False),
                tier=receipt.get("verification_tier", 0),
                passed=test_results.get("passed", 0),
                failed=test_results.get("failed", 0),
                total=test_results.get("total", 0),
                details=test_results.get("details", []),
                receipt=receipt,
                verification_time_ms=elapsed_ms,
            )

        except urllib.error.URLError as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return VerificationResult(
                verified=False, tier=0,
                error=f"Connection error: {e}",
                verification_time_ms=elapsed_ms,
            )
        except Exception as e:
            elapsed_ms = int((time.time() - start) * 1000)
            return VerificationResult(
                verified=False, tier=0,
                error=f"Verification error: {e}",
                verification_time_ms=elapsed_ms,
            )

    def health(self) -> Dict[str, Any]:
        """Check verify_server health."""
        try:
            req = urllib.request.Request(f"{self.base_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read().decode())
        except Exception as e:
            return {"status": "error", "error": str(e)}


def format_acp_delivery_memo(
    deliverable: str,
    verification: VerificationResult,
    notes: str = ""
) -> Dict[str, Any]:
    """
    Format a complete ACP delivery memo with verification attachment.
    
    This is what the provider sends via acp.deliver_job(job_id, deliverable).
    The deliverable string includes both the work output and verification proof.
    """
    memo = {
        "deliverable": deliverable,
        "verification": verification.to_acp_memo(),
    }
    if notes:
        memo["notes"] = notes
    if verification.receipt:
        # Include full receipt for buyer to independently verify
        memo["vrf_receipt"] = verification.receipt
    return memo


# ── Demo / Self-test ──────────────────────────────────────────────

def _run_demo():
    """Demo: provider verifies code before ACP delivery."""
    print("=" * 60)
    print("Provider-Side Verification Demo")
    print("=" * 60)

    # Simulate: provider wrote a fibonacci function for an ACP job
    code = '''
def fibonacci(n):
    """Return nth Fibonacci number."""
    if n <= 0:
        return 0
    if n == 1:
        return 1
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
'''

    test_suite = [
        {"input": "fibonacci(0)", "expected": "0"},
        {"input": "fibonacci(1)", "expected": "1"},
        {"input": "fibonacci(5)", "expected": "5"},
        {"input": "fibonacci(10)", "expected": "55"},
        {"input": "fibonacci(20)", "expected": "6765"},
    ]

    request = VerificationRequest(
        code=code,
        language="python",
        test_suite=test_suite,
        acp_job_id="acp-job-12345",
        provider_address="0xProvider...",
        buyer_address="0xBuyer...",
        service_description="Fibonacci function implementation",
    )

    client = ProviderVerifyClient()

    # Check health
    health = client.health()
    print(f"\nVerify server health: {health.get('status', 'unknown')}")

    if health.get("status") != "ok":
        print("⚠️  Verify server not running — showing offline flow")
        print("\nIn production, provider would:")
        print("  1. Call client.verify(request)")
        print("  2. Get VerificationResult with pass/fail + receipt")
        print("  3. Format as ACP memo: format_acp_delivery_memo(code, result)")
        print("  4. Call acp.deliver_job(job_id, memo)")

        # Show what the ACP memo would look like
        mock_result = VerificationResult(
            verified=True, tier=0,
            passed=5, failed=0, total=5,
            verification_time_ms=150,
            receipt={"mock": True, "verified": True},
        )
        memo = format_acp_delivery_memo(code, mock_result,
                                         notes="Pre-verified via ClawBizarre")
        print(f"\nSample ACP delivery memo:")
        print(json.dumps(memo["verification"], indent=2))
        return

    # Run actual verification
    print("\nVerifying code...")
    result = client.verify(request)

    print(f"\nResult: {'✅ VERIFIED' if result.verified else '❌ FAILED'}")
    print(f"Tier: {result.tier}")
    print(f"Tests: {result.passed}/{result.total} passed")
    print(f"Time: {result.verification_time_ms}ms")

    if result.error:
        print(f"Error: {result.error}")

    # Format for ACP delivery
    memo = format_acp_delivery_memo(
        code, result,
        notes="Pre-verified via ClawBizarre structural verification"
    )
    print(f"\nACP delivery memo (verification section):")
    print(json.dumps(memo["verification"], indent=2))


# ── Tests ──────────────────────────────────────────────────────────

def _run_tests():
    """Unit tests (no server needed)."""
    import sys
    passed = 0
    failed = 0

    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            failed += 1
            print(f"  ❌ {name}")

    print("\n--- Unit Tests ---\n")

    # Test VerificationRequest
    req = VerificationRequest(code="x=1", language="python")
    check("VerificationRequest creation", req.code == "x=1")
    check("VerificationRequest defaults", req.test_suite is None)

    # Test VerificationResult
    res = VerificationResult(verified=True, tier=0, passed=5, failed=1, total=6)
    check("VerificationResult pass_rate", abs(res.pass_rate - 5/6) < 0.01)
    check("VerificationResult verified", res.verified is True)

    # Test to_acp_memo
    memo = res.to_acp_memo()
    check("to_acp_memo has version", memo["clawbizarre_verification"]["version"] == "1.0")
    check("to_acp_memo has pass_rate", memo["clawbizarre_verification"]["pass_rate"] == res.pass_rate)
    check("to_acp_memo has timestamp", "timestamp" in memo["clawbizarre_verification"])

    # Test zero-total edge case
    res0 = VerificationResult(verified=False, tier=1, passed=0, failed=0, total=0)
    check("Zero total pass_rate", res0.pass_rate == 0.0)

    # Test format_acp_delivery_memo
    full_memo = format_acp_delivery_memo("code here", res, notes="test")
    check("Full memo has deliverable", full_memo["deliverable"] == "code here")
    check("Full memo has verification", "verification" in full_memo)
    check("Full memo has notes", full_memo["notes"] == "test")

    # Test with receipt
    res_with_receipt = VerificationResult(
        verified=True, tier=0, passed=3, failed=0, total=3,
        receipt={"id": "abc", "verified": True}
    )
    memo_with = format_acp_delivery_memo("code", res_with_receipt)
    check("Memo includes vrf_receipt", "vrf_receipt" in memo_with)
    check("Receipt hash in memo", 
          memo_with["verification"]["clawbizarre_verification"]["receipt_hash"] is not None)

    # Test client creation (no server needed)
    client = ProviderVerifyClient("http://localhost:9999")
    check("Client creation", client.base_url == "http://localhost:9999")

    # Test offline health check
    h = client.health()
    check("Offline health returns error", h["status"] == "error")

    # Test offline verify
    req = VerificationRequest(code="x=1", test_suite=[{"input": "x", "expected": "1"}])
    res = client.verify(req)
    check("Offline verify returns not verified", res.verified is False)
    check("Offline verify has error", res.error is not None)

    print(f"\n--- Results: {passed}/{passed+failed} passed ---")
    return failed == 0


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        success = _run_tests()
        sys.exit(0 if success else 1)
    else:
        _run_demo()
