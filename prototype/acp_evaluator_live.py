#!/usr/bin/env python3
"""
ClawBizarre ACP Evaluator — Live Integration

Plugs ClawBizarre's structural verification into Virtuals ACP's evaluation phase.
Instead of LLM-subjective "looks good", runs actual test suites against deliverables.

Requirements:
  pip install virtuals-acp

Environment variables:
  WHITELISTED_WALLET_PRIVATE_KEY  — Dev wallet private key
  EVALUATOR_AGENT_WALLET_ADDRESS  — ClawBizarre evaluator wallet on Base
  EVALUATOR_ENTITY_ID             — Registered entity ID on Virtuals
  VERIFY_SERVER_URL               — ClawBizarre verify_server endpoint (default: http://localhost:8900)

Usage:
  python3 acp_evaluator_live.py                    # Connect to ACP, listen for eval jobs
  python3 acp_evaluator_live.py --test             # Run local integration test
"""

import json
import logging
import os
import sys
import threading
import time
import urllib.request
import urllib.error
from typing import Optional, Dict, Any

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("ClawBizarre-Evaluator")

VERIFY_SERVER_URL = os.environ.get("VERIFY_SERVER_URL", "http://localhost:8900")


def verify_deliverable(deliverable: str, test_suite: list, language: str = "python",
                       timeout: int = 30) -> Dict[str, Any]:
    """
    Send deliverable to ClawBizarre verify_server for structural verification.
    
    Returns dict with: decision (approve/reject), receipt, details
    """
    payload = {
        "code": deliverable,
        "verification": {
            "test_suite": {
                "tests": test_suite
            },
            "language": language,
            "timeout_seconds": timeout,
        }
    }
    
    req = urllib.request.Request(
        f"{VERIFY_SERVER_URL}/verify",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    try:
        with urllib.request.urlopen(req, timeout=timeout + 5) as resp:
            result = json.loads(resp.read())
            
            receipt = result.get("receipt", {})
            passed = receipt.get("tests_passed", 0)
            total = receipt.get("tests_total", 0)
            all_passed = passed == total and total > 0
            
            return {
                "decision": "approve" if all_passed else "reject",
                "passed": passed,
                "total": total,
                "receipt": receipt,
                "reason": f"Structural verification: {passed}/{total} tests passed"
                          + (" — all tests pass, deliverable approved."
                             if all_passed else
                             f" — {total - passed} test(s) failed.")
            }
    except urllib.error.URLError as e:
        logger.error(f"verify_server unreachable: {e}")
        return {
            "decision": "reject",
            "passed": 0,
            "total": 0,
            "receipt": None,
            "reason": f"Verification service unavailable: {e}"
        }
    except Exception as e:
        logger.error(f"Verification error: {e}")
        return {
            "decision": "reject",
            "passed": 0,
            "total": 0,
            "receipt": None,
            "reason": f"Verification error: {e}"
        }


def parse_job_for_verification(job) -> Optional[Dict[str, Any]]:
    """
    Extract deliverable and test suite from ACP job memos.
    
    Expected memo structure (in the service_requirement or deliverable memo):
    {
        "content": "<code>",
        "clawbizarre": {
            "test_suite": [
                {"input": "func(1)", "expected": "1"},
                {"input": "func(5)", "expected": "120"}
            ],
            "language": "python"
        }
    }
    
    If no clawbizarre field, falls back to LLM-free heuristic evaluation.
    """
    memos = getattr(job, 'memos', []) or []
    
    deliverable_content = None
    test_suite = None
    language = "python"
    
    for memo in memos:
        memo_data = memo if isinstance(memo, dict) else getattr(memo, '__dict__', {})
        content = memo_data.get("content", "")
        
        # Look for deliverable (last non-empty content)
        if content:
            deliverable_content = content
        
        # Look for clawbizarre verification spec
        cb = memo_data.get("clawbizarre", {})
        if cb:
            test_suite = cb.get("test_suite", test_suite)
            language = cb.get("language", language)
    
    # Also check the original service requirement for test specs
    # (buyer may embed test suite in the job request)
    requirement = getattr(job, 'service_requirement', None)
    if requirement and isinstance(requirement, dict):
        cb = requirement.get("clawbizarre", {})
        if cb:
            test_suite = cb.get("test_suite", test_suite)
            language = cb.get("language", language)
    
    if not deliverable_content:
        return None
    
    if not test_suite:
        # No test suite provided — can't do structural verification
        # Return None to signal fallback to simple approval
        logger.warning(f"Job {getattr(job, 'id', '?')}: No test suite in memos, cannot verify structurally")
        return None
    
    return {
        "deliverable": deliverable_content,
        "test_suite": test_suite,
        "language": language,
    }


def run_acp_evaluator():
    """Main entry point: connect to ACP and listen for evaluation jobs."""
    try:
        from virtuals_acp.client import VirtualsACP
        from virtuals_acp.contract_clients.contract_client_v2 import ACPContractClientV2
        from virtuals_acp.env import EnvSettings
    except ImportError:
        logger.error("virtuals-acp not installed. Run: pip install virtuals-acp")
        sys.exit(1)
    
    env = EnvSettings()
    
    def on_evaluate(job):
        job_id = getattr(job, 'id', '?')
        logger.info(f"[eval] Job {job_id}: evaluation requested")
        
        # Try to extract verification parameters
        params = parse_job_for_verification(job)
        
        if params is None:
            # No test suite available — approve with disclaimer
            # (ACP v2 made eval optional; if someone chose us as evaluator
            # but didn't provide tests, we approve with a note)
            logger.info(f"[eval] Job {job_id}: no test suite, approving with disclaimer")
            try:
                job.evaluate(
                    accept=True,
                    reason="ClawBizarre: No test suite provided for structural verification. "
                           "Approved by default. For deterministic evaluation, include "
                           "clawbizarre.test_suite in your service requirement."
                )
            except Exception as e:
                logger.error(f"[eval] Job {job_id}: evaluate call failed: {e}")
            return
        
        # Run structural verification
        logger.info(f"[eval] Job {job_id}: running structural verification "
                     f"({len(params['test_suite'])} tests, {params['language']})")
        
        result = verify_deliverable(
            deliverable=params["deliverable"],
            test_suite=params["test_suite"],
            language=params["language"],
        )
        
        accept = result["decision"] == "approve"
        reason = f"ClawBizarre structural verification: {result['reason']}"
        
        logger.info(f"[eval] Job {job_id}: {'APPROVE' if accept else 'REJECT'} — {result['reason']}")
        
        try:
            job.evaluate(accept=accept, reason=reason)
            logger.info(f"[eval] Job {job_id}: evaluation submitted successfully")
        except Exception as e:
            logger.error(f"[eval] Job {job_id}: evaluate call failed: {e}")
    
    logger.info("Starting ClawBizarre ACP Evaluator...")
    logger.info(f"Verify server: {VERIFY_SERVER_URL}")
    
    VirtualsACP(
        acp_contract_clients=ACPContractClientV2(
            wallet_private_key=env.WHITELISTED_WALLET_PRIVATE_KEY,
            agent_wallet_address=env.EVALUATOR_AGENT_WALLET_ADDRESS,
            entity_id=env.EVALUATOR_ENTITY_ID,
        ),
        on_evaluate=on_evaluate,
    )
    
    logger.info("ClawBizarre Evaluator listening for jobs...")
    threading.Event().wait()


def run_local_test():
    """Test against local verify_server without ACP."""
    print("=== ClawBizarre ACP Evaluator — Local Test ===\n")
    
    # Test 1: Good code should pass
    good_code = "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return b"
    
    tests = [
        {"input": "fibonacci(0)", "expected": "0"},
        {"input": "fibonacci(1)", "expected": "1"},
        {"input": "fibonacci(10)", "expected": "55"},
        {"input": "fibonacci(20)", "expected": "6765"},
    ]
    
    print("Test 1: Good fibonacci code")
    result = verify_deliverable(good_code, tests)
    print(f"  Decision: {result['decision']}")
    print(f"  Reason: {result['reason']}")
    assert result["decision"] == "approve", f"Expected approve, got {result['decision']}"
    print("  ✅ PASS\n")
    
    # Test 2: Buggy code should fail
    buggy_code = "def fibonacci(n):\n    if n <= 1: return n\n    a, b = 0, 1\n    for _ in range(2, n+1):\n        a, b = b, a+b\n    return a  # BUG: returns a instead of b"
    
    print("Test 2: Buggy fibonacci code")
    result = verify_deliverable(buggy_code, tests)
    print(f"  Decision: {result['decision']}")
    print(f"  Reason: {result['reason']}")
    assert result["decision"] == "reject", f"Expected reject, got {result['decision']}"
    print("  ✅ PASS\n")
    
    # Test 3: Verify server down
    print("Test 3: Unreachable verify server")
    old_url = globals().get("VERIFY_SERVER_URL")
    import acp_evaluator_live
    acp_evaluator_live.VERIFY_SERVER_URL = "http://localhost:19999"
    result = verify_deliverable(good_code, tests)
    acp_evaluator_live.VERIFY_SERVER_URL = old_url or VERIFY_SERVER_URL
    print(f"  Decision: {result['decision']}")
    assert result["decision"] == "reject", "Should reject when server unreachable"
    print("  ✅ PASS\n")
    
    print("=== All local tests passed ===")


if __name__ == "__main__":
    if "--test" in sys.argv:
        run_local_test()
    else:
        run_acp_evaluator()
