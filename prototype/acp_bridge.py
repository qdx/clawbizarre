"""
ClawBizarre ACP Evaluator Bridge

Bridges Virtuals Protocol ACP (Agent Commerce Protocol) evaluation requests
to ClawBizarre's structural verification server.

ACP evaluators receive (requirement, deliverable) and return approve/reject.
This bridge:
1. Parses the ACP deliverable for embedded test suites or schema constraints
2. Routes to ClawBizarre verify_server (Tier 0 or Tier 1)
3. Returns a deterministic, structural verdict (not LLM-subjective)

Usage:
    # As ACP evaluator agent callback:
    bridge = ACPBridge(verify_url="http://localhost:8700")
    result = bridge.evaluate(requirement, deliverable)
    
    # Standalone test:
    python3 acp_bridge.py --test
"""

import json
import os
import sys
import urllib.request
import urllib.error
from dataclasses import dataclass, asdict
from typing import Optional, Any
from datetime import datetime, timezone

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)


@dataclass
class EvaluationResult:
    """Result of an ACP evaluation, mapped from ClawBizarre verification."""
    approved: bool
    confidence: float       # 0.0-1.0, based on test pass rate
    receipt_id: str         # ClawBizarre VRF receipt ID
    tier: int               # Verification tier used
    verdict: str            # pass/fail/partial/error
    details: dict           # Full verification results
    reason: str             # Human-readable explanation
    
    def to_acp_response(self) -> dict:
        """Format as ACP evaluator response."""
        return {
            "approved": self.approved,
            "score": self.confidence,
            "feedback": self.reason,
            "metadata": {
                "verifier": "clawbizarre-verify/1.0",
                "receipt_id": self.receipt_id,
                "tier": self.tier,
                "verdict": self.verdict,
                "structural": True,  # Flag: this is structural, not LLM-subjective
            }
        }


class ACPBridge:
    """Bridge between ACP evaluation requests and ClawBizarre verification."""
    
    def __init__(self, verify_url: str = "http://localhost:8700"):
        self.verify_url = verify_url.rstrip("/")
    
    def evaluate(self, requirement: dict, deliverable: dict) -> EvaluationResult:
        """
        Evaluate an ACP deliverable against a requirement.
        
        ACP requirement format (expected):
        {
            "description": "...",
            "verification": {
                "kind": "test_suite" | "schema",
                "test_suite": { "tests": [...] },   # for kind=test_suite
                "schema": {...},                      # for kind=schema
                "constraints": [...]                  # for kind=schema
            }
        }
        
        ACP deliverable format (expected):
        {
            "content": "...",       # The actual work output
            "content_type": "...",  # e.g., "text/python", "application/json"
        }
        """
        verification = requirement.get("verification", {})
        kind = verification.get("kind", self._infer_kind(requirement, deliverable))
        
        if kind == "test_suite":
            return self._evaluate_tier0(requirement, deliverable, verification)
        elif kind == "schema":
            return self._evaluate_tier1(requirement, deliverable, verification)
        else:
            # No structural verification possible — return uncertain
            return EvaluationResult(
                approved=False,
                confidence=0.0,
                receipt_id="",
                tier=-1,
                verdict="unsupported",
                details={"error": f"No structural verification for kind={kind}"},
                reason=f"Cannot structurally verify: no test_suite or schema provided. "
                       f"This deliverable requires peer review (Tier 2+)."
            )
    
    def _infer_kind(self, requirement: dict, deliverable: dict) -> str:
        """Try to infer verification kind from content."""
        v = requirement.get("verification", {})
        if "test_suite" in v or "tests" in v:
            return "test_suite"
        if "schema" in v or "constraints" in v:
            return "schema"
        # Check if deliverable looks like JSON (→ schema check possible)
        content = deliverable.get("content", "")
        try:
            json.loads(content)
            return "schema" if "schema" in requirement else "unknown"
        except (json.JSONDecodeError, TypeError):
            pass
        return "unknown"
    
    def _evaluate_tier0(self, requirement: dict, deliverable: dict, 
                         verification: dict) -> EvaluationResult:
        """Tier 0: Test suite verification."""
        payload = {
            "tier": 0,
            "task_type": requirement.get("task_type", "acp_evaluation"),
            "task_id": requirement.get("task_id", ""),
            "specification": {"description": requirement.get("description", "")},
            "output": {"content": deliverable.get("content", "")},
            "verification": {
                "kind": "test_suite",
                "test_suite": verification.get("test_suite", 
                              {"tests": verification.get("tests", [])}),
            }
        }
        
        result = self._call_verify("/verify", payload)
        return self._to_evaluation(result)
    
    def _evaluate_tier1(self, requirement: dict, deliverable: dict,
                         verification: dict) -> EvaluationResult:
        """Tier 1: Schema/constraint verification."""
        payload = {
            "output": {"content": deliverable.get("content", "")},
            "schema": verification.get("schema", {}),
            "constraints": verification.get("constraints", []),
            "task_type": requirement.get("task_type", "acp_evaluation"),
            "task_id": requirement.get("task_id", ""),
        }
        
        result = self._call_verify("/verify/schema", payload)
        return self._to_evaluation(result)
    
    def _call_verify(self, path: str, payload: dict) -> dict:
        """Call the ClawBizarre verification server."""
        url = f"{self.verify_url}{path}"
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        try:
            resp = urllib.request.urlopen(req, timeout=60)
            return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            return {"error": body, "status": e.code}
        except Exception as e:
            return {"error": str(e)}
    
    def _to_evaluation(self, result: dict) -> EvaluationResult:
        """Convert verification server response to EvaluationResult."""
        if "error" in result and "verdict" not in result:
            return EvaluationResult(
                approved=False, confidence=0.0, receipt_id="",
                tier=-1, verdict="error", details=result,
                reason=f"Verification server error: {result['error']}"
            )
        
        verdict = result.get("verdict", "error")
        results = result.get("results", {})
        total = results.get("total", 0)
        passed = results.get("passed", 0)
        
        confidence = passed / total if total > 0 else 0.0
        approved = verdict == "pass"
        
        # Build reason
        if verdict == "pass":
            reason = f"All {total} checks passed. Structurally verified (Tier {result.get('tier', '?')})."
        elif verdict == "partial":
            reason = f"{passed}/{total} checks passed ({results.get('failed', 0)} failed). Partial compliance."
        elif verdict == "fail":
            failed_names = [d["name"] for d in results.get("details", []) if d.get("status") == "fail"]
            reason = f"All checks failed. Failed: {', '.join(failed_names[:5])}."
        else:
            reason = f"Verification error: {results.get('errors', 0)} errors encountered."
        
        return EvaluationResult(
            approved=approved,
            confidence=round(confidence, 3),
            receipt_id=result.get("receipt_id", ""),
            tier=result.get("tier", -1),
            verdict=verdict,
            details=result,
            reason=reason,
        )


# ── Tests ────────────────────────────────────────────────────────────

def run_tests():
    from threading import Thread
    from http.server import HTTPServer
    from verify_server import VerificationEngine, VerifyHandler
    
    # Start verify server
    try:
        from identity import AgentIdentity
        identity = AgentIdentity.generate()
    except ImportError:
        identity = None
    
    engine = VerificationEngine(identity=identity)
    VerifyHandler.engine = engine
    server = HTTPServer(("127.0.0.1", 0), VerifyHandler)
    port = server.server_address[1]
    Thread(target=server.serve_forever, daemon=True).start()
    
    bridge = ACPBridge(verify_url=f"http://127.0.0.1:{port}")
    
    passed = total = 0
    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
    
    # Test 1: Tier 0 — passing code
    print("\n1. ACP Tier 0 evaluation (pass)")
    result = bridge.evaluate(
        requirement={
            "description": "Write a function to double a number",
            "task_type": "code_generation",
            "verification": {
                "kind": "test_suite",
                "test_suite": {
                    "tests": [
                        {"name": "basic", "input": "double(5)", "expected_output": "10"},
                        {"name": "zero", "input": "double(0)", "expected_output": "0"},
                        {"name": "negative", "input": "double(-3)", "expected_output": "-6"},
                    ]
                }
            }
        },
        deliverable={"content": "def double(x): return x * 2"}
    )
    check("approved", result.approved)
    check("confidence 1.0", result.confidence == 1.0)
    check("verdict pass", result.verdict == "pass")
    check("has receipt_id", len(result.receipt_id) > 0)
    check("tier 0", result.tier == 0)
    
    # ACP format
    acp = result.to_acp_response()
    check("acp approved", acp["approved"])
    check("acp score 1.0", acp["score"] == 1.0)
    check("acp structural flag", acp["metadata"]["structural"])
    
    # Test 2: Tier 0 — failing code
    print("\n2. ACP Tier 0 evaluation (fail)")
    result = bridge.evaluate(
        requirement={
            "description": "Write a sort function",
            "verification": {
                "kind": "test_suite",
                "test_suite": {
                    "tests": [
                        {"name": "basic", "input": "my_sort([3,1,2])", "expected_output": "[1,2,3]"},
                    ]
                }
            }
        },
        deliverable={"content": "def my_sort(x): return x"}  # Wrong!
    )
    check("not approved", not result.approved)
    check("confidence 0.0", result.confidence == 0.0)
    check("verdict fail", result.verdict == "fail")
    
    # Test 3: Tier 1 — schema check
    print("\n3. ACP Tier 1 evaluation (schema)")
    result = bridge.evaluate(
        requirement={
            "description": "Return user profile JSON",
            "verification": {
                "kind": "schema",
                "schema": {"type": "object", "required": ["name", "age"]},
                "constraints": [
                    {"name": "age_positive", "expr": "output['age'] > 0"},
                ]
            }
        },
        deliverable={"content": '{"name": "Alice", "age": 30}'}
    )
    check("approved", result.approved)
    check("tier 1", result.tier == 1)
    check("confidence 1.0", result.confidence == 1.0)
    
    # Test 4: No verification possible
    print("\n4. ACP evaluation (no verification)")
    result = bridge.evaluate(
        requirement={"description": "Write a poem about the sea"},
        deliverable={"content": "The waves crash upon the shore..."}
    )
    check("not approved", not result.approved)
    check("tier -1 (unsupported)", result.tier == -1)
    check("confidence 0.0", result.confidence == 0.0)
    check("reason mentions peer review", "peer review" in result.reason.lower() or "Tier 2" in result.reason)
    
    # Test 5: Partial pass
    print("\n5. ACP Tier 0 evaluation (partial)")
    result = bridge.evaluate(
        requirement={
            "description": "Reverse a string",
            "verification": {
                "kind": "test_suite",
                "test_suite": {
                    "tests": [
                        {"name": "basic", "input": "rev('hello')", "expected_output": "'olleh'"},
                        {"name": "empty", "input": "rev('')", "expected_output": "''"},
                        {"name": "single", "input": "rev('a')", "expected_output": "'a'"},
                    ]
                }
            }
        },
        deliverable={"content": "def rev(s): return s[::-1] if len(s) > 1 else 'x'"}
    )
    check("not approved (partial)", not result.approved)
    check("confidence between 0 and 1", 0 < result.confidence < 1)
    check("verdict partial", result.verdict == "partial")
    
    # Test 6: Signed receipt in result
    print("\n6. Receipt signature in ACP result")
    result = bridge.evaluate(
        requirement={
            "description": "Add two numbers",
            "verification": {
                "kind": "test_suite",
                "test_suite": {"tests": [
                    {"name": "basic", "input": "add(2,3)", "expected_output": "5"},
                ]}
            }
        },
        deliverable={"content": "def add(a,b): return a+b"}
    )
    check("receipt has signature", result.details.get("signature") is not None if identity else True)
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    
    server.shutdown()
    return passed == total


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        print("ACP Evaluator Bridge — use --test to run tests")
        print("Usage in ACP agent: bridge = ACPBridge(); result = bridge.evaluate(req, deliverable)")
