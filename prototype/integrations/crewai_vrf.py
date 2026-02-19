"""
crewai-vrf: VRF verification integration for CrewAI agents.

Three integration levels:
1. VerifyCodeTool — agent-initiated verification (tool)
2. vrf_task_callback — automatic post-task verification (callback)
3. VRFGuardedCrew — structural verification gate (crew-level)

Requires: verify_server running (default http://localhost:8700)
Dependencies: vrf-client (shared foundation)
"""

import json
from typing import Optional, Callable

from vrf_client import VRFClient, VRFReceipt, extract_code, DEFAULT_VERIFY_URL


# =============================================================================
# Level 1: CrewAI Tool (agent-initiated)
# =============================================================================

def make_verify_tool(verify_url: str = DEFAULT_VERIFY_URL):
    """
    Create a CrewAI-compatible tool function for code verification.
    
    Usage:
        from crewai import Agent
        agent = Agent(role="Coder", tools=[make_verify_tool()])
    
    The tool accepts a JSON string with keys: code, test_suite, language (optional).
    """
    client = VRFClient(url=verify_url)

    def verify_code_tool(input_json: str) -> str:
        """Verify code against a test suite. Input: JSON with 'code', 'test_suite', optional 'language'.
        Returns verification result with pass/fail verdict and receipt."""
        try:
            params = json.loads(input_json)
        except json.JSONDecodeError:
            return "Error: Input must be valid JSON with 'code' and 'test_suite' keys."
        
        code = params.get("code", "")
        test_suite = params.get("test_suite", "")
        language = params.get("language", "python")
        
        if not code or not test_suite:
            return "Error: Both 'code' and 'test_suite' are required."
        
        try:
            receipt = client.verify(code, test_suite, language=language)
            return receipt.summary()
        except ConnectionError as e:
            return f"Error: {e}"
    
    verify_code_tool.__name__ = "verify_code"
    return verify_code_tool


# =============================================================================
# Level 2: Task Callback (automatic post-task verification)
# =============================================================================

def vrf_task_callback(
    test_suite: str,
    language: str = "python",
    verify_url: str = DEFAULT_VERIFY_URL,
    on_fail: Optional[Callable[[VRFReceipt], None]] = None,
) -> Callable:
    """
    Create a CrewAI task callback that auto-verifies task output.
    
    Usage:
        task = Task(
            description="Write fibonacci",
            callback=vrf_task_callback(
                test_suite="assert fibonacci(10) == 55",
                language="python"
            )
        )
    """
    client = VRFClient(url=verify_url, language=language)

    def callback(task_output) -> None:
        code = str(task_output)
        # Strip markdown code blocks if present
        extracted = extract_code(code)
        if extracted != code:
            code = extracted
        
        try:
            receipt = client.verify(code, test_suite)
            if receipt.passed:
                print(f"✅ {receipt.summary()}")
            else:
                print(f"❌ {receipt.summary()}")
                if on_fail:
                    on_fail(receipt)
        except ConnectionError:
            print(f"⚠️ VRF: verify_server unreachable at {verify_url}, skipping verification")
    
    return callback


# =============================================================================
# Level 3: Guarded Execution (structural verification gate)
# =============================================================================

class VRFGuard:
    """
    Wraps a code-execution step with VRF verification and retry logic.
    
    Not CrewAI-specific — works with any callable that returns code.
    For CrewAI Flows, use as a @listen step.
    
    Usage:
        guard = VRFGuard(
            test_suite="assert sort_list([3,1,2]) == [1,2,3]",
            max_retries=2
        )
        
        # In a Flow:
        @listen(generate_code)
        def verify_step(self, code: str) -> str:
            result = guard.verify_or_retry(code, retry_fn=self.regenerate)
            return result.receipt_id
    """
    
    def __init__(
        self,
        test_suite: str,
        language: str = "python",
        verify_url: str = DEFAULT_VERIFY_URL,
        max_retries: int = 2,
        use_docker: bool = False,
    ):
        self.client = VRFClient(url=verify_url, language=language)
        self.test_suite = test_suite
        self.max_retries = max_retries
        self.use_docker = use_docker
        self.attempts: list[VRFReceipt] = []
    
    def verify(self, code: str) -> VRFReceipt:
        """Single verification attempt."""
        receipt = self.client.verify(
            code, self.test_suite, use_docker=self.use_docker
        )
        self.attempts.append(receipt)
        return receipt
    
    def verify_or_retry(
        self,
        code: str,
        retry_fn: Optional[Callable[[str, VRFReceipt], str]] = None,
    ) -> VRFReceipt:
        """
        Verify code. On failure, call retry_fn to get new code and re-verify.
        
        Args:
            code: Initial code to verify
            retry_fn: Called with (code, failed_result) → new code. If None, no retry.
        
        Returns:
            Final VRFReceipt (pass or last failure)
        """
        receipt = self.verify(code)
        
        attempts = 0
        while not receipt.passed and attempts < self.max_retries and retry_fn:
            attempts += 1
            code = retry_fn(code, receipt)
            receipt = self.verify(code)
        
        return receipt


# =============================================================================
# Tests
# =============================================================================

def _run_tests():
    """Self-tests (no verify_server needed for unit tests)."""
    import sys
    passed = 0
    total = 0
    
    # Test 1: VRFReceipt from shared client
    total += 1
    r = VRFReceipt(verdict="pass", tests_passed=3, tests_total=3, receipt_id="abc123", raw={"receipt_id": "abc123"})
    assert r.passed
    assert "PASS" in r.summary()
    assert "3/3" in r.summary()
    passed += 1
    
    # Test 2: VRFReceipt fail
    total += 1
    r = VRFReceipt(verdict="fail", tests_passed=1, tests_total=3, receipt_id="def456", raw={"receipt_id": "def456"})
    assert not r.passed
    assert "FAIL" in r.summary()
    passed += 1
    
    # Test 3: make_verify_tool returns callable
    total += 1
    tool = make_verify_tool()
    assert callable(tool)
    assert tool.__name__ == "verify_code"
    passed += 1
    
    # Test 4: Tool rejects bad input
    total += 1
    result = make_verify_tool()("not json")
    assert "Error" in result
    passed += 1
    
    # Test 5: Tool rejects missing fields
    total += 1
    result = make_verify_tool()(json.dumps({"code": "x"}))
    assert "Error" in result
    passed += 1
    
    # Test 6: VRFGuard init
    total += 1
    guard = VRFGuard("assert True", max_retries=3)
    assert guard.max_retries == 3
    assert len(guard.attempts) == 0
    passed += 1
    
    # Test 7: Code extraction via shared extract_code
    total += 1
    code = extract_code("```python\ndef add(a,b): return a+b\n```")
    assert code == "def add(a,b): return a+b"
    passed += 1
    
    # Test 8: Unreachable server returns error receipt
    total += 1
    client = VRFClient(url="http://localhost:99999")
    receipt = client.verify("x", "y")
    assert receipt.verdict == "error"
    assert receipt.error is not None
    passed += 1
    
    print(f"crewai_vrf: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    success = _run_tests()
    sys.exit(0 if success else 1)
