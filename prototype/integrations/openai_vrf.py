"""
openai-vrf: VRF verification for OpenAI Assistants and Swarm agents.

Three integration levels:
1. Function tool definition for Assistants API
2. Swarm verification agent (hand-off target)  
3. Run step interceptor (automatic)

Requires: verify_server running (default http://localhost:8700)
Dependencies: vrf-client (shared foundation)
"""

import json
from typing import Optional

from vrf_client import VRFClient, VRFReceipt, DEFAULT_VERIFY_URL


# =============================================================================
# Level 1: OpenAI Function Tool Definition
# =============================================================================

def get_tool_definition() -> dict:
    """
    Returns an OpenAI function tool definition for code verification.
    
    Usage with Assistants API:
        assistant = client.beta.assistants.create(
            tools=[{"type": "function", "function": get_tool_definition()}],
            ...
        )
    """
    return {
        "name": "verify_code",
        "description": "Verify code against a test suite using deterministic VRF verification. "
                       "Returns a cryptographic receipt with pass/fail verdict. "
                       "Use this to prove code correctness before delivering to the user.",
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "The source code to verify"
                },
                "test_suite": {
                    "type": "string", 
                    "description": "Test assertions to run (e.g., 'assert add(1,2) == 3\\nassert add(0,0) == 0')"
                },
                "language": {
                    "type": "string",
                    "enum": ["python", "javascript", "bash"],
                    "description": "Programming language of the code",
                    "default": "python"
                }
            },
            "required": ["code", "test_suite"]
        }
    }


def handle_tool_call(arguments: dict, verify_url: str = DEFAULT_VERIFY_URL) -> str:
    """
    Handle a verify_code tool call from OpenAI.
    
    Usage:
        if tool_call.function.name == "verify_code":
            result = handle_tool_call(json.loads(tool_call.function.arguments))
    """
    code = arguments.get("code", "")
    test_suite = arguments.get("test_suite", "")
    language = arguments.get("language", "python")
    
    if not code or not test_suite:
        return json.dumps({"error": "Both 'code' and 'test_suite' are required"})
    
    client = VRFClient(url=verify_url)
    receipt = client.verify(code, test_suite, language=language)
    if receipt.error:
        return json.dumps({"error": receipt.error})
    return json.dumps({
        "verdict": receipt.verdict,
        "tests_passed": receipt.tests_passed,
        "tests_total": receipt.tests_total,
        "receipt_id": receipt.receipt_id,
    })


# =============================================================================
# Level 2: Swarm Verification Agent
# =============================================================================

def make_swarm_agent_config(verify_url: str = DEFAULT_VERIFY_URL) -> dict:
    """
    Returns a Swarm agent configuration for a verification agent.
    
    Usage:
        from swarm import Agent
        config = make_swarm_agent_config()
        verifier = Agent(**config)
    """
    client = VRFClient(url=verify_url)

    def verify_code_fn(code: str, test_suite: str, language: str = "python") -> str:
        """Verify code against a test suite. Returns verdict with receipt."""
        try:
            receipt = client.verify(code, test_suite, language=language)
            return f"Verification {receipt.verdict.upper()}: {receipt.tests_passed}/{receipt.tests_total} tests passed (receipt: {receipt.receipt_id[:12]})"
        except ConnectionError as e:
            return f"Verification unavailable: {e}"
    
    return {
        "name": "Verifier",
        "instructions": (
            "You are a code verification agent. When you receive code from another agent, "
            "verify it using the verify_code function. Report the verdict clearly. "
            "If verification fails, explain which tests failed and suggest fixes."
        ),
        "functions": [verify_code_fn],
    }


# =============================================================================
# Level 3: Assistants Run Step Processor
# =============================================================================

class VRFRunProcessor:
    """
    Processes OpenAI Assistant run steps, auto-verifying code outputs.
    
    Usage:
        processor = VRFRunProcessor(verify_url="http://localhost:8700")
        
        # In your run loop:
        for step in run.steps:
            if step.type == "tool_calls":
                for tc in step.step_details.tool_calls:
                    if tc.type == "code_interpreter":
                        result = processor.check_code_output(tc)
    """
    
    def __init__(self, verify_url: str = DEFAULT_VERIFY_URL):
        self.client = VRFClient(url=verify_url)
        self.results: list[dict] = []
    
    def verify(self, code: str, test_suite: str, language: str = "python") -> dict:
        """Verify code and store result."""
        try:
            receipt = self.client.verify(code, test_suite, language=language)
            entry = {
                "verdict": receipt.verdict,
                "tests_passed": receipt.tests_passed,
                "tests_total": receipt.tests_total,
                "receipt_id": receipt.receipt_id,
            }
        except ConnectionError:
            entry = {"verdict": "error", "error": "verify_server unreachable"}
        
        self.results.append(entry)
        return entry
    
    def stats(self) -> dict:
        """Summary of all verifications in this session."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.get("verdict") == "pass")
        failed = sum(1 for r in self.results if r.get("verdict") == "fail")
        errors = sum(1 for r in self.results if r.get("verdict") == "error")
        return {"total": total, "passed": passed, "failed": failed, "errors": errors}


# =============================================================================
# Tests
# =============================================================================

def _run_tests():
    passed = 0
    total = 0
    
    # Test 1: Tool definition schema
    total += 1
    td = get_tool_definition()
    assert td["name"] == "verify_code"
    assert "code" in td["parameters"]["properties"]
    assert "test_suite" in td["parameters"]["properties"]
    assert td["parameters"]["required"] == ["code", "test_suite"]
    passed += 1
    
    # Test 2: handle_tool_call missing fields
    total += 1
    result = json.loads(handle_tool_call({}))
    assert "error" in result
    passed += 1
    
    # Test 3: handle_tool_call unreachable server
    total += 1
    result = json.loads(handle_tool_call(
        {"code": "x", "test_suite": "y"},
        verify_url="http://localhost:99999"
    ))
    assert "error" in result
    passed += 1
    
    # Test 4: Swarm agent config
    total += 1
    config = make_swarm_agent_config()
    assert config["name"] == "Verifier"
    assert len(config["functions"]) == 1
    assert callable(config["functions"][0])
    passed += 1
    
    # Test 5: VRFRunProcessor init and stats
    total += 1
    proc = VRFRunProcessor()
    assert proc.stats() == {"total": 0, "passed": 0, "failed": 0, "errors": 0}
    passed += 1
    
    # Test 6: VRFRunProcessor stores results
    total += 1
    proc = VRFRunProcessor(verify_url="http://localhost:99999")
    proc.verify("x", "y")  # Will error (no server)
    stats = proc.stats()
    assert stats["total"] == 1
    assert stats["errors"] == 1
    passed += 1
    
    # Test 7: Tool definition is valid OpenAI format
    total += 1
    wrapped = {"type": "function", "function": get_tool_definition()}
    assert wrapped["type"] == "function"
    assert "parameters" in wrapped["function"]
    passed += 1
    
    print(f"openai_vrf: {passed}/{total} tests passed")
    return passed == total


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent))
    success = _run_tests()
    sys.exit(0 if success else 1)
