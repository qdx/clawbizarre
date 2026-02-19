"""
langchain-vrf: VRF verification integration for LangChain/LangGraph agents.

Three integration levels:
1. verify_code tool — agent-initiated verification (LangChain Tool)
2. VRFCallbackHandler — automatic post-tool verification (callback)
3. VRFVerifyNode — structural verification gate (LangGraph node)

Requires: verify_server running (default http://localhost:8700)
No external dependencies beyond langchain/langgraph themselves.
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Optional, Any
from dataclasses import dataclass, field


DEFAULT_VERIFY_URL = "http://localhost:8700"


# ── Shared VRF client ──────────────────────────────────────────────

@dataclass
class VRFResult:
    """Result of a VRF verification."""
    verdict: str
    tests_passed: int
    tests_total: int
    receipt_id: str
    receipt: dict
    error: Optional[str] = None

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    def summary(self) -> str:
        if self.error:
            return f"VRF: ERROR — {self.error}"
        return f"VRF: {self.verdict.upper()} ({self.tests_passed}/{self.tests_total} tests) — receipt {self.receipt_id[:12]}"


def _call_verify(code: str, test_suite: Any, language: str = "python",
                 verify_url: Optional[str] = None, timeout: float = 30.0) -> VRFResult:
    """POST to verify_server and return VRFResult."""
    url = (verify_url or os.environ.get("VERIFY_URL", DEFAULT_VERIFY_URL)).rstrip("/")
    if isinstance(test_suite, str):
        test_suite = json.loads(test_suite)

    payload = json.dumps({
        "code": code,
        "test_suite": test_suite,
        "language": language,
    }).encode()
    req = urllib.request.Request(
        f"{url}/verify",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read())
        return VRFResult(
            verdict=data.get("verdict", "error"),
            tests_passed=data.get("tests_passed", 0),
            tests_total=data.get("tests_total", 0),
            receipt_id=data.get("receipt_id", ""),
            receipt=data,
        )
    except Exception as e:
        return VRFResult(
            verdict="error", tests_passed=0, tests_total=0,
            receipt_id="", receipt={}, error=str(e),
        )


def _extract_code(text: str) -> str:
    """Extract code from markdown fenced blocks if present."""
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ── Level 1: LangChain Tool ────────────────────────────────────────

def make_verify_tool(verify_url: Optional[str] = None, language: str = "python"):
    """
    Create a LangChain-compatible tool function for code verification.
    Returns a plain function with proper docstring (works with @tool or StructuredTool).
    """
    def verify_code(code: str, test_suite: str, language: str = language) -> str:
        """Verify code against a test suite using VRF. Returns PASS/FAIL with details.

        Args:
            code: The source code to verify
            test_suite: JSON string of test cases, e.g. [{"input": "f(1)", "expected": "2"}]
            language: Programming language (python, javascript, bash)
        """
        result = _call_verify(code, test_suite, language, verify_url)
        return result.summary()

    verify_code.__name__ = "verify_code"
    return verify_code


# ── Level 2: Callback Handler ──────────────────────────────────────

class VRFCallbackHandler:
    """
    Auto-verify tool outputs containing code.

    Usage:
        handler = VRFCallbackHandler(test_suites={"code_writer": [{"input": "f(1)", "expected": "2"}]})
        # Pass as callback to LangChain agent/chain

    The handler intercepts tool outputs, checks if the tool name has a registered
    test suite, extracts code from the output, and verifies it.
    """

    def __init__(self, verify_url: Optional[str] = None,
                 test_suites: Optional[dict] = None,
                 on_fail: Optional[str] = "log"):
        """
        Args:
            verify_url: URL of verify_server
            test_suites: {tool_name: test_suite_list} mapping
            on_fail: "log" (default), "raise", or "inject"
        """
        self.verify_url = verify_url
        self.test_suites = test_suites or {}
        self.on_fail = on_fail
        self.receipts: list[VRFResult] = []

    def on_tool_end(self, output: str, *, name: str = "", **kwargs) -> Optional[str]:
        """Called after a tool returns output. Returns modified output or None."""
        if name not in self.test_suites:
            return None

        code = _extract_code(str(output))
        result = _call_verify(code, self.test_suites[name], verify_url=self.verify_url)
        self.receipts.append(result)

        if not result.passed:
            if self.on_fail == "raise":
                raise VerificationError(result)
            elif self.on_fail == "inject":
                return f"{output}\n\n⚠️ VERIFICATION FAILED: {result.summary()}"
        return None

    def get_receipts(self) -> list[VRFResult]:
        return list(self.receipts)

    def stats(self) -> dict:
        total = len(self.receipts)
        passed = sum(1 for r in self.receipts if r.passed)
        return {"total": total, "passed": passed, "failed": total - passed}


class VerificationError(Exception):
    """Raised when verification fails and on_fail='raise'."""
    def __init__(self, result: VRFResult):
        self.result = result
        super().__init__(result.summary())


# ── Level 3: LangGraph Verification Node ───────────────────────────

@dataclass
class VerifyState:
    """State fields used by the verification node."""
    code: str = ""
    test_suite: Any = None
    language: str = "python"
    needs_verification: bool = False
    vrf_receipt: Optional[dict] = None
    vrf_verdict: str = ""
    retry_count: int = 0
    max_retries: int = 3
    feedback: str = ""


def make_verify_node(verify_url: Optional[str] = None, max_retries: int = 3):
    """
    Create a LangGraph verification node function.

    The node reads `code`, `test_suite`, `language` from state,
    runs verification, and sets `vrf_receipt`, `vrf_verdict`, `feedback`.

    Routing logic (add_conditional_edges):
      - vrf_verdict == "pass" → proceed to output
      - vrf_verdict == "retry" → loop back to code generation
      - vrf_verdict == "fail" → proceed with failure info
      - vrf_verdict == "skip" → no verification needed
    """
    def verify_node(state: dict) -> dict:
        if not state.get("needs_verification", False):
            return {**state, "vrf_verdict": "skip"}

        code = _extract_code(state.get("code", ""))
        test_suite = state.get("test_suite")
        language = state.get("language", "python")

        if not code or not test_suite:
            return {**state, "vrf_verdict": "skip"}

        result = _call_verify(code, test_suite, language, verify_url)
        retry_count = state.get("retry_count", 0)

        new_state = {
            **state,
            "vrf_receipt": result.receipt,
            "vrf_verdict": "",
            "retry_count": retry_count,
        }

        if result.error:
            # Verify server unreachable — degrade gracefully
            new_state["vrf_verdict"] = "skip"
            new_state["feedback"] = f"Verification unavailable: {result.error}"
        elif result.passed:
            new_state["vrf_verdict"] = "pass"
        elif retry_count < max_retries:
            new_state["vrf_verdict"] = "retry"
            new_state["retry_count"] = retry_count + 1
            # Build feedback for the code generator
            failures = result.receipt.get("failures", [])
            failure_text = "; ".join(
                f"Test '{f.get('input', '?')}': expected {f.get('expected', '?')}, got {f.get('actual', '?')}"
                for f in failures[:5]
            ) if failures else f"{result.tests_total - result.tests_passed} tests failed"
            new_state["feedback"] = f"Verification failed ({result.tests_passed}/{result.tests_total}): {failure_text}"
        else:
            new_state["vrf_verdict"] = "fail"
            new_state["feedback"] = f"Failed after {max_retries} retries: {result.summary()}"

        return new_state

    return verify_node


def route_on_verdict(state: dict) -> str:
    """Conditional edge router for LangGraph. Returns next node name."""
    verdict = state.get("vrf_verdict", "skip")
    if verdict == "pass":
        return "respond"
    elif verdict == "retry":
        return "generate"
    elif verdict == "fail":
        return "respond"
    else:  # skip
        return "respond"


# ── Tests ──────────────────────────────────────────────────────────

def _run_tests():
    """Self-contained tests (no verify_server needed — mocks HTTP)."""
    import unittest
    from unittest.mock import patch, MagicMock
    from io import BytesIO

    MOCK_PASS_RESPONSE = json.dumps({
        "verdict": "pass", "tests_passed": 3, "tests_total": 3,
        "receipt_id": "abc123def456", "failures": [],
    }).encode()

    MOCK_FAIL_RESPONSE = json.dumps({
        "verdict": "fail", "tests_passed": 1, "tests_total": 3,
        "receipt_id": "fail789xyz",
        "failures": [
            {"input": "f(2)", "expected": "4", "actual": "3"},
            {"input": "f(3)", "expected": "9", "actual": "6"},
        ],
    }).encode()

    def mock_urlopen(response_bytes):
        def _mock(req, **kw):
            resp = MagicMock()
            resp.read.return_value = response_bytes
            resp.__enter__ = lambda s: s
            resp.__exit__ = lambda s, *a: None
            return resp
        return _mock

    class TestVRFResult(unittest.TestCase):
        def test_pass_result(self):
            r = VRFResult(verdict="pass", tests_passed=3, tests_total=3,
                          receipt_id="abc123", receipt={})
            self.assertTrue(r.passed)
            self.assertIn("PASS", r.summary())

        def test_fail_result(self):
            r = VRFResult(verdict="fail", tests_passed=1, tests_total=3,
                          receipt_id="xyz789", receipt={})
            self.assertFalse(r.passed)
            self.assertIn("FAIL", r.summary())

        def test_error_result(self):
            r = VRFResult(verdict="error", tests_passed=0, tests_total=0,
                          receipt_id="", receipt={}, error="timeout")
            self.assertIn("ERROR", r.summary())

    class TestExtractCode(unittest.TestCase):
        def test_plain_code(self):
            self.assertEqual(_extract_code("def f(x): return x"), "def f(x): return x")

        def test_markdown_block(self):
            md = "Here's the code:\n```python\ndef f(x):\n    return x * 2\n```\nDone."
            self.assertEqual(_extract_code(md), "def f(x):\n    return x * 2")

    class TestVerifyTool(unittest.TestCase):
        @patch("urllib.request.urlopen", side_effect=mock_urlopen(MOCK_PASS_RESPONSE))
        def test_tool_pass(self, mock):
            tool = make_verify_tool()
            result = tool("def f(x): return x*x", '[{"input":"f(2)","expected":"4"}]')
            self.assertIn("PASS", result)

    class TestCallbackHandler(unittest.TestCase):
        @patch("urllib.request.urlopen", side_effect=mock_urlopen(MOCK_FAIL_RESPONSE))
        def test_callback_logs_failure(self, mock):
            handler = VRFCallbackHandler(
                test_suites={"coder": [{"input": "f(2)", "expected": "4"}]},
                on_fail="log",
            )
            handler.on_tool_end("def f(x): return x+1", name="coder")
            self.assertEqual(len(handler.receipts), 1)
            self.assertFalse(handler.receipts[0].passed)

        @patch("urllib.request.urlopen", side_effect=mock_urlopen(MOCK_FAIL_RESPONSE))
        def test_callback_raise(self, mock):
            handler = VRFCallbackHandler(
                test_suites={"coder": [{"input": "f(2)", "expected": "4"}]},
                on_fail="raise",
            )
            with self.assertRaises(VerificationError):
                handler.on_tool_end("def f(x): return x+1", name="coder")

        @patch("urllib.request.urlopen", side_effect=mock_urlopen(MOCK_PASS_RESPONSE))
        def test_callback_inject(self, mock):
            handler = VRFCallbackHandler(
                test_suites={"coder": [{"input": "f(2)", "expected": "4"}]},
                on_fail="inject",
            )
            result = handler.on_tool_end("def f(x): return x*x", name="coder")
            # Pass → no injection
            self.assertIsNone(result)

        def test_callback_skip_unknown_tool(self):
            handler = VRFCallbackHandler(test_suites={"coder": []})
            result = handler.on_tool_end("output", name="unknown_tool")
            self.assertIsNone(result)
            self.assertEqual(len(handler.receipts), 0)

    class TestVerifyNode(unittest.TestCase):
        @patch("urllib.request.urlopen", side_effect=mock_urlopen(MOCK_PASS_RESPONSE))
        def test_node_pass(self, mock):
            node = make_verify_node()
            state = {
                "code": "def f(x): return x*x",
                "test_suite": [{"input": "f(2)", "expected": "4"}],
                "needs_verification": True,
            }
            result = node(state)
            self.assertEqual(result["vrf_verdict"], "pass")
            self.assertIn("receipt_id", result["vrf_receipt"])

        @patch("urllib.request.urlopen", side_effect=mock_urlopen(MOCK_FAIL_RESPONSE))
        def test_node_retry(self, mock):
            node = make_verify_node(max_retries=3)
            state = {
                "code": "def f(x): return x+1",
                "test_suite": [{"input": "f(2)", "expected": "4"}],
                "needs_verification": True,
                "retry_count": 0,
            }
            result = node(state)
            self.assertEqual(result["vrf_verdict"], "retry")
            self.assertEqual(result["retry_count"], 1)
            self.assertIn("failed", result["feedback"].lower())

        @patch("urllib.request.urlopen", side_effect=mock_urlopen(MOCK_FAIL_RESPONSE))
        def test_node_fail_after_retries(self, mock):
            node = make_verify_node(max_retries=2)
            state = {
                "code": "def f(x): return x+1",
                "test_suite": [{"input": "f(2)", "expected": "4"}],
                "needs_verification": True,
                "retry_count": 2,
            }
            result = node(state)
            self.assertEqual(result["vrf_verdict"], "fail")

        def test_node_skip(self):
            node = make_verify_node()
            state = {"code": "x", "needs_verification": False}
            result = node(state)
            self.assertEqual(result["vrf_verdict"], "skip")

    class TestRouteOnVerdict(unittest.TestCase):
        def test_routing(self):
            self.assertEqual(route_on_verdict({"vrf_verdict": "pass"}), "respond")
            self.assertEqual(route_on_verdict({"vrf_verdict": "retry"}), "generate")
            self.assertEqual(route_on_verdict({"vrf_verdict": "fail"}), "respond")
            self.assertEqual(route_on_verdict({"vrf_verdict": "skip"}), "respond")

    class TestStats(unittest.TestCase):
        def test_stats(self):
            handler = VRFCallbackHandler()
            handler.receipts = [
                VRFResult("pass", 3, 3, "a", {}),
                VRFResult("fail", 1, 3, "b", {}),
                VRFResult("pass", 2, 2, "c", {}),
            ]
            s = handler.stats()
            self.assertEqual(s["total"], 3)
            self.assertEqual(s["passed"], 2)
            self.assertEqual(s["failed"], 1)

    suite = unittest.TestLoader().loadTestsFromTestCase(TestVRFResult)
    for cls in [TestExtractCode, TestVerifyTool, TestCallbackHandler,
                TestVerifyNode, TestRouteOnVerdict, TestStats]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))

    runner = unittest.TextTestRunner(verbosity=2)
    return runner.run(suite)


if __name__ == "__main__":
    _run_tests()
