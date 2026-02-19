"""
vrf-client: Minimal, zero-dependency Python client for VRF verify_server.

This is the shared foundation that langchain-vrf, crewai-vrf, and openai-vrf
all depend on. It can also be used standalone.

Zero external dependencies (stdlib urllib only).
"""

import json
import os
import re
import time
import urllib.request
import urllib.error
from typing import Optional, Any, Union
from dataclasses import dataclass, field


__version__ = "0.1.0"
DEFAULT_VERIFY_URL = "http://localhost:8700"


@dataclass
class VRFReceipt:
    """Structured VRF verification receipt."""
    verdict: str  # "pass", "fail", "error"
    tests_passed: int
    tests_total: int
    receipt_id: str
    raw: dict  # Full server response
    error: Optional[str] = None
    elapsed_ms: float = 0.0

    @property
    def passed(self) -> bool:
        return self.verdict == "pass"

    @property
    def failures(self) -> list:
        return self.raw.get("failures", [])

    def summary(self) -> str:
        if self.error:
            return f"VRF ERROR: {self.error}"
        return f"VRF {self.verdict.upper()}: {self.tests_passed}/{self.tests_total} tests ({self.elapsed_ms:.0f}ms) — {self.receipt_id[:12]}"

    def to_json(self) -> str:
        return json.dumps({
            "verdict": self.verdict,
            "tests_passed": self.tests_passed,
            "tests_total": self.tests_total,
            "receipt_id": self.receipt_id,
            "elapsed_ms": self.elapsed_ms,
            "error": self.error,
        })


class VRFClient:
    """
    HTTP client for VRF verify_server.

    Usage:
        client = VRFClient()  # uses VERIFY_URL env or localhost:8700
        receipt = client.verify("def f(x): return x*2", [{"input": "f(3)", "expected": "6"}])
        print(receipt.passed)  # True
    """

    def __init__(self, url: Optional[str] = None, timeout: float = 30.0,
                 language: str = "python"):
        self.url = (url or os.environ.get("VERIFY_URL", DEFAULT_VERIFY_URL)).rstrip("/")
        self.timeout = timeout
        self.default_language = language

    def verify(self, code: str, test_suite: Any,
               language: Optional[str] = None,
               use_docker: bool = False) -> VRFReceipt:
        """Verify code against a test suite. Returns VRFReceipt."""
        if isinstance(test_suite, str):
            stripped = test_suite.strip()
            if stripped.startswith("[") or stripped.startswith("{"):
                test_suite = json.loads(test_suite)
            # else: pass string as-is (e.g., assertion strings)

        payload = {
            "code": code,
            "test_suite": test_suite,
            "language": language or self.default_language,
        }
        if use_docker:
            payload["use_docker"] = True

        t0 = time.monotonic()
        try:
            data = self._post("/verify", payload)
            elapsed = (time.monotonic() - t0) * 1000
            return VRFReceipt(
                verdict=data.get("verdict", "error"),
                tests_passed=data.get("tests_passed", 0),
                tests_total=data.get("tests_total", 0),
                receipt_id=data.get("receipt_id", ""),
                raw=data,
                elapsed_ms=elapsed,
            )
        except Exception as e:
            elapsed = (time.monotonic() - t0) * 1000
            return VRFReceipt(
                verdict="error", tests_passed=0, tests_total=0,
                receipt_id="", raw={}, error=str(e), elapsed_ms=elapsed,
            )

    def health(self) -> dict:
        """Check verify_server health."""
        try:
            return self._get("/health")
        except Exception as e:
            return {"status": "unreachable", "error": str(e)}

    def receipts(self, verdict: Optional[str] = None, limit: int = 10,
                 offset: int = 0) -> list:
        """Query stored receipts (if persistence enabled)."""
        params = f"?limit={limit}&offset={offset}"
        if verdict:
            params += f"&verdict={verdict}"
        try:
            return self._get(f"/receipts{params}")
        except Exception:
            return []

    def _post(self, path: str, payload: dict) -> dict:
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.url}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())

    def _get(self, path: str) -> Any:
        req = urllib.request.Request(f"{self.url}{path}")
        with urllib.request.urlopen(req, timeout=self.timeout) as resp:
            return json.loads(resp.read())


# ── Utilities ──────────────────────────────────────────────────────

def extract_code(text: str) -> str:
    """Extract code from markdown fenced blocks if present."""
    m = re.search(r"```(?:\w+)?\s*\n(.*?)```", text, re.DOTALL)
    return m.group(1).strip() if m else text.strip()


# ── Tests ──────────────────────────────────────────────────────────

def _run_tests():
    import unittest
    from unittest.mock import patch, MagicMock

    MOCK_PASS = json.dumps({
        "verdict": "pass", "tests_passed": 3, "tests_total": 3,
        "receipt_id": "abc123def456", "failures": [],
    }).encode()

    MOCK_FAIL = json.dumps({
        "verdict": "fail", "tests_passed": 1, "tests_total": 3,
        "receipt_id": "fail789",
        "failures": [{"input": "f(2)", "expected": "4", "actual": "3"}],
    }).encode()

    MOCK_HEALTH = json.dumps({"status": "ok", "version": "1.0"}).encode()

    def mock_open(resp_bytes):
        def _mock(req, **kw):
            r = MagicMock()
            r.read.return_value = resp_bytes
            r.__enter__ = lambda s: s
            r.__exit__ = lambda s, *a: None
            return r
        return _mock

    class TestVRFReceipt(unittest.TestCase):
        def test_pass(self):
            r = VRFReceipt("pass", 3, 3, "abc", {})
            self.assertTrue(r.passed)
            self.assertIn("PASS", r.summary())

        def test_fail(self):
            r = VRFReceipt("fail", 1, 3, "xyz", {"failures": [{"input": "x"}]})
            self.assertFalse(r.passed)
            self.assertEqual(len(r.failures), 1)

        def test_error(self):
            r = VRFReceipt("error", 0, 0, "", {}, error="timeout")
            self.assertIn("ERROR", r.summary())

        def test_to_json(self):
            r = VRFReceipt("pass", 2, 2, "id1", {}, elapsed_ms=42.5)
            d = json.loads(r.to_json())
            self.assertEqual(d["verdict"], "pass")
            self.assertEqual(d["elapsed_ms"], 42.5)

    class TestExtractCode(unittest.TestCase):
        def test_plain(self):
            self.assertEqual(extract_code("x = 1"), "x = 1")

        def test_markdown(self):
            self.assertEqual(extract_code("```python\nx = 1\n```"), "x = 1")

        def test_no_lang(self):
            self.assertEqual(extract_code("```\nfoo\n```"), "foo")

    class TestVRFClient(unittest.TestCase):
        @patch("urllib.request.urlopen", side_effect=mock_open(MOCK_PASS))
        def test_verify_pass(self, mock):
            c = VRFClient(url="http://test:8700")
            r = c.verify("code", [{"input": "f(1)", "expected": "1"}])
            self.assertTrue(r.passed)
            self.assertGreater(r.elapsed_ms, 0)

        @patch("urllib.request.urlopen", side_effect=mock_open(MOCK_FAIL))
        def test_verify_fail(self, mock):
            c = VRFClient()
            r = c.verify("code", '[{"input": "f(1)", "expected": "1"}]')
            self.assertFalse(r.passed)
            self.assertEqual(r.tests_passed, 1)

        def test_verify_unreachable(self):
            c = VRFClient(url="http://localhost:1", timeout=0.1)
            r = c.verify("code", [])
            self.assertEqual(r.verdict, "error")
            self.assertIsNotNone(r.error)

        @patch("urllib.request.urlopen", side_effect=mock_open(MOCK_HEALTH))
        def test_health(self, mock):
            c = VRFClient()
            h = c.health()
            self.assertEqual(h["status"], "ok")

        def test_health_unreachable(self):
            c = VRFClient(url="http://localhost:1", timeout=0.1)
            h = c.health()
            self.assertEqual(h["status"], "unreachable")

        @patch("urllib.request.urlopen", side_effect=mock_open(MOCK_PASS))
        def test_docker_flag(self, mock):
            c = VRFClient()
            r = c.verify("code", [], use_docker=True)
            # Verify the request included use_docker
            call_args = mock.call_args
            body = json.loads(call_args[0][0].data)
            self.assertTrue(body["use_docker"])

    suite = unittest.TestSuite()
    for cls in [TestVRFReceipt, TestExtractCode, TestVRFClient]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(cls))
    return unittest.TextTestRunner(verbosity=2).run(suite)


if __name__ == "__main__":
    _run_tests()
