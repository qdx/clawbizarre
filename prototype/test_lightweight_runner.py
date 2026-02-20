"""
Tests for lightweight_runner.py — Docker-free VRF execution backend.
All tests work without Docker installed.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
from lightweight_runner import LightweightRunner, _find_node, _check_docker


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def runner():
    """LightweightRunner with Docker disabled for predictable test behavior."""
    return LightweightRunner(prefer_docker=False, timeout=8)


@pytest.fixture
def node_runner():
    """Runner for JavaScript tests — skip if Node not found."""
    node = _find_node()
    if not node:
        pytest.skip("Node.js not found")
    return LightweightRunner(prefer_docker=False, timeout=8, node_path=node)


# ── Python: Expression tests ──────────────────────────────────────────────────

class TestPythonExpressions:

    def test_simple_addition(self, runner):
        code = "def add(a, b): return a + b"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "add(2, 3)", "expected_output": "5"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 1
        assert r["failed"] == 0

    def test_string_manipulation(self, runner):
        code = "def reverse(s): return s[::-1]"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "reverse('hello')", "expected_output": "olleh"},
                {"id": "t2", "type": "expression", "expression": "reverse('abc')", "expected_output": "cba"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 2

    def test_failing_test(self, runner):
        code = "def add(a, b): return a - b  # bug"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "add(2, 3)", "expected_output": "5"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 0
        assert r["failed"] == 1

    def test_mixed_pass_fail(self, runner):
        code = "def double(x): return x * 2"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "double(3)", "expected_output": "6"},
                {"id": "t2", "type": "expression", "expression": "double(0)", "expected_output": "1"},  # wrong expected
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 1
        assert r["failed"] == 1
        assert r["total"] == 2

    def test_empty_suite(self, runner):
        code = "x = 1"
        suite = {"tests": []}
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 0
        assert r["failed"] == 0
        assert r["total"] == 0


# ── Python: IO tests ──────────────────────────────────────────────────────────

class TestPythonIO:

    def test_io_single_arg(self, runner):
        code = "def greet(name): return f'Hello, {name}!'"
        suite = {
            "tests": [
                {"id": "t1", "type": "io", "input": "World", "expected_output": "Hello, World!"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 1

    def test_io_multi_arg(self, runner):
        code = "def multiply(a, b): return a * b"
        suite = {
            "tests": [
                {"id": "t1", "type": "io", "input": [4, 5], "expected_output": "20"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 1

    def test_io_no_args(self, runner):
        code = "def ping(): return 'pong'"
        suite = {
            "tests": [
                {"id": "t1", "type": "io", "input": "", "expected_output": "pong"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 1


# ── Python: Assert tests ──────────────────────────────────────────────────────

class TestPythonAssert:

    def test_assert_passes(self, runner):
        code = "result = [1, 2, 3]"
        suite = {
            "tests": [
                {"id": "t1", "type": "assert", "expression": "len(result) == 3"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["passed"] == 1

    def test_assert_fails(self, runner):
        code = "result = []"
        suite = {
            "tests": [
                {"id": "t1", "type": "assert", "expression": "len(result) > 0"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["failed"] == 1


# ── Python: Security tests ────────────────────────────────────────────────────

class TestPythonSecurity:

    def test_network_blocked(self, runner):
        """Network access should be blocked."""
        code = textwrap.dedent("""
            import socket
            def attempt_connection():
                s = socket.socket()
                try:
                    s.connect(('8.8.8.8', 80))
                    return 'connected'
                except Exception as e:
                    return str(type(e).__name__)
        """).strip()
        suite = {
            "tests": [
                {
                    "id": "t1",
                    "type": "io",
                    "input": "",
                    "expected_output": "PermissionError",
                }
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        # Connection should be blocked → PermissionError
        result = r["results"][0]
        # Either blocked (PermissionError) or passed with PermissionError output
        assert (
            result["passed"]
            or "PermissionError" in str(result.get("output", ""))
            or "PermissionError" in str(result.get("error", ""))
        )

    def test_timeout_enforced(self, runner):
        """Infinite loop should be killed by timeout."""
        slow_runner = LightweightRunner(prefer_docker=False, timeout=2)
        code = "def infinite(): \n    while True: pass"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "infinite()", "expected_output": "never"},
            ]
        }
        r = slow_runner.run_test_suite(code, suite, "python")
        result = r["results"][0]
        assert not result["passed"]
        assert "timeout" in result.get("error", "").lower() or "Timeout" in result.get("error", "")

    def test_runtime_error_captured(self, runner):
        """Runtime errors are captured, not propagated."""
        code = "def crash(): raise ValueError('boom')"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "crash()", "expected_output": "never"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        result = r["results"][0]
        assert not result["passed"]
        assert result.get("error") is not None

    def test_syntax_error_captured(self, runner):
        """Syntax errors in user code are captured."""
        code = "def broken( return 42"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "broken()", "expected_output": "42"},
            ]
        }
        r = runner.run_test_suite(code, suite, "python")
        assert r["failed"] == 1


# ── JavaScript tests ──────────────────────────────────────────────────────────

class TestJavaScript:

    def test_js_expression(self, node_runner):
        code = "function add(a, b) { return a + b; }"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "add(2, 3)", "expected_output": "5"},
            ]
        }
        r = node_runner.run_test_suite(code, suite, "javascript")
        assert r["passed"] == 1
        assert r["runner"] == "lightweight"

    def test_js_io(self, node_runner):
        code = "function greet(name) { return 'Hello, ' + name + '!'; }"
        suite = {
            "tests": [
                {"id": "t1", "type": "io", "input": "World", "expected_output": "Hello, World!"},
            ]
        }
        r = node_runner.run_test_suite(code, suite, "javascript")
        assert r["passed"] == 1

    def test_js_failing(self, node_runner):
        code = "function add(a, b) { return a - b; }  // bug"
        suite = {
            "tests": [
                {"id": "t1", "type": "expression", "expression": "add(2, 3)", "expected_output": "5"},
            ]
        }
        r = node_runner.run_test_suite(code, suite, "javascript")
        assert r["failed"] == 1

    def test_js_assert(self, node_runner):
        code = "const result = [1, 2, 3];"
        suite = {
            "tests": [
                {"id": "t1", "type": "assert", "expression": "result.length === 3"},
            ]
        }
        r = node_runner.run_test_suite(code, suite, "javascript")
        assert r["passed"] == 1


# ── Capabilities ──────────────────────────────────────────────────────────────

class TestCapabilities:

    def test_capabilities_dict(self, runner):
        caps = runner.capabilities()
        assert caps["runner"] == "lightweight"
        assert caps["python"] is True
        assert "supported_languages" in caps

    def test_capabilities_docker_disabled(self, runner):
        caps = runner.capabilities()
        assert caps["docker_available"] is False
        assert caps["security_model"] == "subprocess"

    def test_result_has_runner_field(self, runner):
        code = "x = 1"
        suite = {"tests": [{"id": "t1", "type": "expression", "expression": "x", "expected_output": "1"}]}
        r = runner.run_test_suite(code, suite)
        assert r.get("runner") == "lightweight"

    def test_result_has_runtime_ms(self, runner):
        code = "x = 1"
        suite = {"tests": [{"id": "t1", "type": "expression", "expression": "x", "expected_output": "1"}]}
        r = runner.run_test_suite(code, suite)
        assert "runtime_ms" in r
        assert r["runtime_ms"] >= 0

    def test_unsupported_language(self, runner):
        code = "puts 'hello'"
        suite = {"tests": [{"id": "t1", "type": "expression", "expression": "code", "expected_output": "x"}]}
        r = runner.run_test_suite(code, suite, language="ruby")
        assert r["failed"] == 1
        assert "Unsupported" in r["results"][0].get("error", "")


import textwrap

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
