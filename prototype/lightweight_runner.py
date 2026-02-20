"""
lightweight_runner.py — Docker-free code execution backend for VRF verification.

Provides the same interface as docker_runner.py but without requiring Docker.
Uses subprocess isolation for Python and Node.js worker_threads for JavaScript.

Security model:
  - DEVELOPMENT/CI: subprocess with timeouts, output limits, no Docker required
  - PRODUCTION: use docker_runner.py for true sandboxing (namespaces + seccomp)

This module enables:
  1. Running verify_server without Docker installed (development, CI/CD)
  2. Faster test execution (no container startup overhead: <50ms vs ~1s)
  3. Fly.io / Cloudflare Workers deployment without Docker-in-Docker
  4. Future WASM-based execution (replace subprocess with WASM runtime)

Usage:
  runner = LightweightRunner(prefer_docker=False)
  results = runner.run_test_suite(code, test_suite, language="python")

Security tradeoffs vs Docker:
  - No Linux namespace isolation (filesystem, network, PID visible)
  - No seccomp-bpf filter
  - Memory limit via resource module (Unix only) or ulimit
  - Network blocked via socket override (Python only)
  - Recommend Docker in production; this module for dev/edge

Compatibility: drop-in replacement for docker_runner.py interface.
"""

import subprocess
import json
import os
import sys
import time
import tempfile
import textwrap
import resource
import signal
from typing import Optional

# -- Constants --
DEFAULT_TIMEOUT = 10      # seconds per test case
DEFAULT_MEMORY_MB = 128   # soft memory limit
MAX_OUTPUT = 4096         # bytes
PYTHON_EXEC = sys.executable


def _resource_limit_script(memory_mb: int) -> str:
    """
    Returns a Python preamble that sets resource limits and blocks network.
    Injected before user code in subprocess executions.
    """
    return textwrap.dedent(f"""
import resource as _res
import socket as _socket_mod

# Memory: set soft limit to {memory_mb}MB
_mem_bytes = {memory_mb} * 1024 * 1024
try:
    _soft, _hard = _res.getrlimit(_res.RLIMIT_AS)
    _res.setrlimit(_res.RLIMIT_AS, (_mem_bytes, max(_hard, _mem_bytes)))
except Exception:
    pass

# Block network: override socket to raise on connection
_original_socket_init = _socket_mod.socket.__init__
class _NoNetSocket(_socket_mod.socket):
    def connect(self, *a, **kw):
        raise PermissionError("Network access disabled in sandbox")
    def connect_ex(self, *a, **kw):
        raise PermissionError("Network access disabled in sandbox")
    def bind(self, *a, **kw):
        raise PermissionError("Network access disabled in sandbox")
_socket_mod.socket = _NoNetSocket

""").lstrip()


def _build_python_test_runner(code: str, test_case: dict, memory_mb: int) -> str:
    """
    Build a Python script that:
    1. Applies resource limits and network block
    2. Defines the user's code
    3. Runs one test case
    4. Outputs JSON result to stdout
    """
    preamble = _resource_limit_script(memory_mb)

    test_type = test_case.get("type", "expression")
    test_input = test_case.get("input", "")
    expected = test_case.get("expected_output", "")
    expression = test_case.get("expression", "")
    test_id = test_case.get("id", "test")

    runner = textwrap.dedent(f"""
import json as _json
import sys as _sys
import traceback as _tb

_result = {{"id": {json.dumps(test_id)}, "passed": False, "output": "", "error": None}}

try:
    # User code namespace
    _ns = {{}}
    exec({json.dumps(code)}, _ns)

    test_type = {json.dumps(test_type)}
    
    if test_type == "expression":
        _expr = {json.dumps(expression)}
        _output = str(eval(_expr, _ns)).strip()
        _expected = str({json.dumps(expected)}).strip()
        _result["output"] = _output
        _result["passed"] = _output == _expected

    elif test_type == "io":
        # Find a callable with test_input as argument
        _test_input = {json.dumps(test_input)}
        _expected = {json.dumps(expected)}
        # Try calling the first function defined in user code
        _funcs = [v for k, v in _ns.items() if callable(v) and not k.startswith("_")]
        if not _funcs:
            raise ValueError("No callable found in user code")
        _fn = _funcs[0]
        if isinstance(_test_input, list):
            _output = str(_fn(*_test_input)).strip()
        elif _test_input == "":
            _output = str(_fn()).strip()
        else:
            _output = str(_fn(_test_input)).strip()
        _result["output"] = _output
        _result["passed"] = _output == str(_expected).strip()

    elif test_type == "assert":
        _assert_expr = {json.dumps(expression or expected)}
        _assert_result = eval(compile(_assert_expr, "<assert>", "eval"), _ns)
        if not _assert_result:
            raise AssertionError(f"Expression evaluated to falsy: {{_assert_result!r}}")
        _result["passed"] = True
        _result["output"] = "assertion passed"

    else:
        raise ValueError(f"Unknown test type: {{test_type}}")

except AssertionError as e:
    _result["passed"] = False
    _result["error"] = f"AssertionError: {{e}}"
except Exception as e:
    _result["passed"] = False
    _result["error"] = _tb.format_exc(limit=3)

print(_json.dumps(_result))
""").lstrip()

    return preamble + runner


def _build_js_runner_script(code: str, test_case: dict) -> str:
    """
    Build a Node.js script that runs one test case using vm module for isolation.
    Node.js vm provides a separate V8 context, blocking module access.
    """
    test_type = test_case.get("type", "expression")
    test_input = test_case.get("input", "")
    expected = test_case.get("expected_output", "")
    expression = test_case.get("expression", "")
    test_id = test_case.get("id", "test")

    # We use vm.runInNewContext to isolate code from Node globals
    script = textwrap.dedent(f"""
const vm = require('vm');

const result = {{
  id: {json.dumps(test_id)},
  passed: false,
  output: '',
  error: null
}};

try {{
  const ctx = vm.createContext({{
    console: {{
      log: (...args) => {{ ctx.__stdout = (ctx.__stdout || '') + args.join(' ') + '\\n'; }},
      error: (...args) => {{ ctx.__stderr = (ctx.__stderr || '') + args.join(' ') + '\\n'; }}
    }},
    JSON, Math, parseInt, parseFloat, String, Number, Boolean, Array, Object,
    Error, TypeError, ValueError: class extends Error {{}},
    __stdout: '', __stderr: ''
  }});

  // Snapshot keys before running user code
  const _preKeys = new Set(Object.keys(ctx));

  const userCode = {json.dumps(code)};
  vm.runInContext(userCode, ctx, {{ timeout: 5000 }});

  // Find functions ADDED by user code (not pre-existing context keys)
  const _userKeys = Object.keys(ctx).filter(k => !_preKeys.has(k) && typeof ctx[k] === 'function');

  const testType = {json.dumps(test_type)};
  const expected = {json.dumps(expected)};

  if (testType === 'expression') {{
    const expr = {json.dumps(expression)};
    const output = String(vm.runInContext(expr, ctx, {{ timeout: 2000 }})).trim();
    result.output = output;
    result.passed = output === String(expected).trim();
  }} else if (testType === 'io') {{
    const testInput = {json.dumps(test_input)};
    // Prefer user-defined functions; fall back to any callable if expression-style
    const funcs = _userKeys.length ? _userKeys :
      Object.keys(ctx).filter(k => typeof ctx[k] === 'function' && !k.startsWith('_') &&
        !['parseInt','parseFloat','String','Number','Boolean','Array','Object','Error','TypeError','isFinite','isNaN'].includes(k));
    if (!funcs.length) throw new Error('No callable found in user code');
    const fn = ctx[funcs[0]];
    const args = Array.isArray(testInput) ? testInput : (testInput === '' ? [] : [testInput]);
    const output = String(fn(...args)).trim();
    result.output = output;
    result.passed = output === String(expected).trim();
  }} else if (testType === 'assert') {{
    const expr = {json.dumps(expression or expected)};
    const val = vm.runInContext(expr, ctx, {{ timeout: 2000 }});
    result.passed = !!val;
    result.output = 'assertion ' + (val ? 'passed' : 'failed');
  }}
}} catch(e) {{
  result.error = e.stack || e.message || String(e);
  result.passed = false;
}}

console.log(JSON.stringify(result));
""").lstrip()
    return script


class LightweightRunner:
    """
    Docker-free code execution backend for VRF verification.

    Provides the same interface as docker_runner.DockerRunner but uses
    subprocess isolation instead of containers.

    For production workloads handling untrusted code, use docker_runner.py.
    This runner is designed for:
      - Development and CI environments
      - Edge deployments (Fly.io, Cloudflare Workers via WASM future)
      - Trusted-code verification (framework integrations, PyPI packages)
    """

    def __init__(
        self,
        timeout: int = DEFAULT_TIMEOUT,
        memory_mb: int = DEFAULT_MEMORY_MB,
        prefer_docker: bool = True,
        node_path: Optional[str] = None,
        python_path: Optional[str] = None,
    ):
        self.timeout = timeout
        self.memory_mb = memory_mb
        self.python_path = python_path or PYTHON_EXEC
        self.node_path = node_path or _find_node()

        # Try docker first if preferred
        self.docker_available = prefer_docker and _check_docker()
        if self.docker_available:
            try:
                from prototype.docker_runner import DockerRunner
                self._docker = DockerRunner(timeout=timeout, memory_mb=memory_mb)
            except ImportError:
                try:
                    # Try relative import
                    sys.path.insert(0, os.path.dirname(__file__))
                    from docker_runner import DockerRunner
                    self._docker = DockerRunner(timeout=timeout, memory_mb=memory_mb)
                except ImportError:
                    self.docker_available = False
                    self._docker = None
        else:
            self._docker = None

    def run_test_suite(self, code: str, test_suite: dict, language: str = "python") -> dict:
        """
        Run a complete test suite against code.
        
        Returns dict matching docker_runner format:
        {
          "passed": int,
          "failed": int, 
          "total": int,
          "results": [...],
          "runtime_ms": float,
          "runner": "lightweight" | "docker",
          "language": str
        }
        """
        # Delegate to Docker if available and preferred
        if self.docker_available and self._docker:
            try:
                result = self._docker.run_test_suite(code, test_suite, language)
                result["runner"] = "docker"
                return result
            except Exception as e:
                # Fall through to lightweight on Docker failure
                pass

        start_ms = time.monotonic() * 1000
        tests = test_suite.get("tests", [])
        results = []

        for test in tests:
            r = self._run_one_test(code, test, language)
            results.append(r)

        elapsed = time.monotonic() * 1000 - start_ms
        passed = sum(1 for r in results if r.get("passed", False))

        return {
            "passed": passed,
            "failed": len(results) - passed,
            "total": len(results),
            "results": results,
            "runtime_ms": round(elapsed, 1),
            "runner": "lightweight",
            "language": language,
        }

    def _run_one_test(self, code: str, test_case: dict, language: str) -> dict:
        """Run a single test case, returning result dict."""
        if language in ("python", "py"):
            return self._run_python_test(code, test_case)
        elif language in ("javascript", "js", "node"):
            return self._run_js_test(code, test_case)
        else:
            return {
                "id": test_case.get("id", "test"),
                "passed": False,
                "output": "",
                "error": f"Unsupported language: {language}. Supported: python, javascript",
            }

    def _run_python_test(self, code: str, test_case: dict) -> dict:
        """Run one Python test case in a sandboxed subprocess."""
        script = _build_python_test_runner(code, test_case, self.memory_mb)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script)
            tmpfile = f.name

        try:
            proc = subprocess.run(
                [self.python_path, tmpfile],
                capture_output=True,
                text=True,
                timeout=self.timeout,
                env={**os.environ, "PYTHONPATH": ""},  # clean pythonpath
            )
            stdout = proc.stdout.strip()
            if not stdout:
                return {
                    "id": test_case.get("id", "test"),
                    "passed": False,
                    "output": "",
                    "error": proc.stderr[:MAX_OUTPUT] if proc.stderr else "No output",
                }
            try:
                return json.loads(stdout.split("\n")[-1])
            except json.JSONDecodeError:
                return {
                    "id": test_case.get("id", "test"),
                    "passed": False,
                    "output": stdout[:MAX_OUTPUT],
                    "error": "JSON parse error on runner output",
                }
        except subprocess.TimeoutExpired:
            return {
                "id": test_case.get("id", "test"),
                "passed": False,
                "output": "",
                "error": f"Timeout ({self.timeout}s)",
            }
        except Exception as e:
            return {
                "id": test_case.get("id", "test"),
                "passed": False,
                "output": "",
                "error": str(e),
            }
        finally:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass

    def _run_js_test(self, code: str, test_case: dict) -> dict:
        """Run one JavaScript test case using Node.js vm module."""
        if not self.node_path:
            return {
                "id": test_case.get("id", "test"),
                "passed": False,
                "output": "",
                "error": "Node.js not found. Install Node.js to run JavaScript tests.",
            }

        script = _build_js_runner_script(code, test_case)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".js", delete=False) as f:
            f.write(script)
            tmpfile = f.name

        try:
            proc = subprocess.run(
                [self.node_path, tmpfile],
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = proc.stdout.strip()
            if not stdout:
                return {
                    "id": test_case.get("id", "test"),
                    "passed": False,
                    "output": "",
                    "error": proc.stderr[:MAX_OUTPUT] if proc.stderr else "No output",
                }
            try:
                return json.loads(stdout.split("\n")[-1])
            except json.JSONDecodeError:
                return {
                    "id": test_case.get("id", "test"),
                    "passed": False,
                    "output": stdout[:MAX_OUTPUT],
                    "error": "JSON parse error on runner output",
                }
        except subprocess.TimeoutExpired:
            return {
                "id": test_case.get("id", "test"),
                "passed": False,
                "output": "",
                "error": f"Timeout ({self.timeout}s)",
            }
        except Exception as e:
            return {
                "id": test_case.get("id", "test"),
                "passed": False,
                "output": "",
                "error": str(e),
            }
        finally:
            try:
                os.unlink(tmpfile)
            except Exception:
                pass

    def capabilities(self) -> dict:
        """Return runner capabilities (mirrors docker_runner interface)."""
        return {
            "runner": "lightweight",
            "docker_available": self.docker_available,
            "python": True,
            "javascript": self.node_path is not None,
            "node_path": self.node_path,
            "python_path": self.python_path,
            "security_model": "subprocess" if not self.docker_available else "docker",
            "wasm_ready": False,  # Future: WASM-based execution
            "supported_languages": (
                ["python", "javascript"] if self.node_path else ["python"]
            ),
        }


# -- Helpers --

def _find_node() -> Optional[str]:
    """Find Node.js executable."""
    candidates = [
        os.environ.get("NODE_PATH"),
        "/home/dexin/.nvm/versions/node/v22.22.0/bin/node",
        "/usr/local/bin/node",
        "/usr/bin/node",
    ]
    for path in candidates:
        if path and os.path.isfile(path):
            return path
    # Try which
    try:
        result = subprocess.run(
            ["which", "node"], capture_output=True, text=True, timeout=2
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except Exception:
        pass
    return None


def _check_docker() -> bool:
    """Check if Docker daemon is available."""
    try:
        result = subprocess.run(
            ["docker", "info"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


# -- CLI --

def main():
    """CLI interface for lightweight_runner."""
    import argparse

    parser = argparse.ArgumentParser(description="Lightweight VRF code execution runner")
    parser.add_argument("--code", required=True, help="Python or JS code to test")
    parser.add_argument("--test-suite", required=True, help="JSON test suite")
    parser.add_argument("--language", default="python", choices=["python", "javascript"])
    parser.add_argument("--no-docker", action="store_true", help="Disable Docker even if available")
    parser.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT)
    args = parser.parse_args()

    try:
        test_suite = json.loads(args.test_suite)
    except json.JSONDecodeError as e:
        print(f"Invalid test suite JSON: {e}", file=sys.stderr)
        sys.exit(1)

    runner = LightweightRunner(
        prefer_docker=not args.no_docker,
        timeout=args.timeout,
    )
    result = runner.run_test_suite(args.code, test_suite, args.language)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
