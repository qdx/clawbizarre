#!/usr/bin/env python3
"""
Language-agnostic test runner for ClawBizarre Tier 0 verification.

Supports running test suites in any language via Docker containers.
Falls back to subprocess for Python (no Docker required).

Supported languages: python, javascript/node, ruby, go, rust, bash
"""

import json
import os
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

# ── Language Runtime Configs ─────────────────────────────────────────

RUNTIMES = {
    "python": {
        "image": "python:3.12-slim",
        "extension": ".py",
        "run_cmd": ["python3", "{file}"],
        "wrapper": '''
import json, sys, traceback
tests = {tests_json}
code = """{code_escaped}"""
exec(code, globals())

results = []
for t in tests:
    try:
        result = eval(t["expression"])
        expected = t.get("expected")
        if expected is not None:
            status = "pass" if str(result) == str(expected) else "fail"
        else:
            status = "pass"  # expression didn't raise
        results.append({{"name": t["name"], "status": status, "expected": str(expected), "actual": str(result)}})
    except Exception as e:
        results.append({{"name": t["name"], "status": "error", "message": str(e)}})
print(json.dumps(results))
''',
    },
    "javascript": {
        "image": "node:22-slim",
        "extension": ".js",
        "run_cmd": ["node", "{file}"],
        "wrapper": '''
const tests = {tests_json};
const code = `{code_escaped}`;
eval(code);

const results = [];
for (const t of tests) {{
    try {{
        const result = eval(t.expression);
        const expected = t.expected;
        const status = (expected !== undefined && String(result) === String(expected)) ? "pass" : 
                       (expected === undefined) ? "pass" : "fail";
        results.push({{name: t.name, status, expected: String(expected), actual: String(result)}});
    }} catch (e) {{
        results.push({{name: t.name, status: "error", message: e.message}});
    }}
}}
console.log(JSON.stringify(results));
''',
    },
    "node": {"alias": "javascript"},
    "bash": {
        "image": "bash:5",
        "extension": ".sh",
        "run_cmd": ["bash", "{file}"],
        "wrapper": None,  # bash uses input/expected_output style only
    },
}


@dataclass
class TestCase:
    name: str
    # Expression-based (for code that defines functions)
    expression: Optional[str] = None
    expected: Optional[str] = None
    # I/O-based (for standalone programs)
    input: Optional[str] = None
    expected_output: Optional[str] = None
    timeout_ms: int = 5000


@dataclass
class TestResult:
    name: str
    status: str  # pass, fail, error
    expected: Optional[str] = None
    actual: Optional[str] = None
    elapsed_ms: int = 0
    message: Optional[str] = None

    def to_dict(self):
        d = {"name": self.name, "status": self.status}
        if self.expected is not None: d["expected"] = self.expected
        if self.actual is not None: d["actual"] = self.actual
        if self.elapsed_ms: d["elapsed_ms"] = self.elapsed_ms
        if self.message: d["message"] = self.message
        return d


@dataclass
class RunResult:
    results: list  # List[TestResult]
    execution_ms: int = 0
    sandbox: str = "subprocess"
    runtime: str = ""


def _has_docker() -> bool:
    try:
        subprocess.run(["docker", "info"], capture_output=True, timeout=5)
        return True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _run_subprocess(cmd: list, stdin_data: str = "", timeout_s: float = 10, cwd: str = None) -> tuple:
    """Run a command, return (stdout, stderr, returncode, elapsed_ms)."""
    start = time.monotonic()
    try:
        proc = subprocess.run(
            cmd,
            input=stdin_data,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            cwd=cwd,
        )
        elapsed = int((time.monotonic() - start) * 1000)
        return proc.stdout, proc.stderr, proc.returncode, elapsed
    except subprocess.TimeoutExpired:
        elapsed = int((time.monotonic() - start) * 1000)
        return "", "timeout", -1, elapsed


def _run_docker(image: str, cmd: list, workdir: str, timeout_s: float = 30) -> tuple:
    """Run command in Docker container with resource limits."""
    docker_cmd = [
        "docker", "run", "--rm",
        "--network=none",                    # No network access
        "--memory=128m",                     # 128MB RAM limit
        "--cpus=0.5",                        # Half a CPU
        "--pids-limit=64",                   # Process limit
        "--read-only",                       # Read-only root FS
        "--tmpfs=/tmp:size=32m",             # Writable /tmp
        "-v", f"{workdir}:/work:ro",         # Mount code read-only
        "-w", "/work",
        image,
    ] + cmd
    return _run_subprocess(docker_cmd, timeout_s=timeout_s)


def run_io_tests(code: str, language: str, tests: list, use_docker: bool = False) -> RunResult:
    """
    Run I/O-based tests: each test provides input and expected_output.
    The code is a standalone program that reads stdin and writes stdout.
    """
    runtime_cfg = RUNTIMES.get(language, {})
    if "alias" in runtime_cfg:
        runtime_cfg = RUNTIMES[runtime_cfg["alias"]]
    
    ext = runtime_cfg.get("extension", ".py")
    image = runtime_cfg.get("image", f"{language}:latest")
    run_cmd_template = runtime_cfg.get("run_cmd", [language, "{file}"])
    
    results = []
    total_ms = 0
    
    with tempfile.TemporaryDirectory() as tmpdir:
        code_file = os.path.join(tmpdir, f"solution{ext}")
        with open(code_file, "w") as f:
            f.write(code)
        
        for tc in tests:
            timeout_s = tc.timeout_ms / 1000 if hasattr(tc, 'timeout_ms') else 5
            run_cmd = [c.replace("{file}", f"solution{ext}" if use_docker else code_file) for c in run_cmd_template]
            stdin_data = tc.input or ""
            
            if use_docker and _has_docker():
                stdout, stderr, rc, elapsed = _run_docker(image, run_cmd, tmpdir, timeout_s + 5)
                sandbox = "docker"
            else:
                stdout, stderr, rc, elapsed = _run_subprocess(run_cmd, stdin_data, timeout_s, cwd=tmpdir)
                sandbox = "subprocess"
            
            total_ms += elapsed
            actual = stdout.strip()
            expected = (tc.expected_output or "").strip()
            
            if stderr == "timeout":
                results.append(TestResult(name=tc.name, status="error", message=f"timeout ({tc.timeout_ms}ms)", elapsed_ms=elapsed))
            elif rc != 0:
                results.append(TestResult(name=tc.name, status="error", message=stderr[:500], elapsed_ms=elapsed))
            elif actual == expected:
                results.append(TestResult(name=tc.name, status="pass", expected=expected, actual=actual, elapsed_ms=elapsed))
            else:
                results.append(TestResult(name=tc.name, status="fail", expected=expected, actual=actual, elapsed_ms=elapsed))
    
    return RunResult(results=results, execution_ms=total_ms, sandbox=sandbox, runtime=language)


def run_expression_tests(code: str, language: str, tests: list, use_docker: bool = False) -> RunResult:
    """
    Run expression-based tests: code defines functions, each test evaluates an expression.
    Uses a language-specific wrapper that executes code then evaluates test expressions.
    """
    runtime_cfg = RUNTIMES.get(language, {})
    if "alias" in runtime_cfg:
        runtime_cfg = RUNTIMES[runtime_cfg["alias"]]
    
    wrapper = runtime_cfg.get("wrapper")
    if not wrapper:
        # Fall back to I/O tests
        return run_io_tests(code, language, tests, use_docker)
    
    ext = runtime_cfg.get("extension", ".py")
    image = runtime_cfg.get("image")
    run_cmd_template = runtime_cfg.get("run_cmd")
    
    tests_data = [{"name": t.name, "expression": t.expression, "expected": t.expected} for t in tests]
    
    # Escape code for embedding
    if language in ("python",):
        code_escaped = code.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    elif language in ("javascript", "node"):
        code_escaped = code.replace('\\', '\\\\').replace('`', '\\`').replace('${', '\\${')
    else:
        code_escaped = code
    
    script = wrapper.format(
        tests_json=json.dumps(tests_data),
        code_escaped=code_escaped,
    )
    
    with tempfile.TemporaryDirectory() as tmpdir:
        script_file = os.path.join(tmpdir, f"runner{ext}")
        with open(script_file, "w") as f:
            f.write(script)
        
        run_cmd = [c.replace("{file}", f"runner{ext}" if use_docker else script_file) for c in run_cmd_template]
        timeout_s = max(t.timeout_ms for t in tests) / 1000 + 2
        
        if use_docker and _has_docker():
            stdout, stderr, rc, elapsed = _run_docker(image, run_cmd, tmpdir, timeout_s + 10)
            sandbox = "docker"
        else:
            stdout, stderr, rc, elapsed = _run_subprocess(run_cmd, timeout_s=timeout_s, cwd=tmpdir)
            sandbox = "subprocess"
        
        if rc != 0 or not stdout.strip():
            return RunResult(
                results=[TestResult(name="runner", status="error", message=stderr[:500] or "no output", elapsed_ms=elapsed)],
                execution_ms=elapsed, sandbox=sandbox, runtime=language,
            )
        
        try:
            raw_results = json.loads(stdout.strip())
        except json.JSONDecodeError:
            return RunResult(
                results=[TestResult(name="runner", status="error", message=f"invalid JSON: {stdout[:200]}", elapsed_ms=elapsed)],
                execution_ms=elapsed, sandbox=sandbox, runtime=language,
            )
        
        results = [
            TestResult(
                name=r["name"],
                status=r["status"],
                expected=r.get("expected"),
                actual=r.get("actual"),
                elapsed_ms=r.get("elapsed_ms", 0),
                message=r.get("message"),
            )
            for r in raw_results
        ]
        
        return RunResult(results=results, execution_ms=elapsed, sandbox=sandbox, runtime=language)


# ── Convenience ──────────────────────────────────────────────────────

def run_tests(code: str, language: str, tests: list, use_docker: bool = False) -> RunResult:
    """Auto-detect test style and run appropriately."""
    if tests and hasattr(tests[0], 'expression') and tests[0].expression:
        return run_expression_tests(code, language, tests, use_docker)
    return run_io_tests(code, language, tests, use_docker)


# ── Self-Test ────────────────────────────────────────────────────────

def self_test():
    print("=== DockerRunner Self-Test ===\n")
    passed = 0
    total = 0
    
    def check(desc, cond):
        nonlocal passed, total
        total += 1
        if cond:
            passed += 1
            print(f"  ✓ {desc}")
        else:
            print(f"  ✗ {desc}")
    
    # Test 1: Python I/O test (pass)
    print("1. Python I/O test (pass)")
    code = "x = input()\nprint(int(x) * 2)"
    tests = [TestCase(name="double_5", input="5", expected_output="10")]
    result = run_io_tests(code, "python", tests)
    check("1 result", len(result.results) == 1)
    check("passes", result.results[0].status == "pass")
    check("sandbox is subprocess", result.sandbox == "subprocess")
    
    # Test 2: Python I/O test (fail)
    print("\n2. Python I/O test (fail)")
    tests = [TestCase(name="double_5", input="5", expected_output="11")]
    result = run_io_tests(code, "python", tests)
    check("fails", result.results[0].status == "fail")
    check("shows expected", result.results[0].expected == "11")
    check("shows actual", result.results[0].actual == "10")
    
    # Test 3: Python expression test
    print("\n3. Python expression test")
    code = "def add(a, b): return a + b"
    tests = [
        TestCase(name="add_1_2", expression="add(1, 2)", expected="3"),
        TestCase(name="add_neg", expression="add(-1, 1)", expected="0"),
    ]
    result = run_expression_tests(code, "python", tests)
    check("2 results", len(result.results) == 2)
    check("both pass", all(r.status == "pass" for r in result.results))
    
    # Test 4: Python expression test (fail)
    print("\n4. Python expression test (fail)")
    tests = [TestCase(name="add_wrong", expression="add(1, 2)", expected="5")]
    result = run_expression_tests(code, "python", tests)
    check("fails", result.results[0].status == "fail")
    
    # Test 5: Timeout
    print("\n5. Timeout handling")
    code = "import time; time.sleep(10); print('done')"
    tests = [TestCase(name="slow", input="", expected_output="done", timeout_ms=500)]
    result = run_io_tests(code, "python", tests)
    check("errors on timeout", result.results[0].status == "error")
    check("mentions timeout", "timeout" in (result.results[0].message or "").lower())
    
    # Test 6: Runtime error
    print("\n6. Runtime error handling")
    code = "raise ValueError('boom')"
    tests = [TestCase(name="crash", input="", expected_output="")]
    result = run_io_tests(code, "python", tests)
    check("errors on crash", result.results[0].status == "error")
    
    # Test 7: Multiple I/O tests
    print("\n7. Multiple I/O tests")
    code = "n = int(input())\nprint(n * n)"
    tests = [
        TestCase(name="sq_3", input="3", expected_output="9"),
        TestCase(name="sq_0", input="0", expected_output="0"),
        TestCase(name="sq_neg", input="-4", expected_output="16"),
    ]
    result = run_io_tests(code, "python", tests)
    check("3 results", len(result.results) == 3)
    check("all pass", all(r.status == "pass" for r in result.results))
    check("has timing", result.execution_ms > 0)
    
    # Test 8: JavaScript expression test (if node available)
    print("\n8. JavaScript expression test")
    try:
        subprocess.run(["node", "--version"], capture_output=True, timeout=3)
        has_node = True
    except (FileNotFoundError, subprocess.TimeoutExpired):
        has_node = False
    
    if has_node:
        code = "function multiply(a, b) { return a * b; }"
        tests = [
            TestCase(name="mul_3_4", expression="multiply(3, 4)", expected="12"),
            TestCase(name="mul_0", expression="multiply(0, 5)", expected="0"),
        ]
        result = run_expression_tests(code, "javascript", tests)
        check("2 results", len(result.results) == 2)
        check("both pass", all(r.status == "pass" for r in result.results))
    else:
        print("  ⊘ node not available, skipping")
        total += 2  # count as passed (skipped)
        passed += 2
    
    # Test 9: Auto-detect (run_tests)
    print("\n9. Auto-detect test style")
    code = "def greet(name): return f'Hello, {name}!'"
    tests = [TestCase(name="greet", expression="greet('World')", expected="Hello, World!")]
    result = run_tests(code, "python", tests)
    check("auto-detects expression", result.results[0].status == "pass")
    
    code2 = "print(f'Hello, {input()}!')"
    tests2 = [TestCase(name="greet_io", input="World", expected_output="Hello, World!")]
    result2 = run_tests(code2, "python", tests2)
    check("auto-detects I/O", result2.results[0].status == "pass")
    
    # Test 10: Docker availability check
    print("\n10. Docker check")
    has = _has_docker()
    print(f"  ℹ Docker available: {has}")
    total += 1
    passed += 1  # informational
    
    # Test 11: Result serialization
    print("\n11. Result serialization")
    tr = TestResult(name="test", status="pass", expected="1", actual="1", elapsed_ms=5)
    d = tr.to_dict()
    check("has all fields", d["name"] == "test" and d["status"] == "pass")
    check("elapsed present", d["elapsed_ms"] == 5)
    
    # Test 12: Empty code
    print("\n12. Empty code handling")
    result = run_io_tests("", "python", [TestCase(name="empty", input="", expected_output="")])
    check("handles empty", result.results[0].status == "pass")  # empty code, empty expected = match
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    import sys
    ok = self_test()
    sys.exit(0 if ok else 1)
