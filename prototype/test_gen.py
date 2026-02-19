"""
VRF Auto-Test Generation Service
Converts natural language task descriptions into deterministic test suites.

Usage:
    python3 test_gen.py [--port 8701]
    python3 test_gen.py --test
    
    # One-shot CLI:
    python3 test_gen.py --generate "Write a function that reverses a string"
"""

import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional, Any
from urllib.request import Request, urlopen
from urllib.error import URLError

# ── Config ──────────────────────────────────────────────────────────

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
OPENAI_MODEL = os.environ.get("VRF_TESTGEN_MODEL", "gpt-4o-mini")
VERIFY_SERVER_URL = os.environ.get("VRF_VERIFY_URL", "http://localhost:8700")

# ── Coverage Templates ──────────────────────────────────────────────

COVERAGE_SPECS = {
    "basic": {
        "test_count": "3-5",
        "categories": "happy path, empty/null input, one edge case",
    },
    "standard": {
        "test_count": "8-12",
        "categories": "happy path, empty/null, boundary values, type edge cases, error handling, performance hint",
    },
    "thorough": {
        "test_count": "15-25",
        "categories": "happy path, empty/null, boundary, type errors, large input, adversarial input, unicode, negative numbers, duplicates, ordering, idempotency",
    },
}

# ── Domain Templates ────────────────────────────────────────────────

DOMAIN_TEMPLATES = {
    "function": {
        "hint": "Test the function with various inputs and assert exact outputs.",
        "required_categories": ["happy_path", "empty_input", "edge_case"],
    },
    "api_endpoint": {
        "hint": "Test HTTP status codes, response shapes, and error handling.",
        "required_categories": ["success_response", "invalid_input", "missing_fields"],
    },
    "data_transform": {
        "hint": "Test with sample data, empty data, malformed data, and large data.",
        "required_categories": ["normal_transform", "empty_data", "malformed_data"],
    },
    "cli_tool": {
        "hint": "Test with valid args, missing args, invalid args. Check exit codes and stdout.",
        "required_categories": ["valid_usage", "missing_args", "invalid_args"],
    },
}

# ── Prompt Construction ─────────────────────────────────────────────

SYSTEM_PROMPT = """You are a test suite generator for the VRF (Verification Receipt Format) protocol.
Your job: convert a task description into concrete, deterministic test cases.

RULES:
1. Every test MUST have concrete input and exact expected output — no approximations, no "should contain", no randomness.
2. Tests must be deterministic: same input → same output, every time. No time-dependent, no random, no network calls.
3. Use the EXACT output format specified. Do not add commentary outside the JSON.
4. For code tasks: generate expression tests (eval an expression, check result) and I/O tests (stdin→stdout).
5. For functions: call the function with specific args, assert exact return value.
6. Think about: happy path, empty input, boundary values, type edge cases.
7. Do NOT test implementation details — only test the observable contract.
8. If the task is ambiguous, pick the most common interpretation and note it in test descriptions."""

def build_generation_prompt(task_description: str, language: str, coverage: str,
                           constraints: list[str] | None = None,
                           template: str | None = None) -> str:
    spec = COVERAGE_SPECS.get(coverage, COVERAGE_SPECS["basic"])
    
    prompt = f"""Generate a VRF test suite for this task.

TASK: {task_description}
LANGUAGE: {language}
COVERAGE: {coverage} ({spec['test_count']} tests covering: {spec['categories']})
"""
    if constraints:
        prompt += f"CONSTRAINTS: {', '.join(constraints)}\n"
    
    if template and template in DOMAIN_TEMPLATES:
        t = DOMAIN_TEMPLATES[template]
        prompt += f"DOMAIN: {template} — {t['hint']}\n"
        prompt += f"REQUIRED CATEGORIES: {', '.join(t['required_categories'])}\n"

    prompt += """
OUTPUT FORMAT (JSON only, no markdown fences):
{
  "function_name": "the_function_name_to_test",
  "tests": [
    {
      "input": "<expression or value to pass>",
      "expected_output": "<exact expected result>",
      "description": "<what this test checks>",
      "category": "<happy_path|empty_input|edge_case|boundary|error|performance>"
    }
  ],
  "notes": "<any assumptions made about ambiguous requirements>"
}

For expression tests: "input" is a Python/JS expression calling the function. "expected_output" is the exact return value as a JSON literal.
For I/O tests: "input" is stdin text, "expected_output" is exact stdout text.

Generate ONLY the JSON. No explanation before or after."""
    
    return prompt


# ── LLM Integration ─────────────────────────────────────────────────

def call_openai(system: str, user: str, model: str = OPENAI_MODEL) -> str:
    """Call OpenAI chat completions API. Returns response text."""
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY not set")
    
    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.2,  # Low temp for deterministic test generation
        "max_tokens": 4096,
    }).encode()
    
    req = Request(
        "https://api.openai.com/v1/chat/completions",
        data=body,
        headers={
            "Authorization": f"Bearer {OPENAI_API_KEY}",
            "Content-Type": "application/json",
        },
    )
    
    with urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read())
    
    return data["choices"][0]["message"]["content"]


# ── Test Suite Generation ───────────────────────────────────────────

@dataclass
class GeneratedTestSuite:
    test_suite: dict
    task_description: str
    language: str
    coverage: str
    generated_by: str = "vrf-test-gen/0.1"
    generation_id: str = field(default_factory=lambda: str(uuid.uuid4())[:12])
    generated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    token_estimate: int = 0
    review_suggested: bool = True

    def to_vrf_format(self) -> dict:
        """Convert to VRF verify_server compatible test_suite format."""
        tests = self.test_suite.get("tests", [])
        function_name = self.test_suite.get("function_name", "solution")
        
        vrf_tests = []
        for t in tests:
            vrf_tests.append({
                "input": str(t.get("input", "")),
                "expected_output": str(t.get("expected_output", "")),
                "description": t.get("description", ""),
            })
        
        return {
            "tests": vrf_tests,
            "language": self.language,
            "generated_by": self.generated_by,
            "generation_id": self.generation_id,
            "function_name": function_name,
        }


def generate_test_suite(task_description: str, language: str = "python",
                        coverage: str = "basic", constraints: list[str] | None = None,
                        template: str | None = None) -> GeneratedTestSuite:
    """Generate a VRF-compatible test suite from a task description."""
    
    prompt = build_generation_prompt(task_description, language, coverage, constraints, template)
    raw = call_openai(SYSTEM_PROMPT, prompt)
    
    # Parse JSON from response (strip markdown fences if present)
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```\w*\n?", "", cleaned)
        cleaned = re.sub(r"\n?```$", "", cleaned)
        cleaned = cleaned.strip()
    
    suite = json.loads(cleaned)
    
    # Validate structure
    if "tests" not in suite or not isinstance(suite["tests"], list):
        raise ValueError("Generated suite missing 'tests' array")
    
    for i, t in enumerate(suite["tests"]):
        if "input" not in t or "expected_output" not in t:
            raise ValueError(f"Test {i} missing input or expected_output")
    
    return GeneratedTestSuite(
        test_suite=suite,
        task_description=task_description,
        language=language,
        coverage=coverage,
        token_estimate=len(prompt.split()) + len(raw.split()),
        review_suggested=len(suite["tests"]) < 5 or coverage == "basic",
    )


# ── Verify Integration ──────────────────────────────────────────────

def verify_with_generated_tests(code: str, test_suite: GeneratedTestSuite,
                                verify_url: str = VERIFY_SERVER_URL) -> dict:
    """Submit code + generated tests to verify_server. Returns VRF receipt."""
    vrf_suite = test_suite.to_vrf_format()
    
    body = json.dumps({
        "output": {"content": code},
        "verification": {"test_suite": vrf_suite},
        "specification": {
            "test_generation_id": test_suite.generation_id,
            "generated_by": test_suite.generated_by,
            "task_description": test_suite.task_description,
        },
    }).encode()
    
    req = Request(
        f"{verify_url}/verify",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    
    with urlopen(req, timeout=30) as resp:
        return json.loads(resp.read())


# ── HTTP Server ─────────────────────────────────────────────────────

class TestGenHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):
        pass  # Suppress default logging
    
    def _json_response(self, status: int, data: dict):
        body = json.dumps(data, indent=2).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
    
    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(length)) if length else {}
    
    def do_GET(self):
        if self.path == "/health":
            self._json_response(200, {
                "status": "ok",
                "service": "vrf-test-gen",
                "version": "0.1",
                "model": OPENAI_MODEL,
                "has_api_key": bool(OPENAI_API_KEY),
                "templates": list(DOMAIN_TEMPLATES.keys()),
                "coverage_levels": list(COVERAGE_SPECS.keys()),
            })
        else:
            self._json_response(404, {"error": "not found"})
    
    def do_POST(self):
        if self.path == "/test-gen":
            self._handle_test_gen()
        elif self.path == "/test-gen/verify":
            self._handle_gen_and_verify()
        else:
            self._json_response(404, {"error": "not found"})
    
    def _handle_test_gen(self):
        try:
            body = self._read_body()
            task = body.get("task_description", "")
            if not task:
                self._json_response(400, {"error": "task_description required"})
                return
            
            result = generate_test_suite(
                task_description=task,
                language=body.get("language", "python"),
                coverage=body.get("coverage", "basic"),
                constraints=body.get("constraints"),
                template=body.get("template"),
            )
            
            self._json_response(200, {
                "generation_id": result.generation_id,
                "test_suite": result.test_suite,
                "vrf_format": result.to_vrf_format(),
                "coverage": result.coverage,
                "generated_by": result.generated_by,
                "token_estimate": result.token_estimate,
                "review_suggested": result.review_suggested,
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})
    
    def _handle_gen_and_verify(self):
        """Generate tests AND verify code in one call."""
        try:
            body = self._read_body()
            task = body.get("task_description", "")
            code = body.get("code", "")
            if not task or not code:
                self._json_response(400, {"error": "task_description and code required"})
                return
            
            # Generate tests
            result = generate_test_suite(
                task_description=task,
                language=body.get("language", "python"),
                coverage=body.get("coverage", "basic"),
                constraints=body.get("constraints"),
                template=body.get("template"),
            )
            
            # Verify
            receipt = verify_with_generated_tests(
                code=code,
                test_suite=result,
                verify_url=body.get("verify_url", VERIFY_SERVER_URL),
            )
            
            self._json_response(200, {
                "generation_id": result.generation_id,
                "test_suite": result.test_suite,
                "receipt": receipt,
            })
        except Exception as e:
            self._json_response(500, {"error": str(e)})


# ── Tests ───────────────────────────────────────────────────────────

def run_tests():
    """Run test suite. Uses real OpenAI API for integration tests."""
    passed = 0
    failed = 0
    skipped = 0
    
    def ok(name):
        nonlocal passed; passed += 1; print(f"  ✅ {name}")
    def fail(name, msg):
        nonlocal failed; failed += 1; print(f"  ❌ {name}: {msg}")
    def skip(name, reason):
        nonlocal skipped; skipped += 1; print(f"  ⏭️  {name}: {reason}")
    
    print("=== VRF Test Generation Tests ===\n")
    
    # ── Unit tests (no API needed) ──
    
    print("── Prompt Construction ──")
    
    # 1. Basic prompt
    p = build_generation_prompt("Sort a list", "python", "basic")
    assert "Sort a list" in p and "python" in p and "3-5" in p
    ok("basic prompt construction")
    
    # 2. With constraints
    p = build_generation_prompt("Sort a list", "python", "standard", constraints=["O(n log n)", "stable sort"])
    assert "O(n log n)" in p and "stable sort" in p
    ok("prompt with constraints")
    
    # 3. With template
    p = build_generation_prompt("Sort a list", "python", "basic", template="function")
    assert "happy_path" in p and "empty_input" in p
    ok("prompt with domain template")
    
    # 4. Thorough coverage
    p = build_generation_prompt("Parse JSON", "javascript", "thorough")
    assert "15-25" in p and "adversarial" in p
    ok("thorough coverage spec")
    
    # 5. Unknown template ignored gracefully
    p = build_generation_prompt("Task", "python", "basic", template="nonexistent")
    assert "DOMAIN" not in p
    ok("unknown template ignored")
    
    print("\n── Coverage Specs ──")
    
    # 6. All levels defined
    assert set(COVERAGE_SPECS.keys()) == {"basic", "standard", "thorough"}
    ok("all coverage levels defined")
    
    # 7. Domain templates
    assert len(DOMAIN_TEMPLATES) >= 4
    ok(f"{len(DOMAIN_TEMPLATES)} domain templates defined")
    
    print("\n── GeneratedTestSuite ──")
    
    # 8. VRF format conversion
    suite = GeneratedTestSuite(
        test_suite={
            "function_name": "add",
            "tests": [
                {"input": "add(1, 2)", "expected_output": "3", "description": "basic add", "category": "happy_path"},
                {"input": "add(0, 0)", "expected_output": "0", "description": "zeros", "category": "edge_case"},
            ],
            "notes": "Assumes integer addition",
        },
        task_description="Write an add function",
        language="python",
        coverage="basic",
    )
    vrf = suite.to_vrf_format()
    assert vrf["language"] == "python"
    assert vrf["function_name"] == "add"
    assert len(vrf["tests"]) == 2
    assert vrf["tests"][0]["input"] == "add(1, 2)"
    assert vrf["generated_by"] == "vrf-test-gen/0.1"
    ok("VRF format conversion")
    
    # 9. Generation ID uniqueness
    s1 = GeneratedTestSuite(test_suite={"tests": []}, task_description="", language="python", coverage="basic")
    s2 = GeneratedTestSuite(test_suite={"tests": []}, task_description="", language="python", coverage="basic")
    assert s1.generation_id != s2.generation_id
    ok("unique generation IDs")
    
    print("\n── HTTP Server ──")
    
    # 10. Health endpoint
    from io import BytesIO
    server = HTTPServer(("127.0.0.1", 0), TestGenHandler)
    port = server.server_address[1]
    t = __import__("threading").Thread(target=server.handle_request, daemon=True)
    t.start()
    
    req = Request(f"http://127.0.0.1:{port}/health")
    with urlopen(req, timeout=5) as resp:
        health = json.loads(resp.read())
    assert health["status"] == "ok"
    assert health["service"] == "vrf-test-gen"
    assert "function" in health["templates"]
    ok("health endpoint")
    
    # ── Integration tests (need API key) ──
    
    print("\n── Integration (OpenAI API) ──")
    
    if not OPENAI_API_KEY:
        skip("generate basic test suite", "no OPENAI_API_KEY")
        skip("generate with constraints", "no OPENAI_API_KEY")
        skip("generate JS tests", "no OPENAI_API_KEY")
        skip("generate thorough coverage", "no OPENAI_API_KEY")
    else:
        # 11. Generate basic test suite for a simple function
        try:
            result = generate_test_suite(
                "Write a Python function called 'reverse_string' that takes a string and returns it reversed.",
                language="python",
                coverage="basic",
            )
            assert len(result.test_suite["tests"]) >= 3
            assert result.language == "python"
            assert result.coverage == "basic"
            # Check all tests have required fields
            for t in result.test_suite["tests"]:
                assert "input" in t and "expected_output" in t
            ok(f"generate basic suite ({len(result.test_suite['tests'])} tests)")
        except Exception as e:
            fail("generate basic suite", str(e))
        
        # 12. Generate with constraints
        try:
            result = generate_test_suite(
                "Write a function 'fibonacci' that returns the nth Fibonacci number (0-indexed).",
                language="python",
                coverage="standard",
                constraints=["must handle n=0", "must handle large n efficiently"],
                template="function",
            )
            assert len(result.test_suite["tests"]) >= 5
            ok(f"generate with constraints ({len(result.test_suite['tests'])} tests)")
        except Exception as e:
            fail("generate with constraints", str(e))
        
        # 13. JavaScript test generation
        try:
            result = generate_test_suite(
                "Write a JavaScript function 'isPalindrome' that checks if a string is a palindrome.",
                language="javascript",
                coverage="basic",
            )
            assert len(result.test_suite["tests"]) >= 3
            ok(f"generate JS tests ({len(result.test_suite['tests'])} tests)")
        except Exception as e:
            fail("generate JS tests", str(e))
        
        # 14. Thorough coverage generates more tests
        try:
            result = generate_test_suite(
                "Write a function 'flatten' that recursively flattens a nested list.",
                language="python",
                coverage="thorough",
            )
            assert len(result.test_suite["tests"]) >= 10
            ok(f"generate thorough ({len(result.test_suite['tests'])} tests)")
        except Exception as e:
            fail("generate thorough", str(e))
    
    print(f"\n{'='*40}")
    print(f"Passed: {passed}  Failed: {failed}  Skipped: {skipped}")
    return failed == 0


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    if "--test" in sys.argv:
        sys.exit(0 if run_tests() else 1)
    
    if "--generate" in sys.argv:
        idx = sys.argv.index("--generate")
        if idx + 1 >= len(sys.argv):
            print("Usage: --generate 'task description'")
            sys.exit(1)
        task = sys.argv[idx + 1]
        lang = "python"
        cov = "basic"
        for i, a in enumerate(sys.argv):
            if a == "--language" and i + 1 < len(sys.argv):
                lang = sys.argv[i + 1]
            if a == "--coverage" and i + 1 < len(sys.argv):
                cov = sys.argv[i + 1]
        
        result = generate_test_suite(task, language=lang, coverage=cov)
        print(json.dumps({
            "generation_id": result.generation_id,
            "test_suite": result.test_suite,
            "vrf_format": result.to_vrf_format(),
        }, indent=2))
        sys.exit(0)
    
    # Start server
    port = 8701
    for i, a in enumerate(sys.argv):
        if a == "--port" and i + 1 < len(sys.argv):
            port = int(sys.argv[i + 1])
    
    server = HTTPServer(("0.0.0.0", port), TestGenHandler)
    print(f"VRF Test Generator running on :{port}")
    print(f"  POST /test-gen — generate test suite")
    print(f"  POST /test-gen/verify — generate + verify in one call")
    print(f"  GET  /health")
    server.serve_forever()


if __name__ == "__main__":
    main()
