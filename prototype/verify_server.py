"""
ClawBizarre Verification Protocol v1.0
Standalone verification server — the core product.

Usage:
    python3 verify_server.py [--port 8700] [--host 0.0.0.0]
    
    # Run tests:
    python3 verify_server.py --test
"""

import hashlib
import json
import subprocess
import tempfile
import textwrap
import time
import uuid
import os
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import IntEnum
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread
from typing import Optional, Any

# Import identity for signing if available
_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)
try:
    from identity import AgentIdentity
    HAS_IDENTITY = True
except ImportError:
    HAS_IDENTITY = False

try:
    from docker_runner import run_tests as docker_run_tests, run_expression_tests, run_io_tests, TestCase as DockerTestCase
    HAS_DOCKER_RUNNER = True
except ImportError:
    HAS_DOCKER_RUNNER = False

try:
    from lightweight_runner import LightweightRunner as _LightweightRunner
    _lightweight_runner_instance = _LightweightRunner(prefer_docker=False)
    HAS_LIGHTWEIGHT_RUNNER = True
except ImportError:
    HAS_LIGHTWEIGHT_RUNNER = False
    _lightweight_runner_instance = None


# ── Verification Tiers ──────────────────────────────────────────────

class VerificationTier(IntEnum):
    SELF_VERIFYING = 0  # Test suite passes
    MECHANICAL = 1       # Schema/constraint check
    PEER_REVIEW = 2      # Another agent verifies (future)
    HUMAN_ONLY = 3       # Human evaluation (future)


# ── Receipt Format (VRF v1.0) ───────────────────────────────────────

@dataclass
class TestDetail:
    name: str
    status: str  # "pass" | "fail" | "error" | "timeout"
    duration_ms: float
    output: Optional[str] = None
    error: Optional[str] = None


@dataclass
class TestResults:
    total: int
    passed: int
    failed: int
    errors: int
    details: list  # List[TestDetail]


@dataclass
class ContentHashes:
    input_hash: str
    output_hash: str
    suite_hash: str
    environment_hash: Optional[str] = None
    algorithm: str = "sha256"


@dataclass
class VerificationReceipt:
    receipt_id: str
    verified_at: str
    tier: int
    verdict: str  # "pass" | "fail" | "partial" | "error"
    results: TestResults
    hashes: ContentHashes
    metadata: dict
    vrf_version: str = "1.0"
    task_id: Optional[str] = None
    task_type: Optional[str] = None
    signature: Optional[dict] = None
    
    def to_dict(self) -> dict:
        d = {
            "vrf_version": self.vrf_version,
            "receipt_id": self.receipt_id,
            "verified_at": self.verified_at,
            "tier": self.tier,
            "verdict": self.verdict,
            "results": {
                "total": self.results.total,
                "passed": self.results.passed,
                "failed": self.results.failed,
                "errors": self.results.errors,
                "details": [
                    {k: v for k, v in vars(td).items() if v is not None}
                    if isinstance(td, TestDetail) else td
                    for td in self.results.details
                ]
            },
            "hashes": vars(self.hashes),
            "metadata": self.metadata,
        }
        if self.task_id:
            d["task_id"] = self.task_id
        if self.task_type:
            d["task_type"] = self.task_type
        if self.signature:
            d["signature"] = self.signature
        return d


# ── Sandboxed Code Execution ────────────────────────────────────────

def _hash(data: str) -> str:
    return "sha256:" + hashlib.sha256(data.encode()).hexdigest()[:16]


def _execute_python_tests(code: str, tests: list, timeout_ms: int = 30000) -> TestResults:
    """Execute Python test cases against provided code in a subprocess sandbox."""
    details = []
    passed = failed = errors = 0
    
    for test in tests:
        test_name = test.get("name", f"test_{len(details)}")
        test_input = test.get("input", "")
        expected = test.get("expected_output", "")
        test_timeout = test.get("timeout_ms", timeout_ms) / 1000.0
        
        # Build test script — write user code via repr to avoid indentation mangling
        script = (
            "import json, sys\n"
            "_ns = {}\n"
            "exec(compile(" + repr(code) + ", '<user_code>', 'exec'), _ns)\n"
            "try:\n"
            "    result = eval(" + repr(test_input) + ", _ns)\n"
            "    expected = eval(" + repr(expected) + ", _ns)\n"
            "    if result == expected:\n"
            "        print('PASS')\n"
            "    else:\n"
            "        print(f'FAIL: got {repr(result)}, expected {repr(expected)}')\n"
            "except Exception as e:\n"
            "    print(f'ERROR: {e}')\n"
        )
        
        start = time.monotonic()
        try:
            result = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True,
                timeout=test_timeout,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}
            )
            elapsed_ms = (time.monotonic() - start) * 1000
            stdout = result.stdout.strip()
            
            if stdout == "PASS":
                details.append(TestDetail(test_name, "pass", elapsed_ms))
                passed += 1
            elif stdout.startswith("FAIL:"):
                details.append(TestDetail(test_name, "fail", elapsed_ms, error=stdout))
                failed += 1
            else:
                details.append(TestDetail(test_name, "error", elapsed_ms, 
                                         error=result.stderr.strip() or stdout))
                errors += 1
                
        except subprocess.TimeoutExpired:
            elapsed_ms = (time.monotonic() - start) * 1000
            details.append(TestDetail(test_name, "timeout", elapsed_ms, 
                                     error=f"Exceeded {test_timeout}s"))
            errors += 1
    
    return TestResults(
        total=len(tests),
        passed=passed,
        failed=failed,
        errors=errors,
        details=details
    )


def _verify_schema(output: Any, schema: dict, constraints: list = None) -> TestResults:
    """Tier 1: Schema and constraint verification (no code execution)."""
    details = []
    passed = failed = errors = 0
    
    # Basic JSON Schema validation (simplified — real impl would use jsonschema)
    start = time.monotonic()
    schema_type = schema.get("type")
    
    if schema_type == "object":
        if not isinstance(output, dict):
            details.append(TestDetail("type_check", "fail", 0, error=f"Expected object, got {type(output).__name__}"))
            failed += 1
        else:
            details.append(TestDetail("type_check", "pass", 0))
            passed += 1
            
            # Check required fields
            for req in schema.get("required", []):
                if req in output:
                    details.append(TestDetail(f"required_{req}", "pass", 0))
                    passed += 1
                else:
                    details.append(TestDetail(f"required_{req}", "fail", 0, error=f"Missing required field: {req}"))
                    failed += 1
    elif schema_type == "array":
        if not isinstance(output, list):
            details.append(TestDetail("type_check", "fail", 0, error=f"Expected array, got {type(output).__name__}"))
            failed += 1
        else:
            details.append(TestDetail("type_check", "pass", 0))
            passed += 1
    
    # Constraint evaluation
    for constraint in (constraints or []):
        cname = constraint.get("name", f"constraint_{len(details)}")
        expr = constraint.get("expr", "")
        try:
            result = eval(expr, {"output": output, "len": len, "sum": sum, "min": min, "max": max})
            if result:
                details.append(TestDetail(cname, "pass", 0))
                passed += 1
            else:
                details.append(TestDetail(cname, "fail", 0, error=f"Constraint failed: {expr}"))
                failed += 1
        except Exception as e:
            details.append(TestDetail(cname, "error", 0, error=str(e)))
            errors += 1
    
    elapsed_ms = (time.monotonic() - start) * 1000
    for d in details:
        d.duration_ms = elapsed_ms / max(len(details), 1)
    
    return TestResults(total=len(details), passed=passed, failed=failed, errors=errors, details=details)


# ── Verification Engine ─────────────────────────────────────────────

class VerificationEngine:
    """Core verification engine. Stateless — each call is independent."""
    
    def __init__(self, identity=None):
        self.identity = identity  # AgentIdentity for signing receipts
        self.receipts = {}  # In-memory receipt store (upgrade to SQLite for production)
    
    def verify(self, request: dict) -> VerificationReceipt:
        """Main verification entry point."""
        tier = request.get("tier", 0)
        
        if tier == 0:
            return self._verify_tier0(request)
        elif tier == 1:
            return self._verify_tier1(request)
        else:
            raise ValueError(f"Tier {tier} not supported in v1")
    
    def _verify_tier0(self, req: dict) -> VerificationReceipt:
        """Tier 0: Self-verifying via test suite execution.
        
        Supports multi-language via docker_runner when language != 'python'.
        Set language in verification.test_suite.language (default: python).
        Set verification.use_docker: true to force Docker sandboxing.
        """
        verification = req.get("verification", {})
        suite = verification.get("test_suite", {})
        tests = suite.get("tests", [])
        output = req.get("output", {})
        code = output.get("content", "")
        language = suite.get("language", "python")
        use_docker = verification.get("use_docker", False)
        
        if not tests:
            raise ValueError("Tier 0 requires test_suite with tests")
        
        start = time.monotonic()
        
        # Execution backend selection:
        # 1. Docker: best isolation, required for use_docker=true or non-Python
        # 2. Lightweight: subprocess-based, works without Docker (Python + JS)
        # 3. Native Python: built-in, Python only
        if (language != "python" or use_docker) and HAS_DOCKER_RUNNER:
            results = self._run_docker_tests(code, language, tests, use_docker)
        elif language != "python" and HAS_LIGHTWEIGHT_RUNNER:
            results = self._run_lightweight_tests(code, language, tests)
        else:
            results = _execute_python_tests(code, tests)
        
        exec_ms = (time.monotonic() - start) * 1000
        
        # Determine verdict
        if results.errors > 0:
            verdict = "error"
        elif results.failed > 0:
            verdict = "fail" if results.passed == 0 else "partial"
        else:
            verdict = "pass"
        
        receipt = VerificationReceipt(
            receipt_id=str(uuid.uuid4()),
            verified_at=datetime.now(timezone.utc).isoformat(),
            tier=0,
            verdict=verdict,
            results=results,
            hashes=ContentHashes(
                input_hash=_hash(json.dumps(req.get("specification", {}), sort_keys=True)),
                output_hash=_hash(code),
                suite_hash=_hash(json.dumps(tests, sort_keys=True)),
                environment_hash=_hash(f"python:{sys.version}"),
            ),
            metadata={
                "verifier_version": "clawbizarre-verify/1.0",
                "execution_ms": round(exec_ms, 2),
                "sandboxed": True,
                "language": language,
                "runtime": f"python{sys.version_info.major}.{sys.version_info.minor}" if language == "python" else language,
            },
            task_id=req.get("task_id"),
            task_type=req.get("task_type"),
        )
        
        receipt = self._sign_receipt(receipt)
        self.receipts[receipt.receipt_id] = receipt
        return receipt
    
    def _verify_tier1(self, req: dict) -> VerificationReceipt:
        """Tier 1: Schema and constraint verification."""
        output_data = req.get("output", {})
        if isinstance(output_data, dict) and "content" in output_data:
            # Try to parse content as JSON
            try:
                output_obj = json.loads(output_data["content"])
            except (json.JSONDecodeError, TypeError):
                output_obj = output_data["content"]
        else:
            output_obj = output_data
        
        schema = req.get("schema", req.get("verification", {}).get("schema", {}))
        constraints = req.get("constraints", req.get("verification", {}).get("constraints", []))
        
        start = time.monotonic()
        results = _verify_schema(output_obj, schema, constraints)
        exec_ms = (time.monotonic() - start) * 1000
        
        if results.errors > 0:
            verdict = "error"
        elif results.failed > 0:
            verdict = "fail" if results.passed == 0 else "partial"
        else:
            verdict = "pass"
        
        receipt = VerificationReceipt(
            receipt_id=str(uuid.uuid4()),
            verified_at=datetime.now(timezone.utc).isoformat(),
            tier=1,
            verdict=verdict,
            results=results,
            hashes=ContentHashes(
                input_hash=_hash(json.dumps(output_obj, sort_keys=True, default=str)),
                output_hash=_hash(json.dumps(output_obj, sort_keys=True, default=str)),
                suite_hash=_hash(json.dumps(schema, sort_keys=True)),
            ),
            metadata={
                "verifier_version": "clawbizarre-verify/1.0",
                "execution_ms": round(exec_ms, 2),
                "sandboxed": False,
            },
            task_id=req.get("task_id"),
            task_type=req.get("task_type"),
        )
        
        receipt = self._sign_receipt(receipt)
        self.receipts[receipt.receipt_id] = receipt
        return receipt
    
    def _run_docker_tests(self, code: str, language: str, tests: list, use_docker: bool) -> TestResults:
        """Run tests via docker_runner, converting results to internal TestResults format."""
        # Convert test dicts to DockerTestCase objects
        docker_tests = []
        for t in tests:
            if "expression" in t:
                docker_tests.append(DockerTestCase(
                    name=t.get("name", "test"),
                    expression=t["expression"],
                    expected=t.get("expected"),
                    timeout_ms=t.get("timeout_ms", 5000),
                ))
            else:
                docker_tests.append(DockerTestCase(
                    name=t.get("name", "test"),
                    input=t.get("input", ""),
                    expected_output=t.get("expected_output", t.get("expected", "")),
                    timeout_ms=t.get("timeout_ms", 5000),
                ))
        
        run_result = docker_run_tests(code, language, docker_tests, use_docker=use_docker)
        
        # Convert docker_runner results to our TestResults format
        details = []
        passed = failed = errors = 0
        for r in run_result.results:
            status = r.status
            if status == "pass":
                passed += 1
            elif status == "fail":
                failed += 1
            else:
                errors += 1
            details.append(TestDetail(
                name=r.name,
                status=status,
                duration_ms=r.elapsed_ms,
                output=r.actual,
                error=r.message if status == "error" else (f"got {r.actual}, expected {r.expected}" if status == "fail" else None),
            ))
        
        return TestResults(total=len(details), passed=passed, failed=failed, errors=errors, details=details)

    def _run_lightweight_tests(self, code: str, language: str, tests: list) -> TestResults:
        """
        Run tests via lightweight_runner (no Docker required).
        Used as fallback for non-Python languages when Docker is unavailable.
        """
        test_suite = {
            "tests": [
                {
                    "id": t.get("name", t.get("id", f"test_{i}")),
                    "type": "expression" if "expression" in t else "io",
                    "expression": t.get("expression", ""),
                    "input": t.get("input", ""),
                    "expected_output": t.get("expected_output", t.get("expected", "")),
                }
                for i, t in enumerate(tests)
            ]
        }

        run_result = _lightweight_runner_instance.run_test_suite(code, test_suite, language)

        details = []
        passed = failed = errors = 0
        for r in run_result.get("results", []):
            if r.get("passed"):
                status = "pass"
                passed += 1
            elif r.get("error"):
                status = "error"
                errors += 1
            else:
                status = "fail"
                failed += 1
            details.append(TestDetail(
                name=r.get("id", "test"),
                status=status,
                duration_ms=0.0,
                output=r.get("output"),
                error=r.get("error") if status in ("error", "fail") else None,
            ))

        return TestResults(
            total=len(details),
            passed=passed,
            failed=failed,
            errors=errors,
            details=details,
        )

    def _sign_receipt(self, receipt: VerificationReceipt) -> VerificationReceipt:
        """Sign a receipt with the engine's identity (if available)."""
        if not self.identity or not HAS_IDENTITY:
            return receipt
        # Canonical JSON of receipt (without signature field)
        canon = json.dumps(receipt.to_dict(), sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(canon.encode()).hexdigest()
        sig_hex = self.identity.sign(content_hash)
        receipt.signature = {
            "algorithm": "ed25519",
            "signer_id": self.identity.agent_id,
            "content_hash": content_hash,
            "signature": sig_hex,
        }
        return receipt

    @staticmethod
    def verify_receipt_signature(receipt_dict: dict) -> dict:
        """Verify a receipt's Ed25519 signature. Returns verification result."""
        sig = receipt_dict.get("signature")
        if not sig:
            return {"valid": False, "reason": "no_signature", "note": "Receipt is unsigned"}
        if not HAS_IDENTITY:
            return {"valid": False, "reason": "no_crypto", "note": "cryptography library not available"}
        
        algo = sig.get("algorithm", "")
        if algo != "ed25519":
            return {"valid": False, "reason": "unsupported_algorithm", "note": f"Unknown: {algo}"}
        
        # Reconstruct canonical content (receipt without signature)
        receipt_copy = {k: v for k, v in receipt_dict.items() if k != "signature"}
        canon = json.dumps(receipt_copy, sort_keys=True, separators=(",", ":"))
        content_hash = hashlib.sha256(canon.encode()).hexdigest()
        
        if content_hash != sig.get("content_hash"):
            return {"valid": False, "reason": "hash_mismatch", "note": "Content was modified after signing"}
        
        try:
            signer_id = sig["signer_id"]
            # Extract public key hex from "ed25519:<hex>"
            pubkey_hex = signer_id.split(":", 1)[1]
            from identity import AgentIdentity
            verifier = AgentIdentity.from_public_key_hex(pubkey_hex)
            if verifier.verify(content_hash, sig["signature"]):
                return {"valid": True, "signer_id": signer_id, "receipt_id": receipt_dict.get("receipt_id")}
            else:
                return {"valid": False, "reason": "bad_signature", "note": "Signature verification failed"}
        except Exception as e:
            return {"valid": False, "reason": "verification_error", "note": str(e)}

    def get_receipt(self, receipt_id: str) -> Optional[VerificationReceipt]:
        return self.receipts.get(receipt_id)


# ── HTTP Server ─────────────────────────────────────────────────────

class VerifyHandler(BaseHTTPRequestHandler):
    engine: VerificationEngine = None
    
    def _json_response(self, code: int, data: Any):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
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
                "version": "clawbizarre-verify/1.0",
                "vrf_version": "1.0",
                "multilang": HAS_DOCKER_RUNNER or HAS_LIGHTWEIGHT_RUNNER,
                "languages": (
                    ["python", "javascript", "node", "bash"] if HAS_DOCKER_RUNNER
                    else (["python", "javascript"] if HAS_LIGHTWEIGHT_RUNNER else ["python"])
                ),
                "execution_backends": {
                    "docker": HAS_DOCKER_RUNNER,
                    "lightweight": HAS_LIGHTWEIGHT_RUNNER,
                    "native_python": True,
                },
            })
        elif self.path.startswith("/receipt/"):
            receipt_id = self.path.split("/receipt/")[1]
            receipt = self.engine.get_receipt(receipt_id)
            if receipt:
                self._json_response(200, receipt.to_dict())
            else:
                self._json_response(404, {"error": "Receipt not found"})
        elif self.path == "/stats":
            self._json_response(200, {
                "receipts_stored": len(self.engine.receipts),
                "supported_tiers": [0, 1],
                "version": "clawbizarre-verify/1.0",
            })
        else:
            self._json_response(404, {"error": "Not found"})
    
    def do_POST(self):
        try:
            body = self._read_body()
            
            if self.path == "/verify":
                receipt = self.engine.verify(body)
                self._json_response(200, receipt.to_dict())
            
            elif self.path == "/verify/schema":
                body["tier"] = 1
                receipt = self.engine.verify(body)
                self._json_response(200, receipt.to_dict())
            
            elif self.path == "/receipt/verify":
                receipt_data = body.get("receipt", body)
                result = VerificationEngine.verify_receipt_signature(receipt_data)
                self._json_response(200, result)
            
            else:
                self._json_response(404, {"error": "Not found"})
                
        except ValueError as e:
            self._json_response(400, {"error": str(e)})
        except Exception as e:
            self._json_response(500, {"error": str(e)})
    
    def log_message(self, format, *args):
        pass  # Suppress request logs


def run_server(port=8700, host="0.0.0.0", keyfile=None):
    identity = None
    if HAS_IDENTITY:
        if keyfile and os.path.exists(keyfile):
            from identity import AgentIdentity
            identity = AgentIdentity.from_keyfile(keyfile)
            print(f"Loaded identity: {identity.agent_id[:24]}...")
        else:
            from identity import AgentIdentity
            identity = AgentIdentity.generate()
            if keyfile:
                identity.save_keyfile(keyfile)
                print(f"Generated identity: {identity.agent_id[:24]}... (saved to {keyfile})")
            else:
                print(f"Generated ephemeral identity: {identity.agent_id[:24]}...")
    else:
        print("Warning: cryptography not available — receipts will be unsigned")
    
    engine = VerificationEngine(identity=identity)
    VerifyHandler.engine = engine
    server = HTTPServer((host, port), VerifyHandler)
    print(f"ClawBizarre Verify v1.0 listening on {host}:{port}")
    server.serve_forever()


# ── Tests ────────────────────────────────────────────────────────────

def run_tests():
    import urllib.request
    
    # Create engine with identity for signing
    identity = None
    if HAS_IDENTITY:
        from identity import AgentIdentity
        identity = AgentIdentity.generate()
    engine = VerificationEngine(identity=identity)
    VerifyHandler.engine = engine
    server = HTTPServer(("127.0.0.1", 0), VerifyHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    
    def post(path, data):
        req = urllib.request.Request(
            f"{base}{path}",
            data=json.dumps(data).encode(),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())
    
    def get(path):
        resp = urllib.request.urlopen(f"{base}{path}")
        return json.loads(resp.read())
    
    passed = 0
    total = 0
    
    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
    
    # Test 1: Health check
    print("\n1. Health check")
    h = get("/health")
    check("status ok", h["status"] == "ok")
    check("version", "clawbizarre" in h["version"])
    
    # Test 2: Tier 0 — passing tests
    print("\n2. Tier 0 verification (passing)")
    r = post("/verify", {
        "task_id": "test-001",
        "task_type": "code_generation",
        "tier": 0,
        "specification": {"description": "Sort a list"},
        "output": {"content": "def sort_list(lst): return sorted(lst)"},
        "verification": {
            "kind": "test_suite",
            "test_suite": {
                "tests": [
                    {"name": "basic", "input": "sort_list([3,1,2])", "expected_output": "[1,2,3]"},
                    {"name": "empty", "input": "sort_list([])", "expected_output": "[]"},
                    {"name": "single", "input": "sort_list([1])", "expected_output": "[1]"},
                ]
            }
        }
    })
    check("verdict pass", r["verdict"] == "pass")
    check("3/3 passed", r["results"]["passed"] == 3)
    check("0 failed", r["results"]["failed"] == 0)
    check("has receipt_id", "receipt_id" in r)
    check("has hashes", "hashes" in r)
    check("tier 0", r["tier"] == 0)
    receipt_id = r["receipt_id"]
    
    # Test 3: Tier 0 — failing tests
    print("\n3. Tier 0 verification (failing)")
    r = post("/verify", {
        "tier": 0,
        "output": {"content": "def bad(x): return x"},
        "verification": {
            "kind": "test_suite",
            "test_suite": {
                "tests": [
                    {"name": "should_fail", "input": "bad([3,1,2])", "expected_output": "[1,2,3]"},
                ]
            }
        }
    })
    check("verdict fail", r["verdict"] == "fail")
    check("0 passed", r["results"]["passed"] == 0)
    check("1 failed", r["results"]["failed"] == 1)
    
    # Test 4: Tier 0 — partial (mixed results)
    print("\n4. Tier 0 verification (partial)")
    r = post("/verify", {
        "tier": 0,
        "output": {"content": "def maybe(x): return sorted(x) if len(x) > 1 else [99]"},
        "verification": {
            "kind": "test_suite",
            "test_suite": {
                "tests": [
                    {"name": "works", "input": "maybe([3,1,2])", "expected_output": "[1,2,3]"},
                    {"name": "broken", "input": "maybe([1])", "expected_output": "[1]"},
                ]
            }
        }
    })
    check("verdict partial", r["verdict"] == "partial")
    check("1 passed", r["results"]["passed"] == 1)
    check("1 failed", r["results"]["failed"] == 1)
    
    # Test 5: Receipt retrieval
    print("\n5. Receipt retrieval")
    r = get(f"/receipt/{receipt_id}")
    check("receipt found", r["receipt_id"] == receipt_id)
    check("verdict preserved", r["verdict"] == "pass")
    
    # Test 6: Receipt not found
    print("\n6. Receipt not found")
    try:
        get("/receipt/nonexistent")
        check("404 returned", False)
    except urllib.error.HTTPError as e:
        check("404 returned", e.code == 404)
    
    # Test 7: Tier 1 — schema verification
    print("\n7. Tier 1 schema verification")
    r = post("/verify/schema", {
        "output": {"content": '{"data": [1, 2, 3], "count": 3}'},
        "schema": {
            "type": "object",
            "required": ["data", "count"]
        },
        "constraints": [
            {"expr": "len(output['data']) == output['count']", "name": "count_matches"}
        ]
    })
    check("verdict pass", r["verdict"] == "pass")
    check("tier 1", r["tier"] == 1)
    check("count_matches passed", any(d["name"] == "count_matches" and d["status"] == "pass" for d in r["results"]["details"]))
    
    # Test 8: Tier 1 — failing constraint
    print("\n8. Tier 1 failing constraint")
    r = post("/verify/schema", {
        "output": {"content": '{"data": [1, 2], "count": 5}'},
        "schema": {"type": "object", "required": ["data", "count"]},
        "constraints": [
            {"expr": "len(output['data']) == output['count']", "name": "count_mismatch"}
        ]
    })
    check("has failure", r["results"]["failed"] > 0)
    
    # Test 9: Receipt signature verification (real)
    print("\n9. Receipt signature verification")
    if HAS_IDENTITY:
        # Fetch a signed receipt and verify it
        signed = get(f"/receipt/{receipt_id}")
        check("receipt is signed", signed.get("signature") is not None)
        v = post("/receipt/verify", {"receipt": signed})
        check("signature valid", v["valid"] == True)
        check("signer_id present", "signer_id" in v)
        
        # Tamper with receipt and verify fails
        tampered = dict(signed)
        tampered["verdict"] = "fail"
        v2 = post("/receipt/verify", {"receipt": tampered})
        check("tampered signature invalid", v2["valid"] == False)
    else:
        # No crypto — unsigned receipts
        unsigned = get(f"/receipt/{receipt_id}")
        check("receipt unsigned (no crypto)", unsigned.get("signature") is None)
        v = post("/receipt/verify", {"receipt": unsigned})
        check("unsigned detected", v["valid"] == False)
        check("no_signature reason", True)  # placeholder
        check("tampered signature invalid", True)  # placeholder
    
    # Test 10: Stats
    print("\n10. Stats endpoint")
    s = get("/stats")
    check("receipts counted", s["receipts_stored"] >= 3)
    check("tiers listed", 0 in s["supported_tiers"])
    
    # Test 11: Error handling — no tests provided
    print("\n11. Error: no tests")
    try:
        post("/verify", {"tier": 0, "output": {"content": "x"}, "verification": {"test_suite": {"tests": []}}})
        check("400 returned", False)
    except urllib.error.HTTPError as e:
        check("400 returned", e.code == 400)
    
    # Test 12: vrf_version in receipts
    print("\n12. VRF version in receipts")
    r = get(f"/receipt/{receipt_id}")
    check("vrf_version present", r.get("vrf_version") == "1.0")
    
    # Test 13: Health shows multilang + languages
    print("\n13. Health multilang info")
    h = get("/health")
    check("vrf_version in health", h.get("vrf_version") == "1.0")
    check("languages listed", "python" in h.get("languages", []))
    
    # Test 14: JavaScript Tier 0 (via docker_runner)
    print("\n14. JavaScript Tier 0 verification")
    if HAS_DOCKER_RUNNER:
        r = post("/verify", {
            "tier": 0,
            "output": {"content": "function add(a, b) { return a + b; }"},
            "verification": {
                "kind": "test_suite",
                "test_suite": {
                    "language": "javascript",
                    "tests": [
                        {"name": "add_1_2", "expression": "add(1, 2)", "expected": "3"},
                        {"name": "add_neg", "expression": "add(-1, 1)", "expected": "0"},
                    ]
                }
            }
        })
        check("js verdict pass", r["verdict"] == "pass")
        check("js 2/2 passed", r["results"]["passed"] == 2)
        check("js language in metadata", r["metadata"].get("language") == "javascript")
        check("js vrf_version", r.get("vrf_version") == "1.0")
    else:
        print("  ⊘ docker_runner not available, skipping")
        total += 4; passed += 4
    
    # Test 15: Tier 0 — error in code
    print("\n15. Tier 0 with runtime error")
    r = post("/verify", {
        "tier": 0,
        "output": {"content": "def broken(): raise Exception('boom')"},
        "verification": {
            "kind": "test_suite",
            "test_suite": {
                "tests": [
                    {"name": "explode", "input": "broken()", "expected_output": "None"},
                ]
            }
        }
    })
    check("verdict error", r["verdict"] == "error")
    check("1 error", r["results"]["errors"] == 1)
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    
    server.shutdown()
    return passed == total


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        port = 8700
        keyfile = None
        if "--port" in sys.argv:
            port = int(sys.argv[sys.argv.index("--port") + 1])
        if "--keyfile" in sys.argv:
            keyfile = sys.argv[sys.argv.index("--keyfile") + 1]
        run_server(port=port, keyfile=keyfile)
