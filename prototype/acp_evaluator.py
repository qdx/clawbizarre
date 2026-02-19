#!/usr/bin/env python3
"""
ACP Evaluator Bridge — Phase 1 (Standalone)

Simulates the ClawBizarre evaluator role for ACP jobs.
Accepts a deliverable + test suite, runs verification via verify_server,
returns approve/reject decision with VRF receipt.

Phase 1: HTTP-only, no ACP SDK dependency.
Phase 2: Will add virtuals-acp SDK for on-chain signing.
"""

import json
import time
import uuid
import hashlib
from http.server import HTTPServer, BaseHTTPRequestHandler
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any
from enum import Enum

# Import verify_server components
try:
    from verify_server import VerificationServer, VerificationReceipt
    HAS_VERIFY_SERVER = True
except ImportError:
    HAS_VERIFY_SERVER = False


class EvaluationDecision(Enum):
    APPROVE = "approve"
    REJECT = "reject"
    ERROR = "error"


@dataclass
class EvaluationRequest:
    """Maps ACP job deliverable to ClawBizarre verification request."""
    job_id: str                          # ACP job ID (or external reference)
    deliverable: str                     # The code/data output to verify
    language: str = "python"             # python, javascript, bash
    tier: int = 0                        # Verification tier (0 = test suite)
    test_suite: Optional[List[Dict]] = None  # Explicit tests
    schema: Optional[Dict] = None        # For Tier 1 schema checking
    task_type: str = "code_generation"   # Categorization
    timeout_seconds: int = 30            # Max verification time
    
    @classmethod
    def from_acp_memo(cls, memo: Dict) -> 'EvaluationRequest':
        """Parse an ACP DeliverableMemo + job spec into an EvaluationRequest.
        
        Expected memo structure:
        {
            "job_id": "...",
            "deliverable": { "content": "..." },
            "job_spec": {
                "requirements": {
                    "clawbizarre_verification": {
                        "tier": 0,
                        "language": "python",
                        "tests": [...]
                    }
                }
            }
        }
        """
        job_id = memo.get("job_id", str(uuid.uuid4()))
        
        # Extract deliverable content
        deliverable = memo.get("deliverable", {})
        if isinstance(deliverable, dict):
            content = deliverable.get("content", "")
        else:
            content = str(deliverable)
        
        # Extract verification config from job spec
        job_spec = memo.get("job_spec", {})
        requirements = job_spec.get("requirements", {})
        cb_config = requirements.get("clawbizarre_verification", {})
        
        return cls(
            job_id=job_id,
            deliverable=content,
            language=cb_config.get("language", "python"),
            tier=cb_config.get("tier", 0),
            test_suite=cb_config.get("tests"),
            schema=cb_config.get("schema"),
            task_type=job_spec.get("name", "code_generation"),
            timeout_seconds=cb_config.get("timeout", 30),
        )


@dataclass
class EvaluationResult:
    """The evaluator's decision, including VRF receipt."""
    job_id: str
    decision: EvaluationDecision
    reason: str
    vrf_receipt: Optional[Dict] = None   # Signed VRF receipt
    test_results: Optional[Dict] = None  # Detailed test output
    evaluation_time_ms: int = 0
    evaluator_id: str = "clawbizarre-evaluator-v1"
    
    def to_dict(self) -> Dict:
        d = {
            "job_id": self.job_id,
            "decision": self.decision.value,
            "reason": self.reason,
            "evaluation_time_ms": self.evaluation_time_ms,
            "evaluator_id": self.evaluator_id,
        }
        if self.vrf_receipt:
            d["vrf_receipt"] = self.vrf_receipt
        if self.test_results:
            d["test_results"] = self.test_results
        return d


class ACPEvaluator:
    """Core evaluator logic. Bridges ACP deliverables to verify_server."""
    
    def __init__(self, verify_host: str = "127.0.0.1", verify_port: int = 8340):
        self.verify_host = verify_host
        self.verify_port = verify_port
        self.stats = {
            "total_evaluations": 0,
            "approvals": 0,
            "rejections": 0,
            "errors": 0,
        }
    
    def evaluate(self, request: EvaluationRequest) -> EvaluationResult:
        """Run verification and return approve/reject decision."""
        start = time.time()
        self.stats["total_evaluations"] += 1
        
        # Validate request
        if not request.deliverable:
            self.stats["errors"] += 1
            return EvaluationResult(
                job_id=request.job_id,
                decision=EvaluationDecision.ERROR,
                reason="Empty deliverable",
            )
        
        if request.tier == 0 and not request.test_suite:
            self.stats["errors"] += 1
            return EvaluationResult(
                job_id=request.job_id,
                decision=EvaluationDecision.ERROR,
                reason="Tier 0 evaluation requires test_suite in clawbizarre_verification",
            )
        
        # Build verify_server request
        verify_request = self._build_verify_request(request)
        
        # Call verify_server
        try:
            receipt = self._call_verify_server(verify_request)
        except Exception as e:
            self.stats["errors"] += 1
            return EvaluationResult(
                job_id=request.job_id,
                decision=EvaluationDecision.REJECT,
                reason=f"Verification failed: {str(e)}",
                evaluation_time_ms=int((time.time() - start) * 1000),
            )
        
        # Determine decision from receipt
        elapsed_ms = int((time.time() - start) * 1000)
        
        verdict = receipt.get("verdict", "").lower()
        results = receipt.get("results", {})
        
        if verdict == "pass":
            self.stats["approvals"] += 1
            return EvaluationResult(
                job_id=request.job_id,
                decision=EvaluationDecision.APPROVE,
                reason=f"All tests passed ({results.get('passed', 0)}/{results.get('total', 0)})",
                vrf_receipt=receipt,
                test_results=results.get("details"),
                evaluation_time_ms=elapsed_ms,
            )
        else:
            self.stats["rejections"] += 1
            failed = results.get("failed", 0) + results.get("errors", 0)
            total = results.get("total", 0)
            return EvaluationResult(
                job_id=request.job_id,
                decision=EvaluationDecision.REJECT,
                reason=f"Verification failed: {failed}/{total} tests failed",
                vrf_receipt=receipt,
                test_results=receipt.get("details"),
                evaluation_time_ms=elapsed_ms,
            )
    
    def _build_verify_request(self, request: EvaluationRequest) -> Dict:
        """Convert EvaluationRequest to verify_server format."""
        verify_req: Dict[str, Any] = {
            "task_id": request.job_id,
            "task_type": request.task_type,
            "tier": request.tier,
            "output": {
                "content": request.deliverable,
                "content_hash": f"sha256:{hashlib.sha256(request.deliverable.encode()).hexdigest()}"
            },
        }
        
        if request.tier == 0 and request.test_suite:
            verify_req["verification"] = {
                "kind": "test_suite",
                "test_suite": {
                    "language": request.language,
                    "tests": request.test_suite,
                }
            }
        elif request.tier == 1 and request.schema:
            verify_req["verification"] = {
                "kind": "schema_check",
                "schema": request.schema,
            }
        
        return verify_req
    
    def _call_verify_server(self, verify_request: Dict) -> Dict:
        """Call verify_server's /verify endpoint."""
        import urllib.request
        
        url = f"http://{self.verify_host}:{self.verify_port}/verify"
        data = json.dumps(verify_request).encode()
        req = urllib.request.Request(
            url, data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        
        with urllib.request.urlopen(req, timeout=30) as resp:
            return json.loads(resp.read().decode())


class ACPEvaluatorHandler(BaseHTTPRequestHandler):
    """HTTP handler for standalone evaluator server."""
    
    evaluator: ACPEvaluator = None  # Set by server
    
    def do_POST(self):
        if self.path == "/evaluate":
            self._handle_evaluate()
        elif self.path == "/evaluate/acp":
            self._handle_evaluate_acp()
        else:
            self._send(404, {"error": "Not found"})
    
    def do_GET(self):
        if self.path == "/health":
            self._send(200, {
                "status": "ok",
                "service": "clawbizarre-acp-evaluator",
                "version": "1.0",
                "stats": self.evaluator.stats,
            })
        elif self.path == "/stats":
            self._send(200, self.evaluator.stats)
        else:
            self._send(404, {"error": "Not found"})
    
    def _handle_evaluate(self):
        """Direct evaluation: accepts EvaluationRequest-style JSON."""
        body = self._read_body()
        if not body:
            return
        
        try:
            request = EvaluationRequest(
                job_id=body.get("job_id", str(uuid.uuid4())),
                deliverable=body["deliverable"],
                language=body.get("language", "python"),
                tier=body.get("tier", 0),
                test_suite=body.get("test_suite"),
                schema=body.get("schema"),
                task_type=body.get("task_type", "code_generation"),
            )
        except KeyError as e:
            self._send(400, {"error": f"Missing field: {e}"})
            return
        
        result = self.evaluator.evaluate(request)
        self._send(200, result.to_dict())
    
    def _handle_evaluate_acp(self):
        """ACP-format evaluation: accepts memo-style JSON."""
        body = self._read_body()
        if not body:
            return
        
        try:
            request = EvaluationRequest.from_acp_memo(body)
        except Exception as e:
            self._send(400, {"error": f"Failed to parse ACP memo: {e}"})
            return
        
        result = self.evaluator.evaluate(request)
        self._send(200, result.to_dict())
    
    def _read_body(self) -> Optional[Dict]:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            self._send(400, {"error": "Empty body"})
            return None
        try:
            return json.loads(self.rfile.read(length).decode())
        except json.JSONDecodeError:
            self._send(400, {"error": "Invalid JSON"})
            return None
    
    def _send(self, code: int, data: Dict):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode())
    
    def log_message(self, format, *args):
        pass  # Suppress default logging


def run_server(host: str = "127.0.0.1", port: int = 8350,
               verify_host: str = "127.0.0.1", verify_port: int = 8340):
    """Start the ACP Evaluator HTTP server."""
    evaluator = ACPEvaluator(verify_host, verify_port)
    ACPEvaluatorHandler.evaluator = evaluator
    
    server = HTTPServer((host, port), ACPEvaluatorHandler)
    print(f"ClawBizarre ACP Evaluator listening on {host}:{port}")
    print(f"  → verify_server at {verify_host}:{verify_port}")
    print(f"Endpoints:")
    print(f"  POST /evaluate     — Direct evaluation")
    print(f"  POST /evaluate/acp — ACP memo format")
    print(f"  GET  /health       — Health check")
    print(f"  GET  /stats        — Evaluation statistics")
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


# ── Tests ──────────────────────────────────────────────────────────────

def _run_tests():
    """Self-contained tests (no verify_server dependency for unit tests)."""
    import unittest
    
    class TestEvaluationRequest(unittest.TestCase):
        def test_from_acp_memo_basic(self):
            memo = {
                "job_id": "job-123",
                "deliverable": {"content": "def add(a,b): return a+b"},
                "job_spec": {
                    "name": "code_generation",
                    "requirements": {
                        "clawbizarre_verification": {
                            "tier": 0,
                            "language": "python",
                            "tests": [
                                {"name": "test1", "input": "add(1,2)", "expected": "3"}
                            ]
                        }
                    }
                }
            }
            req = EvaluationRequest.from_acp_memo(memo)
            self.assertEqual(req.job_id, "job-123")
            self.assertEqual(req.deliverable, "def add(a,b): return a+b")
            self.assertEqual(req.language, "python")
            self.assertEqual(req.tier, 0)
            self.assertEqual(len(req.test_suite), 1)
        
        def test_from_acp_memo_missing_verification(self):
            memo = {
                "job_id": "job-456",
                "deliverable": {"content": "some code"},
                "job_spec": {"name": "unknown"}
            }
            req = EvaluationRequest.from_acp_memo(memo)
            self.assertEqual(req.tier, 0)
            self.assertIsNone(req.test_suite)
        
        def test_from_acp_memo_string_deliverable(self):
            memo = {"deliverable": "raw string"}
            req = EvaluationRequest.from_acp_memo(memo)
            self.assertEqual(req.deliverable, "raw string")
    
    class TestEvaluator(unittest.TestCase):
        def setUp(self):
            self.evaluator = ACPEvaluator()
        
        def test_empty_deliverable(self):
            req = EvaluationRequest(job_id="t1", deliverable="")
            result = self.evaluator.evaluate(req)
            self.assertEqual(result.decision, EvaluationDecision.ERROR)
            self.assertIn("Empty", result.reason)
        
        def test_tier0_no_tests(self):
            req = EvaluationRequest(job_id="t2", deliverable="some code", tier=0)
            result = self.evaluator.evaluate(req)
            self.assertEqual(result.decision, EvaluationDecision.ERROR)
            self.assertIn("test_suite", result.reason)
        
        def test_build_verify_request(self):
            req = EvaluationRequest(
                job_id="t3",
                deliverable="def f(): pass",
                language="python",
                tier=0,
                test_suite=[{"name": "t", "input": "f()", "expected": "None"}],
                task_type="code_gen",
            )
            vr = self.evaluator._build_verify_request(req)
            self.assertEqual(vr["task_id"], "t3")
            self.assertEqual(vr["tier"], 0)
            self.assertEqual(vr["verification"]["kind"], "test_suite")
            self.assertEqual(vr["verification"]["test_suite"]["language"], "python")
            self.assertIn("sha256:", vr["output"]["content_hash"])
        
        def test_stats_tracking(self):
            req = EvaluationRequest(job_id="t4", deliverable="")
            self.evaluator.evaluate(req)
            self.assertEqual(self.evaluator.stats["errors"], 1)
            self.assertEqual(self.evaluator.stats["total_evaluations"], 1)
    
    class TestEvaluationResult(unittest.TestCase):
        def test_to_dict(self):
            result = EvaluationResult(
                job_id="j1",
                decision=EvaluationDecision.APPROVE,
                reason="All tests passed",
                evaluation_time_ms=42,
            )
            d = result.to_dict()
            self.assertEqual(d["decision"], "approve")
            self.assertEqual(d["job_id"], "j1")
            self.assertNotIn("vrf_receipt", d)  # None → omitted
        
        def test_to_dict_with_receipt(self):
            result = EvaluationResult(
                job_id="j2",
                decision=EvaluationDecision.REJECT,
                reason="Failed",
                vrf_receipt={"result": "FAIL"},
            )
            d = result.to_dict()
            self.assertIn("vrf_receipt", d)
    
    # HTTP handler tests
    class TestHTTPEndpoints(unittest.TestCase):
        @classmethod
        def setUpClass(cls):
            import threading
            cls.evaluator = ACPEvaluator()
            ACPEvaluatorHandler.evaluator = cls.evaluator
            cls.server = HTTPServer(("127.0.0.1", 0), ACPEvaluatorHandler)
            cls.port = cls.server.server_address[1]
            cls.thread = threading.Thread(target=cls.server.serve_forever)
            cls.thread.daemon = True
            cls.thread.start()
        
        @classmethod
        def tearDownClass(cls):
            cls.server.shutdown()
        
        def _request(self, method, path, body=None):
            import urllib.request
            url = f"http://127.0.0.1:{self.port}{path}"
            data = json.dumps(body).encode() if body else None
            req = urllib.request.Request(
                url, data=data,
                headers={"Content-Type": "application/json"} if data else {},
                method=method,
            )
            try:
                with urllib.request.urlopen(req) as resp:
                    return resp.status, json.loads(resp.read().decode())
            except urllib.error.HTTPError as e:
                return e.code, json.loads(e.read().decode())
        
        def test_health(self):
            code, data = self._request("GET", "/health")
            self.assertEqual(code, 200)
            self.assertEqual(data["service"], "clawbizarre-acp-evaluator")
        
        def test_evaluate_missing_deliverable(self):
            code, data = self._request("POST", "/evaluate", {})
            self.assertEqual(code, 400)
        
        def test_evaluate_empty_deliverable(self):
            code, data = self._request("POST", "/evaluate", {"deliverable": ""})
            self.assertEqual(code, 200)
            self.assertEqual(data["decision"], "error")
        
        def test_evaluate_no_tests(self):
            code, data = self._request("POST", "/evaluate", {
                "deliverable": "def f(): pass",
                "tier": 0,
            })
            self.assertEqual(code, 200)
            self.assertEqual(data["decision"], "error")
        
        def test_evaluate_acp_format(self):
            code, data = self._request("POST", "/evaluate/acp", {
                "job_id": "acp-job-1",
                "deliverable": {"content": "x = 1"},
                "job_spec": {"requirements": {}}
            })
            self.assertEqual(code, 200)
            self.assertEqual(data["job_id"], "acp-job-1")
        
        def test_stats(self):
            code, data = self._request("GET", "/stats")
            self.assertEqual(code, 200)
            self.assertIn("total_evaluations", data)
        
        def test_404(self):
            code, _ = self._request("GET", "/nonexistent")
            self.assertEqual(code, 404)
    
    suite = unittest.TestSuite()
    for tc in [TestEvaluationRequest, TestEvaluator, TestEvaluationResult, TestHTTPEndpoints]:
        suite.addTests(unittest.TestLoader().loadTestsFromTestCase(tc))
    
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    return result


if __name__ == "__main__":
    import sys
    if "--test" in sys.argv:
        result = _run_tests()
        sys.exit(0 if result.wasSuccessful() else 1)
    else:
        run_server()
