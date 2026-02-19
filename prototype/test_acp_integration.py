#!/usr/bin/env python3
"""
Integration test: ACP Evaluator → verify_server → actual code execution.

Starts both servers, sends evaluation requests through the full pipeline.
"""

import json
import time
import threading
import unittest
import urllib.request
import urllib.error
from http.server import HTTPServer

from verify_server import VerifyHandler, VerificationEngine
from acp_evaluator import ACPEvaluator, ACPEvaluatorHandler, EvaluationRequest, EvaluationDecision


class TestACPVerifyIntegration(unittest.TestCase):
    """End-to-end: ACP Evaluator → verify_server → code execution."""
    
    @classmethod
    def setUpClass(cls):
        # Start verify_server
        VerifyHandler.engine = VerificationEngine()
        cls.verify_server = HTTPServer(("127.0.0.1", 0), VerifyHandler)
        cls.verify_port = cls.verify_server.server_address[1]
        cls.verify_thread = threading.Thread(target=cls.verify_server.serve_forever)
        cls.verify_thread.daemon = True
        cls.verify_thread.start()
        
        # Start ACP evaluator pointing at verify_server
        cls.evaluator = ACPEvaluator(verify_port=cls.verify_port)
        ACPEvaluatorHandler.evaluator = cls.evaluator
        cls.eval_server = HTTPServer(("127.0.0.1", 0), ACPEvaluatorHandler)
        cls.eval_port = cls.eval_server.server_address[1]
        cls.eval_thread = threading.Thread(target=cls.eval_server.serve_forever)
        cls.eval_thread.daemon = True
        cls.eval_thread.start()
        
        time.sleep(0.2)
    
    @classmethod
    def tearDownClass(cls):
        cls.eval_server.shutdown()
        cls.verify_server.shutdown()
    
    def _evaluate(self, body):
        url = f"http://127.0.0.1:{self.eval_port}/evaluate"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())
    
    def _evaluate_acp(self, body):
        url = f"http://127.0.0.1:{self.eval_port}/evaluate/acp"
        data = json.dumps(body).encode()
        req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return resp.status, json.loads(resp.read().decode())
        except urllib.error.HTTPError as e:
            return e.code, json.loads(e.read().decode())
    
    def test_01_passing_code(self):
        """Good code → APPROVE with VRF receipt."""
        code, data = self._evaluate({
            "job_id": "int-test-1",
            "deliverable": "def add(a, b): return a + b",
            "language": "python",
            "tier": 0,
            "test_suite": [
                {"name": "basic", "input": "add(1, 2)", "expected_output": "3"},
                {"name": "negative", "input": "add(-1, 1)", "expected_output": "0"},
                {"name": "zero", "input": "add(0, 0)", "expected_output": "0"},
            ]
        })
        self.assertEqual(code, 200)
        self.assertEqual(data["decision"], "approve")
        self.assertIn("vrf_receipt", data)
        self.assertGreater(data["evaluation_time_ms"], 0)
    
    def test_02_failing_code(self):
        """Bad code → REJECT with details."""
        code, data = self._evaluate({
            "job_id": "int-test-2",
            "deliverable": "def add(a, b): return a - b",  # Wrong!
            "language": "python",
            "tier": 0,
            "test_suite": [
                {"name": "basic", "input": "add(1, 2)", "expected_output": "3"},
            ]
        })
        self.assertEqual(code, 200)
        self.assertEqual(data["decision"], "reject")
        self.assertIn("failed", data["reason"].lower())
    
    def test_03_syntax_error(self):
        """Code with syntax error → REJECT."""
        code, data = self._evaluate({
            "job_id": "int-test-3",
            "deliverable": "def add(a, b) return a + b",  # Missing colon
            "language": "python",
            "tier": 0,
            "test_suite": [
                {"name": "basic", "input": "add(1, 2)", "expected_output": "3"},
            ]
        })
        self.assertEqual(code, 200)
        self.assertEqual(data["decision"], "reject")
    
    def test_04_acp_memo_format(self):
        """Full ACP memo → parsed → evaluated → APPROVE."""
        code, data = self._evaluate_acp({
            "job_id": "acp-real-1",
            "deliverable": {"content": "def multiply(a, b): return a * b"},
            "job_spec": {
                "name": "math_function",
                "requirements": {
                    "clawbizarre_verification": {
                        "tier": 0,
                        "language": "python",
                        "tests": [
                            {"name": "basic", "input": "multiply(3, 4)", "expected_output": "12"},
                            {"name": "zero", "input": "multiply(5, 0)", "expected_output": "0"},
                        ]
                    }
                }
            }
        })
        self.assertEqual(code, 200)
        self.assertEqual(data["decision"], "approve")
        self.assertEqual(data["job_id"], "acp-real-1")
    
    def test_05_acp_memo_failing(self):
        """ACP memo with bad deliverable → REJECT."""
        code, data = self._evaluate_acp({
            "job_id": "acp-real-2",
            "deliverable": {"content": "def multiply(a, b): return a + b"},  # Wrong
            "job_spec": {
                "name": "math_function",
                "requirements": {
                    "clawbizarre_verification": {
                        "tier": 0,
                        "language": "python",
                        "tests": [
                            {"name": "basic", "input": "multiply(3, 4)", "expected_output": "12"},
                        ]
                    }
                }
            }
        })
        self.assertEqual(code, 200)
        self.assertEqual(data["decision"], "reject")
    
    def test_06_multiple_tests(self):
        """Deliverable with many tests → APPROVE."""
        code, data = self._evaluate({
            "job_id": "int-test-6",
            "deliverable": "def clamp(x, lo, hi): return max(lo, min(hi, x))",
            "language": "python",
            "tier": 0,
            "test_suite": [
                {"name": "middle", "input": "clamp(5, 0, 10)", "expected_output": "5"},
                {"name": "below", "input": "clamp(-1, 0, 10)", "expected_output": "0"},
                {"name": "above", "input": "clamp(15, 0, 10)", "expected_output": "10"},
                {"name": "edge_lo", "input": "clamp(0, 0, 10)", "expected_output": "0"},
                {"name": "edge_hi", "input": "clamp(10, 0, 10)", "expected_output": "10"},
            ]
        })
        self.assertEqual(code, 200)
        self.assertEqual(data["decision"], "approve")
    
    def test_07_stats_accumulate(self):
        """Stats reflect all evaluations."""
        code, data = self._evaluate({"deliverable": "", "tier": 0})
        code2, stats = urllib.request.Request(f"http://127.0.0.1:{self.eval_port}/stats"), None
        with urllib.request.urlopen(f"http://127.0.0.1:{self.eval_port}/stats") as resp:
            stats = json.loads(resp.read().decode())
        self.assertGreater(stats["total_evaluations"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
