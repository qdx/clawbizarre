#!/usr/bin/env python3
"""
ClawBizarre A2A Protocol Adapter

Exposes the verify_server as an A2A-compatible remote agent.
Handles:
  - Agent Card at /.well-known/agent-card.json
  - JSON-RPC 2.0 task lifecycle (message/send, tasks/get, tasks/cancel)
  - Artifacts with VRF receipts

Thin adapter — all verification logic delegated to verify_server.

Dependencies: None (stdlib only). Requires verify_server running.
"""

import json
import uuid
import time
import urllib.request
import urllib.error
from http.server import HTTPServer, BaseHTTPRequestHandler
from datetime import datetime, timezone


# --- A2A Data Model ---

AGENT_CARD = {
    "name": "ClawBizarre Verification Service",
    "description": "Structural code verification for agent-produced work. "
                   "Executes test suites against code in sandboxed environments "
                   "and returns cryptographic VRF receipts. Protocol-agnostic — "
                   "works with ACP, A2A, MCP, or standalone.",
    "url": "http://localhost:7862/",
    "version": "1.0.0",
    "defaultInputModes": ["application/json"],
    "defaultOutputModes": ["application/json"],
    "capabilities": {
        "streaming": False,
        "pushNotifications": False,
    },
    "skills": [
        {
            "id": "structural_code_verification",
            "name": "Structural Code Verification",
            "description": "Execute a test suite against provided code and return "
                           "a VRF (Verification Receipt Format) receipt with "
                           "pass/fail results. Supports Python, JavaScript, Bash.",
            "tags": ["verification", "testing", "code-quality", "vrf"],
            "examples": [
                "Verify this Python function passes its test suite",
                "Run these unit tests against my code",
                "Check if this JavaScript module meets its specifications",
            ],
            "inputModes": ["application/json"],
            "outputModes": ["application/json"],
        },
        {
            "id": "verification_status",
            "name": "Verification Health Check",
            "description": "Check verification service health and capabilities.",
            "tags": ["health", "status"],
            "examples": ["Is the verification service running?"],
            "inputModes": ["text/plain"],
            "outputModes": ["application/json"],
        },
    ],
    "supportsAuthenticatedExtendedCard": False,
}


# --- Task Store ---

class TaskStore:
    """In-memory task storage."""

    def __init__(self):
        self.tasks = {}

    def create(self, task_id, context_id=None):
        task = {
            "id": task_id,
            "contextId": context_id or str(uuid.uuid4()),
            "status": {
                "state": "submitted",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            },
            "artifacts": [],
            "kind": "task",
        }
        self.tasks[task_id] = task
        return task

    def get(self, task_id):
        return self.tasks.get(task_id)

    def update_status(self, task_id, state, message=None):
        task = self.tasks.get(task_id)
        if task:
            task["status"] = {
                "state": state,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            if message:
                task["status"]["message"] = {
                    "role": "agent",
                    "parts": [{"kind": "text", "text": message}],
                }
        return task

    def add_artifact(self, task_id, artifact):
        task = self.tasks.get(task_id)
        if task:
            task["artifacts"].append(artifact)
        return task


# --- Verification Bridge ---

class VerifyBridge:
    """Bridges A2A requests to verify_server."""

    def __init__(self, verify_url="http://localhost:7860"):
        self.verify_url = verify_url

    def health(self):
        try:
            req = urllib.request.Request(f"{self.verify_url}/health")
            with urllib.request.urlopen(req, timeout=5) as resp:
                return json.loads(resp.read())
        except Exception as e:
            return {"status": "error", "error": str(e)}

    def verify(self, code, test_suite, language="python", use_docker=False):
        """Call verify_server /verify endpoint."""
        payload = {
            "code": code,
            "verification": {
                "test_suite": {
                    "tests": test_suite if isinstance(test_suite, list) else [test_suite],
                    "language": language,
                },
                "use_docker": use_docker,
            },
        }
        data = json.dumps(payload).encode()
        req = urllib.request.Request(
            f"{self.verify_url}/verify",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read())
        except urllib.error.HTTPError as e:
            body = e.read().decode()
            return {"verdict": "ERROR", "error": body}
        except Exception as e:
            return {"verdict": "ERROR", "error": str(e)}


# --- A2A Request Handler ---

class A2AHandler(BaseHTTPRequestHandler):
    """JSON-RPC 2.0 handler for A2A protocol."""

    store = TaskStore()
    bridge = VerifyBridge()

    def log_message(self, format, *args):
        pass  # Silence default logging

    def _send_json(self, obj, status=200):
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/.well-known/agent-card.json", "/agent-card.json"):
            self._send_json(AGENT_CARD)
        elif self.path == "/health":
            health = self.bridge.health()
            self._send_json({
                "a2a": "ok",
                "verify_server": health.get("status", "unknown"),
            })
        else:
            self._send_json({"error": "Not found"}, 404)

    def do_POST(self):
        length = int(self.headers.get("Content-Length", 0))
        body = json.loads(self.rfile.read(length)) if length else {}

        # JSON-RPC 2.0
        method = body.get("method")
        params = body.get("params", {})
        rpc_id = body.get("id")

        handler = {
            "message/send": self._handle_message_send,
            "tasks/get": self._handle_tasks_get,
            "tasks/cancel": self._handle_tasks_cancel,
        }.get(method)

        if not handler:
            self._send_json({
                "jsonrpc": "2.0",
                "id": rpc_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            })
            return

        result = handler(params)
        self._send_json({"jsonrpc": "2.0", "id": rpc_id, "result": result})

    def _handle_message_send(self, params):
        """Process a verification request via A2A message/send."""
        message = params.get("message", {})
        parts = message.get("parts", [])
        task_id = params.get("id") or str(uuid.uuid4())
        context_id = params.get("contextId")

        task = self.store.create(task_id, context_id)
        self.store.update_status(task_id, "working", "Processing verification request...")

        # Extract verification request from parts
        verification_req = None
        for part in parts:
            if part.get("kind") == "data":
                verification_req = part.get("data", {})
                break
            elif part.get("kind") == "text":
                # Try parsing text as JSON
                try:
                    verification_req = json.loads(part["text"])
                except (json.JSONDecodeError, KeyError):
                    pass

        if not verification_req:
            self.store.update_status(task_id, "failed",
                "No verification request found. Send JSON with: code, test_suite, language")
            return task

        code = verification_req.get("code", "")
        test_suite = verification_req.get("test_suite", [])
        language = verification_req.get("language", "python")
        use_docker = verification_req.get("use_docker", False)

        if not code or not test_suite:
            self.store.update_status(task_id, "failed",
                "Missing required fields: code, test_suite")
            return task

        # Execute verification
        result = self.bridge.verify(code, test_suite, language, use_docker)

        # Create artifact with VRF receipt
        artifact = {
            "artifactId": str(uuid.uuid4()),
            "name": "vrf-receipt",
            "parts": [
                {
                    "kind": "data",
                    "data": result,
                    "metadata": {
                        "mimeType": "application/vrf+json",
                        "protocol": "clawbizarre-vrf-v1",
                    },
                }
            ],
        }
        self.store.add_artifact(task_id, artifact)

        verdict = result.get("verdict", "ERROR")
        state = "completed" if verdict in ("PASS", "FAIL") else "failed"
        summary = f"Verification {verdict}"
        if "test_results" in result:
            tr = result["test_results"]
            summary += f": {tr.get('passed', 0)}/{tr.get('total', 0)} tests passed"

        self.store.update_status(task_id, state, summary)
        return self.store.get(task_id)

    def _handle_tasks_get(self, params):
        task_id = params.get("id")
        task = self.store.get(task_id)
        if not task:
            return {"error": f"Task {task_id} not found"}
        return task

    def _handle_tasks_cancel(self, params):
        task_id = params.get("id")
        task = self.store.get(task_id)
        if not task:
            return {"error": f"Task {task_id} not found"}
        self.store.update_status(task_id, "canceled")
        return self.store.get(task_id)


# --- Tests ---

def run_tests():
    """Self-test suite."""
    import threading

    passed = 0
    total = 0

    def test(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            print(f"  ✗ {name}")

    # Test 1: Agent card structure
    card = AGENT_CARD
    test("agent card has name", card["name"] == "ClawBizarre Verification Service")
    test("agent card has skills", len(card["skills"]) == 2)
    test("skill has id", card["skills"][0]["id"] == "structural_code_verification")
    test("skill has tags", "vrf" in card["skills"][0]["tags"])

    # Test 2: Task store
    store = TaskStore()
    t = store.create("test-1")
    test("task created", t["id"] == "test-1")
    test("task has status", t["status"]["state"] == "submitted")

    store.update_status("test-1", "working")
    test("status updated", store.get("test-1")["status"]["state"] == "working")

    store.add_artifact("test-1", {"artifactId": "a1", "parts": []})
    test("artifact added", len(store.get("test-1")["artifacts"]) == 1)

    # Test 3: JSON-RPC parsing
    test("tasks/get missing returns error",
         store.get("nonexistent") is None)

    # Test 4: Verify bridge health (offline)
    bridge = VerifyBridge("http://localhost:99999")
    h = bridge.health()
    test("offline bridge returns error", "error" in h)

    # Test 5: Agent card well-known path serving
    port = 17862
    A2AHandler.store = TaskStore()
    A2AHandler.bridge = VerifyBridge("http://localhost:99999")
    server = HTTPServer(("localhost", port), A2AHandler)
    t_srv = threading.Thread(target=server.handle_request)
    t_srv.daemon = True
    t_srv.start()

    req = urllib.request.Request(f"http://localhost:{port}/.well-known/agent-card.json")
    with urllib.request.urlopen(req, timeout=2) as resp:
        agent_card = json.loads(resp.read())
    test("agent card served via HTTP", agent_card["name"] == "ClawBizarre Verification Service")

    # Test 6: JSON-RPC message/send with missing data
    server2 = HTTPServer(("localhost", port + 1), A2AHandler)
    t_srv2 = threading.Thread(target=server2.handle_request)
    t_srv2.daemon = True
    t_srv2.start()

    rpc = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "message/send",
        "params": {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "hello"}],
            }
        }
    }
    data = json.dumps(rpc).encode()
    req = urllib.request.Request(
        f"http://localhost:{port + 1}/",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        result = json.loads(resp.read())
    test("message/send with bad input returns failed",
         result["result"]["status"]["state"] == "failed")

    # Test 7: JSON-RPC unknown method
    server3 = HTTPServer(("localhost", port + 2), A2AHandler)
    t_srv3 = threading.Thread(target=server3.handle_request)
    t_srv3.daemon = True
    t_srv3.start()

    rpc_bad = {"jsonrpc": "2.0", "id": 2, "method": "foo/bar", "params": {}}
    data = json.dumps(rpc_bad).encode()
    req = urllib.request.Request(
        f"http://localhost:{port + 2}/",
        data=data,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=2) as resp:
        result = json.loads(resp.read())
    test("unknown method returns error", "error" in result)

    print(f"\n{passed}/{total} tests passed")
    return passed == total


def main():
    import sys
    import os

    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)

    verify_url = os.environ.get("CLAWBIZARRE_VERIFY_URL", "http://localhost:7860")
    port = int(os.environ.get("A2A_PORT", "7862"))

    A2AHandler.bridge = VerifyBridge(verify_url)
    server = HTTPServer(("0.0.0.0", port), A2AHandler)
    print(f"ClawBizarre A2A adapter on :{port}")
    print(f"Agent card: http://localhost:{port}/.well-known/agent-card.json")
    print(f"Verify server: {verify_url}")
    server.serve_forever()


if __name__ == "__main__":
    main()
