"""
ClawBizarre HTTP API v7 — Phase 29: Task Board + Compute Credit + Lightweight Runner

Extends v6 with the final marketplace layer:

New endpoints:
  POST /tasks              — buyer posts task
  GET  /tasks              — list available tasks (with filters)
  GET  /tasks/<id>         — task detail
  POST /tasks/<id>/claim   — agent claims task
  POST /tasks/<id>/submit  — agent submits + auto-verifies
  POST /tasks/<id>/cancel  — buyer cancels pending task
  GET  /tasks/stats        — board statistics

  POST /credit/score       — compute credit score from receipt chain
  POST /credit/line        — credit line recommendation
  POST /credit/project     — sustainability projection
  GET  /credit/tiers       — tier policy table

Updated:
  GET /health              — now reports execution_backends
                             (docker / lightweight / native_python)

Architecture complete:
  task_board → verify_server → receipt → compute_credit → treasury

Usage:
  python3 api_server_v7.py [--port 8420] [--db clawbizarre.db]
  python3 api_server_v7.py --test
"""

import json
import sys
import os
import time
from urllib.parse import urlparse, parse_qs

# Extend v6
from api_server_v6 import APIv6Handler, PersistentStateV6, run_server as run_server_v6

from task_board import TaskBoard, TaskStatus, TaskPriority, TaskRequirements, TaskBudget
from compute_credit import CreditScorer, make_credit_handler


# ── State ─────────────────────────────────────────────────────────────────────

class PersistentStateV7(PersistentStateV6):
    """V6 state + task board + credit scorer."""

    def __init__(self, db_path: str = "clawbizarre.db"):
        super().__init__(db_path)

        # Task board — uses same db dir, separate file
        task_db = db_path.replace(".db", "_tasks.db") if db_path != ":memory:" else ":memory:"
        self.task_board = TaskBoard(
            db_path=task_db,
            verify_url=f"http://localhost:{os.environ.get('CLAWBIZARRE_PORT', '8420')}",
            claim_ttl_s=int(os.environ.get("CLAWBIZARRE_CLAIM_TTL", "1800")),
            auto_repost=True,
        )

        # Credit scorer — stateless, uses receipts from persistence layer
        self.credit_scorer = CreditScorer()
        self.credit_handler = make_credit_handler(self.credit_scorer)


# ── Handler ───────────────────────────────────────────────────────────────────

class APIv7Handler(APIv6Handler):
    """Extends v6 with task board and credit endpoints."""

    state: PersistentStateV7

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path
        params = {k: v[0] for k, v in parse_qs(parsed.query).items()}

        # ── Health (v7 override) ──
        if path == "/health":
            self._handle_health_v7()

        # ── Task board endpoints ──
        elif path in ("/tasks", "/tasks/"):
            self._handle_tasks_list(params)
        elif path == "/tasks/stats":
            self._send_json(self.state.task_board.stats())
        elif path.startswith("/tasks/"):
            task_id = path[len("/tasks/"):]
            task = self.state.task_board.get_task(task_id)
            if task:
                self._send_json(task.to_dict())
            else:
                self._send_error(404, f"Task {task_id} not found")

        # ── Credit endpoints ──
        elif path == "/credit/tiers":
            self._send_json(self.state.credit_handler("/credit/tiers", {}))

        else:
            # Delegate to v6 (skip its do_GET header since we handle health above)
            super().do_GET()

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path
        body = self._read_body()

        # ── Task board endpoints ──
        if path == "/tasks" or path == "/tasks/":
            self._handle_task_post(body)
        elif path.endswith("/claim"):
            task_id = path[len("/tasks/"):-len("/claim")]
            self._send_json(self.state.task_board.handle_request("POST", path, body))
        elif path.endswith("/submit"):
            task_id = path[len("/tasks/"):-len("/submit")]
            self._send_json(self.state.task_board.handle_request("POST", path, body))
        elif path.endswith("/cancel"):
            task_id = path[len("/tasks/"):-len("/cancel")]
            self._send_json(self.state.task_board.handle_request("POST", path, body))

        # ── Credit endpoints ──
        elif path in ("/credit/score", "/credit/line", "/credit/project"):
            self._send_json(self.state.credit_handler(path, body))

        else:
            # Delegate to v6
            super().do_POST()

    # ── Task helpers ──────────────────────────────────────────────────────────

    def _handle_tasks_list(self, params: dict):
        """GET /tasks with optional query params."""
        tasks = self.state.task_board.list_tasks(
            task_type=params.get("task_type"),
            min_credits=float(params["min_credits"]) if "min_credits" in params else None,
            max_credits=float(params["max_credits"]) if "max_credits" in params else None,
            required_capability=params.get("capability"),
            receipt_count=int(params.get("receipt_count", 0)),
            limit=int(params.get("limit", 20)),
        )
        self._send_json({"tasks": [t.to_dict() for t in tasks], "count": len(tasks)})

    def _handle_task_post(self, body: dict):
        """POST /tasks — buyer posts a new task."""
        req = TaskRequirements(
            task_type=body.get("task_type", "general"),
            capabilities=body.get("capabilities", []),
            language=body.get("language", "python"),
            test_suite=body.get("test_suite"),
            schema=body.get("schema"),
            min_tier=body.get("min_tier", "bootstrap"),
            verification_tier=int(body.get("verification_tier", 0)),
        )
        bud = TaskBudget(
            credits=float(body.get("credits", 1.0)),
            max_task_usd=float(body.get("max_task_usd", 0.10)),
            payment_protocol=body.get("payment_protocol", "credits"),
            escrow=bool(body.get("escrow", False)),
        )
        try:
            priority = TaskPriority(body.get("priority", "normal"))
        except ValueError:
            priority = TaskPriority.NORMAL

        task = self.state.task_board.post_task(
            title=body.get("title", "Untitled Task"),
            description=body.get("description", ""),
            requirements=req,
            budget=bud,
            buyer_id=body.get("buyer_id", "anonymous"),
            priority=priority,
            deadline_hours=body.get("deadline_hours"),
        )
        self._send_json({"task_id": task.task_id, "status": "pending", "task": task.to_dict()})

    # ── Health ────────────────────────────────────────────────────────────────

    def _handle_health_v7(self):
        """Updated /health with v7 capabilities."""
        try:
            from lightweight_runner import LightweightRunner, _find_node, _check_docker
            docker_avail = _check_docker()
            node_avail = _find_node() is not None
        except ImportError:
            docker_avail = False
            node_avail = False

        stats = self.state.task_board.stats()
        self._send_json({
            "status": "ok",
            "version": "0.9.0",
            "service": "clawbizarre-verify",
            "receipts_stored": self.state.db.count_receipts() if hasattr(self.state.db, 'count_receipts') else 0,
            "capabilities": {
                "verification_tiers": [0, 1],
                "multilang": docker_avail or node_avail,
                "languages": (
                    ["python", "javascript", "node", "bash"] if docker_avail
                    else (["python", "javascript"] if node_avail else ["python"])
                ),
                "execution_backends": {
                    "docker": docker_avail,
                    "lightweight": node_avail,
                    "native_python": True,
                },
                "task_board": True,
                "credit_scoring": True,
                "scitt_transparency": True,
                "acp_support": True,
                "a2a_support": True,
                "mcp_support": True,
                "cose_signing": True,
            },
            "task_board": {
                "total_tasks": stats.get("total_tasks", 0),
                "pending": stats.get("by_status", {}).get("pending", 0),
                "complete": stats.get("by_status", {}).get("complete", 0),
            },
        })

    # ── Body reader ───────────────────────────────────────────────────────────

    def _read_body(self) -> dict:
        content_length = int(self.headers.get("Content-Length", 0))
        if content_length == 0:
            return {}
        try:
            return json.loads(self.rfile.read(content_length))
        except (json.JSONDecodeError, Exception):
            return {}


# ── Server runner ─────────────────────────────────────────────────────────────

def run_server(port: int = 8420, db_path: str = "clawbizarre.db", **kwargs):
    """Start ClawBizarre API v7."""
    state = PersistentStateV7(db_path=db_path)

    class BoundHandler(APIv7Handler):
        pass
    BoundHandler.state = state

    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("", port), BoundHandler)
    print(f"[v7] ClawBizarre API listening on port {port}")
    print(f"[v7] Task board: POST/GET /tasks")
    print(f"[v7] Credit API: POST /credit/{{score,line,project}}")
    print(f"[v7] Health: GET /health")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[v7] Shutting down")
        server.shutdown()


# ── Tests ─────────────────────────────────────────────────────────────────────

def _run_tests():
    """Quick smoke test of v7 endpoints."""
    import urllib.request
    import threading

    state = PersistentStateV7(db_path=":memory:")

    class BoundHandler(APIv7Handler):
        pass
    BoundHandler.state = state

    from http.server import ThreadingHTTPServer
    server = ThreadingHTTPServer(("localhost", 18788), BoundHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    base = "http://localhost:18788"
    errors = []

    def req(method: str, path: str, body=None):
        data = json.dumps(body).encode() if body else None
        r = urllib.request.Request(
            f"{base}{path}",
            data=data,
            headers={"Content-Type": "application/json"},
            method=method,
        )
        with urllib.request.urlopen(r, timeout=5) as resp:
            return json.loads(resp.read())

    tests = [
        ("GET /health", lambda: req("GET", "/health")["version"] == "0.9.0"),
        ("GET /tasks empty", lambda: req("GET", "/tasks")["count"] == 0),
        ("POST /tasks", lambda: "task_id" in req("POST", "/tasks", {
            "title": "Sort test", "task_type": "code",
            "capabilities": ["python"], "credits": 5.0, "buyer_id": "b:001",
        })),
        ("GET /tasks count", lambda: req("GET", "/tasks")["count"] == 1),
        ("GET /tasks/stats", lambda: "total_tasks" in req("GET", "/tasks/stats")),
        ("GET /credit/tiers", lambda: len(req("GET", "/credit/tiers")["tiers"]) == 5),
        ("POST /credit/score", lambda: "total" in req("POST", "/credit/score", {"receipts": []})),
        ("POST /credit/line", lambda: "daily_usd" in req("POST", "/credit/line", {"receipts": []})),
    ]

    passed = 0
    for name, test in tests:
        try:
            result = test()
            if result:
                print(f"  ✓ {name}")
                passed += 1
            else:
                print(f"  ✗ {name} (returned False)")
                errors.append(name)
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            errors.append(name)

    server.shutdown()
    print(f"\n{passed}/{len(tests)} tests passed")
    return len(errors) == 0


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="ClawBizarre API v7")
    parser.add_argument("--port", type=int, default=8420)
    parser.add_argument("--db", default="clawbizarre.db")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    if args.test:
        success = _run_tests()
        sys.exit(0 if success else 1)
    else:
        run_server(port=args.port, db_path=args.db)
