"""
ClawBizarre Verification Server — Production Hardened
Wraps verify_server.py with production-ready middleware:
  - Rate limiting (token bucket per IP)
  - Request size caps (50KB default)
  - Request ID tracking (X-Request-Id)
  - Structured JSON logging
  - CORS headers
  - Graceful shutdown (SIGTERM/SIGINT)
  - API key auth (optional, via --api-key or CLAWBIZARRE_API_KEY)

Usage:
    python3 verify_server_hardened.py [--port 8700] [--host 0.0.0.0]
    python3 verify_server_hardened.py --api-key sk-xxx
    python3 verify_server_hardened.py --test
"""

import json
import os
import signal
import sys
import time
import uuid
import logging
from collections import defaultdict
from dataclasses import dataclass
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock, Event
from typing import Optional, Any

# Import the core engine
_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

from verify_server import VerificationEngine, VerifyHandler as _BaseHandler

try:
    from identity import AgentIdentity
    HAS_IDENTITY = True
except ImportError:
    HAS_IDENTITY = False

try:
    from docker_runner import run_tests as docker_run_tests
    HAS_DOCKER_RUNNER = True
except ImportError:
    HAS_DOCKER_RUNNER = False

try:
    from receipt_store import ReceiptStore
    HAS_RECEIPT_STORE = True
except ImportError:
    HAS_RECEIPT_STORE = False


# ── Structured Logging ──────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts": self.formatTime(record),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        if hasattr(record, "request_id"):
            entry["request_id"] = record.request_id
        if hasattr(record, "client_ip"):
            entry["client_ip"] = record.client_ip
        if hasattr(record, "method"):
            entry["method"] = record.method
        if hasattr(record, "path"):
            entry["path_"] = record.path
        if hasattr(record, "status_code"):
            entry["status"] = record.status_code
        if hasattr(record, "duration_ms"):
            entry["duration_ms"] = record.duration_ms
        if hasattr(record, "extra_data"):
            entry.update(record.extra_data)
        return json.dumps(entry)

logger = logging.getLogger("clawbizarre")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(JSONFormatter())
logger.addHandler(handler)


# ── Rate Limiter (Token Bucket) ─────────────────────────────────────

@dataclass
class Bucket:
    tokens: float
    last_refill: float

class RateLimiter:
    """Per-IP token bucket rate limiter."""
    
    def __init__(self, rate: float = 10.0, burst: int = 20):
        """rate: tokens per second. burst: max bucket size."""
        self.rate = rate
        self.burst = burst
        self.buckets: dict[str, Bucket] = {}
        self.lock = Lock()
    
    def allow(self, key: str) -> bool:
        now = time.monotonic()
        with self.lock:
            b = self.buckets.get(key)
            if b is None:
                self.buckets[key] = Bucket(tokens=self.burst - 1, last_refill=now)
                return True
            # Refill
            elapsed = now - b.last_refill
            b.tokens = min(self.burst, b.tokens + elapsed * self.rate)
            b.last_refill = now
            if b.tokens >= 1:
                b.tokens -= 1
                return True
            return False
    
    def cleanup(self, max_age: float = 3600):
        """Remove stale buckets."""
        now = time.monotonic()
        with self.lock:
            stale = [k for k, b in self.buckets.items() if now - b.last_refill > max_age]
            for k in stale:
                del self.buckets[k]


# ── Config ──────────────────────────────────────────────────────────

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8700
    keyfile: Optional[str] = None
    api_key: Optional[str] = None  # If set, require Authorization: Bearer <key>
    max_body_bytes: int = 50 * 1024  # 50KB
    rate_limit: float = 10.0  # requests/sec per IP
    rate_burst: int = 20
    cors_origins: str = "*"
    db_path: Optional[str] = "receipts.db"  # SQLite receipt persistence (None = in-memory only)


# ── Hardened Handler ────────────────────────────────────────────────

class HardenedVerifyHandler(BaseHTTPRequestHandler):
    engine: VerificationEngine = None
    config: ServerConfig = None
    rate_limiter: RateLimiter = None
    receipt_store: Optional[Any] = None  # ReceiptStore if available
    _request_count: int = 0
    _request_count_lock = Lock()
    
    def _get_request_id(self) -> str:
        return self.headers.get("X-Request-Id", str(uuid.uuid4())[:12])
    
    def _get_client_ip(self) -> str:
        # Support reverse proxy
        forwarded = self.headers.get("X-Forwarded-For")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return self.client_address[0]
    
    def _cors_headers(self):
        origin = self.config.cors_origins if self.config else "*"
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-Id")
        self.send_header("Access-Control-Max-Age", "86400")
    
    def _json_response(self, code: int, data: Any, request_id: str = ""):
        body = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        if request_id:
            self.send_header("X-Request-Id", request_id)
        self._cors_headers()
        self.end_headers()
        self.wfile.write(body)
    
    def _read_body(self, request_id: str) -> Optional[dict]:
        length = int(self.headers.get("Content-Length", 0))
        if length > self.config.max_body_bytes:
            self._json_response(413, {
                "error": f"Request body too large ({length} bytes, max {self.config.max_body_bytes})",
                "request_id": request_id,
            }, request_id)
            return None
        if length == 0:
            return {}
        raw = self.rfile.read(length)
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"Invalid JSON: {e}", "request_id": request_id}, request_id)
            return None
    
    def _check_rate_limit(self, client_ip: str, request_id: str) -> bool:
        if not self.rate_limiter.allow(client_ip):
            self._json_response(429, {
                "error": "Rate limit exceeded",
                "request_id": request_id,
                "retry_after_seconds": 1,
            }, request_id)
            return False
        return True
    
    def _check_auth(self, request_id: str) -> bool:
        if not self.config.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {self.config.api_key}":
            return True
        # Allow unauthenticated access to /health and /metrics
        if self.path in ("/health", "/metrics", "/docs"):
            return True
        self._json_response(401, {"error": "Unauthorized", "request_id": request_id}, request_id)
        return False
    
    def _inc_count(self) -> int:
        with self._request_count_lock:
            HardenedVerifyHandler._request_count += 1
            return HardenedVerifyHandler._request_count
    
    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()
    
    def do_GET(self):
        start = time.monotonic()
        request_id = self._get_request_id()
        client_ip = self._get_client_ip()
        
        if not self._check_rate_limit(client_ip, request_id):
            return
        if not self._check_auth(request_id):
            return
        
        if self.path == "/docs":
            self._json_response(200, {
                "service": "ClawBizarre Structural Verification",
                "version": "1.0-hardened",
                "endpoints": {
                    "POST /verify": {
                        "description": "Verify code against a test suite (Tier 0) or schema (Tier 1)",
                        "auth": "Bearer token via Authorization header (if configured)",
                        "body": {
                            "output": {"content": "<code string>"},
                            "verification": {
                                "tier": 0,
                                "test_suite": {
                                    "language": "python|javascript|bash",
                                    "tests": [
                                        {"name": "test_name", "input": "<eval expression>", "expected_output": "<eval expression>"},
                                        {"name": "test_name", "expression": "<expression>", "expected": "<expected>"}
                                    ]
                                }
                            }
                        },
                        "example": {
                            "output": {"content": "def add(a, b): return a + b"},
                            "verification": {
                                "tier": 0,
                                "test_suite": {
                                    "language": "python",
                                    "tests": [
                                        {"name": "add_basic", "input": "add(1, 2)", "expected_output": "3"},
                                        {"name": "add_neg", "input": "add(-1, 1)", "expected_output": "0"}
                                    ]
                                }
                            }
                        },
                        "response": "VRF receipt with verdict (pass|fail|error|partial), test results, and Ed25519 signature"
                    },
                    "GET /health": "Service health and capabilities",
                    "GET /metrics": "Prometheus-compatible metrics",
                    "GET /stats": "Service statistics",
                    "GET /docs": "This documentation",
                    "GET /receipt/<id>": "Retrieve a stored VRF receipt",
                    "POST /receipt/verify": "Verify an Ed25519-signed receipt",
                    "GET /receipts": "Query receipts (params: verdict, task_type, output_hash, since, limit, offset)"
                },
                "notes": [
                    "Python tests use input/expected_output format (eval'd in code namespace)",
                    "Docker tests (JS/bash) use expression/expected format",
                    "All receipts are Ed25519-signed with VRF v1.0 format",
                    "Rate limiting: token bucket per IP (default 10 req/s)"
                ]
            }, request_id)
        elif self.path == "/health":
            self._json_response(200, {
                "status": "ok",
                "version": "clawbizarre-verify/1.0-hardened",
                "vrf_version": "1.0",
                "multilang": HAS_DOCKER_RUNNER,
                "languages": ["python"] + (["javascript", "node", "bash"] if HAS_DOCKER_RUNNER else []),
                "requests_served": HardenedVerifyHandler._request_count,
                "auth_required": self.config.api_key is not None,
            }, request_id)
        elif self.path == "/metrics":
            # Prometheus-compatible text format
            count = HardenedVerifyHandler._request_count
            receipts = self.receipt_store.count() if self.receipt_store else len(self.engine.receipts)
            body = (
                f"# HELP clawbizarre_requests_total Total requests served\n"
                f"# TYPE clawbizarre_requests_total counter\n"
                f"clawbizarre_requests_total {count}\n"
                f"# HELP clawbizarre_receipts_stored Receipts in memory\n"
                f"# TYPE clawbizarre_receipts_stored gauge\n"
                f"clawbizarre_receipts_stored {receipts}\n"
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path.startswith("/receipt/"):
            receipt_id = self.path.split("/receipt/")[1]
            # Check SQLite first, then in-memory
            receipt_dict = None
            if self.receipt_store:
                receipt_dict = self.receipt_store.get(receipt_id)
            if not receipt_dict:
                receipt = self.engine.get_receipt(receipt_id)
                if receipt:
                    receipt_dict = receipt.to_dict()
            if receipt_dict:
                self._json_response(200, receipt_dict, request_id)
            else:
                self._json_response(404, {"error": "Receipt not found", "request_id": request_id}, request_id)
        elif self.path == "/stats":
            self._json_response(200, {
                "receipts_stored": self.receipt_store.count() if self.receipt_store else len(self.engine.receipts),
                "persistence": "sqlite" if self.receipt_store else "memory",
                "supported_tiers": [0, 1],
                "version": "clawbizarre-verify/1.0-hardened",
                "requests_served": HardenedVerifyHandler._request_count,
            }, request_id)
        elif self.path.startswith("/receipts"):
            if not self.receipt_store:
                self._json_response(501, {"error": "Receipt persistence not enabled", "request_id": request_id}, request_id)
            else:
                # Parse query params: ?verdict=pass&limit=10&offset=0
                import urllib.parse
                parts = urllib.parse.urlparse(self.path)
                params = urllib.parse.parse_qs(parts.query)
                results = self.receipt_store.query(
                    verdict=params.get("verdict", [None])[0],
                    task_type=params.get("task_type", [None])[0],
                    output_hash=params.get("output_hash", [None])[0],
                    since=params.get("since", [None])[0],
                    limit=int(params.get("limit", [50])[0]),
                    offset=int(params.get("offset", [0])[0]),
                )
                self._json_response(200, {"receipts": results, "count": len(results), "request_id": request_id}, request_id)
        else:
            self._json_response(404, {"error": "Not found", "request_id": request_id}, request_id)
        
        self._inc_count()
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        extra = logging.LogRecord("", 0, "", 0, "", (), None)
        logger.info("request", extra={"request_id": request_id, "client_ip": client_ip,
                                       "method": "GET", "path": self.path, 
                                       "duration_ms": duration_ms})
    
    def do_POST(self):
        start = time.monotonic()
        request_id = self._get_request_id()
        client_ip = self._get_client_ip()
        
        if not self._check_rate_limit(client_ip, request_id):
            return
        if not self._check_auth(request_id):
            return
        
        body = self._read_body(request_id)
        if body is None:
            return
        
        status_code = 200
        try:
            if self.path == "/verify":
                receipt = self.engine.verify(body)
                if self.receipt_store:
                    self.receipt_store.save(receipt)
                self._json_response(200, {**receipt.to_dict(), "request_id": request_id}, request_id)
            elif self.path == "/verify/schema":
                body["tier"] = 1
                receipt = self.engine.verify(body)
                if self.receipt_store:
                    self.receipt_store.save(receipt)
                self._json_response(200, {**receipt.to_dict(), "request_id": request_id}, request_id)
            elif self.path == "/receipt/verify":
                receipt_data = body.get("receipt", body)
                result = VerificationEngine.verify_receipt_signature(receipt_data)
                self._json_response(200, {**result, "request_id": request_id}, request_id)
            else:
                status_code = 404
                self._json_response(404, {"error": "Not found", "request_id": request_id}, request_id)
        except ValueError as e:
            status_code = 400
            self._json_response(400, {"error": str(e), "request_id": request_id}, request_id)
        except Exception as e:
            status_code = 500
            logger.error(f"Internal error: {e}", extra={"request_id": request_id, "client_ip": client_ip})
            self._json_response(500, {"error": "Internal server error", "request_id": request_id}, request_id)
        
        self._inc_count()
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info("request", extra={"request_id": request_id, "client_ip": client_ip,
                                       "method": "POST", "path": self.path,
                                       "status_code": status_code, "duration_ms": duration_ms})
    
    def log_message(self, format, *args):
        pass  # Suppress default access logs (we use structured logging)


# ── Server with Graceful Shutdown ───────────────────────────────────

class GracefulServer:
    def __init__(self, config: ServerConfig):
        self.config = config
        self.shutdown_event = Event()
        self.server: Optional[HTTPServer] = None
    
    def start(self):
        # Setup identity
        identity = None
        if HAS_IDENTITY:
            if self.config.keyfile and os.path.exists(self.config.keyfile):
                identity = AgentIdentity.from_keyfile(self.config.keyfile)
                logger.info(f"Loaded identity: {identity.agent_id[:24]}...")
            else:
                identity = AgentIdentity.generate()
                if self.config.keyfile:
                    identity.save_keyfile(self.config.keyfile)
                    logger.info(f"Generated identity (saved): {identity.agent_id[:24]}...")
                else:
                    logger.info(f"Generated ephemeral identity: {identity.agent_id[:24]}...")
        
        engine = VerificationEngine(identity=identity)
        HardenedVerifyHandler.engine = engine
        HardenedVerifyHandler.config = self.config
        HardenedVerifyHandler.rate_limiter = RateLimiter(self.config.rate_limit, self.config.rate_burst)
        
        # Receipt persistence
        if HAS_RECEIPT_STORE and self.config.db_path:
            self.receipt_store = ReceiptStore(self.config.db_path)
            HardenedVerifyHandler.receipt_store = self.receipt_store
            logger.info(f"Receipt persistence: SQLite ({self.config.db_path})")
        else:
            self.receipt_store = None
            HardenedVerifyHandler.receipt_store = None
            logger.info("Receipt persistence: in-memory only")
        
        self.server = HTTPServer((self.config.host, self.config.port), HardenedVerifyHandler)
        
        # Graceful shutdown on signals
        def _shutdown(signum, frame):
            logger.info(f"Received signal {signum}, shutting down gracefully...")
            self.shutdown_event.set()
            Thread(target=self.server.shutdown, daemon=True).start()
        
        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)
        
        logger.info(f"ClawBizarre Verify v1.0 (hardened) listening on {self.config.host}:{self.config.port}")
        if self.config.api_key:
            logger.info("API key authentication: ENABLED")
        else:
            logger.info("API key authentication: DISABLED (open access)")
        
        self.server.serve_forever()
        logger.info("Server stopped.")


# ── Tests ────────────────────────────────────────────────────────────

def run_tests():
    import urllib.request
    import urllib.error
    import tempfile
    
    identity = None
    if HAS_IDENTITY:
        identity = AgentIdentity.generate()
    engine = VerificationEngine(identity=identity)
    
    # Use temp SQLite DB for persistence tests
    tmpdir = tempfile.mkdtemp()
    test_db_path = os.path.join(tmpdir, "test_receipts.db")
    test_store = ReceiptStore(test_db_path) if HAS_RECEIPT_STORE else None
    
    config = ServerConfig(host="127.0.0.1", port=0, api_key="test-key-123", max_body_bytes=1024, db_path=test_db_path)
    HardenedVerifyHandler.engine = engine
    HardenedVerifyHandler.config = config
    HardenedVerifyHandler.rate_limiter = RateLimiter(rate=100.0, burst=200)  # High for tests
    HardenedVerifyHandler.receipt_store = test_store
    HardenedVerifyHandler._request_count = 0
    
    server = HTTPServer(("127.0.0.1", 0), HardenedVerifyHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    
    def req(method, path, data=None, headers=None):
        hdrs = {"Content-Type": "application/json", "Authorization": "Bearer test-key-123"}
        if headers:
            hdrs.update(headers)
        body = json.dumps(data).encode() if data else None
        r = urllib.request.Request(f"{base}{path}", data=body, headers=hdrs, method=method)
        resp = urllib.request.urlopen(r)
        return json.loads(resp.read()), dict(resp.headers)
    
    def post(path, data, headers=None):
        return req("POST", path, data, headers)
    
    def get(path, headers=None):
        return req("GET", path, headers=headers)
    
    passed = total = 0
    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")
    
    # 1. Health (no auth required)
    print("\n1. Health (unauthenticated)")
    r = urllib.request.urlopen(f"{base}/health")
    h = json.loads(r.read())
    check("status ok", h["status"] == "ok")
    check("hardened version", "hardened" in h["version"])
    check("auth_required flag", h["auth_required"] == True)
    
    # 2. Auth required for /verify
    print("\n2. Auth enforcement")
    try:
        r = urllib.request.Request(f"{base}/verify", data=b'{}', headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(r)
        check("401 without auth", False)
    except urllib.error.HTTPError as e:
        check("401 without auth", e.code == 401)
    
    # 3. Request ID tracking
    print("\n3. Request ID tracking")
    data, hdrs = get("/stats", headers={"X-Request-Id": "my-id-123"})
    check("request_id echoed", hdrs.get("X-Request-Id") == "my-id-123")
    
    # 4. Request size limit
    print("\n4. Request size limit (1KB for test)")
    try:
        big_payload = {"output": {"content": "x" * 2000}}
        r = urllib.request.Request(
            f"{base}/verify",
            data=json.dumps(big_payload).encode(),
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key-123"},
            method="POST"
        )
        urllib.request.urlopen(r)
        check("413 for oversized body", False)
    except urllib.error.HTTPError as e:
        check("413 for oversized body", e.code == 413)
    
    # 5. CORS headers
    print("\n5. CORS headers")
    data, hdrs = get("/health")
    check("CORS origin header", hdrs.get("Access-Control-Allow-Origin") == "*")
    
    # 6. Verify still works through hardened layer
    print("\n6. Verification through hardened handler")
    data_from_verify, _ = post("/verify", {
        "tier": 0,
        "output": {"content": "def f(x): return x+1"},
        "verification": {"test_suite": {"tests": [
            {"name": "t1", "input": "f(1)", "expected_output": "2"},
        ]}}
    })
    check("verdict pass", data_from_verify["verdict"] == "pass")
    check("request_id in response", "request_id" in data_from_verify)
    
    # 7. Metrics endpoint
    print("\n7. Metrics endpoint")
    r = urllib.request.urlopen(f"{base}/metrics")
    body = r.read().decode()
    check("prometheus format", "clawbizarre_requests_total" in body)
    check("receipts metric", "clawbizarre_receipts_stored" in body)
    
    # 8. Bad JSON
    print("\n8. Bad JSON body")
    try:
        r = urllib.request.Request(
            f"{base}/verify",
            data=b"not json{",
            headers={"Content-Type": "application/json", "Authorization": "Bearer test-key-123",
                     "Content-Length": "9"},
            method="POST"
        )
        urllib.request.urlopen(r)
        check("400 for bad JSON", False)
    except urllib.error.HTTPError as e:
        check("400 for bad JSON", e.code == 400)
    
    # 9. Wrong API key
    print("\n9. Wrong API key")
    try:
        r = urllib.request.Request(
            f"{base}/stats",
            headers={"Authorization": "Bearer wrong-key"},
        )
        urllib.request.urlopen(r)
        check("401 for wrong key", False)
    except urllib.error.HTTPError as e:
        check("401 for wrong key", e.code == 401)
    
    # 10. Stats shows request count
    print("\n10. Stats request count")
    data, _ = get("/stats")
    check("requests_served > 0", data["requests_served"] > 0)
    check("persistence type", data.get("persistence") == ("sqlite" if test_store else "memory"))
    
    # 11. Receipt persistence (SQLite)
    if test_store:
        print("\n11. Receipt persistence")
        # The verify in test 6 should have persisted the receipt
        check("receipt persisted to SQLite", test_store.count() >= 1)
        
        # Get receipt by ID from the API
        receipt_id = data_from_verify["receipt_id"] if "data_from_verify" in dir() else None
        # Query via /receipts endpoint
        data_q, _ = get("/receipts?verdict=pass")
        check("/receipts query works", "receipts" in data_q)
        check("pass receipts found", data_q["count"] >= 1)
        
        # Verify receipt survives in DB directly
        all_receipts = test_store.query()
        check("receipts in DB", len(all_receipts) >= 1)
        check("receipt has verdict", all_receipts[0].get("verdict") in ("pass", "fail", "error", "partial"))
        
        # Stats from store
        stats = test_store.stats()
        check("stats total > 0", stats["total_receipts"] > 0)
    else:
        print("\n11. Receipt persistence (SKIPPED - receipt_store not available)")
    
    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    
    server.shutdown()
    if test_store:
        test_store.close()
    return passed == total


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        config = ServerConfig()
        args = sys.argv[1:]
        for i, arg in enumerate(args):
            if arg == "--port" and i + 1 < len(args):
                config.port = int(args[i + 1])
            elif arg == "--host" and i + 1 < len(args):
                config.host = args[i + 1]
            elif arg == "--keyfile" and i + 1 < len(args):
                config.keyfile = args[i + 1]
            elif arg == "--api-key" and i + 1 < len(args):
                config.api_key = args[i + 1]
            elif arg == "--max-body" and i + 1 < len(args):
                config.max_body_bytes = int(args[i + 1])
            elif arg == "--rate-limit" and i + 1 < len(args):
                config.rate_limit = float(args[i + 1])
            elif arg == "--db" and i + 1 < len(args):
                config.db_path = args[i + 1]
            elif arg == "--no-db":
                config.db_path = None
        
        # Env var fallback for API key
        if not config.api_key:
            config.api_key = os.environ.get("CLAWBIZARRE_API_KEY")
        
        gs = GracefulServer(config)
        gs.start()
