#!/usr/bin/env python3
"""
ClawBizarre Unified Verification + Transparency Server

Combines verify_server_hardened.py + transparency_server.py into a single
deployable service. One port, one process, shared auth/rate-limiting.

Features:
  - All verify_server_hardened endpoints (/verify, /health, /receipts, etc.)
  - All transparency endpoints (/transparency/register, /proof, etc.)
  - Auto-register: verified receipts optionally COSE-signed and pushed to transparency log
  - Shared auth, rate limiting, CORS, structured logging
  - Single SQLite DB for receipts + separate DB for transparency log

Usage:
    python3 verify_server_unified.py [--port 8700] [--test]
    python3 verify_server_unified.py --api-key sk-xxx --auto-register
"""

import base64
import json
import os
import signal
import sys
import time
import uuid
import logging
import tempfile
from collections import defaultdict
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Thread, Lock, Event
from typing import Optional, Any
from urllib.parse import urlparse, parse_qs

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

from verify_server import VerificationEngine

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

try:
    from merkle_store import MerkleStore
    HAS_MERKLE_STORE = True
except ImportError:
    HAS_MERKLE_STORE = False

try:
    from vrf_cose import VRFCoseReceipt
    HAS_COSE = True
except ImportError:
    HAS_COSE = False


# ── Structured Logging ──────────────────────────────────────────────

class JSONFormatter(logging.Formatter):
    def format(self, record):
        entry = {
            "ts": self.formatTime(record),
            "level": record.levelname.lower(),
            "msg": record.getMessage(),
        }
        for attr in ("request_id", "client_ip", "method", "status_code", "duration_ms"):
            if hasattr(record, attr):
                entry[attr] = getattr(record, attr)
        if hasattr(record, "path"):
            entry["path_"] = record.path
        return json.dumps(entry)

logger = logging.getLogger("clawbizarre")
logger.setLevel(logging.INFO)
_handler = logging.StreamHandler()
_handler.setFormatter(JSONFormatter())
logger.addHandler(_handler)


# ── Rate Limiter ────────────────────────────────────────────────────

@dataclass
class Bucket:
    tokens: float
    last_refill: float

class RateLimiter:
    def __init__(self, rate: float = 10.0, burst: int = 20):
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
            elapsed = now - b.last_refill
            b.tokens = min(self.burst, b.tokens + elapsed * self.rate)
            b.last_refill = now
            if b.tokens >= 1:
                b.tokens -= 1
                return True
            return False


# ── Config ──────────────────────────────────────────────────────────

@dataclass
class UnifiedConfig:
    host: str = "0.0.0.0"
    port: int = 8700
    keyfile: Optional[str] = None
    api_key: Optional[str] = None
    max_body_bytes: int = 1024 * 1024  # 1MB (transparency needs larger)
    rate_limit: float = 10.0
    rate_burst: int = 20
    cors_origins: str = "*"
    receipt_db: Optional[str] = "receipts.db"
    transparency_db: Optional[str] = "transparency.db"
    auto_register: bool = False  # Auto-push verified receipts to transparency log


# ── Unified Handler ─────────────────────────────────────────────────

class UnifiedHandler(BaseHTTPRequestHandler):
    engine: VerificationEngine = None
    config: UnifiedConfig = None
    rate_limiter: RateLimiter = None
    receipt_store: Optional[Any] = None
    merkle_store: Optional[Any] = None
    identity: Optional[Any] = None
    _request_count: int = 0
    _request_lock = Lock()
    _transparency_lock = Lock()

    def _get_request_id(self) -> str:
        return self.headers.get("X-Request-Id", str(uuid.uuid4())[:12])

    def _get_client_ip(self) -> str:
        fwd = self.headers.get("X-Forwarded-For")
        return fwd.split(",")[0].strip() if fwd else self.client_address[0]

    def _cors_headers(self):
        origin = self.config.cors_origins if self.config else "*"
        self.send_header("Access-Control-Allow-Origin", origin)
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization, X-Request-Id")

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

    def _read_body_raw(self, request_id: str) -> Optional[bytes]:
        length = int(self.headers.get("Content-Length", 0))
        if length > self.config.max_body_bytes:
            self._json_response(413, {"error": "Body too large", "request_id": request_id}, request_id)
            return None
        return self.rfile.read(length) if length > 0 else b""

    def _read_body_json(self, request_id: str) -> Optional[dict]:
        raw = self._read_body_raw(request_id)
        if raw is None:
            return None
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as e:
            self._json_response(400, {"error": f"Invalid JSON: {e}", "request_id": request_id}, request_id)
            return None

    def _check_rate_limit(self, client_ip: str, request_id: str) -> bool:
        if not self.rate_limiter.allow(client_ip):
            self._json_response(429, {"error": "Rate limit exceeded", "request_id": request_id}, request_id)
            return False
        return True

    def _check_auth(self, request_id: str) -> bool:
        if not self.config.api_key:
            return True
        auth = self.headers.get("Authorization", "")
        if auth == f"Bearer {self.config.api_key}":
            return True
        # Allow unauthenticated for health/docs/metrics
        if self.path.rstrip("/") in ("/health", "/metrics", "/docs"):
            return True
        self._json_response(401, {"error": "Unauthorized", "request_id": request_id}, request_id)
        return False

    def _inc_count(self) -> int:
        with self._request_lock:
            UnifiedHandler._request_count += 1
            return UnifiedHandler._request_count

    def _parse_path(self):
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/"), parse_qs(parsed.query)

    def _auto_register_receipt(self, receipt_dict: dict):
        """If auto_register is on and we have COSE + MerkleStore, push receipt to transparency log."""
        if not (self.config.auto_register and self.merkle_store and HAS_COSE and self.identity):
            return None
        try:
            seed = self.identity._private_key.private_bytes_raw()
            pub = self.identity._private_key.public_key().public_bytes_raw()
            cose_receipt = VRFCoseReceipt(
                receipt_data=receipt_dict,
                signing_key=seed,
                public_key=pub,
            )
            cose_data = cose_receipt.to_cose_sign1()
            with self._transparency_lock:
                tr = self.merkle_store.register(cose_data)
            return tr
        except Exception as e:
            logger.warning(f"Auto-register to transparency log failed: {e}")
            return None

    # ── GET routes ──

    def do_GET(self):
        start = time.monotonic()
        request_id = self._get_request_id()
        client_ip = self._get_client_ip()
        if not self._check_rate_limit(client_ip, request_id):
            return
        if not self._check_auth(request_id):
            return

        path, qs = self._parse_path()

        if path == "/health":
            self._handle_health(request_id)
        elif path == "/docs":
            self._handle_docs(request_id)
        elif path == "/metrics":
            self._handle_metrics(request_id)
        elif path == "/stats":
            self._handle_stats(request_id)
        elif path.startswith("/receipt/"):
            self._handle_get_receipt(path, request_id)
        elif path == "/receipts":
            self._handle_query_receipts(qs, request_id)
        # Transparency endpoints
        elif path == "/transparency/stats":
            self._handle_transparency_stats(request_id)
        elif path == "/transparency/root":
            self._handle_transparency_root(request_id)
        elif path.startswith("/transparency/proof/"):
            self._handle_transparency_proof(path, request_id)
        elif path.startswith("/transparency/receipt/"):
            self._handle_transparency_receipt(path, request_id)
        elif path == "/transparency/entries":
            self._handle_transparency_entries(qs, request_id)
        elif path == "/transparency/consistency":
            self._handle_transparency_consistency(qs, request_id)
        else:
            self._json_response(404, {"error": "Not found", "request_id": request_id}, request_id)

        self._inc_count()
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info("request", extra={"request_id": request_id, "client_ip": client_ip,
                                       "method": "GET", "path": self.path, "duration_ms": duration_ms})

    # ── POST routes ──

    def do_POST(self):
        start = time.monotonic()
        request_id = self._get_request_id()
        client_ip = self._get_client_ip()
        if not self._check_rate_limit(client_ip, request_id):
            return
        if not self._check_auth(request_id):
            return

        path, qs = self._parse_path()
        status_code = 200

        try:
            if path == "/verify":
                self._handle_verify(request_id)
            elif path == "/verify/schema":
                self._handle_verify_schema(request_id)
            elif path == "/receipt/verify":
                self._handle_verify_receipt_sig(request_id)
            elif path == "/transparency/register":
                self._handle_transparency_register(request_id)
            elif path == "/transparency/verify":
                self._handle_transparency_verify(request_id)
            else:
                self._json_response(404, {"error": "Not found", "request_id": request_id}, request_id)
        except ValueError as e:
            self._json_response(400, {"error": str(e), "request_id": request_id}, request_id)
        except Exception as e:
            logger.error(f"Internal error: {e}", extra={"request_id": request_id, "client_ip": client_ip})
            self._json_response(500, {"error": "Internal server error", "request_id": request_id}, request_id)

        self._inc_count()
        duration_ms = round((time.monotonic() - start) * 1000, 2)
        logger.info("request", extra={"request_id": request_id, "client_ip": client_ip,
                                       "method": "POST", "path": self.path, "duration_ms": duration_ms})

    def do_OPTIONS(self):
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def log_message(self, fmt, *args):
        pass

    # ── Verification handlers ──

    def _handle_health(self, rid):
        data = {
            "status": "ok",
            "version": "clawbizarre-unified/1.0",
            "vrf_version": "1.0",
            "multilang": HAS_DOCKER_RUNNER,
            "languages": ["python"] + (["javascript", "node", "bash"] if HAS_DOCKER_RUNNER else []),
            "auth_required": self.config.api_key is not None,
            "transparency": HAS_MERKLE_STORE and self.merkle_store is not None,
            "auto_register": self.config.auto_register,
            "requests_served": UnifiedHandler._request_count,
        }
        if self.merkle_store:
            data["tree_size"] = self.merkle_store._tree.size
        self._json_response(200, data, rid)

    def _handle_docs(self, rid):
        self._json_response(200, {
            "service": "ClawBizarre Unified Verification + Transparency",
            "version": "1.0",
            "verification_endpoints": {
                "POST /verify": "Verify code against test suite (Tier 0) or schema (Tier 1)",
                "POST /verify/schema": "Schema-only verification (Tier 1)",
                "POST /receipt/verify": "Verify Ed25519 receipt signature",
                "GET /receipt/<id>": "Get stored receipt by ID",
                "GET /receipts": "Query receipts (?verdict=&limit=&offset=)",
            },
            "transparency_endpoints": {
                "POST /transparency/register": "Submit COSE receipt to transparency log",
                "POST /transparency/verify": "Verify inclusion proof",
                "GET /transparency/proof/<seq>": "Get inclusion proof",
                "GET /transparency/receipt/<id>": "Lookup by receipt_id",
                "GET /transparency/entries": "List entries (?verdict=&issuer=&limit=&offset=)",
                "GET /transparency/consistency": "Consistency proof (?old_size=&new_size=)",
                "GET /transparency/stats": "Log statistics",
                "GET /transparency/root": "Current Merkle root + size",
            },
            "other": {
                "GET /health": "Health check",
                "GET /metrics": "Prometheus metrics",
                "GET /stats": "Service statistics",
                "GET /docs": "This page",
            },
            "notes": [
                "auto_register: verified receipts auto-pushed to transparency log as COSE Sign1",
                "All transparency data uses RFC 9162 Merkle trees (Certificate Transparency model)",
            ],
        }, rid)

    def _handle_metrics(self, rid):
        count = UnifiedHandler._request_count
        receipts = self.receipt_store.count() if self.receipt_store else len(self.engine.receipts)
        tree_size = self.merkle_store._tree.size if self.merkle_store else 0
        body = (
            f"# HELP clawbizarre_requests_total Total requests served\n"
            f"# TYPE clawbizarre_requests_total counter\n"
            f"clawbizarre_requests_total {count}\n"
            f"# HELP clawbizarre_receipts_stored Receipts stored\n"
            f"# TYPE clawbizarre_receipts_stored gauge\n"
            f"clawbizarre_receipts_stored {receipts}\n"
            f"# HELP clawbizarre_transparency_tree_size Transparency log entries\n"
            f"# TYPE clawbizarre_transparency_tree_size gauge\n"
            f"clawbizarre_transparency_tree_size {tree_size}\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; version=0.0.4")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _handle_stats(self, rid):
        data = {
            "receipts_stored": self.receipt_store.count() if self.receipt_store else len(self.engine.receipts),
            "persistence": "sqlite" if self.receipt_store else "memory",
            "transparency_entries": self.merkle_store._tree.size if self.merkle_store else 0,
            "auto_register": self.config.auto_register,
            "requests_served": UnifiedHandler._request_count,
            "version": "clawbizarre-unified/1.0",
        }
        self._json_response(200, data, rid)

    def _handle_get_receipt(self, path, rid):
        receipt_id = path.split("/receipt/")[1]
        receipt_dict = None
        if self.receipt_store:
            receipt_dict = self.receipt_store.get(receipt_id)
        if not receipt_dict:
            receipt = self.engine.get_receipt(receipt_id)
            if receipt:
                receipt_dict = receipt.to_dict()
        if receipt_dict:
            self._json_response(200, receipt_dict, rid)
        else:
            self._json_response(404, {"error": "Receipt not found", "request_id": rid}, rid)

    def _handle_query_receipts(self, qs, rid):
        if not self.receipt_store:
            self._json_response(501, {"error": "Receipt persistence not enabled", "request_id": rid}, rid)
            return
        results = self.receipt_store.query(
            verdict=qs.get("verdict", [None])[0],
            task_type=qs.get("task_type", [None])[0],
            output_hash=qs.get("output_hash", [None])[0],
            since=qs.get("since", [None])[0],
            limit=int(qs.get("limit", [50])[0]),
            offset=int(qs.get("offset", [0])[0]),
        )
        self._json_response(200, {"receipts": results, "count": len(results), "request_id": rid}, rid)

    def _handle_verify(self, rid):
        body = self._read_body_json(rid)
        if body is None:
            return
        receipt = self.engine.verify(body)
        if self.receipt_store:
            self.receipt_store.save(receipt)
        result = {**receipt.to_dict(), "request_id": rid}
        # Auto-register to transparency log
        tr = self._auto_register_receipt(receipt.to_dict())
        if tr:
            result["transparency_receipt"] = tr
        self._json_response(200, result, rid)

    def _handle_verify_schema(self, rid):
        body = self._read_body_json(rid)
        if body is None:
            return
        body["tier"] = 1
        receipt = self.engine.verify(body)
        if self.receipt_store:
            self.receipt_store.save(receipt)
        result = {**receipt.to_dict(), "request_id": rid}
        tr = self._auto_register_receipt(receipt.to_dict())
        if tr:
            result["transparency_receipt"] = tr
        self._json_response(200, result, rid)

    def _handle_verify_receipt_sig(self, rid):
        body = self._read_body_json(rid)
        if body is None:
            return
        receipt_data = body.get("receipt", body)
        result = VerificationEngine.verify_receipt_signature(receipt_data)
        self._json_response(200, {**result, "request_id": rid}, rid)

    # ── Transparency handlers ──

    def _handle_transparency_stats(self, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        self._json_response(200, self.merkle_store.stats(), rid)

    def _handle_transparency_root(self, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        self._json_response(200, {
            "tree_size": self.merkle_store._tree.size,
            "tree_root": self.merkle_store._tree.root.hex(),
        }, rid)

    def _handle_transparency_proof(self, path, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        try:
            seq = int(path.split("/")[-1])
            proof = self.merkle_store.get_proof(seq)
            self._json_response(200, proof, rid)
        except (ValueError, IndexError) as e:
            self._json_response(400, {"error": str(e)}, rid)

    def _handle_transparency_receipt(self, path, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        receipt_id = path.split("/transparency/receipt/", 1)[1]
        entry = self.merkle_store.lookup(receipt_id)
        if entry:
            seq = entry["sequence_number"]
            data = self.merkle_store.get_receipt_data(seq)
            entry["receipt_data"] = data
            self._json_response(200, entry, rid)
        else:
            self._json_response(404, {"error": f"Receipt '{receipt_id}' not found"}, rid)

    def _handle_transparency_entries(self, qs, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        entries = self.merkle_store.list_entries(
            limit=int(qs.get("limit", [50])[0]),
            offset=int(qs.get("offset", [0])[0]),
            verdict=qs.get("verdict", [None])[0],
            issuer=qs.get("issuer", [None])[0],
        )
        self._json_response(200, {"entries": entries, "total": self.merkle_store._tree.size}, rid)

    def _handle_transparency_consistency(self, qs, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        old_size = int(qs.get("old_size", [0])[0])
        new_size = qs.get("new_size", [None])[0]
        new_size = int(new_size) if new_size else None
        try:
            cp = self.merkle_store.consistency_proof(old_size, new_size)
            self._json_response(200, cp, rid)
        except ValueError as e:
            self._json_response(400, {"error": str(e)}, rid)

    def _handle_transparency_register(self, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        content_type = self.headers.get("Content-Type", "")
        raw = self._read_body_raw(rid)
        if raw is None:
            return
        try:
            if "application/cbor" in content_type or "application/cose" in content_type:
                cose_data = raw
            elif "application/json" in content_type:
                payload = json.loads(raw)
                if "cose_hex" in payload:
                    cose_data = bytes.fromhex(payload["cose_hex"])
                elif "cose_base64" in payload:
                    cose_data = base64.b64decode(payload["cose_base64"])
                else:
                    self._json_response(400, {"error": "JSON needs cose_hex or cose_base64"}, rid)
                    return
            else:
                cose_data = raw

            with self._transparency_lock:
                tr = self.merkle_store.register(cose_data)
            self._json_response(201, {**tr, "request_id": rid}, rid)
        except Exception as e:
            self._json_response(400, {"error": f"Registration failed: {e}"}, rid)

    def _handle_transparency_verify(self, rid):
        if not self.merkle_store:
            self._json_response(501, {"error": "Transparency log not enabled"}, rid)
            return
        body = self._read_body_json(rid)
        if body is None:
            return
        try:
            cose_hex = body.get("cose_hex")
            cose_b64 = body.get("cose_base64")
            proof_data = body.get("proof")
            if not proof_data:
                self._json_response(400, {"error": "Missing 'proof'"}, rid)
                return
            if cose_hex:
                cose_data = bytes.fromhex(cose_hex)
            elif cose_b64:
                cose_data = base64.b64decode(cose_b64)
            else:
                self._json_response(400, {"error": "Need cose_hex or cose_base64"}, rid)
                return
            ok = self.merkle_store.verify_inclusion(cose_data, proof_data)
            self._json_response(200, {"verified": ok, "request_id": rid}, rid)
        except Exception as e:
            self._json_response(400, {"error": f"Verification failed: {e}"}, rid)


# ── Server ──────────────────────────────────────────────────────────

class UnifiedServer:
    def __init__(self, config: UnifiedConfig):
        self.config = config
        self.shutdown_event = Event()

    def start(self):
        identity = None
        if HAS_IDENTITY:
            if self.config.keyfile and os.path.exists(self.config.keyfile):
                identity = AgentIdentity.from_keyfile(self.config.keyfile)
            else:
                identity = AgentIdentity.generate()
                if self.config.keyfile:
                    identity.save_keyfile(self.config.keyfile)
            logger.info(f"Identity: {identity.agent_id[:24]}...")

        engine = VerificationEngine(identity=identity)

        UnifiedHandler.engine = engine
        UnifiedHandler.config = self.config
        UnifiedHandler.identity = identity
        UnifiedHandler.rate_limiter = RateLimiter(self.config.rate_limit, self.config.rate_burst)

        # Receipt persistence
        if HAS_RECEIPT_STORE and self.config.receipt_db:
            UnifiedHandler.receipt_store = ReceiptStore(self.config.receipt_db)
            logger.info(f"Receipt store: {self.config.receipt_db}")
        else:
            UnifiedHandler.receipt_store = None

        # Transparency log
        if HAS_MERKLE_STORE and self.config.transparency_db:
            UnifiedHandler.merkle_store = MerkleStore(self.config.transparency_db)
            logger.info(f"Transparency log: {self.config.transparency_db}")
        else:
            UnifiedHandler.merkle_store = None

        server = HTTPServer((self.config.host, self.config.port), UnifiedHandler)

        def _shutdown(signum, frame):
            logger.info(f"Signal {signum}, shutting down...")
            self.shutdown_event.set()
            Thread(target=server.shutdown, daemon=True).start()

        signal.signal(signal.SIGTERM, _shutdown)
        signal.signal(signal.SIGINT, _shutdown)

        logger.info(f"Unified server on {self.config.host}:{self.config.port}")
        if self.config.auto_register:
            logger.info("Auto-register: verified receipts → transparency log")
        server.serve_forever()
        logger.info("Server stopped.")


# ── Tests ───────────────────────────────────────────────────────────

def run_tests():
    import urllib.request
    import urllib.error

    tmpdir = tempfile.mkdtemp()

    identity = AgentIdentity.generate() if HAS_IDENTITY else None
    engine = VerificationEngine(identity=identity)

    config = UnifiedConfig(
        host="127.0.0.1", port=0,
        api_key="test-key",
        receipt_db=os.path.join(tmpdir, "r.db"),
        transparency_db=os.path.join(tmpdir, "t.db"),
        auto_register=True,
    )

    receipt_store = ReceiptStore(config.receipt_db) if HAS_RECEIPT_STORE else None
    merkle_store = MerkleStore(config.transparency_db) if HAS_MERKLE_STORE else None

    UnifiedHandler.engine = engine
    UnifiedHandler.config = config
    UnifiedHandler.identity = identity
    UnifiedHandler.rate_limiter = RateLimiter(rate=100.0, burst=200)
    UnifiedHandler.receipt_store = receipt_store
    UnifiedHandler.merkle_store = merkle_store
    UnifiedHandler._request_count = 0

    server = HTTPServer(("127.0.0.1", 0), UnifiedHandler)
    port = server.server_address[1]
    base = f"http://127.0.0.1:{port}"
    Thread(target=server.serve_forever, daemon=True).start()
    time.sleep(0.1)

    auth_headers = {"Content-Type": "application/json", "Authorization": "Bearer test-key"}

    def get(path, headers=None):
        hdrs = {**auth_headers, **(headers or {})}
        r = urllib.request.Request(f"{base}{path}", headers=hdrs)
        resp = urllib.request.urlopen(r)
        return json.loads(resp.read())

    def post(path, data, headers=None):
        hdrs = {**auth_headers, **(headers or {})}
        body = json.dumps(data).encode() if isinstance(data, dict) else data
        if isinstance(data, bytes):
            hdrs["Content-Type"] = "application/cbor"
        r = urllib.request.Request(f"{base}{path}", data=body, headers=hdrs, method="POST")
        resp = urllib.request.urlopen(r)
        return json.loads(resp.read()), resp.status

    passed = total = 0
    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    # 1. Health
    print("\n1. Health")
    h = get("/health")
    check("status ok", h["status"] == "ok")
    check("unified version", "unified" in h["version"])
    check("transparency enabled", h["transparency"] is True)
    check("auto_register on", h["auto_register"] is True)

    # 2. Auth enforcement
    print("\n2. Auth enforcement")
    try:
        r = urllib.request.Request(f"{base}/verify", data=b'{}', headers={"Content-Type": "application/json"}, method="POST")
        urllib.request.urlopen(r)
        check("401 without auth", False)
    except urllib.error.HTTPError as e:
        check("401 without auth", e.code == 401)

    # 3. Verify + auto-register
    print("\n3. Verify + auto-register to transparency")
    vr, _ = post("/verify", {
        "tier": 0,
        "output": {"content": "def f(x): return x+1"},
        "verification": {"test_suite": {"tests": [
            {"name": "t1", "input": "f(1)", "expected_output": "2"},
        ]}}
    })
    check("verdict pass", vr["verdict"] == "pass")
    has_tr = "transparency_receipt" in vr
    check("auto-registered to transparency", has_tr)
    if has_tr:
        check("transparency has sequence_number", "sequence_number" in vr["transparency_receipt"])

    # 4. Receipt persistence
    print("\n4. Receipt persistence")
    rq = get("/receipts?verdict=pass")
    check("receipt queryable", rq["count"] >= 1)

    # 5. Transparency stats
    print("\n5. Transparency stats")
    ts = get("/transparency/stats")
    check("transparency has entries", ts["total_entries"] >= 1)

    # 6. Transparency root
    print("\n6. Transparency root")
    tr = get("/transparency/root")
    check("tree_size >= 1", tr["tree_size"] >= 1)
    check("tree_root is hex", len(tr["tree_root"]) == 64)

    # 7. Transparency proof
    print("\n7. Transparency proof")
    proof = get("/transparency/proof/0")
    check("proof has inclusion_proof", "inclusion_proof" in proof)

    # 8. Transparency entries
    print("\n8. Transparency entries")
    ents = get("/transparency/entries")
    check("entries returned", len(ents["entries"]) >= 1)

    # 9. Manual COSE registration
    print("\n9. Manual COSE registration")
    if HAS_COSE:
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        seed = os.urandom(32)
        pk = Ed25519PrivateKey.from_private_bytes(seed)
        pub = pk.public_key().public_bytes_raw()
        manual_cose = VRFCoseReceipt(
            receipt_data={"vrf_version": "1.0", "receipt_id": "manual-001", "verdict": "pass", "timestamp": time.time()},
            signing_key=seed, public_key=pub,
        ).to_cose_sign1()
        reg, status = post("/transparency/register", {"cose_hex": manual_cose.hex()})
        check("manual register 201", status == 201)
        check("has sequence_number", "sequence_number" in reg)
    else:
        check("COSE available", False)
        check("(skipped)", True)

    # 10. Transparency verify — use manual COSE entry since we have the raw bytes
    print("\n10. Transparency inclusion verification")
    if HAS_COSE:
        # Use the manual_cose we registered in test 9
        manual_proof = get(f"/transparency/proof/{reg['sequence_number']}")
        vr_resp, _ = post("/transparency/verify", {"cose_hex": manual_cose.hex(), "proof": manual_proof})
        check("inclusion verified", vr_resp["verified"] is True)
    else:
        check("(skipped - no COSE)", True)

    # 11. Docs
    print("\n11. Docs")
    docs = get("/docs")
    check("has verification_endpoints", "verification_endpoints" in docs)
    check("has transparency_endpoints", "transparency_endpoints" in docs)

    # 12. Metrics
    print("\n12. Metrics")
    r = urllib.request.urlopen(f"{base}/metrics")
    body = r.read().decode()
    check("has requests metric", "clawbizarre_requests_total" in body)
    check("has transparency metric", "clawbizarre_transparency_tree_size" in body)

    # 13. Stats unified
    print("\n13. Stats")
    st = get("/stats")
    check("has receipts_stored", "receipts_stored" in st)
    check("has transparency_entries", "transparency_entries" in st)
    check("auto_register in stats", st["auto_register"] is True)

    # 14. Consistency proof
    print("\n14. Consistency proof")
    # We have at least 2 entries now (1 auto + 1 manual)
    cp = get("/transparency/consistency?old_size=1&new_size=2")
    check("consistency proof returned", "old_size" in cp and cp["old_size"] == 1)

    # 15. Verify fail case
    print("\n15. Verify failing code")
    vf, _ = post("/verify", {
        "tier": 0,
        "output": {"content": "def f(x): return x-1"},  # Wrong!
        "verification": {"test_suite": {"tests": [
            {"name": "t1", "input": "f(1)", "expected_output": "2"},
        ]}}
    })
    check("verdict fail", vf["verdict"] == "fail")
    check("fail also auto-registered", "transparency_receipt" in vf)

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")

    server.shutdown()
    if receipt_store:
        receipt_store.close()
    if merkle_store:
        merkle_store.close()
    return passed == total


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
    else:
        config = UnifiedConfig()
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
            elif arg == "--receipt-db" and i + 1 < len(args):
                config.receipt_db = args[i + 1]
            elif arg == "--transparency-db" and i + 1 < len(args):
                config.transparency_db = args[i + 1]
            elif arg == "--auto-register":
                config.auto_register = True
            elif arg == "--no-receipt-db":
                config.receipt_db = None
            elif arg == "--no-transparency":
                config.transparency_db = None

        if not config.api_key:
            config.api_key = os.environ.get("CLAWBIZARRE_API_KEY")

        us = UnifiedServer(config)
        us.start()
