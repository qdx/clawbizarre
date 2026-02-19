#!/usr/bin/env python3
"""
ClawBizarre Transparency Server — HTTP API for SCITT-aligned COSE Receipt Log

Exposes the MerkleStore as an HTTP service for:
- Submitting COSE-signed VRF receipts
- Querying inclusion/consistency proofs
- Looking up receipts
- Verifying inclusion client-side

Endpoints:
    POST   /transparency/register       — Submit COSE receipt (binary body)
    GET    /transparency/proof/<seq>     — Inclusion proof for entry
    GET    /transparency/receipt/<id>    — Lookup by receipt_id
    GET    /transparency/entries         — List entries (?verdict=&issuer=&limit=&offset=)
    GET    /transparency/consistency     — Consistency proof (?old_size=&new_size=)
    POST   /transparency/verify         — Verify inclusion (JSON: {cose_hex, proof})
    GET    /transparency/stats          — Log statistics
    GET    /transparency/root           — Current tree root
    GET    /health                      — Health check

Usage:
    python3 transparency_server.py [--port 8710] [--db transparency.db]
    python3 transparency_server.py --test
"""

import json
import os
import sys
import time
import base64
from http.server import HTTPServer, BaseHTTPRequestHandler
from threading import Lock
from typing import Optional
from urllib.parse import urlparse, parse_qs

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

from merkle_store import MerkleStore
from merkle import proof_from_hex, hash_leaf


class TransparencyHandler(BaseHTTPRequestHandler):
    """HTTP handler for Transparency Service."""

    store: MerkleStore = None
    lock: Lock = None

    def log_message(self, fmt, *args):
        pass  # Suppress default logging

    def _json_response(self, status: int, data: dict):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self, max_size: int = 1024 * 1024) -> bytes:
        length = int(self.headers.get("Content-Length", 0))
        if length > max_size:
            return None
        return self.rfile.read(length) if length > 0 else b""

    def _parse_path(self):
        parsed = urlparse(self.path)
        return parsed.path.rstrip("/"), parse_qs(parsed.query)

    def do_GET(self):
        path, qs = self._parse_path()

        if path == "/health":
            self._json_response(200, {
                "status": "ok",
                "service": "clawbizarre-transparency",
                "tree_size": self.store._tree.size,
            })

        elif path == "/transparency/stats":
            self._json_response(200, self.store.stats())

        elif path == "/transparency/root":
            self._json_response(200, {
                "tree_size": self.store._tree.size,
                "tree_root": self.store._tree.root.hex(),
            })

        elif path.startswith("/transparency/proof/"):
            try:
                seq = int(path.split("/")[-1])
                proof = self.store.get_proof(seq)
                self._json_response(200, proof)
            except (ValueError, IndexError) as e:
                self._json_response(400, {"error": str(e)})

        elif path.startswith("/transparency/receipt/"):
            receipt_id = path.split("/transparency/receipt/", 1)[1]
            entry = self.store.lookup(receipt_id)
            if entry:
                # Also include decoded receipt data
                seq = entry["sequence_number"]
                data = self.store.get_receipt_data(seq)
                entry["receipt_data"] = data
                self._json_response(200, entry)
            else:
                self._json_response(404, {"error": f"Receipt '{receipt_id}' not found"})

        elif path == "/transparency/entries":
            limit = int(qs.get("limit", [50])[0])
            offset = int(qs.get("offset", [0])[0])
            verdict = qs.get("verdict", [None])[0]
            issuer = qs.get("issuer", [None])[0]
            entries = self.store.list_entries(limit=limit, offset=offset,
                                              verdict=verdict, issuer=issuer)
            self._json_response(200, {"entries": entries, "total": self.store._tree.size})

        elif path == "/transparency/consistency":
            old_size = int(qs.get("old_size", [0])[0])
            new_size = qs.get("new_size", [None])[0]
            new_size = int(new_size) if new_size else None
            try:
                cp = self.store.consistency_proof(old_size, new_size)
                self._json_response(200, cp)
            except ValueError as e:
                self._json_response(400, {"error": str(e)})

        else:
            self._json_response(404, {"error": "Not found"})

    def do_POST(self):
        path, qs = self._parse_path()

        if path == "/transparency/register":
            content_type = self.headers.get("Content-Type", "")
            body = self._read_body()
            if body is None:
                self._json_response(413, {"error": "Body too large (max 1MB)"})
                return

            try:
                # Accept binary COSE, hex-encoded, or base64-encoded
                if "application/cbor" in content_type or "application/cose" in content_type:
                    cose_data = body
                elif "application/json" in content_type:
                    payload = json.loads(body)
                    if "cose_hex" in payload:
                        cose_data = bytes.fromhex(payload["cose_hex"])
                    elif "cose_base64" in payload:
                        cose_data = base64.b64decode(payload["cose_base64"])
                    else:
                        self._json_response(400, {"error": "JSON body must have cose_hex or cose_base64"})
                        return
                else:
                    # Try raw binary
                    cose_data = body

                with self.lock:
                    tr = self.store.register(cose_data)
                self._json_response(201, tr)

            except Exception as e:
                self._json_response(400, {"error": f"Registration failed: {e}"})

        elif path == "/transparency/verify":
            body = self._read_body()
            if not body:
                self._json_response(400, {"error": "Empty body"})
                return

            try:
                payload = json.loads(body)
                cose_hex = payload.get("cose_hex")
                cose_b64 = payload.get("cose_base64")
                proof_data = payload.get("proof")

                if not proof_data:
                    self._json_response(400, {"error": "Missing 'proof' (transparency receipt)"})
                    return

                if cose_hex:
                    cose_data = bytes.fromhex(cose_hex)
                elif cose_b64:
                    cose_data = base64.b64decode(cose_b64)
                else:
                    self._json_response(400, {"error": "Need cose_hex or cose_base64"})
                    return

                ok = self.store.verify_inclusion(cose_data, proof_data)
                self._json_response(200, {"verified": ok})

            except Exception as e:
                self._json_response(400, {"error": f"Verification failed: {e}"})

        else:
            self._json_response(404, {"error": "Not found"})

    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()


def create_server(port: int = 8710, db_path: str = "transparency.db") -> HTTPServer:
    store = MerkleStore(db_path)
    lock = Lock()

    TransparencyHandler.store = store
    TransparencyHandler.lock = lock

    server = HTTPServer(("0.0.0.0", port), TransparencyHandler)
    server._store = store  # for cleanup
    return server


# ─── Tests ───

def _run_tests():
    import tempfile
    import threading
    import urllib.request
    import urllib.error
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    from vrf_cose import VRFCoseReceipt

    tmpdir = tempfile.mkdtemp()
    db_path = os.path.join(tmpdir, "test.db")
    server = create_server(port=0, db_path=db_path)
    port = server.server_address[1]
    base = f"http://localhost:{port}"

    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    time.sleep(0.1)

    def get(path):
        req = urllib.request.Request(f"{base}{path}")
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read())

    def post(path, data=None, content_type="application/json"):
        body = json.dumps(data).encode() if isinstance(data, dict) else data
        req = urllib.request.Request(f"{base}{path}", data=body,
                                     headers={"Content-Type": content_type})
        resp = urllib.request.urlopen(req)
        return json.loads(resp.read()), resp.status

    def make_cose(rid, verdict="pass"):
        seed = os.urandom(32)
        pk = Ed25519PrivateKey.from_private_bytes(seed)
        pub = pk.public_key().public_bytes_raw()
        r = VRFCoseReceipt(
            receipt_data={"vrf_version": "1.0", "receipt_id": rid, "verdict": verdict, "timestamp": time.time()},
            signing_key=seed, public_key=pub,
        )
        return r.to_cose_sign1()

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

    # 1. Health
    h = get("/health")
    check("Health check", h["status"] == "ok" and h["tree_size"] == 0)

    # 2. Register via hex
    cose1 = make_cose("http-001", "pass")
    tr1, status = post("/transparency/register", {"cose_hex": cose1.hex()})
    check("Register via hex", status == 201 and tr1["sequence_number"] == 0)

    # 3. Register via base64
    cose2 = make_cose("http-002", "fail")
    tr2, status = post("/transparency/register", {"cose_base64": base64.b64encode(cose2).decode()})
    check("Register via base64", status == 201 and tr2["sequence_number"] == 1)

    # 4. Register via raw binary
    cose3 = make_cose("http-003", "pass")
    tr3, status = post("/transparency/register", cose3, "application/cbor")
    check("Register via binary", status == 201 and tr3["sequence_number"] == 2)

    # 5. Get proof
    proof = get("/transparency/proof/0")
    check("Get proof", proof["sequence_number"] == 0 and "inclusion_proof" in proof)

    # 6. Lookup receipt
    receipt = get("/transparency/receipt/http-002")
    check("Lookup receipt", receipt["verdict"] == "fail" and receipt["receipt_data"]["receipt_id"] == "http-002")

    # 7. List entries
    entries = get("/transparency/entries?limit=10")
    check("List entries", len(entries["entries"]) == 3 and entries["total"] == 3)

    # 8. List with filter
    passes = get("/transparency/entries?verdict=pass")
    check("Filter by verdict", len(passes["entries"]) == 2)

    # 9. Verify inclusion
    verify_resp, _ = post("/transparency/verify", {
        "cose_hex": cose1.hex(),
        "proof": tr1,
    })
    check("Verify inclusion", verify_resp["verified"] is True)

    # 10. Verify with wrong data fails
    verify_bad, _ = post("/transparency/verify", {
        "cose_hex": cose2.hex(),  # wrong cose for tr1's proof
        "proof": tr1,
    })
    check("Wrong data rejected", verify_bad["verified"] is False)

    # 11. Stats
    stats = get("/transparency/stats")
    check("Stats", stats["total_entries"] == 3 and stats["unique_issuers"] == 3)

    # 12. Root
    root = get("/transparency/root")
    check("Root endpoint", root["tree_size"] == 3 and len(root["tree_root"]) == 64)

    # 13. Consistency proof
    # Register more to have a meaningful consistency proof
    for i in range(5):
        post("/transparency/register", {"cose_hex": make_cose(f"batch-{i}").hex()})
    cp = get("/transparency/consistency?old_size=3&new_size=8")
    check("Consistency proof", cp["old_size"] == 3 and cp["new_size"] == 8)

    # 14. 404 handling
    try:
        get("/transparency/receipt/nonexistent")
        check("404 for missing receipt", False)
    except urllib.error.HTTPError as e:
        check("404 for missing receipt", e.code == 404)

    server.shutdown()
    server._store.close()

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = _run_tests()
        sys.exit(0 if success else 1)
    else:
        import argparse
        parser = argparse.ArgumentParser(description="ClawBizarre Transparency Server")
        parser.add_argument("--port", type=int, default=8710)
        parser.add_argument("--db", default="transparency.db")
        args = parser.parse_args()

        server = create_server(port=args.port, db_path=args.db)
        print(f"Transparency Server on :{args.port} (db: {args.db})")
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            server.shutdown()
            server._store.close()
