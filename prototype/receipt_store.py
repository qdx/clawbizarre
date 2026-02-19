"""
ClawBizarre Receipt Store — SQLite Persistence for VRF Receipts

Append-only receipt storage with WAL mode for concurrent reads.
Designed to plug into verify_server_hardened.py.

Usage:
    store = ReceiptStore("receipts.db")
    store.save(receipt)           # Save a VerificationReceipt
    r = store.get("receipt-id")   # Get by ID
    rs = store.query(verdict="pass", limit=10)  # Query receipts
    stats = store.stats()         # Aggregate stats
    store.close()

    python3 receipt_store.py --test
"""

import json
import os
import sqlite3
import sys
import time
from datetime import datetime, timezone
from typing import Optional

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

from verify_server import VerificationReceipt, TestResults, TestDetail, ContentHashes


# ── Schema ──────────────────────────────────────────────────────────

SCHEMA = """
CREATE TABLE IF NOT EXISTS receipts (
    receipt_id    TEXT PRIMARY KEY,
    verified_at   TEXT NOT NULL,
    tier          INTEGER NOT NULL,
    verdict       TEXT NOT NULL,
    language      TEXT,
    task_id       TEXT,
    task_type     TEXT,
    input_hash    TEXT,
    output_hash   TEXT,
    suite_hash    TEXT,
    execution_ms  REAL,
    tests_total   INTEGER,
    tests_passed  INTEGER,
    tests_failed  INTEGER,
    tests_errors  INTEGER,
    signer_id     TEXT,
    signature     TEXT,
    data_json     TEXT NOT NULL,  -- full receipt as JSON (source of truth)
    created_at    TEXT NOT NULL DEFAULT (strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
);

CREATE INDEX IF NOT EXISTS idx_receipts_verdict ON receipts(verdict);
CREATE INDEX IF NOT EXISTS idx_receipts_verified_at ON receipts(verified_at);
CREATE INDEX IF NOT EXISTS idx_receipts_task_id ON receipts(task_id);
CREATE INDEX IF NOT EXISTS idx_receipts_output_hash ON receipts(output_hash);
"""


class ReceiptStore:
    """SQLite-backed append-only receipt store."""

    def __init__(self, db_path: str = "receipts.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA synchronous=NORMAL")
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def save(self, receipt: VerificationReceipt) -> str:
        """Save a receipt. Returns receipt_id. Idempotent (upsert)."""
        d = receipt.to_dict()
        meta = d.get("metadata", {})
        hashes = d.get("hashes", {})
        sig = d.get("signature", {})

        self.conn.execute(
            """INSERT OR REPLACE INTO receipts
               (receipt_id, verified_at, tier, verdict, language, task_id, task_type,
                input_hash, output_hash, suite_hash, execution_ms,
                tests_total, tests_passed, tests_failed, tests_errors,
                signer_id, signature, data_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (
                receipt.receipt_id,
                receipt.verified_at,
                receipt.tier,
                receipt.verdict,
                meta.get("language"),
                receipt.task_id,
                receipt.task_type,
                hashes.get("output_hash"),  # Note: using output_hash for the column
                hashes.get("output_hash"),
                hashes.get("suite_hash"),
                meta.get("execution_ms"),
                receipt.results.total if receipt.results else None,
                receipt.results.passed if receipt.results else None,
                receipt.results.failed if receipt.results else None,
                receipt.results.errors if receipt.results else None,
                sig.get("signer_id") if isinstance(sig, dict) else None,
                sig.get("value") if isinstance(sig, dict) else None,
                json.dumps(d, default=str),
            ),
        )
        self.conn.commit()
        return receipt.receipt_id

    def get(self, receipt_id: str) -> Optional[dict]:
        """Get receipt by ID. Returns full dict or None."""
        row = self.conn.execute(
            "SELECT data_json FROM receipts WHERE receipt_id = ?", (receipt_id,)
        ).fetchone()
        if row:
            return json.loads(row[0])
        return None

    def query(
        self,
        verdict: Optional[str] = None,
        task_type: Optional[str] = None,
        output_hash: Optional[str] = None,
        since: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict]:
        """Query receipts with filters."""
        clauses = []
        params = []
        if verdict:
            clauses.append("verdict = ?")
            params.append(verdict)
        if task_type:
            clauses.append("task_type = ?")
            params.append(task_type)
        if output_hash:
            clauses.append("output_hash = ?")
            params.append(output_hash)
        if since:
            clauses.append("verified_at >= ?")
            params.append(since)

        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        params.extend([limit, offset])

        rows = self.conn.execute(
            f"SELECT data_json FROM receipts {where} ORDER BY verified_at DESC LIMIT ? OFFSET ?",
            params,
        ).fetchall()
        return [json.loads(r[0]) for r in rows]

    def stats(self) -> dict:
        """Aggregate stats."""
        row = self.conn.execute(
            """SELECT
                 COUNT(*) as total,
                 SUM(CASE WHEN verdict='pass' THEN 1 ELSE 0 END) as passed,
                 SUM(CASE WHEN verdict='fail' THEN 1 ELSE 0 END) as failed,
                 SUM(CASE WHEN verdict='error' THEN 1 ELSE 0 END) as errors,
                 SUM(CASE WHEN verdict='partial' THEN 1 ELSE 0 END) as partial,
                 AVG(execution_ms) as avg_ms,
                 MIN(verified_at) as first_receipt,
                 MAX(verified_at) as last_receipt
               FROM receipts"""
        ).fetchone()
        return {
            "total_receipts": row[0],
            "verdicts": {
                "pass": row[1] or 0,
                "fail": row[2] or 0,
                "error": row[3] or 0,
                "partial": row[4] or 0,
            },
            "avg_execution_ms": round(row[5], 2) if row[5] else None,
            "first_receipt": row[6],
            "last_receipt": row[7],
        }

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM receipts").fetchone()[0]

    def close(self):
        self.conn.close()


# ── Tests ────────────────────────────────────────────────────────────

def run_tests():
    import tempfile

    passed = total = 0
    def check(name, condition):
        nonlocal passed, total
        total += 1
        if condition:
            passed += 1
            print(f"  ✅ {name}")
        else:
            print(f"  ❌ {name}")

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = os.path.join(tmpdir, "test_receipts.db")
        store = ReceiptStore(db_path)

        # Create a test receipt
        receipt = VerificationReceipt(
            receipt_id="test-001",
            verified_at=datetime.now(timezone.utc).isoformat(),
            tier=0,
            verdict="pass",
            results=TestResults(total=2, passed=2, failed=0, errors=0, details=[
                TestDetail(name="t1", status="pass", duration_ms=5.0).__dict__,
                TestDetail(name="t2", status="pass", duration_ms=3.0).__dict__,
            ]),
            hashes=ContentHashes(
                input_hash="abc123", output_hash="def456", suite_hash="ghi789"
            ),
            metadata={"language": "python", "execution_ms": 8.0, "verifier_version": "test"},
        )

        print("\n1. Save receipt")
        rid = store.save(receipt)
        check("save returns id", rid == "test-001")

        print("\n2. Get receipt")
        r = store.get("test-001")
        check("get returns dict", isinstance(r, dict))
        check("verdict matches", r["verdict"] == "pass")
        check("receipt_id matches", r["receipt_id"] == "test-001")

        print("\n3. Get non-existent")
        check("returns None", store.get("nonexistent") is None)

        print("\n4. Count")
        check("count is 1", store.count() == 1)

        print("\n5. Save another (fail)")
        receipt2 = VerificationReceipt(
            receipt_id="test-002",
            verified_at=datetime.now(timezone.utc).isoformat(),
            tier=0,
            verdict="fail",
            results=TestResults(total=1, passed=0, failed=1, errors=0, details=[]),
            hashes=ContentHashes(input_hash="x", output_hash="y", suite_hash="z"),
            metadata={"language": "python", "execution_ms": 12.0},
        )
        store.save(receipt2)
        check("count is 2", store.count() == 2)

        print("\n6. Query by verdict")
        passes = store.query(verdict="pass")
        check("1 pass receipt", len(passes) == 1)
        fails = store.query(verdict="fail")
        check("1 fail receipt", len(fails) == 1)

        print("\n7. Query all")
        all_r = store.query()
        check("2 total", len(all_r) == 2)

        print("\n8. Stats")
        s = store.stats()
        check("total 2", s["total_receipts"] == 2)
        check("1 pass", s["verdicts"]["pass"] == 1)
        check("1 fail", s["verdicts"]["fail"] == 1)
        check("avg_ms exists", s["avg_execution_ms"] is not None)

        print("\n9. Idempotent save (upsert)")
        receipt.verdict = "pass"  # same
        store.save(receipt)
        check("still 2 after re-save", store.count() == 2)

        print("\n10. Persistence across connections")
        store.close()
        store2 = ReceiptStore(db_path)
        check("survives reconnect", store2.count() == 2)
        r2 = store2.get("test-001")
        check("data intact", r2["verdict"] == "pass")
        store2.close()

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{total} passed")
    return passed == total


if __name__ == "__main__":
    if "--test" in sys.argv:
        success = run_tests()
        sys.exit(0 if success else 1)
