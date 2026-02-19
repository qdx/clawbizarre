#!/usr/bin/env python3
"""
ClawBizarre Merkle Store — SQLite Persistence for Transparency Log

Persists MerkleTree leaves and TransparencyLog entries to SQLite (WAL mode).
On startup, reconstructs the in-memory Merkle tree from stored leaves.

Usage:
    store = MerkleStore("transparency.db")
    receipt = store.register(cose_bytes)     # Append + persist
    ok = store.verify_inclusion(cose_bytes, receipt)
    store.close()

    python3 merkle_store.py --test
"""

import hashlib
import json
import os
import sqlite3
import sys
import time
from typing import Optional, List, Dict

_proto_dir = os.path.dirname(os.path.abspath(__file__))
if _proto_dir not in sys.path:
    sys.path.insert(0, _proto_dir)

from merkle import MerkleTree, proof_to_hex, proof_from_hex, hash_leaf
from vrf_cose import VRFCoseReceipt, TransparencyLog, cose_to_json_receipt


SCHEMA = """
CREATE TABLE IF NOT EXISTS merkle_leaves (
    seq           INTEGER PRIMARY KEY,
    cose_blob     BLOB NOT NULL,
    leaf_hash     TEXT NOT NULL,
    entry_id      TEXT NOT NULL,
    receipt_id    TEXT,
    issuer_did    TEXT,
    verdict       TEXT,
    registered_at REAL NOT NULL,
    meta_json     TEXT
);

CREATE INDEX IF NOT EXISTS idx_ml_receipt_id ON merkle_leaves(receipt_id);
CREATE INDEX IF NOT EXISTS idx_ml_issuer_did ON merkle_leaves(issuer_did);
CREATE INDEX IF NOT EXISTS idx_ml_verdict ON merkle_leaves(verdict);

CREATE TABLE IF NOT EXISTS merkle_roots (
    tree_size     INTEGER PRIMARY KEY,
    root_hash     TEXT NOT NULL,
    recorded_at   REAL NOT NULL
);
"""


class MerkleStore:
    """SQLite-backed Transparency Log with in-memory Merkle tree."""

    def __init__(self, db_path: str = "transparency.db"):
        self.db_path = db_path
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA synchronous=NORMAL")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

        # Reconstruct in-memory tree from stored leaves
        self._tree = MerkleTree()
        self._cose_blobs: List[bytes] = []
        self._index: Dict[str, int] = {}  # receipt_id → seq
        self._roots: Dict[int, bytes] = {}  # size → root

        self._rebuild()

    def _rebuild(self):
        """Reconstruct Merkle tree from stored leaves."""
        cursor = self._conn.execute(
            "SELECT seq, cose_blob, receipt_id FROM merkle_leaves ORDER BY seq"
        )
        for seq, cose_blob, receipt_id in cursor:
            if isinstance(cose_blob, memoryview):
                cose_blob = bytes(cose_blob)
            self._tree.append(cose_blob)
            self._cose_blobs.append(cose_blob)
            if receipt_id:
                self._index[receipt_id] = seq

        # Load saved roots
        cursor = self._conn.execute("SELECT tree_size, root_hash FROM merkle_roots")
        for size, root_hex in cursor:
            self._roots[size] = bytes.fromhex(root_hex)

        # Current root
        if self._tree.size > 0:
            self._roots[self._tree.size] = self._tree.root

    def register(self, cose_data: bytes) -> dict:
        """Register a COSE-signed VRF receipt. Returns transparency receipt with Merkle proof."""
        receipt = VRFCoseReceipt.from_cose_sign1(cose_data)
        entry = receipt.to_transparency_entry()
        seq = self._tree.size

        # Append to in-memory tree
        self._tree.append(cose_data)
        self._cose_blobs.append(cose_data)

        root = self._tree.root
        self._roots[self._tree.size] = root

        receipt_id = entry.get("receipt_id")
        if receipt_id:
            self._index[receipt_id] = seq

        # Persist
        leaf_hash_hex = hash_leaf(cose_data).hex()
        meta = {
            "entry_id": entry["entry_id"],
            "content_type": entry.get("content_type"),
            "signed_statement_size": entry.get("signed_statement_size"),
        }
        self._conn.execute(
            """INSERT INTO merkle_leaves (seq, cose_blob, leaf_hash, entry_id,
               receipt_id, issuer_did, verdict, registered_at, meta_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (seq, cose_data, leaf_hash_hex, entry["entry_id"],
             receipt_id, entry.get("issuer"), entry.get("verdict"),
             time.time(), json.dumps(meta)),
        )
        self._conn.execute(
            "INSERT OR REPLACE INTO merkle_roots (tree_size, root_hash, recorded_at) VALUES (?, ?, ?)",
            (self._tree.size, root.hex(), time.time()),
        )
        self._conn.commit()

        # Generate inclusion proof
        proof = self._tree.inclusion_proof(seq)
        return {
            "entry_id": entry["entry_id"],
            "sequence_number": seq,
            "log_size": self._tree.size,
            "tree_root": root.hex(),
            "inclusion_proof": proof_to_hex(proof),
        }

    def verify_inclusion(self, cose_data: bytes, transparency_receipt: dict) -> bool:
        """Verify that cose_data is in the log at the claimed position."""
        seq = transparency_receipt["sequence_number"]
        log_size = transparency_receipt["log_size"]
        expected_root = bytes.fromhex(transparency_receipt["tree_root"])
        proof = proof_from_hex(transparency_receipt["inclusion_proof"])
        leaf_hash_val = hash_leaf(cose_data)
        return MerkleTree.verify_inclusion(leaf_hash_val, seq, log_size, proof, expected_root)

    def get_proof(self, index: int) -> dict:
        """Get fresh inclusion proof for entry at index."""
        if index >= self._tree.size:
            raise ValueError(f"Index {index} out of range (size={self._tree.size})")
        proof = self._tree.inclusion_proof(index)
        # Get entry metadata
        row = self._conn.execute(
            "SELECT entry_id FROM merkle_leaves WHERE seq = ?", (index,)
        ).fetchone()
        return {
            "entry_id": row[0] if row else None,
            "sequence_number": index,
            "log_size": self._tree.size,
            "tree_root": self._tree.root.hex(),
            "inclusion_proof": proof_to_hex(proof),
        }

    def consistency_proof(self, old_size: int, new_size: int = None) -> dict:
        """Consistency proof showing log is append-only."""
        if new_size is None:
            new_size = self._tree.size
        proof = self._tree.consistency_proof(old_size, new_size)
        return {
            "old_size": old_size,
            "new_size": new_size,
            "old_root": self._roots.get(old_size, b'').hex() if isinstance(self._roots.get(old_size), bytes) else self._roots.get(old_size, ''),
            "new_root": self._roots.get(new_size, self._tree.root).hex() if isinstance(self._roots.get(new_size, self._tree.root), bytes) else '',
            "proof": proof_to_hex(proof),
        }

    def lookup(self, receipt_id: str) -> Optional[dict]:
        """Look up entry by receipt_id."""
        row = self._conn.execute(
            "SELECT seq, entry_id, issuer_did, verdict, registered_at, meta_json FROM merkle_leaves WHERE receipt_id = ?",
            (receipt_id,)
        ).fetchone()
        if not row:
            return None
        return {
            "sequence_number": row[0],
            "entry_id": row[1],
            "issuer_did": row[2],
            "verdict": row[3],
            "registered_at": row[4],
            "meta": json.loads(row[5]) if row[5] else {},
        }

    def get_receipt_data(self, index: int) -> Optional[dict]:
        """Decode the VRF receipt data at a given index."""
        if index < 0 or index >= len(self._cose_blobs):
            return None
        return cose_to_json_receipt(self._cose_blobs[index])

    def list_entries(self, limit: int = 50, offset: int = 0,
                     verdict: str = None, issuer: str = None) -> List[dict]:
        """Query log entries with filters."""
        query = "SELECT seq, entry_id, receipt_id, issuer_did, verdict, registered_at FROM merkle_leaves WHERE 1=1"
        params = []
        if verdict:
            query += " AND verdict = ?"
            params.append(verdict)
        if issuer:
            query += " AND issuer_did = ?"
            params.append(issuer)
        query += " ORDER BY seq DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        rows = self._conn.execute(query, params).fetchall()
        return [
            {"sequence_number": r[0], "entry_id": r[1], "receipt_id": r[2],
             "issuer_did": r[3], "verdict": r[4], "registered_at": r[5]}
            for r in rows
        ]

    def stats(self) -> dict:
        """Aggregate statistics."""
        total = self._conn.execute("SELECT COUNT(*) FROM merkle_leaves").fetchone()[0]
        verdicts = {}
        for row in self._conn.execute("SELECT verdict, COUNT(*) FROM merkle_leaves GROUP BY verdict"):
            verdicts[row[0] or "unknown"] = row[1]
        issuers = self._conn.execute("SELECT COUNT(DISTINCT issuer_did) FROM merkle_leaves").fetchone()[0]
        return {
            "total_entries": total,
            "tree_size": self._tree.size,
            "tree_root": self._tree.root.hex(),
            "verdicts": verdicts,
            "unique_issuers": issuers,
            "db_path": self.db_path,
        }

    def close(self):
        self._conn.close()


# ─── Tests ───

def _make_test_cose(receipt_id: str, verdict: str = "pass", issuer_seed: bytes = None) -> bytes:
    """Helper: create a signed COSE receipt."""
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
    import os as _os

    seed = issuer_seed or _os.urandom(32)
    pk = Ed25519PrivateKey.from_private_bytes(seed)
    pub = pk.public_key().public_bytes_raw()

    receipt = VRFCoseReceipt(
        receipt_data={
            "vrf_version": "1.0",
            "receipt_id": receipt_id,
            "verdict": verdict,
            "timestamp": time.time(),
        },
        signing_key=seed,
        public_key=pub,
    )
    return receipt.to_cose_sign1()


def test_basic_register_and_verify():
    """Register a receipt and verify inclusion."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    cose = _make_test_cose("basic-001", "pass")
    tr = store.register(cose)

    assert tr["sequence_number"] == 0
    assert tr["log_size"] == 1
    assert len(tr["tree_root"]) == 64
    assert store.verify_inclusion(cose, tr)

    store.close()
    print(f"  Registered + verified ✅")
    return True


def test_multiple_entries():
    """Register multiple and verify all."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    coses = []
    trs = []
    for i in range(10):
        c = _make_test_cose(f"multi-{i}", "pass" if i % 2 == 0 else "fail")
        coses.append(c)
        trs.append(store.register(c))

    assert store.stats()["total_entries"] == 10

    for i in range(10):
        assert store.verify_inclusion(coses[i], trs[i])

    # Fresh proof
    fresh = store.get_proof(3)
    assert fresh["log_size"] == 10
    assert store.verify_inclusion(coses[3], fresh)

    store.close()
    print(f"  10 entries registered, all verified ✅")
    return True


def test_persistence_across_restart():
    """Close and reopen — tree should be reconstructed."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")

    # First session
    store1 = MerkleStore(db)
    coses = []
    for i in range(5):
        c = _make_test_cose(f"persist-{i}")
        coses.append(c)
        store1.register(c)
    root1 = store1._tree.root.hex()
    store1.close()

    # Second session (restart)
    store2 = MerkleStore(db)
    assert store2._tree.size == 5
    assert store2._tree.root.hex() == root1, "Root mismatch after restart"

    # Can still verify old entries
    fresh = store2.get_proof(2)
    assert store2.verify_inclusion(coses[2], fresh)

    # Can append more
    c6 = _make_test_cose("persist-5")
    tr = store2.register(c6)
    assert tr["sequence_number"] == 5
    assert store2._tree.size == 6

    store2.close()
    print(f"  Persistence verified: 5 entries survived restart, root matches ✅")
    return True


def test_lookup():
    """Lookup by receipt_id."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    store.register(_make_test_cose("lookup-a", "pass"))
    store.register(_make_test_cose("lookup-b", "fail"))

    a = store.lookup("lookup-a")
    assert a is not None
    assert a["verdict"] == "pass"
    assert a["sequence_number"] == 0

    b = store.lookup("lookup-b")
    assert b is not None
    assert b["verdict"] == "fail"

    assert store.lookup("nonexistent") is None

    store.close()
    print(f"  Lookup by receipt_id ✅")
    return True


def test_list_entries():
    """Query with filters."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    for i in range(8):
        store.register(_make_test_cose(f"list-{i}", "pass" if i < 5 else "fail"))

    all_entries = store.list_entries()
    assert len(all_entries) == 8

    passes = store.list_entries(verdict="pass")
    assert len(passes) == 5

    fails = store.list_entries(verdict="fail")
    assert len(fails) == 3

    limited = store.list_entries(limit=3, offset=0)
    assert len(limited) == 3

    store.close()
    print(f"  List with filters ✅")
    return True


def test_stats():
    """Aggregate stats."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    seed_a = os.urandom(32)
    seed_b = os.urandom(32)
    store.register(_make_test_cose("s1", "pass", seed_a))
    store.register(_make_test_cose("s2", "pass", seed_a))
    store.register(_make_test_cose("s3", "fail", seed_b))

    stats = store.stats()
    assert stats["total_entries"] == 3
    assert stats["tree_size"] == 3
    assert stats["verdicts"]["pass"] == 2
    assert stats["verdicts"]["fail"] == 1
    assert stats["unique_issuers"] == 2

    store.close()
    print(f"  Stats: {stats['verdicts']} ✅")
    return True


def test_consistency_proof():
    """Consistency proof between sizes."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    for i in range(8):
        store.register(_make_test_cose(f"cons-{i}"))

    cp = store.consistency_proof(4, 8)
    assert cp["old_size"] == 4
    assert cp["new_size"] == 8
    assert len(cp["proof"]) > 0

    store.close()
    print(f"  Consistency proof (4→8): {len(cp['proof'])} nodes ✅")
    return True


def test_get_receipt_data():
    """Decode stored receipt data."""
    import tempfile
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    store.register(_make_test_cose("decode-001", "pass"))
    data = store.get_receipt_data(0)
    assert data is not None
    assert data["receipt_id"] == "decode-001"
    assert data["verdict"] == "pass"
    assert store.get_receipt_data(99) is None

    store.close()
    print(f"  Receipt data decode ✅")
    return True


def test_concurrent_reads():
    """Multiple reads don't block (WAL mode)."""
    import tempfile
    import threading
    db = os.path.join(tempfile.mkdtemp(), "test.db")
    store = MerkleStore(db)

    for i in range(20):
        store.register(_make_test_cose(f"conc-{i}"))

    results = []
    def reader(idx):
        try:
            p = store.get_proof(idx)
            results.append(p is not None)
        except Exception as e:
            results.append(False)

    threads = [threading.Thread(target=reader, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(results), f"Some reads failed: {results}"
    store.close()
    print(f"  20 concurrent reads ✅")
    return True


if __name__ == "__main__":
    if "--test" in sys.argv:
        tests = [
            ("Basic register + verify", test_basic_register_and_verify),
            ("Multiple entries", test_multiple_entries),
            ("Persistence across restart", test_persistence_across_restart),
            ("Lookup by receipt_id", test_lookup),
            ("List with filters", test_list_entries),
            ("Aggregate stats", test_stats),
            ("Consistency proof", test_consistency_proof),
            ("Receipt data decode", test_get_receipt_data),
            ("Concurrent reads", test_concurrent_reads),
        ]

        passed = 0
        for name, fn in tests:
            print(f"\n[TEST] {name}")
            try:
                if fn():
                    print(f"  ✅ PASS")
                    passed += 1
                else:
                    print(f"  ❌ FAIL")
            except Exception as e:
                import traceback
                print(f"  ❌ ERROR: {e}")
                traceback.print_exc()

        print(f"\n{'='*40}")
        print(f"Results: {passed}/{len(tests)} passed")
    else:
        print("Usage: python3 merkle_store.py --test")
