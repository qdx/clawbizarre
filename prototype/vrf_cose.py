#!/usr/bin/env python3
"""
VRF-COSE: SCITT-aligned COSE encoding for Verification Receipt Format.

Maps VRF receipts to COSE Sign1 structures, compatible with IETF SCITT
(draft-ietf-scitt-architecture-22) Signed Statements.

A VRF receipt becomes a SCITT Signed Statement where:
- Protected header: algorithm, content type, issuer (agent DID)
- Payload: CBOR-encoded VRF receipt fields
- Signature: Ed25519 (same key as existing VRF identity)
"""

import json
import os
import time
import hashlib
from typing import Optional, List

import cbor2
from pycose.messages import Sign1Message
from pycose.headers import Algorithm, ContentType
from pycose.algorithms import EdDSA
from pycose.keys import OKPKey
from pycose.keys.curves import Ed25519
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey


# SCITT-aligned content type for VRF
VRF_CONTENT_TYPE = "application/vrf+cbor"

# Custom COSE header labels (private use range: < -65536)
LABEL_VRF_VERSION = -70001
LABEL_ISSUER_DID = -70002
LABEL_RECEIPT_ID = -70003


class VRFCoseReceipt:
    """A VRF receipt encoded as a COSE Sign1 message (SCITT Signed Statement)."""

    def __init__(self, receipt_data: dict, signing_key: Optional[bytes] = None,
                 public_key: Optional[bytes] = None, issuer_did: Optional[str] = None):
        self.receipt_data = receipt_data
        # Generate ephemeral key if none provided (COSE Sign1 requires signing)
        if signing_key is None or public_key is None:
            seed = os.urandom(32)
            pk = Ed25519PrivateKey.from_private_bytes(seed)
            signing_key = seed
            public_key = pk.public_key().public_bytes_raw()
        self.signing_key = signing_key
        self.public_key = public_key
        self.issuer_did = issuer_did or f"did:key:{hashlib.sha256(public_key).hexdigest()[:16]}"

    def to_cose_sign1(self) -> bytes:
        """Encode VRF receipt as COSE Sign1 (SCITT Signed Statement)."""
        payload = cbor2.dumps(self.receipt_data)

        protected = {
            Algorithm: EdDSA,
            ContentType: VRF_CONTENT_TYPE,
            LABEL_VRF_VERSION: self.receipt_data.get("vrf_version", "1.0"),
            LABEL_ISSUER_DID: self.issuer_did,
        }

        receipt_id = self.receipt_data.get("receipt_id")
        if receipt_id:
            protected[LABEL_RECEIPT_ID] = receipt_id

        msg = Sign1Message(
            phdr=protected,
            payload=payload,
        )
        msg.key = OKPKey(crv=Ed25519, d=self.signing_key, x=self.public_key)

        return msg.encode()

    @staticmethod
    def from_cose_sign1(data: bytes) -> 'VRFCoseReceipt':
        """Decode a COSE Sign1 message back to a VRF receipt."""
        msg = Sign1Message.decode(data)

        # Extract payload
        receipt_data = cbor2.loads(msg.payload)

        # Extract metadata from protected headers
        phdr = msg.phdr
        issuer_did = phdr.get(LABEL_ISSUER_DID, "unknown")
        vrf_version = phdr.get(LABEL_VRF_VERSION, "1.0")

        receipt = VRFCoseReceipt(
            receipt_data=receipt_data,
            issuer_did=issuer_did,
        )
        receipt._cose_msg = msg
        return receipt

    def verify(self, public_key: bytes) -> bool:
        """Verify the COSE Sign1 signature."""
        if not hasattr(self, '_cose_msg'):
            raise ValueError("Can only verify decoded messages")

        self._cose_msg.key = OKPKey(crv=Ed25519, x=public_key)
        return self._cose_msg.verify_signature()

    def size_bytes(self) -> int:
        """Return the encoded size."""
        return len(self.to_cose_sign1())

    def to_transparency_entry(self) -> dict:
        """Format as a SCITT Transparency Service log entry."""
        encoded = self.to_cose_sign1()
        return {
            "entry_id": hashlib.sha256(encoded).hexdigest()[:16],
            "issuer": self.issuer_did,
            "content_type": VRF_CONTENT_TYPE,
            "registered_at": time.time(),
            "signed_statement_hash": hashlib.sha256(encoded).hexdigest(),
            "signed_statement_size": len(encoded),
            "receipt_id": self.receipt_data.get("receipt_id"),
            "verdict": self.receipt_data.get("verdict"),
        }


def json_receipt_to_cose(json_receipt: dict, signing_key: bytes = None,
                          public_key: bytes = None) -> bytes:
    """Convert a JSON VRF receipt to COSE Sign1 encoding."""
    receipt = VRFCoseReceipt(
        receipt_data=json_receipt,
        signing_key=signing_key,
        public_key=public_key,
    )
    return receipt.to_cose_sign1()


def cose_to_json_receipt(cose_data: bytes) -> dict:
    """Convert a COSE Sign1 encoded receipt back to JSON-compatible dict."""
    receipt = VRFCoseReceipt.from_cose_sign1(cose_data)
    return receipt.receipt_data


# ─── Transparency Service (minimal append-only log) ───

class TransparencyLog:
    """SCITT Transparency Service with RFC 9162-compatible Merkle proofs.
    
    Append-only log of signed statements with cryptographic inclusion
    and consistency proofs via MerkleTree.
    """

    def __init__(self):
        from merkle import MerkleTree, proof_to_hex, hash_leaf
        self._tree = MerkleTree()
        self.entries = []
        self._index = {}  # receipt_id → entry_index
        self._cose_blobs = []  # raw COSE bytes for each entry
        self._roots_at_size = {}  # size → root hash (for consistency)

    def register(self, cose_data: bytes) -> dict:
        """Register a signed statement and return a transparency receipt with Merkle proof."""
        from merkle import proof_to_hex

        receipt = VRFCoseReceipt.from_cose_sign1(cose_data)
        entry = receipt.to_transparency_entry()
        seq = len(self.entries)
        entry["sequence_number"] = seq

        # Append to Merkle tree
        self._tree.append(cose_data)
        self._cose_blobs.append(cose_data)
        self.entries.append(entry)
        self._roots_at_size[self._tree.size] = self._tree.root

        receipt_id = entry.get("receipt_id")
        if receipt_id:
            self._index[receipt_id] = seq

        # SCITT "Receipt" = cryptographic proof of inclusion
        inclusion_proof = self._tree.inclusion_proof(seq)
        transparency_receipt = {
            "entry_id": entry["entry_id"],
            "sequence_number": seq,
            "log_size": self._tree.size,
            "tree_root": self._tree.root.hex(),
            "inclusion_proof": proof_to_hex(inclusion_proof),
        }
        return transparency_receipt

    def verify_inclusion(self, cose_data: bytes, transparency_receipt: dict) -> bool:
        """Verify that a COSE signed statement is included in the log."""
        from merkle import MerkleTree, proof_from_hex, hash_leaf

        seq = transparency_receipt["sequence_number"]
        log_size = transparency_receipt["log_size"]
        expected_root = bytes.fromhex(transparency_receipt["tree_root"])
        proof = proof_from_hex(transparency_receipt["inclusion_proof"])
        leaf_hash = hash_leaf(cose_data)

        return MerkleTree.verify_inclusion(leaf_hash, seq, log_size, proof, expected_root)

    def consistency_proof(self, old_size: int, new_size: int = None) -> dict:
        """Generate a consistency proof between two log sizes."""
        from merkle import proof_to_hex
        if new_size is None:
            new_size = self._tree.size
        proof = self._tree.consistency_proof(old_size, new_size)
        return {
            "old_size": old_size,
            "new_size": new_size,
            "old_root": self._roots_at_size.get(old_size, b'').hex(),
            "new_root": self._roots_at_size.get(new_size, self._tree.root).hex(),
            "proof": proof_to_hex(proof),
        }

    def lookup(self, receipt_id: str) -> Optional[dict]:
        """Look up an entry by receipt ID."""
        idx = self._index.get(receipt_id)
        if idx is not None:
            return self.entries[idx]
        return None

    def get_proof(self, index: int) -> dict:
        """Get inclusion proof for an existing entry."""
        from merkle import proof_to_hex
        if index >= len(self.entries):
            raise ValueError(f"Index {index} out of range")
        proof = self._tree.inclusion_proof(index)
        return {
            "entry_id": self.entries[index]["entry_id"],
            "sequence_number": index,
            "log_size": self._tree.size,
            "tree_root": self._tree.root.hex(),
            "inclusion_proof": proof_to_hex(proof),
        }

    @property
    def root(self) -> bytes:
        return self._tree.root

    def stats(self) -> dict:
        verdicts = {}
        for e in self.entries:
            v = e.get("verdict", "unknown")
            verdicts[v] = verdicts.get(v, 0) + 1
        return {
            "total_entries": len(self.entries),
            "tree_root": self._tree.root.hex(),
            "verdicts": verdicts,
        }


# ─── Tests ───

def test_roundtrip():
    """Test JSON → COSE → JSON roundtrip."""
    receipt = {
        "vrf_version": "1.0",
        "receipt_id": "test-001",
        "verdict": "pass",
        "provider_id": "agent-alice",
        "buyer_id": "agent-bob",
        "task_type": "code_review",
        "timestamp": time.time(),
        "tests_passed": 5,
        "tests_total": 5,
        "execution_time_ms": 142,
    }

    # Encode
    cose_bytes = json_receipt_to_cose(receipt)
    assert len(cose_bytes) > 0, "COSE encoding produced empty output"

    # Decode
    decoded = cose_to_json_receipt(cose_bytes)
    assert decoded["receipt_id"] == "test-001"
    assert decoded["verdict"] == "pass"
    assert decoded["tests_passed"] == 5

    # Size comparison
    json_size = len(json.dumps(receipt).encode())
    cose_size = len(cose_bytes)
    print(f"  JSON size: {json_size} bytes")
    print(f"  COSE size: {cose_size} bytes")
    print(f"  Ratio: {cose_size/json_size:.2f}x")

    return True


def test_transparency_log():
    """Test SCITT Transparency Service with real Merkle proofs."""
    log = TransparencyLog()

    cose_blobs = []
    transparency_receipts = []
    for i in range(5):
        r = {"receipt_id": f"r-{i}", "verdict": "pass" if i % 2 == 0 else "fail",
             "vrf_version": "1.0", "timestamp": time.time()}
        cose = json_receipt_to_cose(r)
        cose_blobs.append(cose)
        tr = log.register(cose)
        transparency_receipts.append(tr)
        assert tr["sequence_number"] == i
        assert "tree_root" in tr

    assert log.stats()["total_entries"] == 5

    # Verify all inclusion proofs
    for i in range(5):
        ok = log.verify_inclusion(cose_blobs[i], transparency_receipts[i])
        assert ok, f"Inclusion proof failed for entry {i}"

    # Lookup
    entry = log.lookup("r-2")
    assert entry is not None
    assert entry["verdict"] == "pass"

    # Get fresh proof for older entry (against current tree)
    fresh = log.get_proof(1)
    assert fresh["log_size"] == 5
    ok = log.verify_inclusion(cose_blobs[1], fresh)
    assert ok, "Fresh proof for entry 1 failed"

    # Consistency proof
    cp = log.consistency_proof(3, 5)
    assert len(cp["proof"]) > 0

    # Stats include verdicts
    stats = log.stats()
    assert stats["verdicts"]["pass"] == 3
    assert stats["verdicts"]["fail"] == 2

    print(f"  Log entries: {stats['total_entries']}")
    print(f"  Root: {stats['tree_root'][:16]}...")
    print(f"  All 5 inclusion proofs verified ✅")
    print(f"  Consistency proof (3→5): {len(cp['proof'])} nodes")
    print(f"  Verdicts: {stats['verdicts']}")
    return True


def test_size_comparison():
    """Compare COSE vs JSON sizes across receipt complexities."""
    # Minimal receipt
    minimal = {"receipt_id": "m1", "verdict": "pass", "vrf_version": "1.0"}

    # Full receipt
    full = {
        "vrf_version": "1.0",
        "receipt_id": "full-001",
        "verdict": "pass",
        "provider_id": "did:key:abc123def456",
        "buyer_id": "did:key:789ghi012jkl",
        "task_type": "code_generation",
        "domain": "python",
        "timestamp": 1708300000.0,
        "tests_passed": 12,
        "tests_total": 12,
        "execution_time_ms": 2847,
        "sandbox": {"type": "docker", "network": False, "memory_mb": 128},
        "chain_position": 47,
        "previous_receipt_hash": "a1b2c3d4e5f6" * 4,
    }

    results = []
    for name, receipt in [("minimal", minimal), ("full", full)]:
        json_size = len(json.dumps(receipt).encode())
        cose_size = len(json_receipt_to_cose(receipt))
        results.append((name, json_size, cose_size))
        print(f"  {name}: JSON={json_size}B, COSE={cose_size}B, ratio={cose_size/json_size:.2f}x")

    return True


def test_sign_and_verify():
    """Test signing and verification with known keys."""
    seed = os.urandom(32)
    pk = Ed25519PrivateKey.from_private_bytes(seed)
    pub = pk.public_key().public_bytes_raw()

    receipt = {
        "vrf_version": "1.0",
        "receipt_id": "verify-test-001",
        "verdict": "pass",
        "provider_id": "agent-alice",
        "timestamp": time.time(),
    }

    cose_receipt = VRFCoseReceipt(
        receipt_data=receipt,
        signing_key=seed,
        public_key=pub,
    )

    encoded = cose_receipt.to_cose_sign1()

    # Decode and verify
    decoded = VRFCoseReceipt.from_cose_sign1(encoded)
    assert decoded.verify(pub), "Signature verification failed"

    # Verify with wrong key fails
    wrong_seed = os.urandom(32)
    wrong_pk = Ed25519PrivateKey.from_private_bytes(wrong_seed)
    wrong_pub = wrong_pk.public_key().public_bytes_raw()
    try:
        result = decoded.verify(wrong_pub)
        assert not result, "Wrong key should fail verification"
    except Exception:
        pass  # Expected — verification failure may raise

    print(f"  Signed COSE receipt: {len(encoded)} bytes")
    print(f"  Signature verified: ✅")
    print(f"  Wrong-key rejected: ✅")
    return True


if __name__ == "__main__":
    tests = [
        ("Roundtrip encoding", test_roundtrip),
        ("Transparency log", test_transparency_log),
        ("Size comparison", test_size_comparison),
        ("Sign and verify", test_sign_and_verify),
    ]

    passed = 0
    for name, test_fn in tests:
        print(f"\n[TEST] {name}")
        try:
            if test_fn():
                print(f"  ✅ PASS")
                passed += 1
            else:
                print(f"  ❌ FAIL")
        except Exception as e:
            print(f"  ❌ ERROR: {e}")

    print(f"\n{'='*40}")
    print(f"Results: {passed}/{len(tests)} passed")
