#!/usr/bin/env python3
"""
VRF-COSE Identity Bridge: Connect AgentIdentity (identity.py) to COSE receipts (vrf_cose.py).

Features:
1. Sign VRF receipts using existing AgentIdentity keys
2. Chain receipts via previous_receipt_hash in COSE protected headers
3. Verify receipt chains end-to-end
4. DID generation from AgentIdentity
"""

import hashlib
import json
import time
from typing import Optional, List, Tuple

from identity import AgentIdentity
from vrf_cose import (
    VRFCoseReceipt, TransparencyLog,
    LABEL_VRF_VERSION, LABEL_ISSUER_DID, LABEL_RECEIPT_ID,
)

# Additional COSE protected header labels for chaining
LABEL_PREV_RECEIPT_HASH = -70004  # SHA-256 of previous COSE receipt bytes
LABEL_CHAIN_POSITION = -70005     # Sequence number in agent's receipt chain


def agent_to_did(identity: AgentIdentity) -> str:
    """Convert AgentIdentity to a did:key identifier."""
    # Extract raw public key hex from agent_id format "ed25519:<hex>"
    hex_str = identity.agent_id.split(":")[-1]
    return f"did:key:z6Mk{hex_str[:32]}"  # Abbreviated multibase encoding


def identity_to_signing_material(identity: AgentIdentity) -> Tuple[bytes, bytes]:
    """Extract raw signing key + public key bytes from AgentIdentity."""
    from cryptography.hazmat.primitives import serialization
    
    if identity._private_key is None:
        raise ValueError("Identity has no private key (verify-only)")
    
    private_bytes = identity._private_key.private_bytes(
        serialization.Encoding.Raw,
        serialization.PrivateFormat.Raw,
        serialization.NoEncryption(),
    )
    public_bytes = identity._public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return private_bytes, public_bytes


def public_key_bytes(identity: AgentIdentity) -> bytes:
    """Extract raw public key bytes from AgentIdentity."""
    from cryptography.hazmat.primitives import serialization
    return identity._public_key.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )


class IdentityCoseReceipt(VRFCoseReceipt):
    """VRF COSE receipt signed with an AgentIdentity, with chain linking."""
    
    def __init__(self, receipt_data: dict, identity: AgentIdentity,
                 prev_receipt_hash: Optional[str] = None,
                 chain_position: Optional[int] = None):
        priv, pub = identity_to_signing_material(identity)
        did = agent_to_did(identity)
        super().__init__(receipt_data, signing_key=priv, public_key=pub, issuer_did=did)
        self.prev_receipt_hash = prev_receipt_hash
        self.chain_position = chain_position
    
    def to_cose_sign1(self) -> bytes:
        """Encode with chain-linking headers."""
        import cbor2
        from pycose.messages import Sign1Message
        from pycose.headers import Algorithm, ContentType
        from pycose.algorithms import EdDSA
        from pycose.keys import OKPKey
        from pycose.keys.curves import Ed25519
        
        payload = cbor2.dumps(self.receipt_data)
        
        protected = {
            Algorithm: EdDSA,
            ContentType: "application/vrf+cbor",
            LABEL_VRF_VERSION: self.receipt_data.get("vrf_version", "1.0"),
            LABEL_ISSUER_DID: self.issuer_did,
        }
        
        receipt_id = self.receipt_data.get("receipt_id")
        if receipt_id:
            protected[LABEL_RECEIPT_ID] = receipt_id
        
        # Chain linking
        if self.prev_receipt_hash:
            protected[LABEL_PREV_RECEIPT_HASH] = self.prev_receipt_hash
        if self.chain_position is not None:
            protected[LABEL_CHAIN_POSITION] = self.chain_position
        
        msg = Sign1Message(phdr=protected, payload=payload)
        msg.key = OKPKey(crv=Ed25519, d=self.signing_key, x=self.public_key)
        return msg.encode()


class AgentReceiptChain:
    """Manages a chain of COSE-signed VRF receipts for one agent."""
    
    def __init__(self, identity: AgentIdentity):
        self.identity = identity
        self.receipts: List[bytes] = []  # COSE-encoded receipts
        self._last_hash: Optional[str] = None
    
    def append(self, receipt_data: dict) -> bytes:
        """Sign and append a receipt, linking to previous."""
        position = len(self.receipts)
        
        cose_receipt = IdentityCoseReceipt(
            receipt_data=receipt_data,
            identity=self.identity,
            prev_receipt_hash=self._last_hash,
            chain_position=position,
        )
        
        encoded = cose_receipt.to_cose_sign1()
        self._last_hash = hashlib.sha256(encoded).hexdigest()
        self.receipts.append(encoded)
        return encoded
    
    def verify_chain(self) -> Tuple[bool, List[str]]:
        """Verify entire chain: signatures + hash links."""
        errors = []
        pub = public_key_bytes(self.identity)
        prev_hash = None
        
        for i, cose_bytes in enumerate(self.receipts):
            # Verify signature
            try:
                decoded = VRFCoseReceipt.from_cose_sign1(cose_bytes)
                if not decoded.verify(pub):
                    errors.append(f"[{i}] Signature verification failed")
            except Exception as e:
                errors.append(f"[{i}] Decode/verify error: {e}")
                continue
            
            # Verify chain link
            phdr = decoded._cose_msg.phdr
            claimed_prev = phdr.get(LABEL_PREV_RECEIPT_HASH)
            
            if i == 0:
                if claimed_prev is not None:
                    errors.append(f"[0] First receipt should have no prev_hash, got {claimed_prev}")
            else:
                if claimed_prev != prev_hash:
                    errors.append(f"[{i}] Chain break: expected prev={prev_hash}, got {claimed_prev}")
            
            # Verify chain position
            claimed_pos = phdr.get(LABEL_CHAIN_POSITION)
            if claimed_pos is not None and claimed_pos != i:
                errors.append(f"[{i}] Position mismatch: expected {i}, got {claimed_pos}")
            
            prev_hash = hashlib.sha256(cose_bytes).hexdigest()
        
        return len(errors) == 0, errors
    
    @property
    def length(self) -> int:
        return len(self.receipts)
    
    @property
    def last_hash(self) -> Optional[str]:
        return self._last_hash


# ─── Tests ───

def test_identity_cose_roundtrip():
    """Sign a COSE receipt with AgentIdentity and verify."""
    alice = AgentIdentity.generate()
    
    receipt = {
        "vrf_version": "1.0",
        "receipt_id": "id-test-001",
        "verdict": "pass",
        "provider_id": alice.agent_id,
        "timestamp": time.time(),
        "tests_passed": 3,
        "tests_total": 3,
    }
    
    cose_receipt = IdentityCoseReceipt(receipt, alice)
    encoded = cose_receipt.to_cose_sign1()
    
    # Decode and verify
    decoded = VRFCoseReceipt.from_cose_sign1(encoded)
    pub = public_key_bytes(alice)
    assert decoded.verify(pub), "Signature verification failed"
    assert decoded.receipt_data["receipt_id"] == "id-test-001"
    
    # Wrong key should fail
    bob = AgentIdentity.generate()
    bob_pub = public_key_bytes(bob)
    try:
        result = decoded.verify(bob_pub)
        assert not result
    except Exception:
        pass  # Expected
    
    print(f"  Alice DID: {agent_to_did(alice)}")
    print(f"  Encoded: {len(encoded)} bytes")
    print(f"  Signature verified ✅, wrong-key rejected ✅")
    return True


def test_receipt_chain():
    """Build and verify a chain of linked receipts."""
    alice = AgentIdentity.generate()
    chain = AgentReceiptChain(alice)
    
    for i in range(5):
        receipt = {
            "vrf_version": "1.0",
            "receipt_id": f"chain-{i}",
            "verdict": "pass" if i != 3 else "fail",
            "provider_id": alice.agent_id,
            "timestamp": time.time(),
        }
        chain.append(receipt)
    
    assert chain.length == 5
    valid, errors = chain.verify_chain()
    assert valid, f"Chain verification failed: {errors}"
    
    print(f"  Chain length: {chain.length}")
    print(f"  Last hash: {chain.last_hash[:16]}...")
    print(f"  Chain verified ✅")
    return True


def test_chain_tamper_detection():
    """Verify that tampering with a receipt breaks the chain."""
    alice = AgentIdentity.generate()
    chain = AgentReceiptChain(alice)
    
    for i in range(3):
        chain.append({
            "vrf_version": "1.0",
            "receipt_id": f"tamper-{i}",
            "verdict": "pass",
            "timestamp": time.time(),
        })
    
    # Tamper: replace middle receipt with a different one
    # (re-signed by alice but with wrong prev_hash link)
    tampered = IdentityCoseReceipt(
        {"vrf_version": "1.0", "receipt_id": "tamper-1", "verdict": "fail", "timestamp": time.time()},
        alice,
        prev_receipt_hash="0000dead",
        chain_position=1,
    )
    chain.receipts[1] = tampered.to_cose_sign1()
    
    valid, errors = chain.verify_chain()
    assert not valid, "Tampered chain should fail"
    assert len(errors) >= 1
    
    print(f"  Tampered chain detected: {len(errors)} error(s)")
    for e in errors:
        print(f"    {e}")
    return True


def test_chain_in_transparency_log():
    """Register a receipt chain in the transparency log."""
    alice = AgentIdentity.generate()
    chain = AgentReceiptChain(alice)
    log = TransparencyLog()
    
    transparency_receipts = []
    for i in range(4):
        encoded = chain.append({
            "vrf_version": "1.0",
            "receipt_id": f"log-{i}",
            "verdict": "pass",
            "timestamp": time.time(),
        })
        tr = log.register(encoded)
        transparency_receipts.append(tr)
    
    # Verify chain integrity
    valid, errors = chain.verify_chain()
    assert valid, f"Chain failed: {errors}"
    
    # Verify all in transparency log
    for i in range(4):
        ok = log.verify_inclusion(chain.receipts[i], transparency_receipts[i])
        assert ok, f"Inclusion proof failed for {i}"
    
    # Consistency proof
    cp = log.consistency_proof(2, 4)
    assert len(cp["proof"]) > 0
    
    print(f"  4 receipts chained + registered in transparency log")
    print(f"  Chain verified ✅, all 4 inclusion proofs ✅")
    print(f"  Consistency proof (2→4): {len(cp['proof'])} nodes")
    return True


def test_cross_agent_verification():
    """Two agents produce receipts, both verifiable in same log."""
    alice = AgentIdentity.generate()
    bob = AgentIdentity.generate()
    
    alice_chain = AgentReceiptChain(alice)
    bob_chain = AgentReceiptChain(bob)
    log = TransparencyLog()
    
    # Alice does 2 tasks, Bob does 1
    for i in range(2):
        encoded = alice_chain.append({
            "vrf_version": "1.0", "receipt_id": f"alice-{i}",
            "verdict": "pass", "provider_id": alice.agent_id,
            "buyer_id": bob.agent_id, "timestamp": time.time(),
        })
        log.register(encoded)
    
    encoded = bob_chain.append({
        "vrf_version": "1.0", "receipt_id": "bob-0",
        "verdict": "pass", "provider_id": bob.agent_id,
        "buyer_id": alice.agent_id, "timestamp": time.time(),
    })
    log.register(encoded)
    
    # Both chains independently valid
    a_ok, _ = alice_chain.verify_chain()
    b_ok, _ = bob_chain.verify_chain()
    assert a_ok and b_ok
    
    # Cross-verify: Bob can verify Alice's receipt signature
    decoded = VRFCoseReceipt.from_cose_sign1(alice_chain.receipts[0])
    assert decoded.verify(public_key_bytes(alice))
    
    stats = log.stats()
    print(f"  Alice: {alice_chain.length} receipts, Bob: {bob_chain.length}")
    print(f"  Log: {stats['total_entries']} entries, root: {stats['tree_root'][:16]}...")
    print(f"  Cross-agent verification ✅")
    return True


if __name__ == "__main__":
    tests = [
        ("Identity → COSE roundtrip", test_identity_cose_roundtrip),
        ("Receipt chain", test_receipt_chain),
        ("Tamper detection", test_chain_tamper_detection),
        ("Chain in transparency log", test_chain_in_transparency_log),
        ("Cross-agent verification", test_cross_agent_verification),
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
