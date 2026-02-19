"""
ClawBizarre Identity & Signing — Ed25519 keypairs for agent identity.

An agent's identity = a keypair. The public key IS the agent_id.
Receipts are signed with the private key, verifiable by anyone with the public key.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives import serialization


def _pubkey_to_id(pubkey: Ed25519PublicKey) -> str:
    """Derive a stable agent_id from a public key."""
    raw = pubkey.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    return f"ed25519:{raw.hex()}"


def _pubkey_to_fingerprint(pubkey: Ed25519PublicKey) -> str:
    """Short fingerprint for display (first 8 bytes of SHA-256 of pubkey)."""
    raw = pubkey.public_bytes(
        serialization.Encoding.Raw,
        serialization.PublicFormat.Raw,
    )
    h = hashlib.sha256(raw).hexdigest()[:16]
    return h


@dataclass
class AgentIdentity:
    """An agent's cryptographic identity."""
    agent_id: str
    fingerprint: str
    _private_key: Optional[Ed25519PrivateKey] = field(default=None, repr=False)
    _public_key: Optional[Ed25519PublicKey] = field(default=None, repr=False)

    @classmethod
    def generate(cls) -> "AgentIdentity":
        """Generate a new random identity."""
        private_key = Ed25519PrivateKey.generate()
        public_key = private_key.public_key()
        return cls(
            agent_id=_pubkey_to_id(public_key),
            fingerprint=_pubkey_to_fingerprint(public_key),
            _private_key=private_key,
            _public_key=public_key,
        )

    @classmethod
    def from_keyfile(cls, path: str) -> "AgentIdentity":
        """Load identity from a PEM private key file."""
        with open(path, "rb") as f:
            private_key = serialization.load_pem_private_key(f.read(), password=None)
        assert isinstance(private_key, Ed25519PrivateKey)
        public_key = private_key.public_key()
        return cls(
            agent_id=_pubkey_to_id(public_key),
            fingerprint=_pubkey_to_fingerprint(public_key),
            _private_key=private_key,
            _public_key=public_key,
        )

    @classmethod
    def from_public_key_hex(cls, hex_str: str) -> "AgentIdentity":
        """Load a verify-only identity from a public key hex string."""
        raw = bytes.fromhex(hex_str)
        public_key = Ed25519PublicKey.from_public_bytes(raw)
        return cls(
            agent_id=_pubkey_to_id(public_key),
            fingerprint=_pubkey_to_fingerprint(public_key),
            _private_key=None,
            _public_key=public_key,
        )

    def save_keyfile(self, path: str):
        """Save private key to PEM file."""
        if not self._private_key:
            raise ValueError("No private key to save (verify-only identity)")
        pem = self._private_key.private_bytes(
            serialization.Encoding.PEM,
            serialization.PrivateFormat.PKCS8,
            serialization.NoEncryption(),
        )
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(pem)
        os.chmod(path, 0o600)

    def sign(self, data: str) -> str:
        """Sign a string, return hex signature."""
        if not self._private_key:
            raise ValueError("No private key (verify-only identity)")
        sig = self._private_key.sign(data.encode())
        return sig.hex()

    def verify(self, data: str, signature_hex: str) -> bool:
        """Verify a signature. Returns True if valid."""
        try:
            self._public_key.verify(bytes.fromhex(signature_hex), data.encode())
            return True
        except Exception:
            return False

    @property
    def public_key_hex(self) -> str:
        raw = self._public_key.public_bytes(
            serialization.Encoding.Raw,
            serialization.PublicFormat.Raw,
        )
        return raw.hex()


@dataclass
class SignedReceipt:
    """A work receipt with a cryptographic signature from its creator."""
    receipt_json: str       # The canonical JSON of the WorkReceipt
    content_hash: str       # sha256 of canonical JSON (from WorkReceipt.content_hash)
    signer_id: str          # ed25519:<pubkey_hex>
    signature: str          # hex-encoded Ed25519 signature of content_hash

    def to_json(self) -> str:
        return json.dumps({
            "receipt": json.loads(self.receipt_json),
            "content_hash": self.content_hash,
            "signer_id": self.signer_id,
            "signature": self.signature,
        }, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "SignedReceipt":
        d = json.loads(data)
        return cls(
            receipt_json=json.dumps(d["receipt"], indent=2),
            content_hash=d["content_hash"],
            signer_id=d["signer_id"],
            signature=d["signature"],
        )

    def verify(self, identity: AgentIdentity) -> bool:
        """Verify this signed receipt against a known identity."""
        if identity.agent_id != self.signer_id:
            return False
        return identity.verify(self.content_hash, self.signature)


def sign_receipt(identity: AgentIdentity, receipt) -> SignedReceipt:
    """Sign a WorkReceipt, producing a SignedReceipt."""
    receipt_json = receipt.to_json()
    content_hash = receipt.content_hash
    signature = identity.sign(content_hash)
    return SignedReceipt(
        receipt_json=receipt_json,
        content_hash=content_hash,
        signer_id=identity.agent_id,
        signature=signature,
    )


# --- Demo ---

if __name__ == "__main__":
    # Generate two identities
    alice = AgentIdentity.generate()
    bob = AgentIdentity.generate()

    print(f"Alice: {alice.agent_id[:30]}... (fp: {alice.fingerprint})")
    print(f"Bob:   {bob.agent_id[:30]}... (fp: {bob.fingerprint})")

    # Alice signs a message
    msg = "sha256:abc123def456"
    sig = alice.sign(msg)
    print(f"\nAlice signs: {msg}")
    print(f"Signature: {sig[:40]}...")

    # Verify with Alice's key — should pass
    assert alice.verify(msg, sig), "Alice's own signature should verify"
    print("✓ Alice verifies own signature")

    # Verify with Bob's key — should fail
    assert not bob.verify(msg, sig), "Bob's key should NOT verify Alice's signature"
    print("✓ Bob cannot forge Alice's signature")

    # Round-trip: save and load keyfile
    alice.save_keyfile("/tmp/clawbizarre_test_key.pem")
    alice_restored = AgentIdentity.from_keyfile("/tmp/clawbizarre_test_key.pem")
    assert alice_restored.agent_id == alice.agent_id
    assert alice_restored.verify(msg, sig)
    print("✓ Keyfile round-trip works")

    # Verify-only identity from public key
    alice_pub = AgentIdentity.from_public_key_hex(alice.public_key_hex)
    assert alice_pub.verify(msg, sig)
    print("✓ Public-key-only verification works")

    # Sign a receipt
    from receipt import WorkReceipt, TestResults, VerificationTier, hash_content

    receipt = WorkReceipt(
        agent_id=alice.agent_id,
        task_type="code_review",
        verification_tier=VerificationTier.SELF_VERIFYING,
        input_hash=hash_content("some code"),
        output_hash=hash_content("review output"),
        test_results=TestResults(passed=3, failed=0, suite_hash=hash_content("tests")),
    )

    signed = sign_receipt(alice, receipt)
    print(f"\n=== Signed Receipt ===")
    print(f"Content hash: {signed.content_hash}")
    print(f"Signer: {signed.signer_id[:30]}...")
    print(f"Signature: {signed.signature[:40]}...")
    print(f"Verified: {signed.verify(alice)}")
    print(f"Forged check: {signed.verify(bob)}")

    # Round-trip signed receipt
    sr_json = signed.to_json()
    sr_restored = SignedReceipt.from_json(sr_json)
    assert sr_restored.verify(alice)
    print("✓ SignedReceipt round-trip works")

    os.unlink("/tmp/clawbizarre_test_key.pem")
    print("\nAll identity tests passed ✓")
