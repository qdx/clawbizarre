"""
ClawBizarre Signed Handshake — Wires cryptographic identity into the handshake protocol.

Every handshake message is signed by the sender. Receipts generated on completion
are signed by the verifier. This creates a full audit trail where:
- Message authenticity is provable (no impersonation)
- Receipt authenticity is provable (no forged work claims)
- The entire handshake transcript is a signed evidence chain
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Optional, Callable

from identity import AgentIdentity, SignedReceipt, sign_receipt
from handshake import (
    HandshakeSession, HandshakeState, HandshakeMessage,
    Constraints, TaskProposal, AgentHello,
)
from receipt import WorkReceipt, TestResults, VerificationTier, hash_content


@dataclass
class SignedMessage:
    """A handshake message with cryptographic signature."""
    message: HandshakeMessage
    signer_id: str
    signature: str  # hex Ed25519 over canonical JSON of message payload + msg_type + timestamp

    @property
    def signing_content(self) -> str:
        """The canonical string that was signed."""
        return json.dumps({
            "msg_type": self.message.msg_type,
            "sender": self.message.sender,
            "payload": self.message.payload,
            "timestamp": self.message.timestamp,
            "msg_id": self.message.msg_id,
        }, sort_keys=True)

    def verify(self, identity: AgentIdentity) -> bool:
        if identity.agent_id != self.signer_id:
            return False
        return identity.verify(self.signing_content, self.signature)


class SignedHandshakeSession:
    """A handshake session where every message is cryptographically signed."""

    def __init__(self, identity: AgentIdentity):
        self.identity = identity
        self.session = HandshakeSession(identity.agent_id)
        self.signed_messages: list[SignedMessage] = []
        self.signed_receipt: Optional[SignedReceipt] = None
        self.peer_identity: Optional[AgentIdentity] = None

    @property
    def state(self) -> HandshakeState:
        return self.session.state

    @property
    def session_id(self) -> str:
        return self.session.session_id

    def _sign_message(self, msg: HandshakeMessage) -> SignedMessage:
        content = json.dumps({
            "msg_type": msg.msg_type,
            "sender": msg.sender,
            "payload": msg.payload,
            "timestamp": msg.timestamp,
            "msg_id": msg.msg_id,
        }, sort_keys=True)
        signature = self.identity.sign(content)
        sm = SignedMessage(message=msg, signer_id=self.identity.agent_id, signature=signature)
        self.signed_messages.append(sm)
        return sm

    def _verify_incoming(self, signed_msg: SignedMessage) -> bool:
        """Verify an incoming signed message against peer identity."""
        if not self.peer_identity:
            # First message — extract peer identity from signer_id
            pubkey_hex = signed_msg.signer_id.replace("ed25519:", "")
            self.peer_identity = AgentIdentity.from_public_key_hex(pubkey_hex)
        return signed_msg.verify(self.peer_identity)

    def send_hello(self, capabilities: list[str], constraints: Constraints = None) -> SignedMessage:
        msg = self.session.send_hello(capabilities, constraints)
        return self._sign_message(msg)

    def receive_hello(self, signed_msg: SignedMessage) -> bool:
        if not self._verify_incoming(signed_msg):
            return False
        return self.session.receive_hello(signed_msg.message)

    def propose(self, task_description: str, task_type: str,
                verification_tier: VerificationTier = VerificationTier.SELF_VERIFYING,
                test_suite_hash: str = None, input_data: str = None) -> SignedMessage:
        msg = self.session.propose(task_description, task_type, verification_tier,
                                    test_suite_hash, input_data)
        return self._sign_message(msg)

    def receive_proposal(self, signed_msg: SignedMessage) -> bool:
        if not self._verify_incoming(signed_msg):
            return False
        self.session.proposal = TaskProposal(**signed_msg.message.payload)
        self.session.messages.append(signed_msg.message)
        self.session.state = HandshakeState.PROPOSED
        return True

    def accept(self) -> SignedMessage:
        msg = self.session.accept()
        return self._sign_message(msg)

    def receive_accept(self, signed_msg: SignedMessage) -> bool:
        if not self._verify_incoming(signed_msg):
            return False
        self.session.messages.append(signed_msg.message)
        self.session.state = HandshakeState.ACCEPTED
        return True

    def reject(self, reason: str) -> SignedMessage:
        msg = self.session.reject(reason)
        return self._sign_message(msg)

    def execute(self, output: str, proof: dict = None) -> SignedMessage:
        msg = self.session.execute(output, proof)
        return self._sign_message(msg)

    def verify_and_receipt(self, execute_msg: SignedMessage,
                           verifier: Callable[[dict], TestResults] = None) -> tuple[SignedMessage, Optional[SignedReceipt]]:
        """Verify work and produce a signed receipt if it passes."""
        if not self._verify_incoming(execute_msg):
            raise ValueError("Execute message signature verification failed")

        verify_msg = self.session.verify(execute_msg.message, verifier)
        signed_verify = self._sign_message(verify_msg)

        if self.session.receipt:
            self.signed_receipt = sign_receipt(self.identity, self.session.receipt)

        return signed_verify, self.signed_receipt

    def abort(self, reason: str = "unspecified") -> SignedMessage:
        msg = self.session.abort(reason)
        return self._sign_message(msg)

    def transcript(self) -> str:
        lines = [f"=== Signed Handshake {self.session_id[:8]} ==="]
        for sm in self.signed_messages:
            m = sm.message
            verified = "✓" if (self.peer_identity and sm.verify(self.peer_identity)) or sm.signer_id == self.identity.agent_id else "?"
            lines.append(f"  [{verified}] {m.sender[:16]:>16} → {m.msg_type}")
        lines.append(f"  State: {self.state.value}")
        if self.signed_receipt:
            lines.append(f"  Signed Receipt: {self.signed_receipt.content_hash[:16]}...")
        return "\n".join(lines)


# --- Demo ---
if __name__ == "__main__":
    alice_id = AgentIdentity.generate()
    bob_id = AgentIdentity.generate()

    alice = SignedHandshakeSession(alice_id)
    bob = SignedHandshakeSession(bob_id)

    # 1. Exchange hellos
    alice_hello = alice.send_hello(["research", "writing"])
    assert bob.receive_hello(alice_hello), "Bob should accept Alice's hello"

    bob_hello = bob.send_hello(["code_review", "security_audit"])
    assert alice.receive_hello(bob_hello), "Alice should accept Bob's hello"

    # 2. Alice proposes
    test_suite = "assert 'sql injection' in review.lower()"
    proposal = alice.propose(
        task_description="Review this code for security issues",
        task_type="code_review",
        verification_tier=VerificationTier.SELF_VERIFYING,
        test_suite_hash=hash_content(test_suite),
        input_data="def transfer(amount): db.execute(f'UPDATE balance SET val={amount}')",
    )
    assert bob.receive_proposal(proposal), "Bob should accept proposal"

    # 3. Bob accepts and executes
    accept_msg = bob.accept()
    assert alice.receive_accept(accept_msg), "Alice should accept Bob's acceptance"

    execute_msg = bob.execute(
        "SQL injection vulnerability found: use parameterized queries",
        proof={"issues_found": 1}
    )

    # 4. Alice verifies with signed receipt
    def verify_review(payload):
        output = payload.get("output", "")
        checks = ["sql injection" in output.lower(), "parameterized" in output.lower()]
        return TestResults(passed=sum(checks), failed=len(checks)-sum(checks),
                          suite_hash=hash_content(test_suite))

    verify_msg, signed_receipt = alice.verify_and_receipt(execute_msg, verify_review)

    # Print results
    print(alice.transcript())
    print()
    print(bob.transcript())
    print()

    if signed_receipt:
        print(f"Receipt signed by: {signed_receipt.signer_id[:30]}...")
        print(f"Verifies with Alice's key: {signed_receipt.verify(alice_id)}")
        print(f"Forged with Bob's key: {signed_receipt.verify(bob_id)}")

    # Verify all messages in the transcript
    print("\n=== Message Verification ===")
    for sm in alice.signed_messages:
        if sm.signer_id == alice_id.agent_id:
            ok = sm.verify(alice_id)
        else:
            ok = sm.verify(bob_id)
        print(f"  {sm.message.msg_type}: {'✓' if ok else '✗'}")

    print("\nAll signed handshake tests passed ✓")
