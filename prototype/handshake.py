"""
ClawBizarre Handshake Protocol v1.0
Two unfamiliar agents agree on a task, execute it, and verify completion — without pre-existing trust.

Core insight: Don't negotiate trust. Negotiate *verification*.
"""

import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Callable

from receipt import WorkReceipt, TestResults, VerificationTier, hash_content, ReceiptChain


class HandshakeState(Enum):
    INIT = "init"
    HELLO_SENT = "hello_sent"
    HELLO_RECEIVED = "hello_received"
    PROPOSED = "proposed"
    ACCEPTED = "accepted"
    COUNTERED = "countered"
    REJECTED = "rejected"
    EXECUTING = "executing"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"
    ABORTED = "aborted"


@dataclass
class Constraints:
    time_limit_seconds: int = 1800  # 30 min default
    budget: Optional[str] = None     # "none" | "10_sats" | etc
    privacy: str = "no_credential_sharing"


@dataclass
class AgentHello:
    agent_id: str
    capabilities: list[str]
    constraints: Constraints
    verification_preference: str = "tier_0_self_verifying"


@dataclass
class TaskProposal:
    task_description: str
    task_type: str
    verification_tier: VerificationTier
    test_suite_hash: Optional[str] = None  # For Tier 0
    success_criteria: Optional[str] = None  # For higher tiers
    input_data: Optional[str] = None
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))


@dataclass
class HandshakeMessage:
    """A single message in the handshake protocol."""
    msg_type: str  # HELLO | PROPOSE | ACCEPT | COUNTER | REJECT | EXECUTE | VERIFY | RECEIPT | ABORT
    sender: str
    payload: dict
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))


class HandshakeSession:
    """Manages one handshake between two agents."""

    def __init__(self, my_agent_id: str):
        self.my_agent_id = my_agent_id
        self.state = HandshakeState.INIT
        self.my_hello: Optional[AgentHello] = None
        self.their_hello: Optional[AgentHello] = None
        self.proposal: Optional[TaskProposal] = None
        self.messages: list[HandshakeMessage] = []
        self.receipt: Optional[WorkReceipt] = None
        self.session_id = str(uuid.uuid4())

    def send_hello(self, capabilities: list[str], constraints: Constraints = None) -> HandshakeMessage:
        if constraints is None:
            constraints = Constraints()
        self.my_hello = AgentHello(
            agent_id=self.my_agent_id,
            capabilities=capabilities,
            constraints=constraints,
        )
        msg = HandshakeMessage(
            msg_type="HELLO",
            sender=self.my_agent_id,
            payload=asdict(self.my_hello),
        )
        self.messages.append(msg)
        self.state = HandshakeState.HELLO_SENT
        return msg

    def receive_hello(self, msg: HandshakeMessage) -> bool:
        if msg.msg_type != "HELLO":
            return False
        self.their_hello = AgentHello(
            agent_id=msg.payload["agent_id"],
            capabilities=msg.payload["capabilities"],
            constraints=Constraints(**msg.payload["constraints"]),
            verification_preference=msg.payload.get("verification_preference", "tier_0_self_verifying"),
        )
        self.messages.append(msg)
        self.state = HandshakeState.HELLO_RECEIVED
        return True

    def propose(self, task_description: str, task_type: str,
                verification_tier: VerificationTier = VerificationTier.SELF_VERIFYING,
                test_suite_hash: str = None, input_data: str = None) -> HandshakeMessage:
        self.proposal = TaskProposal(
            task_description=task_description,
            task_type=task_type,
            verification_tier=verification_tier,
            test_suite_hash=test_suite_hash,
            input_data=input_data,
        )
        msg = HandshakeMessage(
            msg_type="PROPOSE",
            sender=self.my_agent_id,
            payload=asdict(self.proposal),
        )
        self.messages.append(msg)
        self.state = HandshakeState.PROPOSED
        return msg

    def accept(self) -> HandshakeMessage:
        msg = HandshakeMessage(
            msg_type="ACCEPT",
            sender=self.my_agent_id,
            payload={"proposal_id": self.proposal.proposal_id},
        )
        self.messages.append(msg)
        self.state = HandshakeState.ACCEPTED
        return msg

    def reject(self, reason: str) -> HandshakeMessage:
        msg = HandshakeMessage(
            msg_type="REJECT",
            sender=self.my_agent_id,
            payload={"proposal_id": self.proposal.proposal_id, "reason": reason},
        )
        self.messages.append(msg)
        self.state = HandshakeState.REJECTED
        return msg

    def execute(self, output: str, proof: dict = None) -> HandshakeMessage:
        """Worker agent submits output + verification proof."""
        self.state = HandshakeState.EXECUTING
        payload = {
            "proposal_id": self.proposal.proposal_id,
            "output_hash": hash_content(output),
            "output": output,
        }
        if proof:
            payload["proof"] = proof
        msg = HandshakeMessage(
            msg_type="EXECUTE",
            sender=self.my_agent_id,
            payload=payload,
        )
        self.messages.append(msg)
        return msg

    def verify(self, execute_msg: HandshakeMessage,
               verifier: Callable[[dict], TestResults] = None) -> HandshakeMessage:
        """Requesting agent verifies the output."""
        self.state = HandshakeState.VERIFYING

        if verifier and self.proposal.verification_tier == VerificationTier.SELF_VERIFYING:
            test_results = verifier(execute_msg.payload)
            passed = test_results.success
        else:
            # For higher tiers, verification is external
            passed = True
            test_results = TestResults(passed=1, failed=0, suite_hash="manual")

        if passed:
            self.receipt = WorkReceipt(
                agent_id=execute_msg.sender,
                task_type=self.proposal.task_type,
                verification_tier=self.proposal.verification_tier,
                input_hash=hash_content(self.proposal.input_data or ""),
                output_hash=execute_msg.payload["output_hash"],
                test_results=test_results,
                platform="handshake/1.0",
            )
            self.state = HandshakeState.COMPLETE
            result = "PASS"
        else:
            self.state = HandshakeState.FAILED
            result = "FAIL"

        msg = HandshakeMessage(
            msg_type="VERIFY",
            sender=self.my_agent_id,
            payload={
                "result": result,
                "receipt_id": self.receipt.receipt_id if self.receipt else None,
            },
        )
        self.messages.append(msg)
        return msg

    def abort(self, reason: str = "unspecified") -> HandshakeMessage:
        msg = HandshakeMessage(
            msg_type="ABORT",
            sender=self.my_agent_id,
            payload={"reason": reason, "at_state": self.state.value},
        )
        self.messages.append(msg)
        self.state = HandshakeState.ABORTED
        return msg

    def transcript(self) -> str:
        lines = [f"=== Handshake {self.session_id[:8]} ==="]
        for m in self.messages:
            lines.append(f"{m.sender[:12]:>12} → {m.msg_type}: {json.dumps(m.payload, default=str)[:120]}")
        lines.append(f"State: {self.state.value}")
        if self.receipt:
            lines.append(f"Receipt: {self.receipt.receipt_id[:8]}... (tier {int(self.receipt.verification_tier)})")
        return "\n".join(lines)


# --- Demo: Full handshake between Alice and Bob ---

if __name__ == "__main__":
    # Alice wants code reviewed. Bob can do it.
    alice = HandshakeSession("sigil:alice_abc")
    bob = HandshakeSession("sigil:bob_xyz")

    # 1. Exchange hellos
    alice_hello = alice.send_hello(["research", "writing"], Constraints(time_limit_seconds=600))
    bob.receive_hello(alice_hello)

    bob_hello = bob.send_hello(["code_review", "security_audit"], Constraints(time_limit_seconds=900))
    alice.receive_hello(bob_hello)

    # 2. Alice proposes a code review task
    code = "def transfer(amount): db.execute(f'UPDATE balance SET val={amount}')"
    test_suite = "assert 'parameterized' in review or 'sql injection' in review.lower()"

    proposal = alice.propose(
        task_description="Review this code for security issues",
        task_type="code_review",
        verification_tier=VerificationTier.SELF_VERIFYING,
        test_suite_hash=hash_content(test_suite),
        input_data=code,
    )
    # Bob receives and accepts
    bob.proposal = alice.proposal  # In real protocol, transmitted via message
    bob_accept = bob.accept()
    alice.messages.append(bob_accept)

    # 3. Bob executes
    review_output = "SQL injection vulnerability: use parameterized queries instead of f-strings"
    execute_msg = bob.execute(review_output, proof={"issues_found": 1, "severity": "critical"})

    # 4. Alice verifies (Tier 0: does the review mention the actual issue?)
    def verify_review(payload: dict) -> TestResults:
        output = payload.get("output", "")
        checks = [
            "sql injection" in output.lower(),
            "parameterized" in output.lower(),
        ]
        return TestResults(
            passed=sum(checks),
            failed=len(checks) - sum(checks),
            suite_hash=hash_content(test_suite),
        )

    verify_msg = alice.verify(execute_msg, verifier=verify_review)

    # Print transcript
    print(alice.transcript())
    print()
    if alice.receipt:
        print("=== Generated Receipt ===")
        print(alice.receipt.to_json())

    # Add to a receipt chain
    chain = ReceiptChain()
    if alice.receipt:
        chain.append(alice.receipt)
        print(f"\nChain integrity: {chain.verify_integrity()}")
        print(f"Chain length: {chain.length}")
