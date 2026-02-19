"""
ClawBizarre ERC-8004 Identity Adapter

Bridges ClawBizarre's Ed25519 identity system with the ERC-8004 on-chain
agent identity standard (live on Ethereum mainnet since Jan 29, 2026).

ERC-8004 provides:
  - Identity Registry: NFT-minted agent IDs with agent cards
  - Reputation Registry: On-chain structured feedback
  - Validation Registry: Third-party verification records

This adapter allows ClawBizarre agents to:
  1. Register with an existing ERC-8004 token ID
  2. Map between ed25519 agent_ids and ERC-8004 token IDs
  3. Push/pull reputation data between ClawBizarre receipts and on-chain feedback
  4. Generate agent cards from ClawBizarre service listings

Design: Off-chain first, on-chain as persistence/portability layer.
ClawBizarre handles the fast path (matching, handshake, receipts).
ERC-8004 handles the durable path (identity, cross-platform reputation).
"""

import json
import hashlib
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
from enum import Enum


class IdentitySource(Enum):
    """Where the agent's identity originates."""
    NATIVE = "native"       # ClawBizarre Ed25519 keypair
    ERC8004 = "erc8004"     # Ethereum ERC-8004 NFT
    HYBRID = "hybrid"       # Both (linked)


@dataclass
class AgentCard:
    """
    ERC-8004 Agent Card — standardized capability/contact/payment info.
    
    Based on the ERC-8004 spec: every registered agent has a card that
    describes what it does, how to reach it, and where it receives payments.
    """
    # Identity
    token_id: Optional[int] = None          # ERC-8004 NFT token ID
    native_id: Optional[str] = None         # ClawBizarre ed25519:... ID
    name: Optional[str] = None
    description: Optional[str] = None
    
    # Capabilities
    capabilities: List[str] = field(default_factory=list)
    verification_tiers: List[int] = field(default_factory=lambda: [0])  # Which tiers this agent can serve
    
    # Contact
    api_endpoint: Optional[str] = None      # Where to send work requests
    notification_endpoint: Optional[str] = None  # SSE endpoint for events
    
    # Payment
    payment_protocols: List[str] = field(default_factory=lambda: ["x402"])
    wallet_address: Optional[str] = None    # For x402/crypto payments
    
    # Metadata
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    
    def to_json(self) -> str:
        d = {k: v for k, v in asdict(self).items() if v is not None}
        return json.dumps(d, indent=2)
    
    @classmethod
    def from_json(cls, data: str) -> "AgentCard":
        d = json.loads(data)
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
    
    def content_hash(self) -> str:
        canonical = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return f"sha256:{hashlib.sha256(canonical.encode()).hexdigest()}"


@dataclass
class ERC8004Identity:
    """
    Bridged identity: links a ClawBizarre native ID to an ERC-8004 token.
    
    The bridge is established by:
    1. Agent signs their ERC-8004 token_id with their Ed25519 key
    2. Agent signs their Ed25519 pubkey with their Ethereum key (on-chain)
    3. Both signatures prove the same entity controls both identities
    """
    native_id: str                          # ed25519:... from ClawBizarre
    token_id: Optional[int] = None          # ERC-8004 NFT token ID  
    eth_address: Optional[str] = None       # Ethereum address that minted the NFT
    source: IdentitySource = IdentitySource.NATIVE
    
    # Linking proof
    native_signature: Optional[str] = None  # Ed25519 sig of f"link:{token_id}"
    chain_tx: Optional[str] = None          # On-chain tx that links back to native_id
    linked_at: Optional[str] = None
    
    # Agent card
    card: Optional[AgentCard] = None
    
    def is_linked(self) -> bool:
        """True if both native and on-chain identities are linked."""
        return (self.token_id is not None and 
                self.native_signature is not None and
                self.source in (IdentitySource.ERC8004, IdentitySource.HYBRID))
    
    def link_request(self) -> Dict:
        """Generate the data needed to create an on-chain link."""
        return {
            "action": "link",
            "native_id": self.native_id,
            "token_id": self.token_id,
            "native_signature": self.native_signature,
            "message": f"link:{self.token_id}",
        }


@dataclass
class OnChainFeedback:
    """
    Maps a ClawBizarre work receipt to ERC-8004 on-chain feedback format.
    
    ERC-8004 feedback: performance metrics, success rates, response quality,
    all tagged and timestamped.
    
    We map from our richer receipt format to their simpler feedback format.
    """
    token_id: int                           # Agent being rated
    rater_token_id: int                     # Agent giving feedback
    receipt_id: str                         # ClawBizarre receipt that generated this
    
    # Mapped from receipt
    task_type: str
    success: bool                           # passed > 0 and failed == 0
    verification_tier: int
    
    # Metrics (subset of what our receipts capture)
    metrics: Dict[str, float] = field(default_factory=dict)
    # e.g. {"pass_rate": 1.0, "response_time_ms": 1200}
    
    timestamp: Optional[str] = None
    
    @classmethod
    def from_receipt(cls, receipt_data: Dict, provider_token_id: int, 
                     buyer_token_id: int) -> "OnChainFeedback":
        """Convert a ClawBizarre work receipt to on-chain feedback format."""
        test_results = receipt_data.get("test_results", {})
        passed = test_results.get("passed", 0)
        failed = test_results.get("failed", 0)
        
        metrics = {}
        if passed + failed > 0:
            metrics["pass_rate"] = passed / (passed + failed)
        
        return cls(
            token_id=provider_token_id,
            rater_token_id=buyer_token_id,
            receipt_id=receipt_data.get("receipt_id", ""),
            task_type=receipt_data.get("task_type", "unknown"),
            success=failed == 0 and passed > 0,
            verification_tier=receipt_data.get("verification_tier", 0),
            metrics=metrics,
            timestamp=receipt_data.get("timestamp"),
        )
    
    def to_chain_format(self) -> Dict:
        """Format for submitting to ERC-8004 Reputation Registry."""
        return {
            "agentTokenId": self.token_id,
            "raterTokenId": self.rater_token_id,
            "tags": [self.task_type, f"tier{self.verification_tier}"],
            "metrics": self.metrics,
            "success": self.success,
            "metadata": {
                "source": "clawbizarre",
                "receipt_id": self.receipt_id,
            },
        }


class IdentityBridge:
    """
    Manages the mapping between ClawBizarre native IDs and ERC-8004 token IDs.
    
    In-memory for now. Would be backed by persistence.py in production.
    """
    
    def __init__(self):
        self._by_native: Dict[str, ERC8004Identity] = {}
        self._by_token: Dict[int, ERC8004Identity] = {}
    
    def register_native(self, native_id: str) -> ERC8004Identity:
        """Register a native-only identity (no ERC-8004 link yet)."""
        if native_id in self._by_native:
            return self._by_native[native_id]
        eid = ERC8004Identity(native_id=native_id, source=IdentitySource.NATIVE)
        self._by_native[native_id] = eid
        return eid
    
    def register_erc8004(self, native_id: str, token_id: int, 
                          eth_address: str, native_signature: str) -> ERC8004Identity:
        """Register a linked identity (ClawBizarre + ERC-8004)."""
        eid = ERC8004Identity(
            native_id=native_id,
            token_id=token_id,
            eth_address=eth_address,
            source=IdentitySource.HYBRID,
            native_signature=native_signature,
            linked_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        self._by_native[native_id] = eid
        self._by_token[token_id] = eid
        return eid
    
    def get_by_native(self, native_id: str) -> Optional[ERC8004Identity]:
        return self._by_native.get(native_id)
    
    def get_by_token(self, token_id: int) -> Optional[ERC8004Identity]:
        return self._by_token.get(token_id)
    
    def resolve(self, identifier: str) -> Optional[ERC8004Identity]:
        """Resolve any identifier format to an identity."""
        if identifier.startswith("ed25519:"):
            return self.get_by_native(identifier)
        try:
            return self.get_by_token(int(identifier))
        except (ValueError, TypeError):
            return None
    
    def linked_count(self) -> int:
        return sum(1 for e in self._by_native.values() if e.is_linked())
    
    def stats(self) -> Dict:
        return {
            "total": len(self._by_native),
            "native_only": sum(1 for e in self._by_native.values() 
                             if e.source == IdentitySource.NATIVE),
            "linked": self.linked_count(),
        }


def generate_agent_card_from_listing(native_id: str, listing: Dict, 
                                      api_base: str) -> AgentCard:
    """Convert a ClawBizarre service listing to an ERC-8004 agent card."""
    return AgentCard(
        native_id=native_id,
        name=listing.get("name"),
        description=listing.get("description"),
        capabilities=listing.get("capabilities", []),
        api_endpoint=f"{api_base}/handshake/initiate",
        notification_endpoint=f"{api_base}/events",
        payment_protocols=listing.get("payment_protocols", ["x402"]),
        wallet_address=listing.get("wallet_address"),
        created_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    )


# --- Tests ---

def run_tests():
    passed = 0
    failed = 0
    
    def check(name, condition):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  ✓ {name}")
        else:
            failed += 1
            print(f"  ✗ {name}")
    
    print("\n=== ERC-8004 Adapter Tests ===\n")
    
    # 1. Native-only registration
    bridge = IdentityBridge()
    eid = bridge.register_native("ed25519:abc123")
    check("1. Native registration", eid.source == IdentitySource.NATIVE)
    check("2. Not linked", not eid.is_linked())
    
    # 3. ERC-8004 linked registration
    eid2 = bridge.register_erc8004(
        native_id="ed25519:def456",
        token_id=42,
        eth_address="0xabcdef1234567890",
        native_signature="sig_placeholder",
    )
    check("3. Linked registration", eid2.is_linked())
    check("4. Source is hybrid", eid2.source == IdentitySource.HYBRID)
    
    # 5. Resolve by native ID
    resolved = bridge.resolve("ed25519:def456")
    check("5. Resolve by native ID", resolved is not None and resolved.token_id == 42)
    
    # 6. Resolve by token ID
    resolved = bridge.resolve("42")
    check("6. Resolve by token ID", resolved is not None and resolved.native_id == "ed25519:def456")
    
    # 7. Stats
    stats = bridge.stats()
    check("7. Stats correct", stats["total"] == 2 and stats["linked"] == 1 and stats["native_only"] == 1)
    
    # 8. Agent card generation
    card = generate_agent_card_from_listing(
        "ed25519:abc123",
        {"capabilities": ["code_review", "translation"], "name": "ReviewBot"},
        "https://api.clawbizarre.com"
    )
    check("8. Agent card generated", card.name == "ReviewBot" and len(card.capabilities) == 2)
    
    # 9. Agent card round-trip
    card_json = card.to_json()
    card2 = AgentCard.from_json(card_json)
    check("9. Agent card round-trip", card2.name == card.name and card2.capabilities == card.capabilities)
    
    # 10. Agent card content hash
    h1 = card.content_hash()
    check("10. Card has content hash", h1.startswith("sha256:"))
    
    # 11. On-chain feedback from receipt
    receipt_data = {
        "receipt_id": "rcpt_001",
        "task_type": "code_review",
        "verification_tier": 0,
        "test_results": {"passed": 5, "failed": 0},
        "timestamp": "2026-02-19T04:00:00Z",
    }
    feedback = OnChainFeedback.from_receipt(receipt_data, provider_token_id=42, buyer_token_id=99)
    check("11. Feedback from receipt", feedback.success and feedback.metrics["pass_rate"] == 1.0)
    
    # 12. Chain format
    chain_fmt = feedback.to_chain_format()
    check("12. Chain format correct", 
          chain_fmt["agentTokenId"] == 42 and 
          "clawbizarre" in str(chain_fmt["metadata"]))
    
    # 13. Failed receipt → unsuccessful feedback
    bad_receipt = {
        "receipt_id": "rcpt_002",
        "task_type": "translation",
        "verification_tier": 1,
        "test_results": {"passed": 2, "failed": 3},
    }
    bad_feedback = OnChainFeedback.from_receipt(bad_receipt, 42, 99)
    check("13. Failed receipt → unsuccessful", not bad_feedback.success and bad_feedback.metrics["pass_rate"] == 0.4)
    
    # 14. Link request generation
    link_req = eid2.link_request()
    check("14. Link request format", 
          link_req["action"] == "link" and 
          link_req["token_id"] == 42 and
          link_req["message"] == "link:42")
    
    # Summary
    print(f"\n{'='*50}")
    print(f"  {passed} passed, {failed} failed, {passed+failed} total")
    print(f"{'='*50}\n")
    return failed == 0


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
