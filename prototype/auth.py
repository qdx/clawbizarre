"""
ClawBizarre Auth — Phase 7
Ed25519 challenge-response authentication for the API.

Flow:
1. Agent POST /auth/challenge with agent_id → gets challenge (random bytes)
2. Agent signs challenge with Ed25519 private key
3. Agent POST /auth/verify with challenge_id + signature + pubkey → gets bearer token
4. Agent uses "Authorization: Bearer <token>" on subsequent requests
5. Server validates token on protected endpoints

Public endpoints (no auth required):
- POST /auth/challenge
- POST /auth/verify
- GET /discovery/stats
- POST /discovery/search (read-only)

Protected endpoints (bearer token required, agent_id must match):
- POST /discovery/register (must be own agent_id)
- POST /receipt/create (must be own agent_id)
- POST /receipt/chain/append (must be own agent_id)
- POST /handshake/* (must be participant)
- POST /treasury/* (must be fleet owner)
"""

from identity import AgentIdentity
from persistence import PersistenceLayer
from typing import Optional, Callable
import json


# Endpoints that don't require auth
PUBLIC_ENDPOINTS = {
    ("POST", "/auth/challenge"),
    ("POST", "/auth/verify"),
    ("GET", "/discovery/stats"),
    ("POST", "/discovery/search"),
    ("GET", "/receipt/chain/"),  # prefix match for reading chains
    ("GET", "/reputation/"),     # prefix match for reading reputation
}


def is_public(method: str, path: str) -> bool:
    """Check if endpoint is public (no auth required)."""
    for m, p in PUBLIC_ENDPOINTS:
        if method == m and (path == p or (p.endswith("/") and path.startswith(p))):
            return True
    return False


def extract_bearer_token(authorization_header: Optional[str]) -> Optional[str]:
    """Extract token from 'Bearer <token>' header."""
    if not authorization_header:
        return None
    parts = authorization_header.split(" ", 1)
    if len(parts) != 2 or parts[0] != "Bearer":
        return None
    return parts[1]


def ed25519_verify(pubkey_hex: str, challenge_hex: str, signature_hex: str) -> bool:
    """Verify an Ed25519 signature on a challenge hex string."""
    try:
        ident = AgentIdentity.from_public_key_hex(pubkey_hex)
        return ident.verify(challenge_hex, signature_hex)
    except Exception:
        return False


class AuthMiddleware:
    """Wraps persistence layer for auth operations."""

    def __init__(self, db: PersistenceLayer):
        self.db = db

    def create_challenge(self, agent_id: str) -> dict:
        return self.db.create_challenge(agent_id)

    def verify_and_issue_token(self, challenge_id: str, agent_id: str,
                                signature_hex: str, pubkey_hex: str) -> Optional[str]:
        return self.db.verify_challenge(
            challenge_id, agent_id, signature_hex, pubkey_hex,
            ed25519_verify
        )

    def authenticate(self, authorization_header: Optional[str]) -> Optional[str]:
        """Authenticate a request. Returns agent_id or None."""
        token = extract_bearer_token(authorization_header)
        if not token:
            return None
        return self.db.validate_token(token)

    def revoke(self, token: str):
        self.db.revoke_token(token)


if __name__ == "__main__":
    import tempfile, os

    with tempfile.TemporaryDirectory() as tmp:
        db = PersistenceLayer(os.path.join(tmp, "test.db"))
        auth = AuthMiddleware(db)

        # Create identity
        ident = AgentIdentity.generate()
        agent_id = f"ed25519:{ident.public_key_hex}"

        # Challenge flow
        challenge = auth.create_challenge(agent_id)
        print(f"✓ Challenge: {challenge['challenge'][:16]}...")

        # Sign challenge
        # sign() expects string, returns hex string
        signature_hex = ident.sign(challenge["challenge"])
        sig_hex = signature_hex

        # Verify and get token
        token = auth.verify_and_issue_token(
            challenge["challenge_id"], agent_id, sig_hex, ident.public_key_hex
        )
        assert token is not None
        print(f"✓ Token issued: {token[:16]}...")

        # Authenticate with token
        authenticated = auth.authenticate(f"Bearer {token}")
        assert authenticated == agent_id
        print(f"✓ Auth succeeded: {authenticated[:30]}...")

        # Bad token
        bad = auth.authenticate("Bearer fake-token")
        assert bad is None
        print(f"✓ Bad token rejected")

        # Revoke
        auth.revoke(token)
        revoked = auth.authenticate(f"Bearer {token}")
        assert revoked is None
        print(f"✓ Revoked token rejected")

        # Public endpoint check
        assert is_public("POST", "/auth/challenge")
        assert is_public("GET", "/discovery/stats")
        assert is_public("GET", "/receipt/chain/ed25519:abc")
        assert not is_public("POST", "/discovery/register")
        assert not is_public("POST", "/receipt/create")
        print(f"✓ Public/protected endpoint classification")

        db.close()
        print(f"\n=== All auth tests passed ===")
