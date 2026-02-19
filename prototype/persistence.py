"""
ClawBizarre Persistence Layer — Phase 7
SQLite-backed storage for receipts, discovery, reputation, and auth tokens.

Design principles:
- Append-only receipt storage (immutable once written)
- Indexed by agent_id, task_type, timestamp
- JSON columns for complex nested structures (pragmatic for prototype)
- WAL mode for concurrent reads
"""

import sqlite3
import json
import hashlib
import os
import time
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from receipt import (
    WorkReceipt, ReceiptChain, TestResults, RiskEnvelope,
    Timing, Attestation, VerificationTier, PricingStrategy
)
from discovery import CapabilityAd, AvailabilityStatus


DB_VERSION = 1


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class PersistenceLayer:
    """SQLite storage for ClawBizarre state."""

    def __init__(self, db_path: str = "clawbizarre.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._init_schema()

    def _init_schema(self):
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS schema_version (
                version INTEGER PRIMARY KEY
            );

            -- Auth: challenge-response tokens
            CREATE TABLE IF NOT EXISTS auth_tokens (
                token TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                pubkey_hex TEXT NOT NULL,
                created_at TEXT NOT NULL,
                expires_at TEXT NOT NULL,
                revoked INTEGER DEFAULT 0
            );
            CREATE INDEX IF NOT EXISTS idx_auth_agent ON auth_tokens(agent_id);

            -- Auth challenges (short-lived, for login flow)
            CREATE TABLE IF NOT EXISTS auth_challenges (
                challenge_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                challenge_bytes TEXT NOT NULL,  -- hex-encoded
                created_at TEXT NOT NULL,
                used INTEGER DEFAULT 0
            );

            -- Receipts (append-only)
            CREATE TABLE IF NOT EXISTS receipts (
                receipt_id TEXT PRIMARY KEY,
                agent_id TEXT NOT NULL,
                task_type TEXT NOT NULL,
                verification_tier INTEGER NOT NULL,
                pricing_strategy TEXT NOT NULL,
                input_hash TEXT NOT NULL,
                output_hash TEXT NOT NULL,
                content_hash TEXT NOT NULL,
                platform TEXT DEFAULT 'direct',
                environment_hash TEXT,
                timestamp TEXT NOT NULL,
                test_passed INTEGER,
                test_failed INTEGER,
                test_suite_hash TEXT,
                timing_json TEXT,        -- JSON blob
                risk_envelope_json TEXT, -- JSON blob
                attestations_json TEXT,  -- JSON blob
                receipt_json TEXT NOT NULL,  -- full receipt for reconstruction
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_receipts_agent ON receipts(agent_id);
            CREATE INDEX IF NOT EXISTS idx_receipts_type ON receipts(task_type);
            CREATE INDEX IF NOT EXISTS idx_receipts_time ON receipts(timestamp);
            CREATE INDEX IF NOT EXISTS idx_receipts_tier ON receipts(verification_tier);

            -- Receipt chain links (ordering + integrity)
            CREATE TABLE IF NOT EXISTS chain_links (
                agent_id TEXT NOT NULL,
                position INTEGER NOT NULL,
                receipt_id TEXT NOT NULL REFERENCES receipts(receipt_id),
                chain_hash TEXT NOT NULL,
                previous_hash TEXT,
                PRIMARY KEY (agent_id, position)
            );

            -- Discovery: capability advertisements
            CREATE TABLE IF NOT EXISTS capabilities (
                agent_id TEXT PRIMARY KEY,
                pubkey_hex TEXT,
                capabilities_json TEXT NOT NULL,  -- JSON array
                verification_tier INTEGER NOT NULL,
                pricing_strategy TEXT,
                availability TEXT DEFAULT 'available',
                receipt_chain_length INTEGER DEFAULT 0,
                success_rate REAL DEFAULT 0.0,
                last_heartbeat TEXT,
                registered_at TEXT NOT NULL,
                metadata_json TEXT  -- extra fields
            );

            -- Reputation snapshots (cached, recomputable)
            CREATE TABLE IF NOT EXISTS reputation_snapshots (
                agent_id TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                computed_at TEXT NOT NULL
            );

            -- Settlements (payment tracking linked to receipts)
            CREATE TABLE IF NOT EXISTS settlements (
                receipt_id TEXT PRIMARY KEY,
                registered_by TEXT NOT NULL,
                protocol TEXT NOT NULL DEFAULT 'x402',
                payment_id TEXT DEFAULT '',
                amount REAL DEFAULT 0,
                currency TEXT DEFAULT 'USDC',
                chain TEXT DEFAULT 'base',
                status TEXT NOT NULL DEFAULT 'pending',
                registered_at REAL NOT NULL,
                confirmed_at REAL,
                confirmed_by TEXT
            );
            CREATE INDEX IF NOT EXISTS idx_settlements_status ON settlements(status);
            CREATE INDEX IF NOT EXISTS idx_settlements_agent ON settlements(registered_by);

            -- ERC-8004 Identity Bridge
            CREATE TABLE IF NOT EXISTS identity_bridge (
                native_id TEXT PRIMARY KEY,
                token_id INTEGER UNIQUE,
                eth_address TEXT,
                source TEXT NOT NULL DEFAULT 'native',
                native_signature TEXT,
                chain_tx TEXT,
                linked_at TEXT,
                card_json TEXT,
                created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_identity_token ON identity_bridge(token_id);

            -- Treasury audit log
            CREATE TABLE IF NOT EXISTS treasury_log (
                log_id TEXT PRIMARY KEY,
                request_json TEXT NOT NULL,
                decision TEXT NOT NULL,
                reason TEXT,
                timestamp TEXT NOT NULL
            );
        """)

        # Set version
        existing = self.conn.execute("SELECT version FROM schema_version").fetchone()
        if not existing:
            self.conn.execute("INSERT INTO schema_version VALUES (?)", (DB_VERSION,))
        self.conn.commit()

    def close(self):
        self.conn.close()

    # --- Auth ---

    def create_challenge(self, agent_id: str) -> dict:
        """Create a challenge for Ed25519 authentication."""
        challenge_id = str(uuid.uuid4())
        challenge_bytes = os.urandom(32).hex()
        now = _now_iso()
        self.conn.execute(
            "INSERT INTO auth_challenges (challenge_id, agent_id, challenge_bytes, created_at) VALUES (?, ?, ?, ?)",
            (challenge_id, agent_id, challenge_bytes, now)
        )
        self.conn.commit()
        return {"challenge_id": challenge_id, "challenge": challenge_bytes}

    def verify_challenge(self, challenge_id: str, agent_id: str, signature_hex: str,
                         pubkey_hex: str, verify_fn) -> Optional[str]:
        """Verify a signed challenge and issue a bearer token.

        verify_fn(pubkey_hex, message_bytes, signature_hex) -> bool
        Returns token string or None on failure.
        """
        row = self.conn.execute(
            "SELECT * FROM auth_challenges WHERE challenge_id = ? AND agent_id = ? AND used = 0",
            (challenge_id, agent_id)
        ).fetchone()
        if not row:
            return None

        # Check age (challenges valid for 5 minutes)
        created = datetime.fromisoformat(row["created_at"])
        age = (datetime.now(timezone.utc) - created).total_seconds()
        if age > 300:
            self.conn.execute("DELETE FROM auth_challenges WHERE challenge_id = ?", (challenge_id,))
            self.conn.commit()
            return None

        # Verify signature (pass hex string as message)
        if not verify_fn(pubkey_hex, row["challenge_bytes"], signature_hex):
            return None

        # Mark used
        self.conn.execute("UPDATE auth_challenges SET used = 1 WHERE challenge_id = ?", (challenge_id,))

        # Issue token (valid 24h)
        token = str(uuid.uuid4())
        now = _now_iso()
        expires = datetime.now(timezone.utc).replace(hour=23, minute=59, second=59).isoformat()
        self.conn.execute(
            "INSERT INTO auth_tokens (token, agent_id, pubkey_hex, created_at, expires_at) VALUES (?, ?, ?, ?, ?)",
            (token, agent_id, pubkey_hex, now, expires)
        )
        self.conn.commit()
        return token

    def validate_token(self, token: str) -> Optional[str]:
        """Validate a bearer token. Returns agent_id or None."""
        row = self.conn.execute(
            "SELECT agent_id, expires_at, revoked FROM auth_tokens WHERE token = ?",
            (token,)
        ).fetchone()
        if not row or row["revoked"]:
            return None
        if datetime.fromisoformat(row["expires_at"]) < datetime.now(timezone.utc):
            return None
        return row["agent_id"]

    def revoke_token(self, token: str):
        self.conn.execute("UPDATE auth_tokens SET revoked = 1 WHERE token = ?", (token,))
        self.conn.commit()

    # --- Receipts ---

    def store_receipt(self, receipt: WorkReceipt) -> str:
        """Store a receipt. Returns receipt_id."""
        now = _now_iso()
        tr = receipt.test_results
        self.conn.execute(
            """INSERT OR IGNORE INTO receipts
            (receipt_id, agent_id, task_type, verification_tier, pricing_strategy,
             input_hash, output_hash, content_hash, platform, environment_hash,
             timestamp, test_passed, test_failed, test_suite_hash,
             timing_json, risk_envelope_json, attestations_json, receipt_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                receipt.receipt_id, receipt.agent_id, receipt.task_type,
                int(receipt.verification_tier), receipt.pricing_strategy,
                receipt.input_hash, receipt.output_hash, receipt.content_hash,
                receipt.platform, receipt.environment_hash, receipt.timestamp,
                tr.passed if tr else None, tr.failed if tr else None,
                tr.suite_hash if tr else None,
                json.dumps(asdict(receipt.timing)) if receipt.timing else None,
                json.dumps(asdict(receipt.risk_envelope)) if receipt.risk_envelope else None,
                json.dumps([asdict(a) for a in receipt.attestations]) if receipt.attestations else "[]",
                receipt.to_json(), now
            )
        )
        self.conn.commit()
        return receipt.receipt_id

    def get_receipt(self, receipt_id: str) -> Optional[WorkReceipt]:
        row = self.conn.execute("SELECT receipt_json FROM receipts WHERE receipt_id = ?", (receipt_id,)).fetchone()
        if not row:
            return None
        return WorkReceipt.from_json(row["receipt_json"])

    def get_agent_receipts(self, agent_id: str, limit: int = 100, offset: int = 0) -> list[WorkReceipt]:
        rows = self.conn.execute(
            "SELECT receipt_json FROM receipts WHERE agent_id = ? ORDER BY timestamp ASC LIMIT ? OFFSET ?",
            (agent_id, limit, offset)
        ).fetchall()
        return [WorkReceipt.from_json(r["receipt_json"]) for r in rows]

    def count_agent_receipts(self, agent_id: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) as cnt FROM receipts WHERE agent_id = ?", (agent_id,)).fetchone()
        return row["cnt"]

    def store_chain_link(self, agent_id: str, position: int, receipt_id: str,
                         chain_hash: str, previous_hash: Optional[str]):
        self.conn.execute(
            "INSERT OR REPLACE INTO chain_links (agent_id, position, receipt_id, chain_hash, previous_hash) VALUES (?, ?, ?, ?, ?)",
            (agent_id, position, receipt_id, chain_hash, previous_hash)
        )
        self.conn.commit()

    def get_chain_links(self, agent_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM chain_links WHERE agent_id = ? ORDER BY position ASC",
            (agent_id,)
        ).fetchall()
        return [dict(r) for r in rows]

    def rebuild_chain(self, agent_id: str) -> ReceiptChain:
        """Reconstruct a ReceiptChain from stored data."""
        links = self.get_chain_links(agent_id)
        chain = ReceiptChain()
        for link in links:
            receipt = self.get_receipt(link["receipt_id"])
            if receipt:
                chain.receipts.append(receipt)
                chain.chain_hashes.append(link["chain_hash"])
        return chain

    def append_to_chain(self, agent_id: str, receipt: WorkReceipt) -> str:
        """Store receipt and append to agent's chain. Returns chain_hash."""
        self.store_receipt(receipt)
        links = self.get_chain_links(agent_id)
        position = len(links)
        prev_hash = links[-1]["chain_hash"] if links else "genesis"
        entry = f"{prev_hash}:{receipt.content_hash}"
        chain_hash = f"sha256:{hashlib.sha256(entry.encode()).hexdigest()}"
        self.store_chain_link(agent_id, position, receipt.receipt_id, chain_hash, prev_hash)
        return chain_hash

    # --- Discovery ---

    def register_capability(self, agent_id: str, capabilities: list[str],
                            verification_tier: int = 0, pricing_strategy: str = "reputation_premium",
                            pubkey_hex: str = None, metadata: dict = None):
        now = _now_iso()
        receipt_count = self.count_agent_receipts(agent_id)
        self.conn.execute(
            """INSERT OR REPLACE INTO capabilities
            (agent_id, pubkey_hex, capabilities_json, verification_tier, pricing_strategy,
             availability, receipt_chain_length, success_rate, last_heartbeat, registered_at, metadata_json)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (agent_id, pubkey_hex, json.dumps(capabilities), verification_tier,
             pricing_strategy, "available", receipt_count, 0.0, now, now,
             json.dumps(metadata) if metadata else None)
        )
        self.conn.commit()

    def search_capabilities(self, task_type: str = None, min_receipts: int = 0,
                           limit: int = 20) -> list[dict]:
        query = "SELECT * FROM capabilities WHERE availability = 'available'"
        params = []
        if task_type:
            query += " AND capabilities_json LIKE ?"
            params.append(f'%"{task_type}"%')
        if min_receipts > 0:
            query += " AND receipt_chain_length >= ?"
            params.append(min_receipts)
        query += " ORDER BY receipt_chain_length DESC LIMIT ?"
        params.append(limit)
        rows = self.conn.execute(query, params).fetchall()
        return [dict(r) for r in rows]

    def heartbeat(self, agent_id: str):
        self.conn.execute(
            "UPDATE capabilities SET last_heartbeat = ? WHERE agent_id = ?",
            (_now_iso(), agent_id)
        )
        self.conn.commit()

    def deregister(self, agent_id: str):
        self.conn.execute("DELETE FROM capabilities WHERE agent_id = ?", (agent_id,))
        self.conn.commit()

    # --- Reputation ---

    def store_reputation(self, agent_id: str, snapshot_json: str):
        now = _now_iso()
        self.conn.execute(
            "INSERT OR REPLACE INTO reputation_snapshots (agent_id, snapshot_json, computed_at) VALUES (?, ?, ?)",
            (agent_id, snapshot_json, now)
        )
        self.conn.commit()

    def get_reputation(self, agent_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM reputation_snapshots WHERE agent_id = ?", (agent_id,)
        ).fetchone()
        if not row:
            return None
        return {"agent_id": agent_id, "snapshot": json.loads(row["snapshot_json"]), "computed_at": row["computed_at"]}

    # --- Treasury ---

    def log_treasury_decision(self, request: dict, decision: str, reason: str = None):
        self.conn.execute(
            "INSERT INTO treasury_log (log_id, request_json, decision, reason, timestamp) VALUES (?, ?, ?, ?, ?)",
            (str(uuid.uuid4()), json.dumps(request), decision, reason, _now_iso())
        )
        self.conn.commit()

    def get_treasury_log(self, limit: int = 50) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM treasury_log ORDER BY timestamp DESC LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Settlements ---

    def register_settlement(self, receipt_id: str, registered_by: str,
                           protocol: str = "x402", payment_id: str = "",
                           amount: float = 0, currency: str = "USDC",
                           chain: str = "base") -> dict:
        """Register a payment intent for a receipt."""
        now = time.time()
        record = {
            "receipt_id": receipt_id,
            "registered_by": registered_by,
            "protocol": protocol,
            "payment_id": payment_id,
            "amount": amount,
            "currency": currency,
            "chain": chain,
            "status": "pending",
            "registered_at": now,
            "confirmed_at": None,
            "confirmed_by": None,
        }
        self.conn.execute(
            """INSERT OR REPLACE INTO settlements
            (receipt_id, registered_by, protocol, payment_id, amount, currency, chain,
             status, registered_at, confirmed_at, confirmed_by)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (receipt_id, registered_by, protocol, payment_id, amount, currency, chain,
             "pending", now, None, None)
        )
        self.conn.commit()
        return record

    def confirm_settlement(self, receipt_id: str, confirmed_by: str) -> Optional[dict]:
        """Confirm a pending settlement. Returns updated record or None."""
        row = self.conn.execute(
            "SELECT * FROM settlements WHERE receipt_id = ?", (receipt_id,)
        ).fetchone()
        if not row or row["status"] != "pending":
            return None
        now = time.time()
        self.conn.execute(
            "UPDATE settlements SET status = 'confirmed', confirmed_at = ?, confirmed_by = ? WHERE receipt_id = ?",
            (now, confirmed_by, receipt_id)
        )
        self.conn.commit()
        updated = self.conn.execute("SELECT * FROM settlements WHERE receipt_id = ?", (receipt_id,)).fetchone()
        return dict(updated)

    def get_settlement(self, receipt_id: str) -> Optional[dict]:
        """Get settlement status for a receipt."""
        row = self.conn.execute("SELECT * FROM settlements WHERE receipt_id = ?", (receipt_id,)).fetchone()
        return dict(row) if row else None

    def get_agent_settlements(self, agent_id: str, limit: int = 50) -> list[dict]:
        """Get all settlements registered by an agent."""
        rows = self.conn.execute(
            "SELECT * FROM settlements WHERE registered_by = ? ORDER BY registered_at DESC LIMIT ?",
            (agent_id, limit)
        ).fetchall()
        return [dict(r) for r in rows]

    # --- Identity Bridge (ERC-8004) ---

    def store_identity(self, native_id: str, source: str = "native",
                       token_id: int = None, eth_address: str = None,
                       native_signature: str = None, chain_tx: str = None,
                       card_json: str = None) -> dict:
        """Store or update an identity bridge record. Merges with existing."""
        now = _now_iso()
        existing = self.get_identity(native_id)
        if existing:
            # Merge: only overwrite fields that are explicitly provided (non-None)
            new_token = token_id if token_id is not None else existing["token_id"]
            new_eth = eth_address if eth_address is not None else existing["eth_address"]
            new_source = source if source != "native" or not existing["token_id"] else existing["source"]
            new_sig = native_signature if native_signature is not None else existing["native_signature"]
            new_tx = chain_tx if chain_tx is not None else existing["chain_tx"]
            new_card = card_json if card_json is not None else existing["card_json"]
            new_linked = now if token_id is not None and existing["token_id"] is None else existing["linked_at"]
            self.conn.execute(
                """UPDATE identity_bridge SET
                token_id=?, eth_address=?, source=?, native_signature=?,
                chain_tx=?, linked_at=?, card_json=?
                WHERE native_id=?""",
                (new_token, new_eth, new_source, new_sig,
                 new_tx, new_linked, new_card, native_id)
            )
        else:
            linked_at = now if token_id is not None else None
            self.conn.execute(
                """INSERT INTO identity_bridge
                (native_id, token_id, eth_address, source, native_signature,
                 chain_tx, linked_at, card_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (native_id, token_id, eth_address, source, native_signature,
                 chain_tx, linked_at, card_json, now)
            )
        self.conn.commit()
        return self.get_identity(native_id)

    def get_identity(self, native_id: str) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM identity_bridge WHERE native_id = ?", (native_id,)
        ).fetchone()
        return dict(row) if row else None

    def get_identity_by_token(self, token_id: int) -> Optional[dict]:
        row = self.conn.execute(
            "SELECT * FROM identity_bridge WHERE token_id = ?", (token_id,)
        ).fetchone()
        return dict(row) if row else None

    def resolve_identity(self, identifier: str) -> Optional[dict]:
        """Resolve native_id or token_id string to identity."""
        if identifier.startswith("ed25519:"):
            return self.get_identity(identifier)
        try:
            return self.get_identity_by_token(int(identifier))
        except (ValueError, TypeError):
            return None

    def list_identities(self, linked_only: bool = False, limit: int = 50) -> list[dict]:
        if linked_only:
            rows = self.conn.execute(
                "SELECT * FROM identity_bridge WHERE token_id IS NOT NULL LIMIT ?", (limit,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM identity_bridge LIMIT ?", (limit,)
            ).fetchall()
        return [dict(r) for r in rows]

    def identity_stats(self) -> dict:
        total = self.conn.execute("SELECT COUNT(*) FROM identity_bridge").fetchone()[0]
        linked = self.conn.execute("SELECT COUNT(*) FROM identity_bridge WHERE token_id IS NOT NULL").fetchone()[0]
        return {"total": total, "linked": linked, "native_only": total - linked}

    # --- Stats ---

    def stats(self) -> dict:
        receipts = self.conn.execute("SELECT COUNT(*) as cnt FROM receipts").fetchone()["cnt"]
        agents = self.conn.execute("SELECT COUNT(DISTINCT agent_id) FROM receipts").fetchone()[0]
        capabilities = self.conn.execute("SELECT COUNT(*) as cnt FROM capabilities").fetchone()["cnt"]
        tokens = self.conn.execute("SELECT COUNT(*) as cnt FROM auth_tokens WHERE revoked = 0").fetchone()["cnt"]
        settlements = self.conn.execute("SELECT COUNT(*) as cnt FROM settlements").fetchone()["cnt"]
        confirmed = self.conn.execute("SELECT COUNT(*) as cnt FROM settlements WHERE status = 'confirmed'").fetchone()["cnt"]
        id_stats = self.identity_stats()
        return {
            "receipts": receipts,
            "unique_agents": agents,
            "registered_capabilities": capabilities,
            "active_tokens": tokens,
            "settlements": settlements,
            "settlements_confirmed": confirmed,
            "identities": id_stats["total"],
            "identities_linked": id_stats["linked"],
            "db_version": DB_VERSION
        }


# --- Tests ---

if __name__ == "__main__":
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "test.db")
        db = PersistenceLayer(db_path)

        # Test receipt storage
        from receipt import hash_content
        r1 = WorkReceipt(
            agent_id="ed25519:test_agent_1",
            task_type="code_review",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content("input1"),
            output_hash=hash_content("output1"),
            test_results=TestResults(passed=5, failed=0, suite_hash=hash_content("suite1")),
            timing=Timing(
                proposed_at=_now_iso(),
                started_at=_now_iso(),
                completed_at=_now_iso(),
                deadline="2026-12-31T23:59:59+00:00"
            )
        )

        # Store and retrieve
        db.store_receipt(r1)
        retrieved = db.get_receipt(r1.receipt_id)
        assert retrieved is not None
        assert retrieved.receipt_id == r1.receipt_id
        assert retrieved.content_hash == r1.content_hash
        print("✓ Receipt store/retrieve")

        # Chain append
        chain_hash = db.append_to_chain("ed25519:test_agent_1", r1)
        assert chain_hash.startswith("sha256:")
        print(f"✓ Chain append (hash: {chain_hash[:30]}...)")

        r2 = WorkReceipt(
            agent_id="ed25519:test_agent_1",
            task_type="translation",
            verification_tier=VerificationTier.SELF_VERIFYING,
            input_hash=hash_content("hello"),
            output_hash=hash_content("你好"),
            test_results=TestResults(passed=3, failed=0, suite_hash=hash_content("suite2")),
        )
        chain_hash2 = db.append_to_chain("ed25519:test_agent_1", r2)
        print(f"✓ Chain append #2 (hash: {chain_hash2[:30]}...)")

        # Rebuild chain and verify integrity
        chain = db.rebuild_chain("ed25519:test_agent_1")
        assert chain.length == 2
        assert chain.verify_integrity()
        print(f"✓ Chain rebuild + integrity check (length={chain.length})")

        # Discovery
        db.register_capability("ed25519:test_agent_1", ["code_review", "translation"])
        results = db.search_capabilities(task_type="code_review")
        assert len(results) == 1
        print(f"✓ Discovery register + search")

        # Auth challenge
        challenge = db.create_challenge("ed25519:test_agent_1")
        assert "challenge_id" in challenge
        assert "challenge" in challenge
        print(f"✓ Auth challenge created")

        # Stats
        s = db.stats()
        assert s["receipts"] == 2
        assert s["unique_agents"] == 1
        assert s["registered_capabilities"] == 1
        print(f"✓ Stats: {s}")

        # Treasury log
        db.log_treasury_decision({"amount": 5.0, "category": "compute"}, "approved", "under threshold")
        log = db.get_treasury_log()
        assert len(log) == 1
        print(f"✓ Treasury log")

        # Settlement
        sett = db.register_settlement(r1.receipt_id, "ed25519:test_agent_1",
                                       protocol="x402", payment_id="pay_123", amount=0.5)
        assert sett["status"] == "pending"
        assert sett["protocol"] == "x402"
        print(f"✓ Settlement registered")

        got = db.get_settlement(r1.receipt_id)
        assert got is not None
        assert got["payment_id"] == "pay_123"
        print(f"✓ Settlement retrieved")

        confirmed = db.confirm_settlement(r1.receipt_id, "ed25519:buyer_1")
        assert confirmed is not None
        assert confirmed["status"] == "confirmed"
        assert confirmed["confirmed_by"] == "ed25519:buyer_1"
        print(f"✓ Settlement confirmed")

        # Can't confirm again
        again = db.confirm_settlement(r1.receipt_id, "ed25519:buyer_1")
        assert again is None
        print(f"✓ Double-confirm rejected")

        # Agent settlements list
        agent_setts = db.get_agent_settlements("ed25519:test_agent_1")
        assert len(agent_setts) == 1
        print(f"✓ Agent settlements list")

        db.close()
        print(f"\n=== All persistence tests passed ===")
        print(f"DB size: {os.path.getsize(db_path)} bytes")
