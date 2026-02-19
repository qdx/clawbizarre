# ClawBizarre Architecture

## Overview

ClawBizarre is infrastructure for agent-to-agent commerce, built on three primitives:
**identity** (Ed25519 keypairs), **receipts** (structural work records), and **handshakes** (bilateral negotiation protocol).

## Design Principles

1. **Start trustless, earn trust** — Tier 0 (self-verifying) work needs no trust infrastructure. Higher tiers unlock as receipt history accumulates.
2. **Structural over subjective** — Receipts encode what happened (test results, hashes, timing), not opinions. Aggregators infer quality.
3. **Append-only** — Receipt chains are hash-linked and tamper-evident. History is immutable.
4. **Signed** — Every receipt carries an Ed25519 signature from its creator. Verifiable by anyone with the public key.
5. **Composable** — Attestations are separate from core receipts, added post-hoc without invalidating signatures.

## Components

### 1. Identity (`identity.py`)
```
AgentIdentity
├── generate()           → new Ed25519 keypair
├── from_keyfile(path)   → load from PEM
├── from_public_key_hex  → verify-only (no signing)
├── sign(data) → hex     → sign any string
├── verify(data, sig)    → verify any signature
└── save_keyfile(path)   → export PEM (0600 perms)

SignedReceipt
├── receipt_json         → canonical JSON of WorkReceipt
├── content_hash         → SHA-256 of receipt (excl. attestations)
├── signer_id            → ed25519:<pubkey_hex>
├── signature            → Ed25519 signature of content_hash
└── verify(identity)     → bool
```

**Agent ID format**: `ed25519:<64-char-hex-pubkey>`
**Fingerprint**: first 16 hex chars of SHA-256(pubkey) — for human display.

### 2. Work Receipts (`receipt.py`)
```
WorkReceipt
├── agent_id             → who did the work
├── task_type            → code_review, translation, security_audit, ...
├── verification_tier    → 0 (self-verifying) to 3 (human-only)
├── input_hash           → SHA-256 of input
├── output_hash          → SHA-256 of output
├── test_results?        → { passed, failed, suite_hash }
├── risk_envelope?       → { counterparty_risk, policy_decision, ... }
├── environment_hash?    → nix/docker hash for reproducibility
├── agent_config_hash?   → version of agent that did the work
├── attestations[]       → peer reviews, co-signs (added post-hoc)
├── content_hash         → SHA-256 of everything except attestations
└── verify_tier0()       → bool (tests exist and pass)

ReceiptChain
├── append(receipt)      → hash-linked append
├── verify_integrity()   → validate entire chain
├── tier_breakdown()     → {tier: count}
└── success_rate()       → fraction of Tier 0 receipts that pass
```

### 3. Handshake Protocol (`handshake.py`)
```
State machine:
INIT → HELLO → PROPOSED → ACCEPTED → EXECUTING → VERIFYING → COMPLETE
                       ↘ COUNTERED ↗        ↘ FAILED
                       ↘ REJECTED            ↘ ABORTED

Message types: HELLO, PROPOSE, ACCEPT, COUNTER, REJECT, EXECUTE, VERIFY, ABORT
```

Two agents exchange capabilities (HELLO), negotiate a task (PROPOSE/ACCEPT), execute it, and verify the output. Successful verification produces a signed WorkReceipt.

### 4. CLI (`cli.py`)
```bash
clawbizarre init                    # Generate identity
clawbizarre whoami                  # Show identity
clawbizarre receipt create [opts]   # Create + sign receipt
clawbizarre receipt verify FILE     # Verify signed receipt
clawbizarre chain append CHAIN REC  # Append to chain
clawbizarre chain verify CHAIN      # Check integrity
clawbizarre chain stats CHAIN       # Statistics
```

## Verification Tiers

| Tier | Name | Verification | Trust needed | Example |
|------|------|-------------|-------------|---------|
| 0 | Self-verifying | Tests pass, code compiles | None | Code review with test suite |
| 1 | Mechanical | Machine-checkable | Minimal | Format validation, uptime |
| 2 | Peer review | Another agent's judgment | Moderate | Content quality assessment |
| 3 | Human-only | Human evaluation | Full | Creative work, strategy |

**Design constraint**: Start marketplace with Tier 0 only. Each tier upgrade requires more trust infrastructure.

## Era Model

| Era | Economic unit | Payment | Discovery | Trust |
|-----|-------------|---------|-----------|-------|
| 1 (now) | Fleet (sponsor + agents) | Human-to-human | Word of mouth, Moltbook | Human-to-human |
| 1.5 | Fleet + Treasury Agent | Policy-automated | Registries | Human + automated audit |
| 2 | Semi-autonomous agents | Agent-negotiated, sponsor-approved | Push registries | Receipt history |
| 3 | Autonomous agents | Agent wallets | Emergent | Portable reputation DAGs |

## Open Questions

1. **Transport**: How do agents exchange handshake messages? HTTP? Moltbook DMs? MCP?
2. **Multi-party**: Current handshake is bilateral. Fleet-to-fleet needs N-party extension.
3. **Timeout enforcement**: Handshake tracks time constraints but doesn't enforce them.
4. **Counter-proposals**: State exists but logic not implemented.
5. **Receipt discovery**: How do third parties find and aggregate receipts across platforms?
6. **Key rotation**: What happens when an agent rotates keys? Need migration receipts.

## Dependencies

- Python 3.10+
- `cryptography` (Ed25519 only — no heavyweight crypto)
- Zero other dependencies (stdlib dataclasses, json, hashlib)
