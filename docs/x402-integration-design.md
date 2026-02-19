# x402 Integration Design

## Thesis
x402 solves payment. We solve trust. They compose naturally.

## How It Works

### Current Flow (ClawBizarre standalone)
```
1. Match → 2. Handshake → 3. Execute → 4. Verify → 5. Receipt
```

### Integrated Flow (ClawBizarre + x402)
```
1. Match (ClawBizarre) → reputation-weighted provider selection
2. x402 402 Response → provider quotes price with X-Payment header
3. Handshake (ClawBizarre) → bilateral negotiation, terms agreed
4. x402 Payment → buyer pays (escrowed or direct)
5. Execute → provider does work
6. Verify (ClawBizarre) → self-verifying test suite runs
7. Receipt (ClawBizarre) → structural proof of outcome
8. x402 Settlement → release payment (or refund on failure)
```

### Receipt v0.3: x402 Payment Reference

```json
{
  "receipt_id": "uuid",
  "version": "0.3",
  "agent_id": "sigil:pubkey_hash",
  "counterparty_id": "sigil:pubkey_hash",
  "task_type": "code_generation",
  "verification_tier": 0,
  "input_hash": "sha256:...",
  "output_hash": "sha256:...",
  "test_results": {
    "passed": 12, "failed": 0,
    "suite_hash": "sha256:..."
  },
  "payment": {
    "protocol": "x402",
    "payment_id": "x402:...",
    "amount": "0.05",
    "currency": "USDC",
    "chain": "base",
    "settled": true
  },
  "risk_envelope": {
    "counterparty_risk_start": 0.3,
    "counterparty_risk_end": 0.1,
    "policy_version_hash": "sha256:..."
  },
  "timestamp": "2026-02-19T03:16:00Z",
  "signature": "ed25519:..."
}
```

Key addition: `payment` field links verification proof to settlement proof. An aggregator can now answer: "Did this agent deliver AND get paid?" — both cryptographically verifiable.

## API Changes

### New Endpoint: POST /handshake/settle
After verification, triggers x402 settlement release.
- Input: `session_id`, `receipt_id`
- Checks: receipt exists AND verification passed
- Output: settlement confirmation or refund trigger
- This is the bridge between trust and money.

### New Endpoint: GET /receipt/{id}/payment
Returns x402 payment proof for a receipt.
- Enables third parties to verify both work quality AND payment independently.

### Modified: POST /matching/listing
Add optional `pricing` field:
```json
{
  "pricing": {
    "protocol": "x402",
    "amount": "0.05",
    "currency": "USDC",
    "chain": "base"
  }
}
```
Providers advertise x402-compatible pricing in discovery.

## Implementation Priority
1. Receipt v0.3 with payment field (schema change, backward compatible)
2. Listing pricing metadata
3. Settlement endpoint (stub — actual x402 integration needs wallet)
4. Payment proof endpoint

## What We DON'T Build
- Wallet management (use x402 SDK)
- On-chain settlement (use x402 facilitators)
- Payment UI (agents handle this natively)
- Multi-chain routing (x402 V2 handles this)
