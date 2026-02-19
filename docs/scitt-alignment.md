# VRF ↔ SCITT Alignment Analysis
*Created: 2026-02-19 18:16 GMT+8*

## Summary

IETF SCITT (Supply Chain Integrity, Transparency, and Trust) is a Standards Track working group building architecture for signed, transparent supply chain statements. VRF (Verification Receipt Format) maps naturally onto SCITT concepts. Aligning VRF with SCITT provides:

1. **Formal standards backing** — SCITT is already at draft-22, near completion
2. **Existing infrastructure** — Transparency Services, SCRAPI reference APIs
3. **Credibility for NIST submission** — "VRF is a SCITT content type" >> "VRF is our custom format"
4. **Interoperability** — COSE signing, Merkle-based audit, existing tooling

## Concept Mapping

| SCITT Concept | VRF Equivalent | Notes |
|---|---|---|
| **Issuer** | Verifier (ClawBizarre verify_server) | Entity that signs the statement |
| **Signed Statement** | VRF Receipt (signed) | COSE_Sign1 wrapper around VRF payload |
| **Artifact** | Agent work output (code, data, etc.) | What's being verified |
| **Statement** | Verification result (pass/fail/partial + details) | The claim about the artifact |
| **Receipt** (SCITT) | Transparency proof from a TS | Proof that the VRF receipt was logged |
| **Transparency Service** | VRF receipt ledger / aggregator | append-only log of all verification events |
| **Registration Policy** | Verification tier requirements | What receipts a TS will accept |
| **Artifact Repository** | Agent's receipt chain | Collection of receipts about one agent |

## Key Architectural Alignment

### SCITT's model
```
Issuer → signs Statement about Artifact → registers at Transparency Service → anyone can audit
```

### VRF's model  
```
Verifier → creates Receipt about Agent Work → appends to Receipt Chain → reputation aggregator audits
```

These are the **same architecture** with different terminology.

### Where They Diverge

1. **SCITT uses COSE (CBOR Object Signing)** — VRF currently uses Ed25519 over canonical JSON
   - Migration path: wrap VRF JSON as COSE_Sign1 payload, use COSE key format
   - Keep JSON as human-readable format, COSE as wire format
   
2. **SCITT is about supply chain artifacts** (software packages, SBOMs, etc.) — VRF is about agent work output
   - But SCITT is explicitly extensible: "all supply chains" including digital services
   - Agent-to-agent work IS a supply chain (of labor/computation)

3. **SCITT Receipts are Transparency Service proofs** — VRF receipts are verifier attestations
   - In SCITT terms, our VRF receipt is a Signed Statement; the SCITT Receipt would be what the Transparency Service issues when it logs our VRF receipt
   - This is actually MORE powerful: VRF receipts get SCITT Receipts, providing two layers of trust

## Standards Roadmap

### Phase 1: SCITT Content Type (now)
- Register `application/vrf+json` as SCITT payload content type
- Define VRF-specific COSE header parameters (tier, verdict, task_type)
- Write mapping document: VRF fields → SCITT Signed Statement structure

### Phase 2: SCITT Transparency Service for Agent Work (Q2 2026)
- Deploy a Transparency Service that accepts VRF Signed Statements
- Registration Policy: only Tier 0-1 receipts (deterministic verification)
- SCRAPI-compatible API (POST /entries, GET /entries/{id}, etc.)

### Phase 3: IETF Internet-Draft (Q3 2026)
- `draft-clawbizarre-scitt-vrf-00` — VRF as SCITT content type for agent work verification
- Submit to SCITT WG mailing list for discussion
- Reference existing SCITT architecture, add agent-specific considerations

## COSE Encoding of VRF Receipt

```
COSE_Sign1 = [
  protected: {
    1: -8,              # alg: EdDSA
    3: "application/vrf+json",  # content type
    15: {               # CWT claims
      1: "clawbizarre-verify/1.0",  # iss (verifier)
      6: 1771459000     # iat (verified_at as Unix timestamp)
    },
    # VRF-specific headers (to be registered):
    TBD1: 0,            # vrf-tier (0-3)
    TBD2: "pass",       # vrf-verdict
  },
  unprotected: {},
  payload: <VRF receipt JSON as bytes>,
  signature: <Ed25519 signature>
]
```

## Impact on NIST Submission

Current NIST draft positions VRF as standalone. Strengthening it:

**Before**: "We propose VRF, an open format for verification receipts."
**After**: "We propose VRF as a SCITT-compatible content type, leveraging the IETF SCITT architecture (draft-ietf-scitt-architecture-22) for transparency and auditability. This positions agent work verification within an existing standards framework rather than creating a parallel ecosystem."

This is significantly more credible:
- Shows awareness of existing standards
- Reduces NIH (not-invented-here) concerns  
- Provides concrete interop path
- Aligns with NIST's preference for building on existing standards

## SCITT WG Engagement

- Mailing list: scitt@ietf.org
- Archive: https://mailarchive.ietf.org/arch/browse/scitt/
- GitHub: https://github.com/ietf-wg-scitt/draft-ietf-scitt-architecture
- **Action**: Subscribe to mailing list, introduce VRF use case
- **Action**: Check if there's an agent/AI use case already discussed in SCITT WG

## Open Questions

1. Should VRF be a SCITT "content type" or a full SCITT "use case" document?
2. COSE vs JSON: keep both? COSE for wire, JSON for human readability?
3. Do we need IANA registration for VRF-specific COSE header parameters?
4. Is there a simpler path: just use SCITT as-is with VRF as opaque payload?
