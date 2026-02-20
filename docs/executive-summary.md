# ClawBizarre — 1-Pager for DChar
*Updated February 21, 2026 (6:06 AM)*

---

**What**: Structural verification protocol for AI agent work. Think "SSL certificates for agent output" + "FICO credit score for agent work history." The only open infrastructure for deterministic, cryptographically signed output quality verification in agent economies.

**Why now**: 
- Unicity Labs raised $3M (Feb 19) for p2p agent commerce — validates the market, doesn't do output quality
- Apple shipped agent self-verification in Xcode 26.3 (Feb 20) — validates the paradigm at platform scale  
- NIST AI Agent Standards Initiative launched (Feb 19) — RFI due March 9 (15 days)
- EU AI Act main provisions enforceable August 2026 — max penalty €35M
- Every funded agent marketplace solves "can agents transact?" — nobody solves "was the work correct?"

---

**What's built**: 

| Component | Status |
|-----------|--------|
| Verify server (Tier 0/1, signed receipts) | ✅ Complete |
| Execution backends (Docker + lightweight, no infra needed) | ✅ Complete |
| Protocol adapters (MCP, ACP, A2A) | ✅ Complete |
| Framework integrations (LangChain, CrewAI, OpenAI) | ✅ Complete |
| Task Board (buyer posts, agent claims/submits) | ✅ Complete (Phase 29) |
| Compute Credit (VRF receipts → 0-100 credit score) | ✅ Complete (Phase 28) |
| Full lifecycle end-to-end | ✅ Complete (Phase 29b) |
| REST API v7 (30+ endpoints) | ✅ Complete |
| Python SDK + CLI (`clawbizarre verify`) | ✅ Complete |
| SCITT/COSE transparency stack | ✅ Complete |
| Deployment script (one command) | ✅ `./deploy.sh` |
| CI/CD (622 tests on GitHub push) | ✅ Complete |
| **Total tests** | **622/622 passing** |

---

**Architecture (complete)**:
```
Buyers → Task Board → Verify Server → Compute Credit → Treasury → Agent sustains
             ↑               ↓
         Discovery       Receipt Chain (VRF signed)
```

---

**Revenue model**: $0.005-0.01/verification. Self-sustaining at ~100 verifications/day. Fee is <1% of task value (invisible to agents).

**Agent economic loop**: Each verified task → receipt → credit score → compute access → more tasks → self-sustaining at ~100 tasks/day. The cold-start solution: staked introduction (existing agents vouch for newcomers, stake their own credit).

---

**Status**: Engineering **architecturally complete**. Blocked on 10 minutes of your time.

### The 3 Things That Matter Now

**1. Deploy verify_server** (10 min, **one command**, $0/mo)
```bash
fly auth login            # opens browser
./deploy.sh --setup       # creates app + volume
./deploy.sh               # tests + deploy + health check
# Add Route 53 CNAME: api.rahcd.com → clawbizarre.fly.dev
```

**2. NIST RFI** (March 9, 15 days — v4 ready at `docs/nist-rfi-draft-v3.md`)
```bash
# Just review v4, put your name on it, submit
# 30 min of your time → first-mover in verification standards
```

**3. ACP Registration** (after deployment)
```bash
# Creates Base wallet, access to 18K agents
# Separate decision — see docs/acp-offering-design.md
```

---

**Competitive landscape**: 

| Player | Raises | Layer | Relationship |
|--------|--------|-------|-------------|
| Unicity Labs | $3M seed | Counterparty trust + payment rails | Complementary (they build the roads, we inspect the delivery) |
| Bittensor/AGIX | Ongoing | Token-staked compute | Different model (subjective consensus vs deterministic tests) |
| Apple Xcode | Platform | IDE-level visual verification | Validates paradigm; doesn't solve portable third-party receipts |
| *(everyone else)* | — | Commerce infrastructure | Future VRF customers |

**Bottom line**: The window is open. 622 tests. 66 laws. Architecture complete. `./deploy.sh` when ready.
