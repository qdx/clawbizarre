# ClawBizarre — 1-Pager for DChar

**What**: Structural verification protocol for AI agent work. Think "SSL certificates for agent output."

**Why now**: ACP has 18K agents, zero structural verification. NIST just launched agent standards (Feb 19). Security researchers are publicly asking "who's verifying?" Nobody is. We are.

**What's built**: Verification server (test suites, Docker sandbox, multi-language, signed receipts), 3 protocol adapters (MCP/ACP/A2A), 250+ tests, 25 empirical laws. Feature-complete.

**Revenue model**: $0.005/verification. Self-sustaining at ~100 verifications/day. Fee is <1% of task value (invisible to agents).

**Status**: Engineering done. Blocked on 6 decisions (see `decision-brief-v1.md`). The two that matter most:

1. **Deploy verify_server** → Fly.io free tier, $0/mo, low risk → unlocks everything
2. **Register on ACP** → `acp setup`, creates Base wallet → access to 18K agents + $1M/mo subsidy pool

Optional but time-sensitive:
3. **NIST RFI** → March 9 deadline → first-mover in verification standards
4. **NCCoE feedback** → April 2 deadline
5. **ClawHub publish** → free, open source
6. **NIST listening sessions** → March 20 registration

**Bottom line**: Window is open. Nobody else does this. Deployment is 6 commands. Say the word.
