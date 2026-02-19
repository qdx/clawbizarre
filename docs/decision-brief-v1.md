# ClawBizarre Decision Brief for DChar
*Updated: 2026-02-19 10:31 GMT+8*

## TL;DR
ClawBizarre has a complete prototype (49 files, 17 empirical laws, 200+ tests). The strategic pivot to **verification protocol** (not marketplace) is validated — nobody else does structural work verification for agents. Three decisions are needed to go live.

## 🔥 NEW: OpenClaw × ACP Official Integration
**Virtuals Protocol announced OpenClaw integration with ACP** (Feb 1, 2026). V1 enables OpenClaw agents as buyers on ACP with crypto signatures, escrow, and x402 settlement. Future versions add bidirectional services and group coordination. **This means our verification service could be the first graduated OpenClaw agent on ACP.** The timing is perfect — we have the code, they're building the bridge.

## What We Built
- **Verification server**: Tiered (0-3) structural verification with test suites, Docker sandboxing, multi-language (Python/JS/Bash), Ed25519 signed receipts
- **Full marketplace engine**: Identity, discovery, matching, handshake, settlement, reputation, treasury
- **Distribution**: MCP server (14 tools), OpenClaw skill, WebMCP manifest
- **ACP bridge**: Standalone evaluator that can plug into Virtuals' 18K-agent marketplace
- **10 economic simulations**: 17 laws governing agent marketplace dynamics

## The Strategic Thesis
The marketplace layer is commoditizing (ACP, x402, ERC-8004). **Verification is the durable moat.** Nobody has structural tiered verification with test-suite-as-proof. ACP uses LLM-subjective evaluation (expensive, unreliable). We offer deterministic verification at $0.001-0.005/receipt.

## Three Decisions Needed

### 1. Deploy verify_server publicly — ~$0/mo (Fly.io free tier)
- **What**: Put the verification server on `api.rahcd.com`
- **Why**: Can't be an ACP evaluator without a public endpoint
- **Cost**: $0 (Fly.io free tier: 3 VMs, adequate for bootstrap)
- **Risk**: Low — it's a stateless verification service, no user data
- **Alternative**: EC2 on existing AWS (~$15/mo)
- **I need**: Permission to create Fly.io account, or approval for EC2 terraform

### 2. Register as ACP Service Provider — requires Base wallet
- **What**: Run `acp setup` (auto-provisions wallet on Base), register verification offering
- **Why**: Access to 18K existing agents. $1M/mo subsidy pool. OpenClaw is officially integrating with ACP.
- **Cost**: Minimal gas (Base L2 = pennies). Agent registration is free. $0.005/verification fee.
- **Risk**: Crypto wallet = key management. `acp setup` auto-creates it. Can backup private key to 1Password.
- **Path**: `npm install` openclaw-acp → `acp setup` → `acp sell create` → `acp serve start` → 10 sandbox txns → graduation review (7 days)
- **I need**: Permission to run `acp setup` (creates on-chain identity) and store wallet key in rahcd vault
- **Repo**: github.com/Virtual-Protocol/openclaw-acp — the official OpenClaw×ACP CLI

### 3. Publish to ClawHub — $0
- **What**: `clawhub publish skills/clawbizarre` — makes it installable by any OpenClaw agent
- **Why**: Distribution to OpenClaw ecosystem
- **Cost**: Free
- **Risk**: None — it's open source
- **I need**: ClawHub auth (may need your account)

## What Happens After
1. **Week 1-2**: Deploy, register as ACP evaluator, handle first verification jobs
2. **Month 1**: Prove unit economics (cost per verification, revenue per job)
3. **Month 2-3**: If viable, scale. If not, we learned what breaks and where the real market is.

## What I Can Do Without Approval
- Continue simulation research (diminishing returns — 17 laws is comprehensive)
- Write more docs (also diminishing returns)
- Build more prototype code (already feature-complete for v1)

**Bottom line**: The prototype phase is done. The next step is deployment, and that requires your call.
