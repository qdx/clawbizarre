# First Week Operational Playbook
*What happens after DChar approves deployment*

## Day 0: Deploy (2-3 hours)

### verify_server on Fly.io
```bash
cd prototype
fly launch --name clawbizarre-verify --region nrt  # Tokyo, closest to Shanghai
fly deploy
# Health check: curl https://clawbizarre-verify.fly.dev/health
```

Point `api.rahcd.com` → Fly.io (CNAME in Route 53).

### ACP Registration
```bash
npm install -g openclaw-acp
acp setup  # Creates Base wallet, store key in 1Password
cd acp-deploy
acp sell create  # Registers offering from offering.json
acp serve start  # WebSocket runtime, accepts jobs
```

### ClawHub Publish
```bash
clawhub publish skills/clawbizarre
```

### Monitoring Setup
- Fly.io built-in metrics (requests, latency, errors)
- Add a cron job: health check every 5 min, alert on failure
- Log all verification requests to SQLite (already built into persistence.py)

## Day 1-2: Sandbox Graduation

ACP requires 10 sandbox transactions (3 consecutive successes) before graduation.

### Self-test strategy
- Use two separate agent identities (already have keypair generation)
- Submit 10 verification jobs covering:
  - Python code (3 jobs)
  - JavaScript code (3 jobs)
  - Bash scripts (2 jobs)
  - Edge cases: timeout, syntax error (2 jobs)
- Verify all receipts are properly signed and chain-linked

### Record everything
- Video recordings of each transaction (ACP graduation requires this)
- Concurrent request handling demo (2-3 simultaneous verifications)
- Capture latency metrics for each

## Day 3-4: Graduation Submission

Submit to Virtuals team:
- Video recordings from sandbox testing
- API documentation link (rahcd.com or GitHub)
- Metadata: offering description, pricing, capabilities
- Wait for review (up to 7 working days)

While waiting:
- Monitor verify_server logs for any issues
- Run the benchmark suite periodically (should stay <200ms avg)

## Day 5-7: First Real Customers

### Finding customers on ACP
Once graduated, we appear in "Agent to Agent" tab and are discoverable via Butler.

**Outreach strategy** (within ACP ecosystem):
1. **Passive**: listing is discoverable, Butler can recommend us
2. **Active**: identify ACP agents that do code-related work, offer pre-verification
   - Search ACP marketplace for code/dev offerings
   - Provider-side pitch: "Verify your deliverables before submission → higher buyer satisfaction → more repeat business"

### Pricing validation
- Start at $0.005/verification (simulation-validated sweet spot)
- Track: jobs/day, pass/fail ratio, avg execution time, revenue
- If <10 jobs/day after week 2, consider dropping to $0.001 temporarily

### Success metrics (Week 1)
| Metric | Target | Critical |
|--------|--------|----------|
| Uptime | 99%+ | >95% |
| Avg latency | <200ms | <500ms |
| Sandbox txns | 10/10 | 10/10 |
| Graduation submitted | ✅ | ✅ |
| Real verifications | 1+ | nice to have |

## Failure Modes & Responses

### verify_server goes down
- Fly.io auto-restarts on crash
- If persistent: check logs (`fly logs`), likely Docker daemon issue
- Fallback: Python-only mode (no Docker sandbox, still functional for Python tasks)

### ACP graduation rejected
- Most likely: incomplete metadata or missing video
- Fix and resubmit (no penalty for resubmission)
- Worst case: SDK version mismatch — update and redo sandbox tests

### No customers after 2 weeks
- Pivot to provider-side model: reach out directly to ACP code agents
- Consider free tier (first 100 verifications) to build track record
- Cross-post on OpenClaw Discord, agent communities

### Verification false positive/negative
- Log the failure case
- Add to test suite
- If systematic: adjust verification tier routing

## Cost Projections (Month 1)

| Item | Cost |
|------|------|
| Fly.io (free tier) | $0 |
| Base gas (registration + 10 sandbox) | <$0.50 |
| Domain (already owned) | $0 |
| **Total** | **<$1** |

Revenue at 10 jobs/day × $0.005 = $0.05/day = $1.50/month. Won't cover compute, but proves the model. At 100 jobs/day = $15/month → self-sustaining on free tier.

## Week 2+ Decision Points

- **If >50 jobs/day**: Scale Fly.io, consider paid tier for reliability
- **If <5 jobs/day**: Double down on outreach, consider free tier
- **If 0 jobs/day**: Re-evaluate — is ACP the right marketplace? Consider direct MCP distribution instead.
- **Quality signal**: If pass rate >95%, we're getting good submissions. If <70%, buyers may be testing our limits.
