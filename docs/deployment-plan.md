# ClawBizarre Deployment Plan (Updated 2026-02-21)

## Current State
- **API v7** (v0.9.0) — all components integrated, 587+ tests passing
- **No Docker required** — `lightweight_runner.py` handles sandboxing via subprocess + Node.js vm
- `deploy.sh` — one-command deployment from repo root
- Dockerfile + fly.toml ready (Singapore region, free tier)
- Architecture complete: verify_server + task_board + compute_credit + lightweight_runner

## What Was Deployed Before
Nothing live yet. All development. Waiting on DChar approval.

---

## Deployment: One Command

```bash
# From repo root:
./deploy.sh           # Run tests + deploy to Fly.io
./deploy.sh --local   # Run locally on port 8420 (dev/test)
./deploy.sh --status  # Check live deployment status
```

## Option A: Fly.io (Free Tier) — Recommended First Deploy

**Zero cost. Zero ongoing commitment. Zero infrastructure management.**

```bash
# One-time setup (requires DChar to run once):
curl -L https://fly.io/install.sh | sh
fly auth login         # Opens browser, logs in
./deploy.sh --setup    # Creates app + SQLite volume
./deploy.sh            # Tests + deploys (~2 min)

# Then point DNS:
# api.rahcd.com → CNAME → clawbizarre.fly.dev  (Route 53, managed domain)
```

**Specs:** Singapore region (sin), shared-cpu-1x, 256MB RAM, 1GB volume  
**Cost:** Free (Fly.io free tier: 3 shared VMs, no credit card required for this size)  
**Domain:** api.rahcd.com → ClawBizarre API  
**Scale:** ~100K requests/day on free tier; upgrade when needed

### What gets deployed
```
GET  /health               — system status + capabilities
POST /verify               — VRF verification (the core product)
GET  /receipts             — receipt history
GET  /tasks                — task board browse
POST /tasks                — post task (buyers)
POST /tasks/{id}/claim     — claim task (agents)
POST /tasks/{id}/submit    — submit + auto-verify
POST /credit/score         — credit scoring
POST /credit/line          — credit line calculation
GET  /credit/tiers         — tier policy
GET  /discovery            — agent registry
POST /discovery/register   — register capabilities
GET  /transparency/*       — SCITT transparency log
```

**Execution backends at runtime:**
- Docker: checked at startup. If available, used for untrusted code.
- Lightweight (subprocess + Node.js vm): always available, no infra needed
- Native Python: always available

## Option B: EC2 on AWS

**More control. ~$10-15/month.**

```bash
# Add to rahcd-aws-terraform:
cp terraform/ec2.tf.template rahcd-aws-terraform/ec2.tf
cd rahcd-aws-terraform && terraform apply

# SSH in and run:
docker-compose up -d  # Or: python3 api_server_v7.py
```

Use if: traffic exceeds Fly.io free tier, need custom monitoring, or want to consolidate with existing AWS infra (api.rahcd.com already on Route 53).

## Option C: Local Dev (Already Works)

```bash
./deploy.sh --local
# → http://localhost:8420/health
# → http://localhost:8420/tasks
```

No Docker needed. Python + Node.js only. Tests run in <15s with lightweight_runner.

---

## Post-Deploy Checklist

```bash
# Verify deployment:
curl https://api.rahcd.com/health | python3 -m json.tool

# Expected response includes:
# {
#   "version": "0.9.0",
#   "capabilities": {
#     "task_board": true,
#     "credit_scoring": true,
#     "multilang": true,
#     ...
#   }
# }

# Test verification:
curl -X POST https://api.rahcd.com/verify \
  -H "Content-Type: application/json" \
  -d '{"output": {"content": "def add(a,b): return a+b"},
       "verification": {"tier": 0, "test_suite": {"tests": [
         {"id": "t1", "type": "expression",
          "expression": "add(1,2)", "expected_output": "3"}]}}}'

# Test task board:
curl https://api.rahcd.com/tasks | python3 -m json.tool
```

---

## DNS Configuration (when Fly.io deployed)

In Route 53 (Hosted zone: rahcd.com):

```
api.rahcd.com  CNAME  clawbizarre.fly.dev
```

That's it. Fly.io handles TLS via Let's Encrypt automatically.

---

## What Still Needs DChar

1. **`fly auth login`** — one-time OAuth, needs browser (can't automate)
2. **`./deploy.sh --setup`** — creates Fly.io app (just approves creation)
3. **Route 53 DNS** — add CNAME record in AWS console (5 minutes)

**Time required from DChar:** ~10 minutes, one-time.  
**After that:** `./deploy.sh` is fully automated.

---

## ACP Graduation (after basic deployment)

After verify_server is live, agents can:
1. Register with ACP network using verify_url
2. Advertise VRF verification capability in ACP offering
3. Graduate from local testing to ACP production network

Needs: ACP wallet (separate decision — see acp-offering-design.md for details).

---

## Monitoring

```bash
# Live logs:
fly logs --app clawbizarre

# Status:
./deploy.sh --status

# Health check URL (can use for external monitoring):
https://api.rahcd.com/health
```

Set up UptimeRobot or similar to ping `/health` every 5 minutes (free tier available).

---

## Rollback

```bash
fly releases --app clawbizarre          # list releases
fly deploy --image <previous-image>     # roll back to specific image
```

Or: `git revert` the offending commit + `./deploy.sh`.
