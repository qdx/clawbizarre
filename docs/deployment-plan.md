# ClawBizarre Deployment Plan

## Current State
- API server v6 (v0.8.0) runs locally, all tests pass
- Dockerfile + docker-compose + nginx ready
- OpenClaw skill (`skills/clawbizarre/`) ready
- MCP server ready (`prototype/mcp_server.py`)

## Option A: EC2 on AWS (Recommended)
- Add `ec2.tf` to rahcd-aws-terraform
- t3.micro (free tier eligible) or t3.small (~$15/mo)
- Docker Compose deployment
- api.rahcd.com subdomain (Route 53 already managed)
- Let's Encrypt via certbot for HTTPS
- **Needs DChar approval** (recurring cost)

## Option B: Fly.io (Free Tier)
- 3 shared-cpu-1x VMs free
- `fly launch` from prototype dir
- fly.toml + Dockerfile already compatible
- SQLite + Fly Volumes for persistence
- Custom domain: api.rahcd.com CNAME → fly.dev
- **No cost, but platform dependency**

## Option C: This VM (Dev Only)
- Already works (`python3 api_server_v6.py`)
- No public IP → not accessible externally
- Good for development/testing only

## Recommendation
Fly.io for zero-cost public deployment. Migrate to EC2 if traffic warrants it.

## Steps (Fly.io)
1. `curl -L https://fly.io/install.sh | sh`
2. `fly auth login`
3. `fly launch` in prototype/
4. `fly volumes create clawbizarre_data`
5. Point api.rahcd.com CNAME → app.fly.dev
6. Test: `curl https://api.rahcd.com/health`

## ClawHub Publishing
- Skill at `skills/clawbizarre/`
- `clawhub publish skills/clawbizarre`
- Needs ClawHub auth (check if available)
