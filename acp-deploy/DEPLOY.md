# ClawBizarre ACP Deployment — Quick Start

## Prerequisites
- [ ] DChar approval for wallet creation + verify_server deployment
- [ ] Node.js 18+ installed
- [ ] Docker installed (for sandboxed verification)

## Step 1: Deploy verify_server

### Option A: Fly.io (free tier)
```bash
cd memory/projects/clawbizarre/prototype
fly launch --name clawbizarre-verify --region nrt  # Tokyo, closest to Shanghai
fly deploy
```

### Option B: EC2 (existing AWS)
```bash
cd ~/rahcd-aws-terraform
# Add verify_server EC2 instance to terraform
terraform apply
```

### Option C: Local (testing only)
```bash
cd memory/projects/clawbizarre/prototype
python3 verify_server.py --port 8340
```

## Step 2: Install openclaw-acp
```bash
git clone https://github.com/Virtual-Protocol/openclaw-acp ~/openclaw-acp
cd ~/openclaw-acp
npm install
```

## Step 3: Setup agent identity
```bash
cd ~/openclaw-acp
acp setup
# → Auto-provisions Base chain wallet
# → Backup private key to 1Password: op item create --vault rahcd ...

acp profile update description "Structural code verification for AI agents. Deterministic test-suite-based verification with signed VRF receipts. Supports Python, JavaScript, Bash."
```

## Step 4: Register offering
```bash
# Scaffold (then replace files with our versions)
acp sell init structural_code_verification

# Copy our files over the scaffold
cp /path/to/acp-deploy/offering.json src/seller/offerings/*/structural_code_verification/
cp /path/to/acp-deploy/handlers.ts src/seller/offerings/*/structural_code_verification/

# Set verify_server URL
export CLAWBIZARRE_VERIFY_URL=https://clawbizarre-verify.fly.dev  # or your EC2 URL

# Register on ACP
acp sell create structural_code_verification

# Start serving
acp serve start
```

## Step 5: Sandbox graduation (10 txns)
```bash
# Create a test buyer agent
acp agent create clawbizarre_test_buyer
acp agent switch clawbizarre_test_buyer

# Run test transactions
acp job create <seller-wallet> structural_code_verification \
  --requirements '{"code": "def add(a, b): return a + b", "test_suite": "assert add(1, 2) == 3\nassert add(0, 0) == 0", "language": "python"}'

# Need 10 successful txns, including 3 consecutive
# Switch back to seller to check
acp agent switch clawbizarre
acp serve logs
```

## Step 6: Submit for graduation
- Video recordings of job execution
- Screenshots of sandbox visualizer
- Demonstrate concurrent request handling
- Submit via Virtuals platform
- Wait 7 working days for review

## Environment Variables
| Var | Default | Description |
|-----|---------|-------------|
| `CLAWBIZARRE_VERIFY_URL` | `http://localhost:8340` | Verify server endpoint |

## Monitoring
```bash
acp serve status --json    # Runtime status
acp serve logs --follow    # Live logs
acp wallet balance         # Revenue
acp sell list --json       # Offering status
```
