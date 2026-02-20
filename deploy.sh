#!/usr/bin/env bash
# deploy.sh — ClawBizarre one-command deployment
#
# Usage:
#   ./deploy.sh              — full deploy (tests + fly deploy)
#   ./deploy.sh --skip-tests — deploy without running tests (faster)
#   ./deploy.sh --tests-only — just run tests, don't deploy
#   ./deploy.sh --setup      — first-time: create app + volume on Fly.io
#   ./deploy.sh --local      — run locally on port 8420 (dev mode)
#   ./deploy.sh --status     — check deployed app status
#
# Requirements:
#   - Python 3.11+ (pip install not needed — pure stdlib)
#   - Node.js (for JavaScript test execution in lightweight_runner)
#   - fly CLI (for Fly.io deployment): curl -L https://fly.io/install.sh | sh
#   - fly auth login (one-time, requires DChar approval)

set -euo pipefail

# ── Config ───────────────────────────────────────────────────────────────────

APP_NAME="clawbizarre"
REGION="sin"                   # Singapore
VOLUME_NAME="clawbizarre_data"
VOLUME_SIZE_GB=1
PORT=8420
PROTOTYPE_DIR="$(cd "$(dirname "$0")/prototype" && pwd)"
ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
HEALTH_ENDPOINT="http://localhost:${PORT}/health"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
BOLD='\033[1m'
NC='\033[0m' # No Color

# ── Helpers ───────────────────────────────────────────────────────────────────

log()    { echo -e "${BLUE}[deploy]${NC} $*"; }
ok()     { echo -e "${GREEN}[✓]${NC} $*"; }
warn()   { echo -e "${YELLOW}[!]${NC} $*"; }
error()  { echo -e "${RED}[✗]${NC} $*"; exit 1; }
header() { echo -e "\n${BOLD}$*${NC}"; }

check_dep() {
    command -v "$1" &>/dev/null || error "$1 not found. $2"
}

# ── Parse args ────────────────────────────────────────────────────────────────

SKIP_TESTS=false
TESTS_ONLY=false
SETUP_MODE=false
LOCAL_MODE=false
STATUS_MODE=false

for arg in "$@"; do
    case $arg in
        --skip-tests)  SKIP_TESTS=true ;;
        --tests-only)  TESTS_ONLY=true ;;
        --setup)       SETUP_MODE=true ;;
        --local)       LOCAL_MODE=true ;;
        --status)      STATUS_MODE=true ;;
        --help|-h)
            echo "Usage: ./deploy.sh [--skip-tests|--tests-only|--setup|--local|--status]"
            exit 0
            ;;
        *) warn "Unknown argument: $arg" ;;
    esac
done

# ── Status check ──────────────────────────────────────────────────────────────

if $STATUS_MODE; then
    header "ClawBizarre Status"
    check_dep fly "Install: curl -L https://fly.io/install.sh | sh"
    fly status --app "$APP_NAME" 2>/dev/null || error "App not deployed or not authenticated"
    fly logs --app "$APP_NAME" --no-tail 2>/dev/null | tail -20
    exit 0
fi

# ── Local mode ────────────────────────────────────────────────────────────────

if $LOCAL_MODE; then
    header "Starting ClawBizarre locally on port $PORT"
    check_dep python3 "Install Python 3.11+"
    cd "$PROTOTYPE_DIR"
    log "Running at http://localhost:$PORT"
    log "Health: http://localhost:$PORT/health"
    log "Press Ctrl+C to stop"
    CLAWBIZARRE_PORT=$PORT python3 api_server_v7.py --port $PORT --db /tmp/clawbizarre_dev.db
    exit 0
fi

# ── Header ────────────────────────────────────────────────────────────────────

header "ClawBizarre Deployment — $(date '+%Y-%m-%d %H:%M %Z')"
log "App: $APP_NAME | Region: $REGION | Dir: $PROTOTYPE_DIR"

# ── Check dependencies ────────────────────────────────────────────────────────

header "Checking dependencies..."
check_dep python3 "Install Python 3.11+"
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
ok "Python $PYTHON_VERSION"

if command -v node &>/dev/null; then
    NODE_VERSION=$(node --version)
    ok "Node.js $NODE_VERSION (JavaScript test support)"
else
    warn "Node.js not found — JavaScript tests will be skipped"
fi

if ! $TESTS_ONLY; then
    check_dep fly "Install: curl -L https://fly.io/install.sh | sh && fly auth login"
    FLY_VERSION=$(fly version 2>&1 | head -1)
    ok "fly CLI: $FLY_VERSION"
fi

# ── Run tests ─────────────────────────────────────────────────────────────────

if ! $SKIP_TESTS; then
    header "Running test suite..."
    cd "$PROTOTYPE_DIR"

    # Check for pytest
    if ! python3 -m pytest --version &>/dev/null; then
        warn "pytest not found, installing..."
        pip install pytest --quiet
    fi

    log "Running 587+ tests (no Docker required — lightweight_runner handles sandboxing)..."
    
    # Run tests; show progress but suppress verbose output unless failing
    if python3 -m pytest tests/ -q --tb=short 2>&1; then
        ok "All tests pass"
    else
        error "Tests failed — aborting deployment. Fix tests before deploying."
    fi
fi

if $TESTS_ONLY; then
    ok "Tests complete. Use ./deploy.sh to also deploy."
    exit 0
fi

# ── First-time setup ──────────────────────────────────────────────────────────

if $SETUP_MODE; then
    header "First-time Fly.io setup..."
    cd "$PROTOTYPE_DIR"

    log "Creating Fly.io app: $APP_NAME (region: $REGION)"
    fly apps create "$APP_NAME" --machines 2>/dev/null || warn "App may already exist"

    log "Creating persistent volume for SQLite..."
    fly volumes create "$VOLUME_NAME" \
        --app "$APP_NAME" \
        --region "$REGION" \
        --size "$VOLUME_SIZE_GB" 2>/dev/null || warn "Volume may already exist"

    ok "Setup complete. Run './deploy.sh' to deploy."
    exit 0
fi

# ── Deploy ────────────────────────────────────────────────────────────────────

header "Deploying to Fly.io..."
cd "$PROTOTYPE_DIR"

log "Building and deploying (this takes ~60s on first deploy, ~20s after)..."
fly deploy \
    --app "$APP_NAME" \
    --region "$REGION" \
    --strategy rolling \
    --wait-timeout 120

# ── Health check ──────────────────────────────────────────────────────────────

header "Verifying deployment..."

# Get the app URL
FLY_URL="https://${APP_NAME}.fly.dev"
log "App URL: $FLY_URL"

# Wait for health check
log "Waiting for health endpoint..."
for i in {1..12}; do
    if curl -sf "${FLY_URL}/health" &>/dev/null; then
        HEALTH_RESP=$(curl -s "${FLY_URL}/health")
        ok "Health check passed: $(echo "$HEALTH_RESP" | python3 -c "import sys,json; d=json.load(sys.stdin); print(f\"v{d.get('version','?')}, {d.get('receipts_stored',0)} receipts\")" 2>/dev/null || echo "$HEALTH_RESP")"
        break
    fi
    if [ $i -eq 12 ]; then
        error "Health check timed out after 60s. Check: fly logs --app $APP_NAME"
    fi
    sleep 5
    log "Waiting... ($((i*5))s)"
done

# ── Post-deploy summary ───────────────────────────────────────────────────────

header "Deployment complete! 🚀"
echo ""
echo "  Verify endpoint:  ${FLY_URL}/verify"
echo "  Health endpoint:  ${FLY_URL}/health"
echo "  Credit score API: ${FLY_URL}/credit/score"
echo "  Task board API:   ${FLY_URL}/tasks"
echo "  Discovery API:    ${FLY_URL}/discovery"
echo ""
echo "  Next steps:"
echo "    1. Point api.rahcd.com CNAME → ${APP_NAME}.fly.dev  (Route 53)"
echo "    2. Test: curl ${FLY_URL}/health | python3 -m json.tool"
echo "    3. MCP config: set verify_url=https://api.rahcd.com in OpenClaw"
echo ""
echo "  Logs: fly logs --app $APP_NAME"
echo "  Status: ./deploy.sh --status"
echo ""
ok "ClawBizarre is live!"
