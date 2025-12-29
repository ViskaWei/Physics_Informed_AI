#!/bin/bash
# ==============================================================================
# SpecViT Remote Check
# Quick check if there are changes in Overleaf/GitHub paper repo
# Non-interactive, suitable for cron or quick status checks
# ==============================================================================

# Configuration
PAPER_DIR="paper/vit/SpecViT"
REMOTE_URL="git@github.com:ViskaWei/SpecViT-paper.git"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

# Get repo root
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo -e "${RED}Not in a git repository${NC}"
    exit 1
}

cd "$REPO_ROOT"

# Create temp dir
TEMP_DIR=$(mktemp -d)
trap "rm -rf $TEMP_DIR" EXIT

# Clone remote (quiet)
if ! git clone --depth 1 -q "$REMOTE_URL" "$TEMP_DIR/remote" 2>/dev/null; then
    echo -e "${RED}✗ Failed to fetch remote${NC}"
    exit 1
fi

# Compare (exclude files that differ between local and remote by design)
DIFF_COUNT=$(diff -rq "$PAPER_DIR" "$TEMP_DIR/remote" \
    --exclude='.git' \
    --exclude='.gitkeep' \
    --exclude='.gitignore' \
    --exclude='Makefile' \
    --exclude='*.aux' --exclude='*.log' --exclude='*.out' \
    --exclude='*.bbl' --exclude='*.blg' --exclude='*.synctex.gz' \
    2>/dev/null | wc -l)

REMOTE_COMMIT=$(cd "$TEMP_DIR/remote" && git rev-parse --short HEAD)
REMOTE_DATE=$(cd "$TEMP_DIR/remote" && git log -1 --format=%cr)

if [ "$DIFF_COUNT" -eq 0 ]; then
    echo -e "${GREEN}✓ Up to date${NC} with remote ($REMOTE_COMMIT, $REMOTE_DATE)"
    exit 0
else
    echo -e "${YELLOW}⚠ $DIFF_COUNT file(s) differ${NC} from remote ($REMOTE_COMMIT, $REMOTE_DATE)"
    echo ""
    echo "Run './tools/specvit_pull_agent.sh' to pull changes"
    exit 1
fi
