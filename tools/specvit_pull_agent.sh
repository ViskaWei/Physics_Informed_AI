#!/bin/bash
# ==============================================================================
# SpecViT Pull Agent
# Intelligent agent to pull and sync Overleaf/GitHub paper changes
# 
# Features:
#   - Check for remote changes before pulling
#   - Preview diff before applying
#   - Auto-commit with meaningful message
#   - Handle conflicts gracefully
# ==============================================================================

set -e

# ==============================================================================
# Configuration
# ==============================================================================
PAPER_DIR="paper/vit/SpecViT"
REMOTE_URL="git@github.com:ViskaWei/SpecViT-paper.git"
BRANCH="main"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# ==============================================================================
# Helper functions
# ==============================================================================
print_header() {
    echo ""
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "${BOLD}${BLUE}  $1${NC}"
    echo -e "${BOLD}${BLUE}════════════════════════════════════════════════════════════${NC}"
}

print_info() {
    echo -e "${CYAN}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1" >&2
}

print_step() {
    echo ""
    echo -e "${BOLD}▶ $1${NC}"
}

cleanup() {
    if [ -n "$TEMP_DIR" ] && [ -d "$TEMP_DIR" ]; then
        rm -rf "$TEMP_DIR"
    fi
}

trap cleanup EXIT

# ==============================================================================
# Main script
# ==============================================================================
print_header "SpecViT Pull Agent 🤖"

# Get the root of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    print_error "Not inside a git repository!"
    exit 1
}

cd "$REPO_ROOT"

echo ""
print_info "Repository: $REPO_ROOT"
print_info "Paper directory: $PAPER_DIR"
print_info "Remote: $REMOTE_URL"
echo ""

# Check if PAPER_DIR exists
if [ ! -d "$PAPER_DIR" ]; then
    print_error "Paper directory does not exist: $PAPER_DIR"
    exit 1
fi

# ==============================================================================
# Step 1: Check local status
# ==============================================================================
print_step "Step 1: Checking local status"

LOCAL_CHANGES=false
if ! git diff --quiet -- "$PAPER_DIR" || ! git diff --cached --quiet -- "$PAPER_DIR"; then
    LOCAL_CHANGES=true
    print_warning "You have uncommitted local changes in $PAPER_DIR"
    echo ""
    git diff --stat -- "$PAPER_DIR"
    echo ""
    echo -e "Options:"
    echo -e "  ${BOLD}c${NC} - Commit local changes first, then continue"
    echo -e "  ${BOLD}s${NC} - Stash local changes, pull, then restore"
    echo -e "  ${BOLD}d${NC} - Discard local changes and pull"
    echo -e "  ${BOLD}q${NC} - Quit"
    echo ""
    read -p "Choose action [c/s/d/q]: " -n 1 -r
    echo ""
    
    case $REPLY in
        c|C)
            print_info "Committing local changes..."
            git add "$PAPER_DIR"
            read -p "Commit message: " commit_msg
            git commit -m "${commit_msg:-'Update paper before pull'}"
            ;;
        s|S)
            print_info "Stashing local changes..."
            git stash push -m "paper-pull-agent-stash" -- "$PAPER_DIR"
            STASHED=true
            ;;
        d|D)
            print_warning "Discarding local changes..."
            git checkout -- "$PAPER_DIR"
            ;;
        *)
            echo "Aborted."
            exit 0
            ;;
    esac
fi

print_success "Local status: OK"

# ==============================================================================
# Step 2: Fetch remote changes
# ==============================================================================
print_step "Step 2: Fetching remote paper repository"

TEMP_DIR=$(mktemp -d)
print_info "Temp directory: $TEMP_DIR"

cd "$TEMP_DIR"
git clone --depth 1 "$REMOTE_URL" paper_repo 2>&1 | grep -v "^Cloning\|^remote:\|^Receiving\|^Resolving" || true

if [ ! -d "paper_repo" ]; then
    print_error "Failed to clone paper repository"
    exit 1
fi

cd paper_repo
REMOTE_COMMIT=$(git rev-parse --short HEAD)
REMOTE_DATE=$(git log -1 --format=%ci)
REMOTE_MSG=$(git log -1 --format=%s)

print_success "Fetched remote: $REMOTE_COMMIT"
print_info "Last commit: $REMOTE_MSG"
print_info "Date: $REMOTE_DATE"

# ==============================================================================
# Step 3: Compare with local
# ==============================================================================
print_step "Step 3: Comparing with local"

cd "$REPO_ROOT"

# Create a temp copy of local paper for comparison
LOCAL_TEMP="$TEMP_DIR/local_paper"
mkdir -p "$LOCAL_TEMP"
cp -r "$PAPER_DIR"/* "$LOCAL_TEMP/" 2>/dev/null || true

# Compare
DIFF_OUTPUT=$(diff -rq "$LOCAL_TEMP" "$TEMP_DIR/paper_repo" \
    --exclude='.git' \
    --exclude='.gitkeep' \
    --exclude='.gitignore' \
    --exclude='Makefile' \
    --exclude='*.aux' \
    --exclude='*.log' \
    --exclude='*.out' \
    --exclude='*.bbl' \
    --exclude='*.blg' \
    --exclude='*.synctex.gz' \
    2>/dev/null || true)

if [ -z "$DIFF_OUTPUT" ]; then
    print_success "No changes detected - local is up to date with remote"
    
    if [ "$STASHED" = true ]; then
        print_info "Restoring stashed changes..."
        git stash pop
    fi
    
    echo ""
    echo -e "${GREEN}✓ Nothing to pull${NC}"
    exit 0
fi

# Count changes
CHANGED_FILES=$(echo "$DIFF_OUTPUT" | wc -l)
print_warning "Found $CHANGED_FILES file difference(s)"

echo ""
echo -e "${BOLD}Changed files:${NC}"
echo "$DIFF_OUTPUT" | head -20 | while read line; do
    echo "  $line"
done

if [ "$CHANGED_FILES" -gt 20 ]; then
    echo "  ... and $((CHANGED_FILES - 20)) more"
fi

# ==============================================================================
# Step 4: Show detailed diff (optional)
# ==============================================================================
echo ""
read -p "Show detailed diff? [y/N]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_step "Detailed diff"
    diff -r "$LOCAL_TEMP" "$TEMP_DIR/paper_repo" \
        --exclude='.git' \
        --exclude='.gitkeep' \
        --exclude='.gitignore' \
        --exclude='Makefile' \
        --exclude='*.aux' \
        --exclude='*.log' \
        --exclude='*.bbl' \
        --exclude='*.blg' \
        -u 2>/dev/null | head -100 || true
    
    if [ "$CHANGED_FILES" -gt 5 ]; then
        echo ""
        echo "(Diff truncated, showing first 100 lines)"
    fi
fi

# ==============================================================================
# Step 5: Apply changes
# ==============================================================================
echo ""
read -p "Apply these changes? [Y/n]: " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Aborted."
    
    if [ "$STASHED" = true ]; then
        print_info "Restoring stashed changes..."
        git stash pop
    fi
    
    exit 0
fi

print_step "Step 5: Applying changes"

# Remove old files and copy new
rm -rf "$PAPER_DIR"/*
cp -r "$TEMP_DIR/paper_repo"/* "$PAPER_DIR/"
rm -rf "$PAPER_DIR/.git"

print_success "Files updated"

# ==============================================================================
# Step 6: Stage and commit
# ==============================================================================
print_step "Step 6: Staging changes"

git add "$PAPER_DIR"

echo ""
echo -e "${BOLD}Changes to be committed:${NC}"
git diff --cached --stat -- "$PAPER_DIR"

echo ""
read -p "Commit these changes? [Y/n]: " -n 1 -r
echo ""

if [[ ! $REPLY =~ ^[Nn]$ ]]; then
    # Generate commit message
    DEFAULT_MSG="Pull paper changes from Overleaf ($REMOTE_COMMIT)"
    read -p "Commit message [$DEFAULT_MSG]: " custom_msg
    COMMIT_MSG="${custom_msg:-$DEFAULT_MSG}"
    
    git commit -m "$COMMIT_MSG"
    print_success "Changes committed"
else
    print_info "Changes staged but not committed"
    echo "Run 'git commit' when ready"
fi

# ==============================================================================
# Step 7: Restore stash if needed
# ==============================================================================
if [ "$STASHED" = true ]; then
    print_step "Step 7: Restoring stashed changes"
    
    if git stash pop; then
        print_success "Stashed changes restored"
    else
        print_warning "Conflict while restoring stash"
        echo "Your stashed changes are preserved. Resolve conflicts manually."
        echo "Use 'git stash show -p' to view stashed changes"
    fi
fi

# ==============================================================================
# Summary
# ==============================================================================
print_header "Pull Complete ✓"

echo ""
echo -e "${GREEN}Summary:${NC}"
echo "  • Pulled from: $REMOTE_URL"
echo "  • Remote commit: $REMOTE_COMMIT"
echo "  • Files changed: $CHANGED_FILES"
echo ""
echo -e "${BOLD}Next steps:${NC}"
echo "  1. Review changes: git show --stat"
echo "  2. Push to main repo: git push origin main"
echo ""
