#!/bin/bash
# ==============================================================================
# SpecViT Paper Push Script
# Push paper/vit/SpecViT to the standalone paper GitHub repository
# Works without git-subtree by using a temporary clone approach
# ==============================================================================

set -e

# ==============================================================================
# Configuration
# ==============================================================================
PAPER_DIR="paper/vit/SpecViT"
REMOTE_NAME="specvit-paper"
REMOTE_URL="git@github.com:ViskaWei/SpecViT-paper.git"
BRANCH="main"

# ==============================================================================
# Helper functions
# ==============================================================================
print_header() {
    echo ""
    echo "============================================================"
    echo "$1"
    echo "============================================================"
}

print_info() {
    echo "[INFO] $1"
}

print_error() {
    echo "[ERROR] $1" >&2
}

print_success() {
    echo "[SUCCESS] $1"
}

# ==============================================================================
# Main script
# ==============================================================================
print_header "SpecViT Paper Push"

# Get the root of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    print_error "Not inside a git repository!"
    exit 1
}

cd "$REPO_ROOT"

print_info "Repository root: $REPO_ROOT"
print_info "Paper directory: $PAPER_DIR"
print_info "Remote: $REMOTE_NAME"
print_info "Branch: $BRANCH"
echo ""

# Check if PAPER_DIR exists
if [ ! -d "$PAPER_DIR" ]; then
    print_error "Paper directory does not exist: $PAPER_DIR"
    exit 1
fi

print_info "Paper directory exists: ✓"

# Check for uncommitted changes in PAPER_DIR
if ! git diff --quiet -- "$PAPER_DIR" || ! git diff --cached --quiet -- "$PAPER_DIR"; then
    print_error "You have uncommitted changes in $PAPER_DIR"
    echo ""
    echo "Please commit your changes first:"
    echo "  git add $PAPER_DIR"
    echo "  git commit -m 'Update paper'"
    exit 1
fi

print_info "No uncommitted changes: ✓"

# Create temp directory
TEMP_DIR=$(mktemp -d)
print_info "Temp directory: $TEMP_DIR"

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

print_header "Preparing Paper Repository"

# Clone the paper repo (or init if empty)
cd "$TEMP_DIR"
if git clone "$REMOTE_URL" paper_repo 2>/dev/null; then
    print_info "Cloned existing paper repository"
    cd paper_repo
    # Clear all files except .git
    find . -maxdepth 1 ! -name '.git' ! -name '.' -exec rm -rf {} \;
else
    print_info "Initializing new paper repository"
    mkdir paper_repo
    cd paper_repo
    git init
    git remote add origin "$REMOTE_URL"
fi

# Copy paper contents
print_info "Copying paper files..."
cp -r "$REPO_ROOT/$PAPER_DIR"/* .

# Stage all files
git add -A

# Check if there are changes to commit
if git diff --cached --quiet; then
    print_info "No changes to push (paper repo is up to date)"
    print_success "Done - nothing to push"
    exit 0
fi

# Get commit info from main repo
MAIN_COMMIT=$(cd "$REPO_ROOT" && git rev-parse --short HEAD)
MAIN_MSG=$(cd "$REPO_ROOT" && git log -1 --format=%s)

# Commit
print_info "Committing changes..."
git commit -m "Sync from main repo (${MAIN_COMMIT}): ${MAIN_MSG}"

# Push
print_header "Pushing to GitHub"
echo "Command: git push origin $BRANCH"
echo ""

if git push origin "$BRANCH" 2>&1; then
    :
else
    # If push fails, try force push for first time or branch mismatch
    print_info "Regular push failed, trying with --force (may be first push)..."
    git push origin "$BRANCH" --force
fi

print_header "Push Complete"
print_success "Successfully pushed $PAPER_DIR to $REMOTE_URL"
echo ""
echo "Commit: $(git rev-parse --short HEAD)"
echo ""
echo "Next steps:"
echo "  1. Verify on GitHub: https://github.com/ViskaWei/SpecViT-paper"
echo "  2. In Overleaf: Menu → GitHub → Pull"
echo ""
echo "Note: Overleaf GitHub sync is NOT automatic - you must manually Pull."
