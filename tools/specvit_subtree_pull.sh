#!/bin/bash
# ==============================================================================
# SpecViT Paper Pull Script
# Pull changes from the standalone paper GitHub repository back to main repo
# Use this to sync Overleaf edits back to the main repository
# Works without git-subtree
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

print_warning() {
    echo "[WARNING] $1"
}

# ==============================================================================
# Main script
# ==============================================================================
print_header "SpecViT Paper Pull"

# Get the root of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    print_error "Not inside a git repository!"
    exit 1
}

cd "$REPO_ROOT"

print_info "Repository root: $REPO_ROOT"
print_info "Paper directory: $PAPER_DIR"
print_info "Remote: $REMOTE_URL"
print_info "Branch: $BRANCH"
echo ""

# Check if PAPER_DIR exists
if [ ! -d "$PAPER_DIR" ]; then
    print_error "Paper directory does not exist: $PAPER_DIR"
    exit 1
fi

# Check for uncommitted changes
if ! git diff --quiet -- "$PAPER_DIR" || ! git diff --cached --quiet -- "$PAPER_DIR"; then
    print_warning "You have uncommitted changes in $PAPER_DIR"
    echo ""
    read -p "Continue and overwrite local changes? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted. Please commit or stash your changes first."
        exit 1
    fi
fi

# Create temp directory
TEMP_DIR=$(mktemp -d)
print_info "Temp directory: $TEMP_DIR"

# Cleanup on exit
cleanup() {
    rm -rf "$TEMP_DIR"
}
trap cleanup EXIT

print_header "Fetching Paper Repository"

# Clone the paper repo
cd "$TEMP_DIR"
git clone "$REMOTE_URL" paper_repo || {
    print_error "Failed to clone paper repository"
    echo "Make sure the repository exists and you have access."
    exit 1
}

cd paper_repo
PAPER_COMMIT=$(git rev-parse --short HEAD)
PAPER_MSG=$(git log -1 --format=%s)
print_info "Paper repo HEAD: $PAPER_COMMIT - $PAPER_MSG"

print_header "Syncing Changes"

# Remove old paper files (except .git stuff that shouldn't exist)
cd "$REPO_ROOT"
rm -rf "$PAPER_DIR"/*

# Copy new paper files
cp -r "$TEMP_DIR/paper_repo"/* "$PAPER_DIR/"

# Remove .git from copied content if exists
rm -rf "$PAPER_DIR/.git"

# Stage changes
git add "$PAPER_DIR"

# Check if there are changes
if git diff --cached --quiet; then
    print_info "No changes detected (local is up to date)"
    print_success "Done - nothing to update"
    exit 0
fi

# Show diff summary
print_header "Changes Summary"
git diff --cached --stat -- "$PAPER_DIR"

echo ""
read -p "Commit these changes? (Y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Changes staged but not committed."
    echo "Run 'git commit' when ready, or 'git checkout -- $PAPER_DIR' to discard."
    exit 0
fi

# Commit
git commit -m "Pull paper changes from SpecViT-paper ($PAPER_COMMIT)"

print_header "Pull Complete"
print_success "Successfully pulled changes from paper repository"
echo ""
echo "Changes from paper repo ($PAPER_COMMIT) have been committed."
echo ""
echo "Next steps:"
echo "  1. Review changes: git show --stat"
echo "  2. Push to main repo: git push origin main"
