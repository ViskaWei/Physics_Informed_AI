#!/bin/bash
# ==============================================================================
# SpecViT Subtree Pull Script
# Pull changes from the standalone paper GitHub repository back to main repo
# Use this to sync Overleaf edits back to the main repository
# ==============================================================================

set -e

# ==============================================================================
# Configuration (modify these as needed)
# ==============================================================================
PAPER_DIR="paper/vit/SpecViT"
REMOTE_NAME="specvit-paper"
REMOTE_URL="<FILL_ME_GITHUB_URL>"
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
print_header "SpecViT Subtree Pull"

# Get the root of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    print_error "Not inside a git repository!"
    exit 1
}

print_info "Repository root: $REPO_ROOT"
print_info "Paper directory: $PAPER_DIR"
print_info "Remote name: $REMOTE_NAME"
print_info "Remote URL: $REMOTE_URL"
print_info "Branch: $BRANCH"
echo ""

# Change to repo root
cd "$REPO_ROOT"
print_info "Working directory: $(pwd)"

# Check if PAPER_DIR exists
if [ ! -d "$PAPER_DIR" ]; then
    print_error "Paper directory does not exist: $PAPER_DIR"
    exit 1
fi

print_info "Paper directory exists: ✓"

# Check if REMOTE_URL is still placeholder
if [ "$REMOTE_URL" = "<FILL_ME_GITHUB_URL>" ]; then
    print_error "REMOTE_URL is not configured!"
    echo ""
    echo "Please edit this script and replace <FILL_ME_GITHUB_URL> with your actual GitHub repository URL."
    exit 1
fi

# Check if remote exists
if ! git remote get-url "$REMOTE_NAME" &>/dev/null; then
    print_error "Remote '$REMOTE_NAME' does not exist!"
    echo ""
    echo "Please add the remote first by running:"
    echo ""
    echo "  git remote add $REMOTE_NAME $REMOTE_URL"
    echo ""
    exit 1
fi

print_info "Remote '$REMOTE_NAME' exists: ✓"

# Show current remote URL
CURRENT_URL=$(git remote get-url "$REMOTE_NAME")
print_info "Remote URL: $CURRENT_URL"
echo ""

# Check for uncommitted changes
if ! git diff-index --quiet HEAD --; then
    print_warning "You have uncommitted changes in your working directory."
    echo "It's recommended to commit or stash changes before pulling."
    echo ""
    read -p "Continue anyway? (y/N) " -n 1 -r
    echo ""
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 1
    fi
fi

# Fetch from remote first
print_header "Fetching from Remote"
echo "Command: git fetch $REMOTE_NAME"
git fetch "$REMOTE_NAME"
echo ""

# Execute subtree pull with --squash
print_header "Executing Subtree Pull"
echo "Command: git subtree pull --prefix $PAPER_DIR $REMOTE_NAME $BRANCH --squash"
echo ""
echo "Note: Using --squash to consolidate remote commits into a single merge commit."
echo ""

git subtree pull --prefix "$PAPER_DIR" "$REMOTE_NAME" "$BRANCH" --squash

print_header "Pull Complete"
print_success "Successfully pulled changes from $REMOTE_NAME/$BRANCH into $PAPER_DIR"
echo ""
echo "What happened:"
echo "  - Changes from the standalone paper repo have been merged into $PAPER_DIR"
echo "  - A squash merge commit was created"
echo ""
echo "Next steps:"
echo "  1. Review the changes: git log --oneline -5"
echo "  2. Check the diff: git diff HEAD~1"
echo "  3. Push to main repo if satisfied: git push origin main"
echo ""
echo "Conflict handling:"
echo "  If Overleaf created a new branch due to conflicts, you need to:"
echo "  1. Merge that branch into main on GitHub"
echo "  2. Run this script again"
