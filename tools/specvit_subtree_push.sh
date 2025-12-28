#!/bin/bash
# ==============================================================================
# SpecViT Subtree Push Script
# Push paper/vit/SpecViT to the standalone paper GitHub repository
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

# ==============================================================================
# Main script
# ==============================================================================
print_header "SpecViT Subtree Push"

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
    echo ""
    echo "Please ensure the paper directory exists with LaTeX files."
    echo "Expected structure:"
    echo "  $PAPER_DIR/"
    echo "  ├── main.tex"
    echo "  ├── sections/"
    echo "  ├── refs.bib"
    echo "  └── figs/"
    exit 1
fi

print_info "Paper directory exists: ✓"

# Check if REMOTE_URL is still placeholder
if [ "$REMOTE_URL" = "<FILL_ME_GITHUB_URL>" ]; then
    print_error "REMOTE_URL is not configured!"
    echo ""
    echo "Please edit this script and replace <FILL_ME_GITHUB_URL> with your actual GitHub repository URL."
    echo "Example: https://github.com/YourUsername/physics_informed_ai-specvit-paper.git"
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
    echo "Then run this script again."
    exit 1
fi

print_info "Remote '$REMOTE_NAME' exists: ✓"

# Show current remote URL
CURRENT_URL=$(git remote get-url "$REMOTE_NAME")
print_info "Remote URL: $CURRENT_URL"
echo ""

# Execute subtree push
print_header "Executing Subtree Push"
echo "Command: git subtree push --prefix $PAPER_DIR $REMOTE_NAME $BRANCH"
echo ""

git subtree push --prefix "$PAPER_DIR" "$REMOTE_NAME" "$BRANCH"

print_header "Push Complete"
print_success "Successfully pushed $PAPER_DIR to $REMOTE_NAME/$BRANCH"
echo ""
echo "Next steps:"
echo "  1. Go to your GitHub repository to verify the push"
echo "  2. In Overleaf: Menu → GitHub → Pull (to sync changes)"
echo ""
echo "Note: Overleaf GitHub sync is NOT automatic. You must manually Pull in Overleaf."
