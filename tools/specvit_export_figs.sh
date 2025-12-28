#!/bin/bash
# ==============================================================================
# SpecViT Figure Export Script
# Copy publication-ready figures from source directory to paper figs/
# ==============================================================================

set -e

# ==============================================================================
# Configuration
# ==============================================================================
PAPER_DIR="paper/vit/SpecViT"
SRC_DIR="assets/figures/specvit"
DST_DIR="${PAPER_DIR}/figs"

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

print_warning() {
    echo "[WARNING] $1"
}

print_success() {
    echo "[SUCCESS] $1"
}

# ==============================================================================
# Main script
# ==============================================================================
print_header "SpecViT Figure Export"

# Get the root of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    echo "[ERROR] Not inside a git repository!"
    exit 1
}

cd "$REPO_ROOT"

print_info "Source: $SRC_DIR"
print_info "Destination: $DST_DIR"
echo ""

# Check/create source directory
if [ ! -d "$SRC_DIR" ]; then
    print_warning "Source directory does not exist. Creating: $SRC_DIR"
    mkdir -p "$SRC_DIR"
    
    cat > "$SRC_DIR/README.md" << 'EOF'
# Figure Source Directory

Place your figure source files here. Run `tools/specvit_export_figs.sh` to copy 
PDF/PNG/JPG files to the paper figs/ directory.
EOF
    print_info "Created source directory with README"
fi

# Check destination directory
mkdir -p "$DST_DIR"

# Find and copy figures
print_header "Copying Figures"

COPY_COUNT=0

for file in "$SRC_DIR"/*.pdf "$SRC_DIR"/*.PDF \
            "$SRC_DIR"/*.png "$SRC_DIR"/*.PNG \
            "$SRC_DIR"/*.jpg "$SRC_DIR"/*.JPG \
            "$SRC_DIR"/*.jpeg "$SRC_DIR"/*.JPEG 2>/dev/null; do
    if [ -f "$file" ]; then
        filename=$(basename "$file")
        echo "  → $filename"
        cp "$file" "$DST_DIR/"
        ((COPY_COUNT++))
    fi
done

echo ""

if [ $COPY_COUNT -eq 0 ]; then
    print_warning "No figures found in $SRC_DIR"
    echo ""
    echo "Add PDF/PNG/JPG figures to: $SRC_DIR"
    echo "Then run this script again."
else
    print_success "Copied $COPY_COUNT figure(s)"
    echo ""
    echo "Next steps:"
    echo "  git add $DST_DIR"
    echo "  git commit -m 'Update figures'"
    echo "  ./tools/specvit_subtree_push.sh"
fi

echo ""
echo "⚠️  Reminder: DO NOT use symlinks - Overleaf doesn't support them!"
