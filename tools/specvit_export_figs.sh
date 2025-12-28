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
print_header "SpecViT Figure Export"

# Get the root of the git repository
REPO_ROOT=$(git rev-parse --show-toplevel 2>/dev/null) || {
    print_error "Not inside a git repository!"
    exit 1
}

print_info "Repository root: $REPO_ROOT"
print_info "Source directory: $SRC_DIR"
print_info "Destination directory: $DST_DIR"
echo ""

# Change to repo root
cd "$REPO_ROOT"

# Check/create source directory
if [ ! -d "$SRC_DIR" ]; then
    print_warning "Source directory does not exist. Creating: $SRC_DIR"
    mkdir -p "$SRC_DIR"
    
    # Write README for the source directory
    cat > "$SRC_DIR/README.md" << 'EOF'
# Figure Source Directory

This directory stores **source figures and generated products** for the SpecViT paper.

## Usage

1. Place your figure source files (scripts, data, intermediate products) here
2. Generate publication-ready figures (PDF/PNG)
3. Run `tools/specvit_export_figs.sh` to copy final figures to `paper/vit/SpecViT/figs/`

## Supported formats for export

- `.pdf` (preferred for vector graphics)
- `.png` (for raster images)
- `.jpg` / `.jpeg`

## Important

- Only `.pdf`, `.png`, `.jpg` files will be copied to the paper figs/ directory
- Large data files should be kept here, NOT in the paper directory
- The paper directory must remain lightweight for Overleaf sync
EOF
    print_info "Created README in source directory"
fi

# Check destination directory
if [ ! -d "$DST_DIR" ]; then
    print_warning "Destination directory does not exist. Creating: $DST_DIR"
    mkdir -p "$DST_DIR"
fi

# Find and copy figures
print_header "Copying Figures"

COPY_COUNT=0
echo ""
echo "Files to copy:"
echo "---"

# Find all PDF, PNG, JPG files in source directory
for ext in pdf png jpg jpeg; do
    for file in "$SRC_DIR"/*.$ext "$SRC_DIR"/*."${ext^^}" 2>/dev/null; do
        if [ -f "$file" ]; then
            filename=$(basename "$file")
            echo "  $filename"
            cp "$file" "$DST_DIR/"
            ((COPY_COUNT++))
        fi
    done
done

echo "---"
echo ""

if [ $COPY_COUNT -eq 0 ]; then
    print_warning "No figures found to copy!"
    echo ""
    echo "Please add figures to: $SRC_DIR"
    echo "Supported formats: .pdf, .png, .jpg"
    echo ""
    echo "Example workflow:"
    echo "  1. Generate figures in your experiment scripts"
    echo "  2. Save to: $SRC_DIR/"
    echo "  3. Run this script again"
else
    print_success "Copied $COPY_COUNT figure(s) to $DST_DIR"
fi

echo ""
print_header "Important Reminders"
echo ""
echo "⚠️  DO NOT use symlinks in figs/ - Overleaf Git sync is incompatible!"
echo ""
echo "✓  Figures are COPIED (not linked)"
echo "✓  This ensures Overleaf can access all files"
echo ""
echo "After exporting figures:"
echo "  1. Commit the changes: git add $DST_DIR && git commit -m 'Update figures'"
echo "  2. Push to paper repo: tools/specvit_subtree_push.sh"
echo "  3. Pull in Overleaf: Menu → GitHub → Pull"
