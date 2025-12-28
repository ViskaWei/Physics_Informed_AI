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

## Naming conventions (suggested)

- `fig1_architecture.pdf` - Main architecture figure
- `fig2_r2_vs_snr.pdf` - R² vs SNR comparison
- `fig3_attention_map.png` - Attention visualization
- `table1_main_results.pdf` - Main results (if as figure)
