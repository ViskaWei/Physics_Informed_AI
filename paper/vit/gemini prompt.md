# 1. SpecViT Pipeline
Create a clean vector infographic titled “SpecViT Pipeline (1D Spectra)” with a left-to-right pipeline of 6 rounded blocks and arrow connectors.

Blocks:
(1) Input Spectrum (length 4096) + optional Error/SNR
(2) Noise Injection (train-time, heteroscedastic)
(3) Tokenization / Patch Embedding (Patch size=16 → 256 tokens) with two options shown inside the block:
    Option A: C1D Conv1D embedding
    Option B: Sliding Window embedding
Add a small inset under Block (3) illustrating patching along the wavelength axis (a simple 1D curve with vertical patch boundaries and one highlighted patch).

IMPORTANT: Make Blocks (4), (5), (6) primarily graphical (icons/mini-diagrams), not text paragraphs.
(4) Add Learned Positional Embedding + [CLS]:
    Draw a token sequence as small rectangles; the first token is a distinct [CLS] tile.
    Show a sinusoidal/wave line or “embedding vector” motif above tokens, plus a “+” symbol indicating addition of positional embedding.
(5) Transformer Encoder ×6 (hidden=256, heads=8):
    Depict a stack of 6 repeated encoder layers (layer blocks).
    Inside each layer, use a simple multi-head self-attention icon (Q/K/V merging to attention) + a feed-forward icon.
    Represent 8 heads visually (e.g., eight small dots or mini-head bars). Use a small “×6” mark near the stack.
(6) Regression Head (MLP) → output log(g):
    Draw a small fully-connected network diagram (nodes and connections) ending in a single output node.
    The output arrow points to “log(g)”.

Style:
Flat vector, white background, consistent line weight, blue outlines, subtle gray fills, clean sans-serif typography.
English labels only. Keep text minimal: short block titles + key numbers only (no long sentences).
Output as SVG or PDF vector.
