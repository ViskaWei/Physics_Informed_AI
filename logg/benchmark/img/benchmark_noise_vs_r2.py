import matplotlib.pyplot as plt
import numpy as np

# Data from benchmark_hub table
# noise_levels = [0, 0.1, 0.2, 0.5, 1.0, 2.0]
# SNR (L2) ≈ 3.3 / noise_level (from snr_noise_level_table.md)
# Skip noise=0 (infinite SNR), use noise 0.1 to 2.0
noise_levels = [0.1, 0.2, 0.5, 1.0, 2.0]
snr_values = [33.4, 16.7, 6.7, 3.3, 1.65]  # L2 SNR

# Ridge data (excluding noise=0)
ridge_32k = [0.900, 0.862, 0.670, 0.458, 0.221]
ridge_100k = [0.9174, 0.8413, 0.6674, 0.4687, 0.2536]

# LightGBM data (excluding noise=0)
lgb_32k = [0.9616, 0.9045, 0.7393, 0.5361, 0.2679]
lgb_100k = [0.9720, 0.9318, 0.7573, 0.5582, 0.3038]

# Create figure with space for table
fig, (ax, ax_table) = plt.subplots(2, 1, figsize=(6, 6.5), 
                                    gridspec_kw={'height_ratios': [3, 1.2]})

# Plot lines with markers (SNR on x-axis, reversed order for left-to-right increasing)
ax.plot(snr_values, ridge_32k, 'o-', label='Ridge 32k', color='#2E86AB', linewidth=2, markersize=7)
ax.plot(snr_values, ridge_100k, 's--', label='Ridge 100k', color='#2E86AB', linewidth=2, markersize=7, alpha=0.7)
ax.plot(snr_values, lgb_32k, 'o-', label='LightGBM 32k', color='#E94F37', linewidth=2, markersize=7)
ax.plot(snr_values, lgb_100k, 's--', label='LightGBM 100k', color='#E94F37', linewidth=2, markersize=7, alpha=0.7)

# Annotate σ=0.2 (SNR=16.7) for Ridge 32k and LightGBM 32k
snr_02 = 16.7
ridge_32k_02 = 0.862
lgb_32k_02 = 0.9045

# Ridge 32k annotation
ax.annotate(f'{ridge_32k_02:.3f}', xy=(snr_02, ridge_32k_02), 
            xytext=(snr_02 + 4, ridge_32k_02 - 0.06),
            fontsize=10, color='#2E86AB', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#2E86AB', lw=1.2))

# LightGBM 32k annotation
ax.annotate(f'{lgb_32k_02:.3f}', xy=(snr_02, lgb_32k_02), 
            xytext=(snr_02 + 4, lgb_32k_02 + 0.08),
            fontsize=10, color='#E94F37', fontweight='bold',
            arrowprops=dict(arrowstyle='->', color='#E94F37', lw=1.2))

# Add vertical line at σ=0.2
ax.axvline(x=snr_02, color='gray', linestyle=':', alpha=0.5, linewidth=1.5)

# Labels and title
ax.set_xlabel(r'SNR = $\Vert$flux$\Vert$ / ($\Vert$error$\Vert$ × σ)', fontsize=11)
ax.set_ylabel('R²', fontsize=12)
ax.set_title('Ridge vs LightGBM: SNR vs Performance', fontsize=13, fontweight='bold')

# Legend
ax.legend(loc='lower right', fontsize=10)

# Grid
ax.grid(True, alpha=0.3, linestyle='--')

# Set axis limits
ax.set_xlim(0, 36)
ax.set_ylim(0.15, 1.02)

# Add noise level as secondary labels on top
ax2 = ax.twiny()
ax2.set_xlim(ax.get_xlim())
noise_ticks = [1.65, 3.3, 6.7, 16.7, 33.4]
noise_labels = ['σ=2', 'σ=1', 'σ=0.5', 'σ=0.2', 'σ=0.1']
ax2.set_xticks(noise_ticks)
ax2.set_xticklabels(noise_labels, fontsize=9, color='gray')
ax2.set_xlabel('Noise Level', fontsize=10, color='gray')

# === Create table below ===
ax_table.axis('off')

# Table data
table_data = [
    ['σ=0.1\n(SNR≈33)', '0.900', '0.917', '0.962', '0.972'],
    ['σ=0.2\n(SNR≈17)', '0.862', '0.841', '0.905', '0.932'],
    ['σ=0.5\n(SNR≈7)', '0.670', '0.667', '0.739', '0.757'],
    ['σ=1.0\n(SNR≈3)', '0.458', '0.469', '0.536', '0.558'],
    ['σ=2.0\n(SNR≈2)', '0.221', '0.254', '0.268', '0.304'],
]
col_labels = ['Noise', 'Ridge\n32k', 'Ridge\n100k', 'LGB\n32k', 'LGB\n100k']

table = ax_table.table(cellText=table_data, colLabels=col_labels,
                       loc='center', cellLoc='center',
                       colColours=['#f0f0f0']*5)
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1.0, 1.4)

# Style header
for i in range(5):
    table[(0, i)].set_fontsize(9)
    table[(0, i)].set_text_props(weight='bold')

# Tight layout
plt.tight_layout()

# Save
plt.savefig('/home/swei20/Physics_Informed_AI/logg/benchmark/img/benchmark_snr_vs_r2.png', dpi=150, bbox_inches='tight')
plt.show()

print("✅ 图表已保存到: logg/benchmark/img/benchmark_snr_vs_r2.png")
