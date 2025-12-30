#!/usr/bin/env python
"""Generate magNoise.png: Flux error vs wavelength for different magnitudes."""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os

# Data path
DATA_DIR = "/datascope/subaru/user/swei20/data/bosz50000/z0/mag205_225_lowT_1M"
OUTPUT_DIR = "/home/swei20/Physics_Informed_AI/paper/vit/SpecViT/figs"

def load_sample_data(n_samples=1000):
    """Load sample spectra for each magnitude bin."""
    data_path = os.path.join(DATA_DIR, "test_10k/dataset.h5")
    
    with h5py.File(data_path, 'r') as f:
        wave = f['spectrumdataset/wave'][()]
        flux = f['dataset/arrays/flux/value'][:n_samples]
        error = f['dataset/arrays/error/value'][:n_samples]
    
    import pandas as pd
    df = pd.read_hdf(data_path)[:n_samples]
    mag = df['mag'].values
    
    return wave, flux, error, mag

def main():
    print("Loading data...")
    wave, flux, error, mag = load_sample_data()
    
    # Define magnitude bins
    mag_bins = [(20.5, 21.0), (21.0, 21.5), (21.5, 22.0), (22.0, 22.5)]
    colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D']
    labels = ['20.5-21.0', '21.0-21.5', '21.5-22.0', '22.0-22.5']
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    for (mag_min, mag_max), color, label in zip(mag_bins, colors, labels):
        mask = (mag >= mag_min) & (mag < mag_max)
        if mask.sum() == 0:
            continue
        
        # Average error for this magnitude bin
        err_mean = error[mask].mean(axis=0)
        flux_mean = flux[mask].mean(axis=0)
        
        # Absolute error
        axes[0].plot(wave, err_mean, color=color, label=f'mag {label}', alpha=0.8, lw=1.5)
        
        # Relative error
        rel_err = err_mean / np.maximum(flux_mean, 1e-20)
        axes[1].plot(wave, rel_err, color=color, label=f'mag {label}', alpha=0.8, lw=1.5)
    
    # Left panel: Absolute error
    axes[0].set_xlabel('Wavelength (Å)', fontsize=12)
    axes[0].set_ylabel(r'$\sigma_F$ (erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$)', fontsize=12)
    axes[0].set_title('Absolute Flux Error', fontsize=13)
    axes[0].legend(loc='upper right', fontsize=10)
    axes[0].set_xlim(wave.min(), wave.max())
    axes[0].ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    # Right panel: Relative error
    axes[1].set_xlabel('Wavelength (Å)', fontsize=12)
    axes[1].set_ylabel(r'$\sigma_F / F$', fontsize=12)
    axes[1].set_title('Relative Flux Error', fontsize=13)
    axes[1].legend(loc='upper right', fontsize=10)
    axes[1].set_xlim(wave.min(), wave.max())
    axes[1].set_ylim(0, None)
    
    plt.tight_layout()
    
    # Save
    output_path = os.path.join(OUTPUT_DIR, "magNoise.png")
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()

if __name__ == "__main__":
    main()
