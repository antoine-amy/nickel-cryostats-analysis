#!/usr/bin/env python3
"""
Simplified plotting and comparison of selected event counts with asymmetric errors.
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants and Data
RADII = ["1026mm", "1226mm", "1510mm", "1691mm"]
ISOTOPES = ["Th232", "U238"]

# Reconstructions: label -> isotope -> values
DATASETS = {
    "My new recon code IC": {
        "Th232": [251.58, 60.0198, 6.1098, 2.1564],
        "U238":  [170.0,    33.0,    1.0,    3.0],
    },
    "Old recon code IC": {
        "Th232": [256.6116, 63.9732, 6.4692, 2.8752],
        "U238":  [161.0,    36.0,    3.0,    2.0],
    },
    "Old recon code OC": {
        "Th232": [264.159,  60.7386, 7.5474, 2.5158],
        "U238":  [165.0,    32.0,    6.0,    0.0],
    },
}

# Gehrels 1986 asymmetric Poisson errors
def asym_errors(n):
    if n == 0:
        return 0.0, 1.841
    up = np.sqrt(n + 0.75) + 1
    lo = np.sqrt(n - 0.25) if n > 0.25 else 0.0
    return lo, up

# Plotting
def plot_counts():
    width = 0.25
    x = np.arange(len(RADII))
    fig, axes = plt.subplots(1, len(ISOTOPES), figsize=(12, 6))

    for ax, isotope in zip(axes, ISOTOPES):
        for k, (label, data) in enumerate(DATASETS.items()):
            vals = np.array(data[isotope])
            lo, up = zip(*[asym_errors(v) for v in vals])
            positions = x + k * width
            ax.bar(positions, vals, width, label=label)
            ax.errorbar(
                positions, vals, yerr=[lo, up], fmt='none', capsize=3, color='black'
            )

        ax.set_xbound(x.min() - width, x.max() + width * len(DATASETS))
        ax.set_xticks(x + width)
        ax.set_xticklabels(RADII, rotation=45)
        ax.set_yscale('log')
        ax.set_title(f"{isotope} Selected Events")
        ax.set_ylabel("Selected Events")
        ax.grid(True, which='both', ls='--', alpha=0.7)
        ax.legend()

    fig.suptitle("Number of Selected Background Events from IC (1e9 events)")
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()

# Text output
def print_comparisons():
    print("\nNumerical Comparisons (with asymmetric errors):")
    for isotope in ISOTOPES:
        print(f"\n{isotope}:")
        for i, radius in enumerate(RADII):
            print(f"  {radius}:")
            for label, data in DATASETS.items():
                val = data[isotope][i]
                lo, up = asym_errors(val)
                print(f"    {label}: {val:.2f} (+{up:.2f}/-{lo:.2f})")

if __name__ == '__main__':
    plot_counts()
    print_comparisons()
