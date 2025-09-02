#!/usr/bin/env python3
"""
Hit efficiency comparison — grouped bar charts
with material labels for IV and OV.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# ─── Helper: Sum contributions ─────────────────────────────
def sum_contributions(contribs, errors, include_po210, po210_indices):
    total_val = 0.0
    total_var = 0.0
    for i, (v, e) in enumerate(zip(contribs, errors)):
        if not include_po210 and i in po210_indices:
            continue
        total_val += v
        total_var += e**2
    return total_val, math.sqrt(total_var)

# ─── Data: Your results ────────────────────────────────────
your_data = {
    "HFE":  ([2.61E-02], [5.86E-03]),
    "IV":   ([5.62E-05], [3.41E-05]),
    "OV":   ([8.15E-05], [3.75E-05]),
    "Water":([1.23E-06], [4.98E-04]),
}

# ─── Data: nEXO 2025 ───────────────────────────────────────
nexo2025_data = {
    "HFE": (
        [4.584E-3, 5.410E-4, 3.264E-3, 1.037E-5, 1.068E-4, 8.956E-3, 1.202E-2],
        [5.630E-3, 6.644E-4, 4.619E-4, 7.796E-7, 8.030E-6, 6.732E-4, 9.036E-4],
    ),
    "IV": (
        [2.831E-4, 1.397E-4],
        [3.477E-4, 6.547E-5],
    ),
    "OV": (
        [3.524E-4],
        [1.688E-4],
    ),
    "Water": (
        [0.0],
        [0.0],
    ),
}
nexo2025_po210_indices = {"HFE": [2], "IV": [], "OV": [], "Water": []}

# ─── Data: nEXO 2022 ───────────────────────────────────────
nexo2022_data = {
    "HFE": (
        [1.955E-2, 6.350E-3, 2.131E-2, 3.264E-3],
        [2.401E-2, 7.799E-3, 1.464E-3, 4.619E-4],
    ),
    "IV": (
        [9.217E-3, 1.014E-3, 1.241E-2, 5.316E-4],
        [3.015E-3, 2.193E-3, 7.346E-3, 1.174E-4],
    ),
    "OV": (
        [1.730E-3, 1.994E-2, 7.471E-3],
        [3.576E-3, 1.305E-2, 4.631E-3],
    ),
    "Water": (
        [0.0],
        [0.0],
    ),
}
nexo2022_po210_indices = {"HFE": [3], "IV": [3], "OV": [], "Water": []}

# ─── Compute totals ────────────────────────────────────────
def compute_totals(include_po210):
    components = ["HFE", "IV", "OV", "Water"]
    you_vals, you_errs = [], []
    n25_vals, n25_errs = [], []
    n22_vals, n22_errs = [], []

    for comp in components:
        yv, ye = sum_contributions(*your_data[comp], True, [])
        n25v, n25e = sum_contributions(
            *nexo2025_data[comp], include_po210, nexo2025_po210_indices[comp]
        )
        n22v, n22e = sum_contributions(
            *nexo2022_data[comp], include_po210, nexo2022_po210_indices[comp]
        )

        you_vals.append(yv); you_errs.append(ye)
        n25_vals.append(n25v); n25_errs.append(n25e)
        n22_vals.append(n22v); n22_errs.append(n22e)

    return components, you_vals, you_errs, n25_vals, n25_errs, n22_vals, n22_errs

# ─── Plotting function ─────────────────────────────────────
def plot_comparison(include_po210):
    comps, y_vals, y_errs, n25_vals, n25_errs, n22_vals, n22_errs = compute_totals(include_po210)

    # Adjust labels for IV and OV
    dataset_labels = {
        "You": [comp + (" (Nickel)" if comp in ["IV", "OV"] else "") for comp in comps],
        "nEXO 2025": [comp + (" (Nickel)" if comp in ["IV", "OV"] else "") for comp in comps],
        "nEXO 2022": [comp + (" (Carbon Fiber)" if comp in ["IV", "OV"] else "") for comp in comps],
    }

    x = np.arange(len(comps))
    width = 0.25

    fig, ax = plt.subplots()
    ax.bar(x - width, y_vals, width, yerr=y_errs, capsize=4, label="You")
    ax.bar(x,         n25_vals, width, yerr=n25_errs, capsize=4, label="nEXO 2025")
    ax.bar(x + width, n22_vals, width, yerr=n22_errs, capsize=4, label="nEXO 2022")

    ax.set_xticks(x, comps)
    ax.set_ylabel("Hit efficiency (per parent decay)")
    ax.set_yscale("log")
    ax.set_title(f"Hit efficiency comparison — {'with' if include_po210 else 'without'} Po-210")
    ax.legend()

    # Add material names as x-axis annotations
    for i, comp in enumerate(comps):
        ax.text(i - width, y_vals[i], dataset_labels["You"][i], ha='center', va='bottom', fontsize=8, rotation=90)
        ax.text(i,         n25_vals[i], dataset_labels["nEXO 2025"][i], ha='center', va='bottom', fontsize=8, rotation=90)
        ax.text(i + width, n22_vals[i], dataset_labels["nEXO 2022"][i], ha='center', va='bottom', fontsize=8, rotation=90)

    plt.tight_layout()
    return fig

# ─── Make both plots ───────────────────────────────────────
fig1 = plot_comparison(include_po210=True)
fig2 = plot_comparison(include_po210=False)

plt.show()
