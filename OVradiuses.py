#!/usr/bin/env python3
"""
Compare OV cryostat background for two OV radii at fixed IV radius (1226 mm).

Goal: show that the OV contribution is (nearly) unchanged between:
  - OV radius = 1765 mm: 1.47e-3 ± 6.02e-4 counts / y / 2t / ROI
  - OV radius = 2230 mm (baseline): 1.58e-3 ± 6.58e-4 counts / y / 2t / ROI

Produces a two-bar plot with error bars + ratio panel.
"""

import numpy as np
import matplotlib.pyplot as plt


def main() -> None:
    # ---- Inputs (counts/y/2t/ROI) ----
    iv_radius_mm = 1226

    ov_r_mm = np.array([1765, 2230], dtype=float)
    y = np.array([1.47e-3, 1.58e-3], dtype=float)
    yerr = np.array([6.02e-4, 6.58e-4], dtype=float)

    labels = ["OV 1765 mm", "OV 2230 mm (baseline)"]

    # ---- Style ----
    fs_label = 14
    fs_tick = 12
    fs_annot = 12
    lw_main = 1.6
    lw_grid = 1.0
    lw_err = 1.6
    capsize = 6
    marker_size = 7
    spine_width = 1.2
    plt.rcParams["axes.linewidth"] = spine_width

    # ---- Derived comparison metrics ----
    baseline_label = "OV 2230 mm (baseline)"
    baseline_index = labels.index(baseline_label) if baseline_label in labels else 1
    baseline = y[baseline_index]
    baseline_err = yerr[baseline_index]

    ratio = y / baseline
    # Propagate ratio uncertainty: r = a/b
    ratio_err = ratio * np.sqrt((yerr / y) ** 2 + (baseline_err / baseline) ** 2)
    ratio_err[baseline_index] = 0.0

    # ---- Plot ----
    fig, (ax, ax_ratio) = plt.subplots(
        2,
        1,
        sharex=True,
        figsize=(12,8),
    )

    x = np.arange(len(ov_r_mm))
    bar_colors = ["#7aa6c2", "#d0a86e"]
    ax.bar(
        x,
        y,
        yerr=yerr,
        capsize=capsize,
        width=0.6,
        color=bar_colors,
        edgecolor="0.2",
        linewidth=lw_main,
        alpha=0.9,
        error_kw={"elinewidth": lw_err, "capthick": lw_err},
    )

    ax.set_ylabel("Background rate [cts/(y·t·FWHM)]", fontsize=fs_label)
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2.5e-3)
    ax.grid(True, axis="y", linestyle=":", alpha=0.4, linewidth=lw_grid)
    ax.tick_params(
        axis="x",
        which="both",
        labelbottom=False,
        labelsize=fs_tick,
        width=spine_width,
    )
    ax.tick_params(axis="y", labelsize=fs_tick, width=spine_width)

    # Also annotate each bar with its value ± error
    scale_pow = int(np.floor(np.log10(np.nanmax(y))))
    scale_factor = 10 ** (-scale_pow)
    y_scaled = y * scale_factor
    yerr_scaled = yerr * scale_factor
    for xi, yi, yi_s, ei_s in zip(x, y, y_scaled, yerr_scaled):
        text_y = yi * 0.55
        label = rf"${yi_s:.2f}\pm{ei_s:.2f}\times 10^{{{scale_pow}}}$"
        ax.text(
            xi,
            text_y,
            label,
            ha="center",
            va="top",
            fontsize=fs_annot,
        )

    # ---- Ratio panel ----
    ax_ratio.axhline(1.0, color="0.4", linestyle="--", linewidth=lw_main)
    baseline_frac_err = baseline_err / baseline
    ax_ratio.axhspan(
        1.0 - baseline_frac_err,
        1.0 + baseline_frac_err,
        color="0.85",
        alpha=0.6,
        zorder=0,
    )
    for xi, ri, rei, color in zip(x, ratio, ratio_err, bar_colors):
        ax_ratio.errorbar(
            xi,
            ri,
            yerr=rei,
            fmt="o",
            capsize=capsize,
            color=color,
            markersize=marker_size,
            elinewidth=lw_err,
            capthick=lw_err,
            markeredgewidth=lw_main,
        )
    ratio_min = np.nanmin(ratio - ratio_err)
    ratio_max = np.nanmax(ratio + ratio_err)
    span = ratio_max - ratio_min
    pad = 0.1 * span if span > 0 else 0.1
    ax_ratio.set_ylim(max(0.0, ratio_min - pad), ratio_max + pad)
    ax_ratio.set_ylabel("Ratio to baseline", fontsize=fs_label)
    ax_ratio.set_xticks(x)
    ax_ratio.set_xticklabels(labels)
    ax_ratio.grid(True, axis="y", linestyle=":", alpha=0.4, linewidth=lw_grid)
    ax_ratio.tick_params(axis="both", labelsize=fs_tick, width=spine_width)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
