#!/usr/bin/env python3
"""
Transition box background vs HFE mass from the Summary sheet.

Fits y(r) = A * exp(-mu r) with fixed mu from the 2.5 MeV HFE attenuation
coefficient and plots data with 1-sigma confidence bands on a shared HFE-mass
x-axis consistent with the other _only plots.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Shared IV-radius -> HFE-mass conversion
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfe_volume_to_iv_radius import (
    BASELINE_TARGET_MASS_KG_1,
    calibration_loss_tonnes_for_target,
    hfe_mass_tonnes,
)

CALIBRATION_LOSS_TONNES = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_1)


def hfe_mass_tonnes_from_radius_mm(radius_mm):
    """Convert IV radius (mm) to HFE mass (tonnes)."""
    radius_m = np.asarray(radius_mm, float) / 1000.0
    return hfe_mass_tonnes(radius_m, CALIBRATION_LOSS_TONNES)


# --- Config ---
XLSX_PATH = Path(__file__).resolve().parent / "Summary_bkgd_vs_hfe-shield.xlsx"
R_GRID = np.linspace(950, 1800, 600)  # mm
Z_BAND = 1.0  # 1-sigma confidence band; set 1.96 for ~95%

# 2.5 MeV HFE-7200 attenuation coefficient at 165 K (approx. LXe temperature)
MU_HFE_2P5MEV_MM = 0.00674403  # 1/mm

# Baseline radius for CFC points; set to None to use max radius in data.
CFC_BASELINE_RADIUS_MM = None
CFC_BASELINE_POINTS = {
    "IV": {"y": 2.76e-2, "e": 1.21e-2},
    "OV": {"y": 4.26e-2, "e": 1.76e-2},
}

TOTAL_INTRINSIC_BACKGROUND = 0.55
BASELINE_LABEL_Y = 0.43
X_AXIS_MAX_TONNES = 35.0

# Font sizes
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14

TRANSITION_BOX_COMPONENT = "Transition Box"


def load_data(filepath):
    """Load data from Excel file and return processed DataFrame."""
    data_frame = pd.read_excel(filepath, sheet_name="Summary", header=None)

    header_row_index = None
    for row_index, (_, row_series) in enumerate(data_frame.iterrows()):
        row_values = [str(value).strip().lower() for value in row_series.values]
        if all(
            header in row_values
            for header in ["component", "iv radius (mm)", "background", "error"]
        ):
            header_row_index = row_index
            break

    if header_row_index is None:
        raise ValueError("Header row with required columns not found.")

    data_frame = data_frame.iloc[header_row_index + 1 :, :4].copy()
    data_frame.columns = ["Component", "r_mm", "y", "e"]

    data_frame["Component"] = data_frame["Component"].ffill().astype(str).str.strip()
    for column in ["r_mm", "y", "e"]:
        data_frame[column] = pd.to_numeric(data_frame[column], errors="coerce")

    mapping = {
        "transition box": TRANSITION_BOX_COMPONENT,
    }
    data_frame["Component"] = (
        data_frame["Component"].str.lower().map(mapping).fillna(data_frame["Component"])
    )
    return data_frame.dropna(subset=["r_mm", "y"]).reset_index(drop=True)


def _safe_sigma(errors: np.ndarray):
    """Return sigma array for curve_fit; if all zeros/NaN, return None (unweighted)."""
    errors = np.asarray(errors, float)
    valid_mask = np.isfinite(errors) & (errors > 0)
    if not np.any(valid_mask):
        return None
    min_positive = np.nanmin(errors[valid_mask])
    return np.where((~np.isfinite(errors)) | (errors <= 0), min_positive, errors)


def _confidence_band_single(curve, jacobian, variance, z_value):
    """Compute confidence bands for a single-parameter fit."""
    if variance is None or not np.isfinite(variance):
        return None, None
    standard_error = np.sqrt(max(variance, 0.0)) * np.abs(jacobian)
    lower_band = np.clip(curve - z_value * standard_error, 1e-300, np.inf)
    upper_band = curve + z_value * standard_error
    return lower_band, upper_band


def fit_attenuation_with_bands(
    radius_mm,
    counts,
    errors,
    mu_mm_inv=MU_HFE_2P5MEV_MM,
    radius_grid_mm=R_GRID,
    z_value=Z_BAND,
):
    """Fit exponential attenuation model with fixed mu and compute bands."""
    radius_mm = np.asarray(radius_mm, float)
    counts = np.asarray(counts, float)
    errors = np.asarray(errors, float)
    valid_mask = np.isfinite(radius_mm) & np.isfinite(counts)
    if valid_mask.sum() < 1:
        return None

    radius_fit = radius_mm[valid_mask]
    counts_fit = counts[valid_mask]
    errors_fit = errors[valid_mask]

    attenuation = np.exp(-mu_mm_inv * radius_fit)
    sigma = _safe_sigma(errors_fit)
    if sigma is None:
        weights = np.ones_like(counts_fit)
    else:
        weights = 1.0 / sigma**2

    denom = np.sum(weights * attenuation**2)
    if denom <= 0.0:
        return None
    amplitude = float(np.sum(weights * attenuation * counts_fit) / denom)
    amplitude = max(amplitude, 0.0)

    curve = amplitude * np.exp(-mu_mm_inv * radius_grid_mm)
    variance = (1.0 / denom) if sigma is not None else None
    jacobian = np.exp(-mu_mm_inv * radius_grid_mm)
    lower_band, upper_band = _confidence_band_single(
        curve,
        jacobian,
        variance,
        z_value,
    )

    return {
        "params": (amplitude, -mu_mm_inv),
        "cov": np.array([[variance]]) if variance is not None else None,
        "curve": curve,
        "lo": lower_band,
        "hi": upper_band,
        "mu": mu_mm_inv,
    }


def main():
    """Run the transition box analysis and plot results."""
    summary_frame = load_data(XLSX_PATH)
    summary_frame = summary_frame[
        summary_frame["Component"] == TRANSITION_BOX_COMPONENT
    ].copy()

    if summary_frame.empty:
        raise ValueError("No transition box data found in the Summary sheet.")

    series = {
        "radius_mm": summary_frame["r_mm"].to_numpy(),
        "counts": summary_frame["y"].to_numpy(),
        "errors": summary_frame["e"].fillna(0).to_numpy(),
    }

    baseline_radius = CFC_BASELINE_RADIUS_MM
    if baseline_radius is None:
        baseline_radius = float(summary_frame["r_mm"].max())
    baseline_mass = float(hfe_mass_tonnes_from_radius_mm(baseline_radius))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color = default_colors[0] if default_colors else "C0"
    cfc_color_map = {
        "IV": "0.15",
        "OV": "0.65",
    }

    fit_curves = {}
    bands = {}

    result = fit_attenuation_with_bands(
        series["radius_mm"],
        series["counts"],
        series["errors"],
    )
    if result is not None:
        amplitude, _ = result["params"]
        mu_mm_inv = result["mu"]
        fit_curves[TRANSITION_BOX_COMPONENT] = result["curve"]
        bands[TRANSITION_BOX_COMPONENT] = (result["lo"], result["hi"])
        print(
            f"{TRANSITION_BOX_COMPONENT}: mu fixed at {mu_mm_inv:.4e} 1/mm "
            f"(A = {amplitude:.3g})"
        )

    _, axis = plt.subplots(figsize=(10, 7))
    mass_grid_t = hfe_mass_tonnes_from_radius_mm(R_GRID)

    curve = fit_curves.get(TRANSITION_BOX_COMPONENT)
    if curve is not None:
        axis.semilogy(
            mass_grid_t,
            curve,
            "--",
            linewidth=1.2,
            alpha=0.9,
            color=color,
            label="_nolegend_",
        )
        lower_band, upper_band = bands.get(TRANSITION_BOX_COMPONENT, (None, None))
        if lower_band is not None:
            lower_band = np.clip(lower_band, 1e-300, np.inf)
            axis.fill_between(
                mass_grid_t,
                lower_band,
                upper_band,
                color=color,
                alpha=0.15,
                linewidth=0,
                label="_nolegend_",
            )

    axis.errorbar(
        hfe_mass_tonnes_from_radius_mm(series["radius_mm"]),
        series["counts"],
        yerr=series["errors"],
        fmt="o",
        ms=5,
        mew=1.1,
        elinewidth=1.4,
        capsize=4,
        capthick=1.2,
        label=TRANSITION_BOX_COMPONENT,
        color=color,
        zorder=3,
    )

    # CFC baseline points
    for component, point in CFC_BASELINE_POINTS.items():
        label = f"CFC {component} @ baseline"
        cfc_color = cfc_color_map.get(component, "0.15")
        axis.errorbar(
            [baseline_mass],
            [point["y"]],
            yerr=[point["e"]],
            fmt="s",
            ms=7,
            color=cfc_color,
            markerfacecolor="none",
            label="_nolegend_",
            zorder=2,
        )
        axis.annotate(
            label,
            xy=(baseline_mass, point["y"]),
            xytext=(-10, 0),
            textcoords="offset points",
            ha="right",
            va="center",
            fontsize=FS_TICK - 2,
            color=cfc_color,
            zorder=3,
        )

    # Total intrinsic background marker
    axis.plot(
        [baseline_mass],
        [TOTAL_INTRINSIC_BACKGROUND],
        marker="s",
        ms=7,
        color="0.15",
        markerfacecolor="none",
        linestyle="None",
        label="_nolegend_",
    )
    axis.annotate(
        "total intrinsic background",
        xy=(baseline_mass, TOTAL_INTRINSIC_BACKGROUND),
        xytext=(-10, -2),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=FS_TICK - 2,
        color="0.15",
    )

    axis.axvline(baseline_mass, color="0.3", linestyle="--", linewidth=1.2)
    axis.text(
        baseline_mass,
        BASELINE_LABEL_Y,
        "Baseline Design",
        rotation=90,
        va="bottom",
        ha="right",
        transform=axis.get_xaxis_transform(),
        fontsize=FS_TICK,
        color="0.3",
    )

    axis.set_xlabel("HFE mass (tonnes)", fontsize=FS_LABEL)
    axis.set_ylabel("Background rate [cts/(y·2t·FWHM)]", fontsize=FS_LABEL)
    axis.set_yscale("log")
    axis.set_xlim(float(mass_grid_t.min()), X_AXIS_MAX_TONNES)
    axis.set_ylim(1e-5, 1e0)
    axis.tick_params(axis="both", which="major", labelsize=FS_TICK)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)

    axis.legend(fontsize=FS_LEGEND)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
