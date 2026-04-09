#!/usr/bin/env python3
"""
IV/OV background vs HFE mass from the 'Summary' sheet.

Fits y(r) = A * exp(k r) for IV/OV only, prints attenuation mu = -k with
its 1-sigma uncertainty, and plots the total intrinsic-background marker on
the shared HFE-mass axis used by the HFE scan.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --- Config ---
XLSX_PATH = Path(__file__).resolve().parent / "Summary_bkgd_vs_hfe-shield.xlsx"
R_GRID = np.linspace(950, 1800, 600)  # mm
Z_BAND = 1.0  # 1-sigma confidence band; set 1.96 for ~95%

# Baseline radius for the total intrinsic-background marker; set to None to use
# the max IV/OV radius in data.
BASELINE_RADIUS_MM = None
TOTAL_INTRINSIC_BACKGROUND = 0.55
BASELINE_LABEL_Y = 0.25
X_AXIS_MAX_TONNES = 35.0

# Font sizes
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14

# Shared IV-radius -> HFE-mass conversion used in the HFE plots.
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

    data_frame = data_frame.iloc[header_row_index + 1:, :4].copy()
    data_frame.columns = ["Component", "r_mm", "y", "e"]

    data_frame["Component"] = data_frame["Component"].ffill().astype(str).str.strip()
    for column in ["r_mm", "y", "e"]:
        data_frame[column] = pd.to_numeric(data_frame[column], errors="coerce")

    mapping = {
        "iv": "IV",
        "ov": "OV",
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


def _confidence_band(curve, jacobian_first, jacobian_second, covariance, z_value):
    """Compute confidence bands via the delta method for two-parameter fits."""
    variance = (
        covariance[0, 0] * jacobian_first**2
        + 2.0 * covariance[0, 1] * jacobian_first * jacobian_second
        + covariance[1, 1] * jacobian_second**2
    )
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    lower_band = np.clip(curve - z_value * standard_error, 1e-300, np.inf)
    upper_band = curve + z_value * standard_error
    return lower_band, upper_band


def fit_attenuation_with_bands(
    radius_mm,
    counts,
    errors,
    radius_grid_mm=R_GRID,
    z_value=Z_BAND,
):
    """Fit exponential attenuation model and compute confidence bands."""
    radius_mm = np.asarray(radius_mm, float)
    counts = np.asarray(counts, float)
    errors = np.asarray(errors, float)
    valid_mask = np.isfinite(radius_mm) & np.isfinite(counts) & (counts > 0)
    if valid_mask.sum() < 2:
        return None

    radius_fit = radius_mm[valid_mask]
    counts_fit = counts[valid_mask]
    errors_fit = errors[valid_mask]
    slope_guess, log_amplitude_guess = np.polyfit(radius_fit, np.log(counts_fit), 1)
    initial_params = (float(np.exp(log_amplitude_guess)), float(slope_guess))

    def model(x_values, amplitude, slope):
        return amplitude * np.exp(slope * x_values)

    sigma = _safe_sigma(errors_fit)
    params, covariance = curve_fit(
        model,
        radius_fit,
        counts_fit,
        p0=initial_params,
        sigma=sigma,
        absolute_sigma=True,
        maxfev=10000,
    )
    amplitude, slope = params
    curve = model(radius_grid_mm, amplitude, slope)

    jac_amplitude = np.exp(slope * radius_grid_mm)
    jac_slope = amplitude * radius_grid_mm * np.exp(slope * radius_grid_mm)
    lower_band, upper_band = _confidence_band(
        curve,
        jac_amplitude,
        jac_slope,
        covariance,
        z_value,
    )

    return {
        "params": (amplitude, slope),
        "cov": covariance,
        "curve": curve,
        "lo": lower_band,
        "hi": upper_band,
        "mu": -slope,
    }


def main():
    """Run the IV/OV analysis and plot results."""
    summary_frame = load_data(XLSX_PATH)
    summary_frame = summary_frame[summary_frame["Component"].isin(["IV", "OV"])].copy()

    series_by_component = {}
    for component in ["IV", "OV"]:
        component_data = summary_frame[summary_frame["Component"] == component]
        if component_data.empty:
            continue
        series_by_component[component] = {
            "radius_mm": component_data["r_mm"].to_numpy(),
            "counts": component_data["y"].to_numpy(),
            "errors": component_data["e"].fillna(0).to_numpy(),
        }

    if not series_by_component:
        raise ValueError("No IV/OV data found in the Summary sheet.")

    baseline_radius = BASELINE_RADIUS_MM
    if baseline_radius is None:
        baseline_radius = float(summary_frame["r_mm"].max())
    baseline_mass = float(hfe_mass_tonnes_from_radius_mm(baseline_radius))

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        "IV": default_colors[2 % len(default_colors)],
        "OV": default_colors[3 % len(default_colors)],
    }

    fit_curves = {}
    bands = {}

    for component in ["IV", "OV"]:
        if component not in series_by_component:
            continue
        component_series = series_by_component[component]
        result = fit_attenuation_with_bands(
            component_series["radius_mm"],
            component_series["counts"],
            component_series["errors"],
        )
        if result is None:
            continue
        mu_mm_inv = result["mu"]
        mu_unc = float(np.sqrt(max(result["cov"][1, 1], 0.0)))
        fit_curves[component] = result["curve"]
        bands[component] = (result["lo"], result["hi"])
        print(f"{component}: mu = {mu_mm_inv:.4e} +/- {mu_unc:.4e} 1/mm")

    # ---------- Plot ----------
    fig, axis = plt.subplots(figsize=(10, 7))
    mass_grid_t = hfe_mass_tonnes_from_radius_mm(R_GRID)

    # data points with error bars
    for component, component_series in series_by_component.items():
        label = "Nickel IV: Th/U" if component == "IV" else "Nickel OV: Th/U"
        axis.errorbar(
            hfe_mass_tonnes_from_radius_mm(component_series["radius_mm"]),
            component_series["counts"],
            yerr=component_series["errors"],
            fmt="o",
            ms=5,
            label=label,
            color=color_map[component],
        )

    # fit curves + shaded confidence bands (mean fit)
    for component, curve in fit_curves.items():
        color = color_map.get(component, None)
        axis.semilogy(
            mass_grid_t,
            curve,
            "--",
            linewidth=1.2,
            alpha=0.9,
            color=color,
            label="_nolegend_",
        )
        if component in bands:
            lower_band, upper_band = bands[component]
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

    axis.set_xlabel("HFE mass (tonnes)", fontsize=FS_LABEL)
    axis.set_ylabel("Background rate [cts/(y·2t·FWHM)]", fontsize=FS_LABEL)
    axis.set_yscale("log")
    axis.set_ylim(1e-5, 1.0)
    axis.set_xlim(float(mass_grid_t.min()), X_AXIS_MAX_TONNES)
    axis.tick_params(axis="both", which="major", labelsize=FS_TICK)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)

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

    axis.legend(fontsize=FS_LEGEND)

    fig.tight_layout()
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path)
    plt.close(fig)


if __name__ == "__main__":
    main()
