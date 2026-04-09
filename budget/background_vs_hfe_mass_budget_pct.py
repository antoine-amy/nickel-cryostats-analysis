#!/usr/bin/env python3
"""
Background vs HFE mass from the 'Summary' sheet.

This reproduces the style of background_vs_hfe_spreadsheet.py with:
- x-axis in HFE mass (tonnes) instead of IV radius (mm)
- y-axis in percent of the total allowed budget (0.55 cts/(y*2t*FWHM))
- legend reduced to:
    HFE (no Rn only),
    Cryostat (IV+OV summed),
    Water shielding,
    Electronics box
- only one baseline budget marker for total cryostat (IV+OV sum)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --- Config ---
XLSX_PATH = Path(__file__).resolve().parent / "Summary_bkgd_vs_hfe-shield.xlsx"
R_GRID_MM = np.linspace(950.0, 1800.0, 600)

# Finite-cylinder TPC geometry for the analytic model (cm)
R_TPC_CM = 56.665
TPC_HALF_HEIGHT_CM = 59.15

# Gauss-Legendre quadrature for the shell-averaged sphere-to-cylinder transmission.
_MU_QUAD_ORDER = 256
_COS_THETA_NODES, _COS_THETA_WEIGHTS = np.polynomial.legendre.leggauss(_MU_QUAD_ORDER)
_SIN_THETA_NODES = np.sqrt(np.maximum(0.0, 1.0 - _COS_THETA_NODES**2))
with np.errstate(divide="ignore", invalid="ignore"):
    _TPC_BOUNDARY_RADIUS_CM = np.minimum(
        R_TPC_CM / np.where(_SIN_THETA_NODES > 0.0, _SIN_THETA_NODES, np.inf),
        TPC_HALF_HEIGHT_CM / np.where(np.abs(_COS_THETA_NODES) > 0.0, np.abs(_COS_THETA_NODES), np.inf),
    )

Z_BAND = 1.0  # 1-sigma confidence band; set 1.96 for ~95%

# 2.5 MeV HFE-7200 attenuation coefficient at 165 K (approx. LXe temperature)
MU_HFE_2P5MEV_MM = 0.00674403

# Total allowed background budget
TOTAL_ALLOWED_BUDGET = 0.55  # cts/(y*2t*FWHM)
PROPOSED_HFE_MASS_T = 6.0
BASELINE_LABEL_Y = 0.4
PROPOSED_LABEL_Y = 0.02
TPC_VESSEL_LABEL_Y = 0.12

# HFE mass conversion constants (from hfe_volume_to_iv_radius.py)
DENSITY_T_PER_M3 = 1.73
VOL_OFFSET_M3 = 2.2 - 0.58
HFE_LOSS_TONNES = 0.430
TPC_VESSEL_RADIUS_MM = 931.0

# Baseline radius for budget marker; if None, use max IV/OV radius in data.
CFC_BASELINE_RADIUS_MM = None
CFC_BASELINE_POINTS = {
    "IV": {"y": 2.76e-2, "e": 1.21e-2},
    "OV": {"y": 4.26e-2, "e": 1.76e-2},
}

# Poster-sized font settings
FS_LABEL = 26
FS_TICK = 24
FS_LEGEND = 24
FS_ANNOTATION = 24

# Internal component names
HFE_NO_RN_LABEL = "HFE (Th/U only)"
WATER_SEMI_ANALYTICAL_LABEL = "Water (semi-analytical)"
TRANSITION_BOX_LABEL = "Transition Box"
CRYOSTAT_SUM_LABEL = "Cryostat"

# Legend labels required by user
LEGEND_LABELS = {
    HFE_NO_RN_LABEL: "HFE",
    CRYOSTAT_SUM_LABEL: "Cryostat",
    WATER_SEMI_ANALYTICAL_LABEL: "Water shielding",
    TRANSITION_BOX_LABEL: "Electronics box",
}


def counts_to_budget_percent(values):
    """Convert absolute background rate to % of allowed budget."""
    return np.asarray(values, float) / TOTAL_ALLOWED_BUDGET * 100.0


def hfe_mass_tonnes_from_radius_mm(radius_mm):
    """Convert IV radius (mm) to HFE mass (tonnes)."""
    radius_m = np.asarray(radius_mm, float) / 1000.0
    total_volume_m3 = (4.0 / 3.0) * np.pi * radius_m**3
    hfe_volume_m3 = total_volume_m3 - VOL_OFFSET_M3
    return hfe_volume_m3 * DENSITY_T_PER_M3 - HFE_LOSS_TONNES


def load_data(filepath):
    """Load Summary sheet and map component names to canonical labels."""
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
        "hfe (no rn-222)": HFE_NO_RN_LABEL,
        "hfe (th/u only)": HFE_NO_RN_LABEL,
        "hfe (th/u & rn)": "HFE (Th/U & Rn)",
        "hfe (th/u and rn)": "HFE (Th/U & Rn)",
        "hfe (th/u & po)": "HFE (Th/U & Po)",
        "hfe (th/u and po)": "HFE (Th/U & Po)",
        "hfe (th/u & rn & po)": "HFE (Th/U & Po)",
        "hfe (th/u and rn and po)": "HFE (Th/U & Po)",
        "hfe": "HFE",
        "iv": "IV",
        "ov": "OV",
        "water": "Water",
        "water (theoretical)": WATER_SEMI_ANALYTICAL_LABEL,
        "transition box": TRANSITION_BOX_LABEL,
    }
    data_frame["Component"] = (
        data_frame["Component"].str.lower().map(mapping).fillna(data_frame["Component"])
    )
    return data_frame.dropna(subset=["r_mm", "y"]).reset_index(drop=True)


def _safe_sigma(errors):
    """Return sigma for weighted fits; None if no valid positive uncertainties."""
    errors = np.asarray(errors, float)
    valid_mask = np.isfinite(errors) & (errors > 0)
    if not np.any(valid_mask):
        return None
    min_positive = np.nanmin(errors[valid_mask])
    return np.where((~np.isfinite(errors)) | (errors <= 0), min_positive, errors)


def _confidence_band(curve, jacobian_first, jacobian_second, covariance, z_value):
    """Delta-method confidence band for two-parameter fits."""
    variance = (
        covariance[0, 0] * jacobian_first**2
        + 2.0 * covariance[0, 1] * jacobian_first * jacobian_second
        + covariance[1, 1] * jacobian_second**2
    )
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    lower_band = np.clip(curve - z_value * standard_error, 1e-300, np.inf)
    upper_band = curve + z_value * standard_error
    return lower_band, upper_band


def _confidence_band_single(curve, jacobian, variance, z_value):
    """Delta-method confidence band for one-parameter fits."""
    if variance is None or not np.isfinite(variance):
        return None, None
    standard_error = np.sqrt(max(variance, 0.0)) * np.abs(jacobian)
    lower_band = np.clip(curve - z_value * standard_error, 1e-300, np.inf)
    upper_band = curve + z_value * standard_error
    return lower_band, upper_band


def _analytic_f_on_grid(mu_mm_inv, radius_grid_mm=R_GRID_MM):
    """Unnormalized cumulative analytic HFE model F(r; mu) for a finite-cylinder TPC."""
    radius_cm_grid = radius_grid_mm / 10.0
    mu_cm_inv = mu_mm_inv * 10.0

    # MC-calibrated geometric acceptance (valid around 0.95-1.70 m)
    a_m2 = 0.19575476
    b_m3 = 0.099439452

    radius_m_grid = radius_cm_grid / 100.0
    solid_fraction = np.where(
        radius_m_grid > 0,
        0.5 * (a_m2 / radius_m_grid**2 + b_m3 / radius_m_grid**3),
        0.0,
    )
    shell_thickness_cm = np.maximum(
        radius_cm_grid[:, np.newaxis] - _TPC_BOUNDARY_RADIUS_CM[np.newaxis, :],
        0.0,
    )
    mean_transmission = 0.5 * np.exp(-mu_cm_inv * shell_thickness_cm) @ _COS_THETA_WEIGHTS
    integrand = 4.0 * np.pi * (radius_cm_grid**2) * solid_fraction * mean_transmission
    delta_r = radius_cm_grid[1] - radius_cm_grid[0]
    return np.cumsum(integrand) * delta_r


def fit_hfe_no_rn_with_bands(radius_mm, counts, errors, radius_grid_mm=R_GRID_MM, z_value=Z_BAND):
    """Fit analytic HFE model (no Rn) and return curve + confidence band."""
    radius_mm = np.asarray(radius_mm, float)
    counts = np.asarray(counts, float)
    errors = np.asarray(errors, float)
    valid_mask = np.isfinite(radius_mm) & np.isfinite(counts)
    radius_fit = radius_mm[valid_mask]
    counts_fit = counts[valid_mask]
    errors_fit = errors[valid_mask]
    if radius_fit.size < 2:
        return None

    def model(radius_mm_input, norm_factor, mu_mm_inv):
        cumulative_shape = _analytic_f_on_grid(mu_mm_inv, radius_grid_mm)
        return norm_factor * np.interp(radius_mm_input, radius_grid_mm, cumulative_shape)

    mu_guess = 0.01
    interp_last = np.interp(
        radius_fit[-1], radius_grid_mm, _analytic_f_on_grid(mu_guess, radius_grid_mm)
    )
    norm_guess = counts_fit[-1] / interp_last if interp_last > 0 else 1.0

    sigma = _safe_sigma(errors_fit)
    params, covariance = curve_fit(
        model,
        radius_fit,
        counts_fit,
        p0=(float(norm_guess), float(mu_guess)),
        sigma=sigma,
        absolute_sigma=True,
        maxfev=20000,
    )
    norm_factor, mu_mm_inv = params

    f_grid = _analytic_f_on_grid(mu_mm_inv, radius_grid_mm)
    curve = norm_factor * f_grid

    delta_mu = max(1e-6, 1e-6 * abs(mu_mm_inv))
    derivative_mu = (
        _analytic_f_on_grid(mu_mm_inv + delta_mu, radius_grid_mm)
        - _analytic_f_on_grid(mu_mm_inv - delta_mu, radius_grid_mm)
    ) / (2.0 * delta_mu)

    lower_band, upper_band = _confidence_band(
        curve, f_grid, norm_factor * derivative_mu, covariance, z_value
    )
    return {"curve": curve, "lo": lower_band, "hi": upper_band}


def fit_exponential_with_bands(radius_mm, counts, errors, radius_grid_mm=R_GRID_MM, z_value=Z_BAND):
    """Fit y=A*exp(k*r) and return curve + confidence band."""
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
        curve, jac_amplitude, jac_slope, covariance, z_value
    )
    return {"curve": curve, "lo": lower_band, "hi": upper_band}


def fit_fixed_mu_with_bands(
    radius_mm,
    counts,
    errors,
    mu_mm_inv=MU_HFE_2P5MEV_MM,
    radius_grid_mm=R_GRID_MM,
    z_value=Z_BAND,
):
    """Fit y=A*exp(-mu*r) with fixed mu and return curve + confidence band."""
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
    lower_band, upper_band = _confidence_band_single(curve, jacobian, variance, z_value)
    return {"curve": curve, "lo": lower_band, "hi": upper_band}


def build_cryostat_sum(data_frame):
    """Build IV+OV summed component with errors added in quadrature."""
    iv = (
        data_frame.loc[data_frame["Component"] == "IV", ["r_mm", "y", "e"]]
        .rename(columns={"y": "y_iv", "e": "e_iv"})
    )
    ov = (
        data_frame.loc[data_frame["Component"] == "OV", ["r_mm", "y", "e"]]
        .rename(columns={"y": "y_ov", "e": "e_ov"})
    )
    merged = iv.merge(ov, on="r_mm", how="inner")
    if merged.empty:
        raise ValueError("Cannot build Cryostat sum: IV/OV overlap on r_mm is empty.")

    merged["e_iv"] = merged["e_iv"].fillna(0.0)
    merged["e_ov"] = merged["e_ov"].fillna(0.0)
    cryostat = pd.DataFrame(
        {
            "Component": CRYOSTAT_SUM_LABEL,
            "r_mm": merged["r_mm"],
            "y": merged["y_iv"] + merged["y_ov"],
            "e": np.sqrt(merged["e_iv"] ** 2 + merged["e_ov"] ** 2),
        }
    )
    return cryostat


def main():
    """Generate the requested plot."""
    summary_frame = load_data(XLSX_PATH)
    cryostat_sum = build_cryostat_sum(summary_frame)

    baseline_radius_mm = CFC_BASELINE_RADIUS_MM
    if baseline_radius_mm is None:
        iv_ov = summary_frame[summary_frame["Component"].isin(["IV", "OV"])]
        baseline_radius_mm = float(iv_ov["r_mm"].max())

    selected_components = [
        HFE_NO_RN_LABEL,
        WATER_SEMI_ANALYTICAL_LABEL,
        TRANSITION_BOX_LABEL,
    ]
    selected = summary_frame[summary_frame["Component"].isin(selected_components)].copy()

    # Drop Transition Box baseline point when it is an upper-limit style entry.
    transition_upper_limit_mask = (
        (selected["Component"] == TRANSITION_BOX_LABEL)
        & np.isclose(selected["r_mm"], baseline_radius_mm)
        & (selected["y"] <= 0.0)
        & (selected["e"] > 0.0)
    )
    selected = selected.loc[~transition_upper_limit_mask].copy()

    selected = pd.concat([selected, cryostat_sum], ignore_index=True)

    components_order = [
        HFE_NO_RN_LABEL,
        CRYOSTAT_SUM_LABEL,
        WATER_SEMI_ANALYTICAL_LABEL,
        TRANSITION_BOX_LABEL,
    ]

    series_by_component = {}
    for component in components_order:
        component_data = selected[selected["Component"] == component]
        if component_data.empty:
            continue
        series_by_component[component] = {
            "radius_mm": component_data["r_mm"].to_numpy(),
            "counts": component_data["y"].to_numpy(),
            "errors": component_data["e"].fillna(0.0).to_numpy(),
        }

    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        component: default_colors[index % len(default_colors)]
        for index, component in enumerate(components_order)
    }

    fit_curves = {}
    bands = {}

    if HFE_NO_RN_LABEL in series_by_component:
        result = fit_hfe_no_rn_with_bands(
            series_by_component[HFE_NO_RN_LABEL]["radius_mm"],
            series_by_component[HFE_NO_RN_LABEL]["counts"],
            series_by_component[HFE_NO_RN_LABEL]["errors"],
        )
        if result is not None:
            fit_curves[HFE_NO_RN_LABEL] = result["curve"]
            bands[HFE_NO_RN_LABEL] = (result["lo"], result["hi"])

    for component in [CRYOSTAT_SUM_LABEL, WATER_SEMI_ANALYTICAL_LABEL]:
        if component not in series_by_component:
            continue
        result = fit_exponential_with_bands(
            series_by_component[component]["radius_mm"],
            series_by_component[component]["counts"],
            series_by_component[component]["errors"],
        )
        if result is not None:
            fit_curves[component] = result["curve"]
            bands[component] = (result["lo"], result["hi"])

    if TRANSITION_BOX_LABEL in series_by_component:
        result = fit_fixed_mu_with_bands(
            series_by_component[TRANSITION_BOX_LABEL]["radius_mm"],
            series_by_component[TRANSITION_BOX_LABEL]["counts"],
            series_by_component[TRANSITION_BOX_LABEL]["errors"],
        )
        if result is not None:
            fit_curves[TRANSITION_BOX_LABEL] = result["curve"]
            bands[TRANSITION_BOX_LABEL] = (result["lo"], result["hi"])

    mass_grid_t = hfe_mass_tonnes_from_radius_mm(R_GRID_MM)
    _, axis = plt.subplots(figsize=(12, 8))

    # Data points with error bars (legend entries only for these).
    for component in components_order:
        if component not in series_by_component:
            continue
        component_series = series_by_component[component]
        axis.errorbar(
            hfe_mass_tonnes_from_radius_mm(component_series["radius_mm"]),
            counts_to_budget_percent(component_series["counts"]),
            yerr=counts_to_budget_percent(component_series["errors"]),
            fmt="o",
            ms=4,
            label=LEGEND_LABELS[component],
            color=color_map[component],
        )

    # Fit curves + confidence bands
    for component, curve in fit_curves.items():
        color = color_map.get(component)
        axis.semilogy(
            mass_grid_t,
            counts_to_budget_percent(curve),
            "--",
            linewidth=1.2,
            alpha=0.9,
            color=color,
            label="_nolegend_",
        )
        if component in bands:
            lower_band, upper_band = bands[component]
            if lower_band is not None:
                axis.fill_between(
                    mass_grid_t,
                    counts_to_budget_percent(np.clip(lower_band, 1e-300, np.inf)),
                    counts_to_budget_percent(upper_band),
                    color=color,
                    alpha=0.15,
                    linewidth=0,
                    label="_nolegend_",
                )

    # Baseline vertical line
    baseline_mass_t = float(hfe_mass_tonnes_from_radius_mm(baseline_radius_mm))
    tpc_vessel_mass_t = float(hfe_mass_tonnes_from_radius_mm(TPC_VESSEL_RADIUS_MM))

    axis.axvline(
        tpc_vessel_mass_t,
        color="0.7",
        linestyle="--",
        linewidth=1.2,
        alpha=0.6,
    )
    axis.text(
        tpc_vessel_mass_t,
        TPC_VESSEL_LABEL_Y,
        "TPC vessel",
        rotation=90,
        va="bottom",
        ha="left",
        transform=axis.get_xaxis_transform(),
        fontsize=FS_TICK,
        color="0.3",
    )
    axis.axvline(baseline_mass_t, color="0.7", linestyle="--", linewidth=1.2, alpha=0.6)
    axis.text(
        baseline_mass_t,
        BASELINE_LABEL_Y,
        "Baseline design",
        rotation=90,
        va="bottom",
        ha="right",
        transform=axis.get_xaxis_transform(),
        fontsize=FS_TICK,
        color="0.3",
    )

    axis.axvline(
        PROPOSED_HFE_MASS_T,
        color="0.5",
        linestyle="--",
        linewidth=1.2,
        alpha=0.7,
    )
    axis.text(
        PROPOSED_HFE_MASS_T,
        PROPOSED_LABEL_Y,
        "Proposed design",
        rotation=90,
        va="bottom",
        ha="right",
        transform=axis.get_xaxis_transform(),
        fontsize=FS_TICK,
        fontweight="bold",
        color="0.35",
    )

    # Single total cryostat budget point (IV+OV only)
    cryostat_budget_y = CFC_BASELINE_POINTS["IV"]["y"] + CFC_BASELINE_POINTS["OV"]["y"]
    cryostat_budget_e = np.sqrt(
        CFC_BASELINE_POINTS["IV"]["e"] ** 2 + CFC_BASELINE_POINTS["OV"]["e"] ** 2
    )
    axis.errorbar(
        [baseline_mass_t],
        [counts_to_budget_percent(cryostat_budget_y)],
        yerr=[counts_to_budget_percent(cryostat_budget_e)],
        fmt="s",
        ms=7,
        color="0.15",
        markerfacecolor="none",
        label="_nolegend_",
    )
    axis.annotate(
        "Cryostat background budget",
        xy=(baseline_mass_t, counts_to_budget_percent(cryostat_budget_y)),
        xytext=(-10, 0),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=FS_ANNOTATION,
        color="0.15",
    )

    axis.set_xlabel("HFE mass (tonnes)", fontsize=FS_LABEL)
    axis.set_ylabel("Background contribution (% budget)", fontsize=FS_LABEL)
    axis.set_yscale("log")
    axis.set_ylim(
        counts_to_budget_percent(1e-5),
        counts_to_budget_percent(2e-1),
    )
    axis.set_xlim(
        min(float(mass_grid_t.min()), tpc_vessel_mass_t),
        float(mass_grid_t.max()),
    )
    axis.tick_params(axis="both", which="major", labelsize=FS_TICK)
    axis.tick_params(axis="both", which="minor", labelsize=FS_TICK)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=FS_LEGEND, loc="lower center", ncol=2)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
