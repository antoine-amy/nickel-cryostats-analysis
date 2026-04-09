#!/usr/bin/env python3
"""
Background vs IV radius analysis from 'Summary' sheet, with confidence bands,
plus printed intersection points against the HFE (Th/U only) curve.

• HFE (Th/U only): analytic geometry-based model
  (explicit 4π r^2 and
  f_solid(r) = 0.5 * (R_TPC / r)^2), NO path-length multiplier.
  We fit normalization C and attenuation μ, and propagate their covariance.

• HFE (Th/U & Rn): exponential attenuation + 1/r^3 term (same as HFE-only).

• Transition Box: exponential attenuation with fixed μ from the 2.5 MeV HFE
  coefficient; we fit only the amplitude.

• Others (HFE, IV, OV, Water semi-analytical):
  y(r) = A * exp(k r). We fit A,k with weights from the y-errors,
  then compute bands via the delta method.

Bands shown are for the MEAN fit (confidence bands). For ~95% bands, set Z_BAND=1.96.
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BUDGET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUDGET_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(BUDGET_DIR) not in sys.path:
    sys.path.insert(0, str(BUDGET_DIR))

from hfe_volume_to_iv_radius import (
    BASELINE_TARGET_MASS_KG_1,
    calibration_loss_tonnes_for_target,
    hfe_mass_tonnes,
)

CALIBRATION_LOSS_TONNES = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_1)

from background_vs_hfe_water_only import (
    build_water_semi_analytical_curve_with_bands,
    R_GRID as _WATER_R_GRID,
)


def hfe_mass_tonnes_from_radius_mm(radius_mm):
    """Convert IV radius (mm) to HFE mass (tonnes)."""
    radius_m = np.asarray(radius_mm, float) / 1000.0
    return hfe_mass_tonnes(radius_m, CALIBRATION_LOSS_TONNES)


# --- Config ---
XLSX_PATH = ("/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/"
             "nickel-cryostats-analysis/budget/Summary_bkgd_vs_hfe-shield.xlsx")
R_GRID = np.linspace(950, 1800, 600)  # mm

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

Z_BAND = 1.0  # 1σ confidence band; set 1.96 for ~95%

# 2.5 MeV HFE-7200 attenuation coefficient at 165 K (approx. LXe temperature)
MU_HFE_2P5MEV_MM = 0.00674403  # 1/mm

# Baseline radius for CFC points; set to None to use max IV/OV radius in data.
CFC_BASELINE_RADIUS_MM = None
CFC_BASELINE_POINTS = {
    "IV": {"y": 2.76e-2, "e": 1.21e-2},
    "OV": {"y": 4.26e-2, "e": 1.76e-2},
}

TOTAL_INTRINSIC_BACKGROUND = 0.55
X_AXIS_MAX_TONNES = 35.0

# Named design configurations [label: (R_IV mm, label_y_frac, va)]
DESIGN_CONFIGS = {
    "Baseline":    (1691.00, 0.03, "bottom"),
    "Recommended": (1308.37, 0.03, "bottom"),
    "Aggressive":  (1121.80, 0.03, "bottom"),
}
DESIGN_COLORS = {
    "Baseline":    "black",
    "Recommended": "black",
    "Aggressive":  "black",
}

# Font sizes
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14

HFE_NO_RN_LABEL = "HFE (Th/U only)"
HFE_RN_LABEL = "HFE (Th/U & Rn)"
HFE_RN_PO_LABEL = "HFE (Th/U & Rn & Po)"
WATER_SEMI_ANALYTICAL_LABEL = "Water (semi-analytical)"

# ---------- Data loading ----------
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
        "hfe (no rn-222)": HFE_NO_RN_LABEL,
        "hfe (th/u only)": HFE_NO_RN_LABEL,
        "hfe (th/u & rn)": HFE_RN_LABEL,
        "hfe (th/u and rn)": HFE_RN_LABEL,
        "hfe (th/u & po)": HFE_RN_PO_LABEL,
        "hfe (th/u and po)": HFE_RN_PO_LABEL,
        "hfe (th/u & rn & po)": HFE_RN_PO_LABEL,
        "hfe (th/u and rn and po)": HFE_RN_PO_LABEL,
        "hfe": "HFE",
        "iv": "IV",
        "ov": "OV",
        "water": "Water",
        "water (theoretical)": WATER_SEMI_ANALYTICAL_LABEL,
        "transition box": "Transition Box",
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


# ---------- Analytic HFE geometry model ----------
def _analytic_f_on_grid(mu_mm_inv, radius_grid_mm=R_GRID):
    """Unnormalized cumulative shape F(r; μ) on R_GRID (mm) for a finite-cylinder TPC."""
    radius_cm_grid = radius_grid_mm / 10.0
    mu_cm_inv = mu_mm_inv * 10.0  # mm^-1 -> cm^-1
    
    # MC-calibrated geometric acceptance (fit valid for ~0.95–1.70 m)
    A_M2 = 0.19575476
    B_M3 = 0.099439452

    radius_m_grid = radius_cm_grid / 100.0

    solid_fraction = np.where(
        radius_m_grid > 0,
        0.5 * (A_M2 / radius_m_grid**2 + B_M3 / radius_m_grid**3),  # f_Omega = Ω/(4π)
        0.0
    )
    shell_thickness_cm = np.maximum(
        radius_cm_grid[:, np.newaxis] - _TPC_BOUNDARY_RADIUS_CM[np.newaxis, :],
        0.0,
    )
    mean_transmission = 0.5 * np.exp(-mu_cm_inv * shell_thickness_cm) @ _COS_THETA_WEIGHTS
    integrand = 4.0 * np.pi * (radius_cm_grid**2) * solid_fraction * mean_transmission
    delta_r = radius_cm_grid[1] - radius_cm_grid[0]
    cumulative = np.cumsum(integrand) * delta_r
    return cumulative


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


def _confidence_band_multi(curve, jacobians, covariance, z_value):
    """Compute confidence bands via the delta method for multi-parameter fits."""
    jacobians = np.asarray(jacobians, float)
    if jacobians.ndim == 1:
        jacobians = jacobians[np.newaxis, :]
    elif (
        jacobians.shape[0] != covariance.shape[0]
        and jacobians.shape[1] == covariance.shape[0]
    ):
        jacobians = jacobians.T
    variance = np.einsum("ip,ij,jp->p", jacobians, covariance, jacobians)
    standard_error = np.sqrt(np.maximum(variance, 0.0))
    lower_band = np.clip(curve - z_value * standard_error, 1e-300, np.inf)
    upper_band = curve + z_value * standard_error
    return lower_band, upper_band


def _confidence_band_single(curve, jacobian, variance, z_value):
    """Compute confidence bands for a single-parameter fit."""
    if variance is None or not np.isfinite(variance):
        return None, None
    standard_error = np.sqrt(max(variance, 0.0)) * np.abs(jacobian)
    lower_band = np.clip(curve - z_value * standard_error, 1e-300, np.inf)
    upper_band = curve + z_value * standard_error
    return lower_band, upper_band


def fit_hfe_no_rn_with_bands(
    radius_mm,
    counts,
    errors,
    radius_grid_mm=R_GRID,
    z_value=Z_BAND,
):
    """Fit analytic HFE geometry model and compute confidence bands."""
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

    mu_guess = 0.01  # 1/mm
    interp_last = np.interp(
        radius_fit[-1],
        radius_grid_mm,
        _analytic_f_on_grid(mu_guess, radius_grid_mm),
    )
    norm_guess = counts_fit[-1] / interp_last if interp_last > 0 else 1.0
    initial_params = (float(norm_guess), float(mu_guess))

    sigma = _safe_sigma(errors_fit)
    params, covariance = curve_fit(
        model,
        radius_fit,
        counts_fit,
        p0=initial_params,
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
        curve,
        f_grid,
        norm_factor * derivative_mu,
        covariance,
        z_value,
    )

    return {
        "params": (norm_factor, mu_mm_inv),
        "cov": covariance,
        "curve": curve,
        "lo": lower_band,
        "hi": upper_band,
    }


def fit_attenuation_with_r3_bands(
    radius_mm,
    counts,
    errors,
    radius_grid_mm=R_GRID,
    z_value=Z_BAND,
):
    """Fit exponential + 1/r^3 attenuation model and compute confidence bands."""
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
    amplitude_guess = float(np.exp(log_amplitude_guess))
    residual_guess = counts_fit - amplitude_guess * np.exp(slope_guess * radius_fit)
    r3_basis = np.where(radius_fit > 0, radius_fit**-3, 0.0)
    r3_denom = float(np.dot(r3_basis, r3_basis))
    r3_guess = float(np.dot(residual_guess, r3_basis) / r3_denom) if r3_denom else 0.0
    initial_params = (amplitude_guess, float(slope_guess), r3_guess)

    def model(x_values, amplitude, slope, r3_coeff):
        r_safe = np.maximum(x_values, 1.0)
        return amplitude * np.exp(slope * x_values) + r3_coeff * r_safe**-3

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
    amplitude, slope, r3_coeff = params
    curve = model(radius_grid_mm, amplitude, slope, r3_coeff)

    jac_amplitude = np.exp(slope * radius_grid_mm)
    jac_slope = amplitude * radius_grid_mm * np.exp(slope * radius_grid_mm)
    jac_r3 = np.maximum(radius_grid_mm, 1.0) ** -3
    lower_band, upper_band = _confidence_band_multi(
        curve,
        [jac_amplitude, jac_slope, jac_r3],
        covariance,
        z_value,
    )

    return {
        "params": (amplitude, slope, r3_coeff),
        "cov": covariance,
        "curve": curve,
        "lo": lower_band,
        "hi": upper_band,
        "mu": -slope,
        "r3_coeff": r3_coeff,
    }


# ---------- Exponential y = A * exp(k r) for others ----------
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


def fit_fixed_mu_with_bands(
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


# ---------- Utilities: curve intersections ----------
def _interp_log(x_interp, x_values, y_values):
    """Log-linear interpolate strictly positive y(x) at x_interp."""
    y_values = np.clip(y_values, 1e-300, np.inf)
    return np.exp(np.interp(x_interp, x_values, np.log(y_values)))


def find_intersections(x_values, first_curve, second_curve):
    """
    Return (x_crossings, y_crossings) where curves first_curve(x) and second_curve(x) cross.
    x_crossings, y_crossings are numpy arrays (could be multiple crossings).
    """
    first_curve = np.asarray(first_curve, float)
    second_curve = np.asarray(second_curve, float)
    delta = first_curve - second_curve
    crossing_indices = np.where(np.sign(delta[:-1]) * np.sign(delta[1:]) <= 0)[0]
    x_crossings, y_crossings = [], []
    for index in crossing_indices:
        x_left, x_right = x_values[index], x_values[index + 1]
        delta_left, delta_right = delta[index], delta[index + 1]
        if delta_right == delta_left:
            fraction = 0.0
        else:
            fraction = delta_left / (delta_left - delta_right)
        fraction = np.clip(fraction, 0.0, 1.0)
        x_interp = x_left + fraction * (x_right - x_left)
        y_first = _interp_log(x_interp, x_values, first_curve)
        y_second = _interp_log(x_interp, x_values, second_curve)
        y_interp = 0.5 * (y_first + y_second)
        x_crossings.append(x_interp)
        y_crossings.append(y_interp)
    return np.array(x_crossings), np.array(y_crossings)


def main():
    """Run the analysis and plot the results."""
    summary_frame = load_data(XLSX_PATH)

    series_by_component = {}
    for component in summary_frame["Component"].unique():
        component_data = summary_frame[summary_frame["Component"] == component]
        series_by_component[component] = {
            "radius_mm": component_data["r_mm"].to_numpy(),
            "counts": component_data["y"].to_numpy(),
            "errors": component_data["e"].fillna(0).to_numpy(),
        }

    excluded_components = {HFE_RN_PO_LABEL, "Water", "HFE"}
    for component in excluded_components:
        series_by_component.pop(component, None)

    baseline_radius = CFC_BASELINE_RADIUS_MM
    if baseline_radius is None:
        ivov_radii = []
        for component in ("IV", "OV"):
            if component in series_by_component:
                ivov_radii.append(series_by_component[component]["radius_mm"])
        if ivov_radii:
            baseline_radius = float(np.nanmax(np.concatenate(ivov_radii)))
        else:
            baseline_radius = float(np.nanmax(summary_frame["r_mm"].to_numpy()))

    preferred_components = [
        HFE_NO_RN_LABEL,
        HFE_RN_LABEL,
        "HFE",
        "IV",
        "OV",
        WATER_SEMI_ANALYTICAL_LABEL,
        "Transition Box",
    ]
    components_order = [
        component
        for component in preferred_components
        if component in series_by_component
    ]
    components_order.extend(
        [component for component in series_by_component if component not in components_order]
    )
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        component: default_colors[color_index % len(default_colors)]
        for color_index, component in enumerate(components_order)
    }

    fit_curves = {}
    bands = {}
    attenuation_by_component = {}

    # HFE (Th/U only): analytic fit + band
    for component in [HFE_NO_RN_LABEL]:
        if component not in series_by_component:
            continue
        component_series = series_by_component[component]
        result = fit_hfe_no_rn_with_bands(
            component_series["radius_mm"],
            component_series["counts"],
            component_series["errors"],
        )
        if result is None:
            continue
        norm_factor, mu_mm_inv = result["params"]
        fit_curves[component] = result["curve"]
        bands[component] = (result["lo"], result["hi"])
        attenuation_by_component[component] = mu_mm_inv
        print(f"{component}: μ = {mu_mm_inv:.4f} 1/mm, C = {norm_factor:.3g}")

    if HFE_RN_LABEL in series_by_component:
        component_series = series_by_component[HFE_RN_LABEL]
        result = fit_attenuation_with_r3_bands(
            component_series["radius_mm"],
            component_series["counts"],
            component_series["errors"],
        )
        if result is not None:
            fit_curves[HFE_RN_LABEL] = result["curve"]
            bands[HFE_RN_LABEL] = (result["lo"], result["hi"])
            attenuation_by_component[HFE_RN_LABEL] = result["mu"]
            print(f"{HFE_RN_LABEL}: μ = {result['mu']:.4e} 1/mm")

    # IV, OV: exponential fit + band
    for component in ["IV", "OV"]:
        if component in series_by_component:
            component_series = series_by_component[component]
            result = fit_attenuation_with_bands(
                component_series["radius_mm"],
                component_series["counts"],
                component_series["errors"],
            )
            if result is None:
                continue
            amplitude, slope = result["params"]
            attenuation_mm_inv = -slope
            fit_curves[component] = result["curve"]
            bands[component] = (result["lo"], result["hi"])
            attenuation_by_component[component] = attenuation_mm_inv
            print(
                f"{component}: μ = {attenuation_mm_inv:.4f} 1/mm "
                f"(A = {amplitude:.3g}, k = {slope:.4g})"
            )

    # Water: semi-analytical model (no Rn)
    water_result = build_water_semi_analytical_curve_with_bands(
        radius_grid_mm=_WATER_R_GRID, include_rn=False
    )
    fit_curves[WATER_SEMI_ANALYTICAL_LABEL] = np.interp(
        R_GRID, _WATER_R_GRID, water_result["curve"]
    )
    bands[WATER_SEMI_ANALYTICAL_LABEL] = (
        np.interp(R_GRID, _WATER_R_GRID, water_result["lo"]),
        np.interp(R_GRID, _WATER_R_GRID, water_result["hi"]),
    )
    print(f"{WATER_SEMI_ANALYTICAL_LABEL}: semi-analytical model (Th/U, no Rn)")

    # Transition box: fixed-mu fit + band
    if "Transition Box" in series_by_component:
        component_series = series_by_component["Transition Box"]
        result = fit_fixed_mu_with_bands(
            component_series["radius_mm"],
            component_series["counts"],
            component_series["errors"],
        )
        if result is not None:
            amplitude, _ = result["params"]
            mu_mm_inv = result["mu"]
            fit_curves["Transition Box"] = result["curve"]
            bands["Transition Box"] = (result["lo"], result["hi"])
            attenuation_by_component["Transition Box"] = mu_mm_inv
            print(
                "Transition Box: mu fixed at "
                f"{mu_mm_inv:.4e} 1/mm (A = {amplitude:.3g})"
            )

    # ---------- Crossings vs HFE (Th/U only) ----------
    baseline = HFE_NO_RN_LABEL
    if baseline in fit_curves:
        for component in [WATER_SEMI_ANALYTICAL_LABEL, "OV", "IV", "Transition Box"]:
            if component in fit_curves:
                crossing_radii, crossing_counts = find_intersections(
                    R_GRID,
                    fit_curves[component],
                    fit_curves[baseline],
                )
                if crossing_radii.size:
                    for crossing_index, (radius_mm, count_value) in enumerate(
                        zip(crossing_radii, crossing_counts),
                        1,
                    ):
                        print(
                            f"{component} crosses {baseline} at r ≈ {radius_mm:.1f} mm, "
                            f"y ≈ {count_value:.3g} counts/yr (#{crossing_index})"
                        )
                else:
                    print(
                        f"{component} does not cross {baseline} on "
                        f"[{R_GRID.min():.0f}, {R_GRID.max():.0f}] mm."
                    )

    # ---------- Plot ----------
    _, axis = plt.subplots(figsize=(10, 7))
    mass_grid_t = hfe_mass_tonnes_from_radius_mm(R_GRID)
    baseline_mass = float(hfe_mass_tonnes_from_radius_mm(baseline_radius))

    display_names = {WATER_SEMI_ANALYTICAL_LABEL: "Water"}

    # fit curves + shaded confidence bands
    for component, curve in fit_curves.items():
        color = color_map.get(component, None)
        axis.semilogy(
            mass_grid_t,
            curve,
            "--",
            linewidth=1.2,
            alpha=0.9,
            color=color,
            label=display_names.get(component, component),
        )
        if component in bands:
            lower_band, upper_band = bands[component]
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

    # Total intrinsic background marker (consistent with other _only plots)
    axis.plot(
        [baseline_mass],
        [TOTAL_INTRINSIC_BACKGROUND],
        marker="s",
        ms=5,
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

    # Design configuration vertical bars
    for design_label, (r_mm, y_frac, va) in DESIGN_CONFIGS.items():
        mass_t = float(hfe_mass_tonnes_from_radius_mm(r_mm))
        col = DESIGN_COLORS[design_label]
        axis.axvline(mass_t, color=col, linestyle=(0, (4, 2)), linewidth=1.5)
        axis.text(
            mass_t, y_frac, design_label,
            rotation=90, ha="right", va=va,
            transform=axis.get_xaxis_transform(),
            fontsize=FS_TICK, color=col,
        )

    axis.set_xlabel("HFE mass (tonnes)", fontsize=FS_LABEL)
    axis.set_ylabel("Background rate [cts/(y·2t·FWHM)]", fontsize=FS_LABEL)
    axis.set_yscale("log")
    axis.set_ylim(1e-5, 1e0)
    axis.set_xlim(float(mass_grid_t.min()), X_AXIS_MAX_TONNES)
    axis.tick_params(axis="both", which="major", labelsize=FS_TICK)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=FS_LEGEND, loc="upper left", ncols=2)

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
