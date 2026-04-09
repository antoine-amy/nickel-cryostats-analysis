#!/usr/bin/env python3
"""
HFE background vs HFE mass from the 'Summary' sheet.

Fits HFE (Th/U only) and HFE (Th/U & Po) with the analytic geometry model and
builds HFE (Th/U & Rn) from the Th/U baseline plus a baseline-normalized radon
model derived from the same transport kernel, with the IV contribution carrying
the additional radius-squared scaling, then plots only those lines.
"""

from pathlib import Path
import sys
from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

# --- Config ---
XLSX_PATH = Path(__file__).resolve().parent / "Summary_bkgd_vs_hfe-shield.xlsx"
R_GRID = np.linspace(950, 1800, 600)  # mm

# Finite-cylinder TPC geometry matching the acceptance calibration.
R_TPC_CM = 56.665
TPC_HALF_HEIGHT_CM = 59.15
TPC_RADIUS_M = R_TPC_CM / 100.0
TPC_HALF_HEIGHT_M = TPC_HALF_HEIGHT_CM / 100.0

# Transport tabulation for the hit-conditioned attenuation factor T_hit(r, mu).
TRANSPORT_RADIUS_GRID_MM = np.linspace(float(R_GRID.min()), float(R_GRID.max()), 96)
TRANSPORT_SAMPLES_PER_RADIUS = 25000
TRANSPORT_SEED = 20260401

Z_BAND = 1.0  # 1-sigma confidence band; set 1.96 for ~95%

# HFE-7200 attenuation at 2.5 MeV:
#   (mu/rho) = 3.898e-2 cm^2/g from NIST XCOM for C6H5F9O
#   rho(T)   = 1.4811 - 0.0023026*T_C g/ml from the 3M Novec 7200 datasheet
# using the repo's nominal cryogenic operating temperature of 165 K.
HFE_OPERATING_TEMPERATURE_K = 165.0
HFE7200_MASS_ATTENUATION_CM2_PER_G = 3.898e-2
HFE7200_DENSITY_G_PER_ML = 1.4811 - 0.0023026 * (HFE_OPERATING_TEMPERATURE_K - 273.15)
MU_HFE_2P5MEV_MM = (
    HFE7200_MASS_ATTENUATION_CM2_PER_G * HFE7200_DENSITY_G_PER_ML / 10.0
)

# Import the radius -> HFE mass conversion from the shared geometry script.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfe_volume_to_iv_radius import (
    BASELINE_TARGET_MASS_KG_1,
    calibration_loss_tonnes_for_target,
    hfe_mass_tonnes,
)

# Baseline radius for annotation; set to None to use max radius in data.
BASELINE_RADIUS_MM = None
TOTAL_INTRINSIC_BACKGROUND = 0.55

# Font sizes
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14
BASELINE_LABEL_Y = 0.12
X_AXIS_MAX_TONNES = 35.0

HFE_NO_RN_LABEL = "HFE (Th/U only)"
HFE_RN_LABEL = "HFE (Th/U & Rn)"
HFE_PO_LABEL = "HFE (Th/U & Po)"
LEGEND_LABELS = {
    HFE_NO_RN_LABEL: "HFE: Th/U only",
    HFE_RN_LABEL: "HFE: Th/U + Rn",
    HFE_PO_LABEL: "HFE: Th/U + Po",
}
CALIBRATION_LOSS_TONNES = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_1)
SOLID_ANGLE_A_M2 = 0.19575476
SOLID_ANGLE_B_M3 = 0.099439452


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

    data_frame = data_frame.iloc[header_row_index + 1 :, :4].copy()
    data_frame.columns = ["Component", "r_mm", "y", "e"]

    data_frame["Component"] = data_frame["Component"].ffill().astype(str).str.strip()
    for column in ["r_mm", "y", "e"]:
        data_frame[column] = pd.to_numeric(data_frame[column], errors="coerce")

    mapping = {
        "hfe (no rn-222)": HFE_NO_RN_LABEL,
        "hfe (th/u only)": HFE_NO_RN_LABEL,
        "hfe (th/u & rn)": HFE_RN_LABEL,
        "hfe (th/u and rn)": HFE_RN_LABEL,
        "hfe (th/u & po)": HFE_PO_LABEL,
        "hfe (th/u and po)": HFE_PO_LABEL,
        "hfe (th/u & rn & po)": HFE_PO_LABEL,
        "hfe (th/u and rn and po)": HFE_PO_LABEL,
        "hfe": HFE_PO_LABEL,
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


def _sample_points_on_sphere(n_samples: int, radius_m: float, rng: np.random.Generator):
    points = rng.normal(size=(n_samples, 3))
    points /= np.linalg.norm(points, axis=1)[:, np.newaxis]
    return radius_m * points


def _sample_dirs_inward_hemisphere(points: np.ndarray, rng: np.random.Generator):
    n_samples = points.shape[0]
    inward_axis = -points / np.linalg.norm(points, axis=1)[:, np.newaxis]
    helper = np.zeros_like(inward_axis)
    use_z_axis = np.abs(inward_axis[:, 2]) < 0.9
    helper[use_z_axis] = np.array([0.0, 0.0, 1.0])
    helper[~use_z_axis] = np.array([1.0, 0.0, 0.0])

    tangent_1 = np.cross(helper, inward_axis)
    tangent_1 /= np.linalg.norm(tangent_1, axis=1)[:, np.newaxis]
    tangent_2 = np.cross(inward_axis, tangent_1)

    cos_polar = rng.random(n_samples)
    sin_polar = np.sqrt(np.maximum(0.0, 1.0 - cos_polar**2))
    azimuth = rng.uniform(0.0, 2.0 * np.pi, size=n_samples)
    cos_azimuth = np.cos(azimuth)
    sin_azimuth = np.sin(azimuth)
    return (
        (sin_polar * cos_azimuth)[:, np.newaxis] * tangent_1
        + (sin_polar * sin_azimuth)[:, np.newaxis] * tangent_2
        + cos_polar[:, np.newaxis] * inward_axis
    )


def _first_hit_distance_m(
    points: np.ndarray,
    directions: np.ndarray,
    cylinder_radius_m: float = TPC_RADIUS_M,
    cylinder_half_height_m: float = TPC_HALF_HEIGHT_M,
    eps: float = 1e-14,
):
    """Return first-hit distances for a finite cylinder; misses are inf."""
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]
    vx, vy, vz = directions[:, 0], directions[:, 1], directions[:, 2]
    distances = np.full(points.shape[0], np.inf)

    quadratic_a = vx * vx + vy * vy
    quadratic_b = 2.0 * (px * vx + py * vy)
    quadratic_c = px * px + py * py - cylinder_radius_m**2

    side_mask = quadratic_a > eps
    if np.any(side_mask):
        a_side = quadratic_a[side_mask]
        b_side = quadratic_b[side_mask]
        c_side = quadratic_c[side_mask]
        discriminant = b_side * b_side - 4.0 * a_side * c_side
        valid_disc = discriminant >= 0.0
        side_indices = np.where(side_mask)[0]
        hit_indices = side_indices[valid_disc]
        if hit_indices.size:
            sqrt_disc = np.sqrt(np.maximum(discriminant[valid_disc], 0.0))
            a_hit = quadratic_a[hit_indices]
            b_hit = quadratic_b[hit_indices]
            t1 = (-b_hit - sqrt_disc) / (2.0 * a_hit)
            t2 = (-b_hit + sqrt_disc) / (2.0 * a_hit)
            side_distance = np.where(
                (t1 > 0.0) & (t2 > 0.0),
                np.minimum(t1, t2),
                np.where(t1 > 0.0, t1, np.where(t2 > 0.0, t2, np.inf)),
            )
            z_at_hit = pz[hit_indices] + side_distance * vz[hit_indices]
            valid_side = (
                (side_distance < np.inf)
                & (z_at_hit >= -cylinder_half_height_m)
                & (z_at_hit <= +cylinder_half_height_m)
            )
            distances[hit_indices[valid_side]] = np.minimum(
                distances[hit_indices[valid_side]],
                side_distance[valid_side],
            )

    cap_mask = np.abs(vz) > eps
    if np.any(cap_mask):
        cap_indices = np.where(cap_mask)[0]
        vz_cap = vz[cap_indices]

        top_distance = (cylinder_half_height_m - pz[cap_indices]) / vz_cap
        x_top = px[cap_indices] + top_distance * vx[cap_indices]
        y_top = py[cap_indices] + top_distance * vy[cap_indices]
        valid_top = (top_distance > 0.0) & (
            x_top * x_top + y_top * y_top <= cylinder_radius_m**2
        )
        distances[cap_indices[valid_top]] = np.minimum(
            distances[cap_indices[valid_top]],
            top_distance[valid_top],
        )

        bottom_distance = (-cylinder_half_height_m - pz[cap_indices]) / vz_cap
        x_bottom = px[cap_indices] + bottom_distance * vx[cap_indices]
        y_bottom = py[cap_indices] + bottom_distance * vy[cap_indices]
        valid_bottom = (bottom_distance > 0.0) & (
            x_bottom * x_bottom + y_bottom * y_bottom <= cylinder_radius_m**2
        )
        distances[cap_indices[valid_bottom]] = np.minimum(
            distances[cap_indices[valid_bottom]],
            bottom_distance[valid_bottom],
        )

    return distances


@lru_cache(maxsize=1)
def _hit_path_samples_by_radius():
    """Tabulate hit-path samples once so mu scans reuse the same transport kernel."""
    rng = np.random.default_rng(TRANSPORT_SEED)
    hit_path_samples_cm = []
    for radius_mm in TRANSPORT_RADIUS_GRID_MM:
        radius_m = radius_mm / 1000.0
        points = _sample_points_on_sphere(TRANSPORT_SAMPLES_PER_RADIUS, radius_m, rng)
        directions = _sample_dirs_inward_hemisphere(points, rng)
        hit_distances_m = _first_hit_distance_m(points, directions)
        hit_mask = np.isfinite(hit_distances_m)
        hit_path_samples_cm.append(
            (100.0 * hit_distances_m[hit_mask]).astype(np.float32, copy=False)
        )
    return tuple(hit_path_samples_cm)


def _hit_conditioned_transmission(mu_mm_inv, radius_grid_mm=R_GRID):
    """
    Return T_hit(r, mu) = <exp(-mu * l)> averaged over accepted rays only.

    Combined with the separate acceptance fit this yields the correct inward-
    hemisphere transport factor P(hit) * T_hit instead of an unconditional
    shell-average attenuation.
    """
    mu_cm_inv = mu_mm_inv * 10.0
    transport = np.empty_like(TRANSPORT_RADIUS_GRID_MM, dtype=float)
    for index, hit_paths_cm in enumerate(_hit_path_samples_by_radius()):
        if hit_paths_cm.size == 0:
            transport[index] = 0.0
            continue
        transport[index] = float(np.mean(np.exp(-mu_cm_inv * hit_paths_cm)))
    return np.interp(radius_grid_mm, TRANSPORT_RADIUS_GRID_MM, transport)


def _analytic_f_on_grid(mu_mm_inv, radius_grid_mm=R_GRID):
    """
    Unnormalized cumulative shape F(r; mu) on R_GRID (mm).

    The model keeps the fitted geometric acceptance separate from the HFE
    self-attenuation, but uses the attenuation averaged over hit rays only.
    """
    radius_cm_grid = radius_grid_mm / 10.0

    # MC-calibrated geometric acceptance (fit valid for ~0.95-1.70 m)
    a_m2 = 0.19575476
    b_m3 = 0.099439452

    radius_m_grid = radius_cm_grid / 100.0
    solid_fraction = np.where(
        radius_m_grid > 0,
        0.5 * (a_m2 / radius_m_grid**2 + b_m3 / radius_m_grid**3),
        0.0,
    )
    transmission_hit = _hit_conditioned_transmission(mu_mm_inv, radius_grid_mm)
    integrand = 4.0 * np.pi * (radius_cm_grid**2) * solid_fraction * transmission_hit
    delta_r = radius_cm_grid[1] - radius_cm_grid[0]
    cumulative = np.cumsum(integrand) * delta_r
    return cumulative


def _confidence_band(curve, jacobians, covariance, z_value):
    """Compute confidence bands via the delta method."""
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


def _solid_angle_acceptance(radius_mm):
    """Return the fitted full-sphere geometric acceptance f_Omega(r)."""
    radius_m = np.asarray(radius_mm, float) / 1000.0
    return np.where(
        radius_m > 0.0,
        0.5 * (SOLID_ANGLE_A_M2 / radius_m**2 + SOLID_ANGLE_B_M3 / radius_m**3),
        0.0,
    )


def _shell_coupling(radius_mm, mu_mm_inv=MU_HFE_2P5MEV_MM):
    """Shell-averaged geometric acceptance times HFE transmission."""
    return _solid_angle_acceptance(radius_mm) * _hit_conditioned_transmission(
        mu_mm_inv,
        radius_mm,
    )


@lru_cache(maxsize=1)
def _load_hfe_rn_baseline_split():
    """
    Read the baseline-design radon split from the HFE sheet.

    The split is used only to decompose the baseline radon residual into:
      - an IV-surface term, whose source strength scales as r_IV^2
      - all other radon terms, treated as radius-independent source strength
    """
    hfe_frame = pd.read_excel(XLSX_PATH, sheet_name="HFE", header=None).iloc[:, :9]

    blocks = []
    current_block = []
    for row in hfe_frame.itertuples(index=False, name=None):
        radius_value = row[0]
        isotope = row[1]
        is_radius_row = (
            isinstance(radius_value, (int, float, np.integer, np.floating))
            and np.isfinite(radius_value)
            and isinstance(isotope, str)
        )
        if is_radius_row:
            if current_block:
                blocks.append(current_block)
            current_block = [row]
            continue
        if current_block and isinstance(isotope, str) and isotope:
            current_block.append(row)
            continue
        if current_block:
            break
    if current_block:
        blocks.append(current_block)

    baseline_block = None
    baseline_radius = -np.inf
    for block in blocks:
        radius_mm = float(block[0][0])
        if radius_mm > baseline_radius:
            baseline_radius = radius_mm
            baseline_block = block

    if baseline_block is None:
        raise ValueError("No baseline radon block found in HFE sheet.")

    rows_by_isotope = {
        str(row[1]).strip(): row
        for row in baseline_block
        if isinstance(row[1], str) and str(row[1]).strip()
    }
    required_isotopes = [
        "Rn222 (Calibration)",
        "Rn222 (Conduits)",
        "Rn222 (Pump+Piping)",
        "Rn222 (IV)",
    ]
    missing = [isotope for isotope in required_isotopes if isotope not in rows_by_isotope]
    if missing:
        raise ValueError(f"Missing baseline radon rows in HFE sheet: {missing}")

    calibration = rows_by_isotope["Rn222 (Calibration)"]
    conduits = rows_by_isotope["Rn222 (Conduits)"]
    pump = rows_by_isotope["Rn222 (Pump+Piping)"]
    iv = rows_by_isotope["Rn222 (IV)"]

    other_background = float(calibration[7] + conduits[7] + pump[7])
    other_error = float(
        np.sqrt(float(calibration[8]) ** 2 + float(conduits[8]) ** 2 + float(pump[8]) ** 2)
    )
    iv_background = float(iv[7])
    iv_error = float(iv[8])

    return {
        "radius_mm": float(baseline_radius),
        "other_background": other_background,
        "other_error": other_error,
        "iv_background": iv_background,
        "iv_error": iv_error,
    }


def _mean_remaining_hfe_coupling(radius_mm, mu_mm_inv=MU_HFE_2P5MEV_MM):
    """
    Mean shell-coupling of the remaining HFE between the TPC corner and R_IV.

    This approximates the statement that, as outer HFE is removed, the mean
    geometric acceptance increases and the mean attenuation decreases because
    the remaining HFE is on average closer to the TPC.
    """
    radius_mm = np.asarray(radius_mm, float)
    corner_radius_mm = 1000.0 * np.hypot(TPC_RADIUS_M, TPC_HALF_HEIGHT_M)
    integration_grid_mm = np.linspace(corner_radius_mm, float(np.max(radius_mm)), 800)
    shell_coupling = _shell_coupling(integration_grid_mm, mu_mm_inv)
    radius_grid_cm = integration_grid_mm / 10.0
    delta_r_cm = (integration_grid_mm[1] - integration_grid_mm[0]) / 10.0
    cumulative_weighted_coupling = np.cumsum(
        4.0 * np.pi * radius_grid_cm**2 * shell_coupling
    ) * delta_r_cm
    cumulative_volume = (4.0 * np.pi / 3.0) * (
        radius_grid_cm**3 - radius_grid_cm[0] ** 3
    )
    mean_coupling = np.full_like(shell_coupling, shell_coupling[0], dtype=float)
    valid_volume = cumulative_volume > 0.0
    mean_coupling[valid_volume] = (
        cumulative_weighted_coupling[valid_volume] / cumulative_volume[valid_volume]
    )
    mean_coupling[0] = shell_coupling[0]
    return np.interp(radius_mm, integration_grid_mm, mean_coupling)


def _radon_transport_ratio(
    radius_mm,
    baseline_radius_mm,
    mu_mm_inv=MU_HFE_2P5MEV_MM,
):
    """
    Baseline-normalized mean transport factor for the radon term.

    This uses the same finite-cylinder acceptance-plus-attenuation kernel as
    the Th/U model, but removes the overall mass scaling by dividing by the
    remaining-HFE volume.
    """
    radius_mm = np.asarray(radius_mm, float)
    mean_coupling = _mean_remaining_hfe_coupling(radius_mm, mu_mm_inv)
    baseline_mean_coupling = float(
        _mean_remaining_hfe_coupling(np.array([baseline_radius_mm]), mu_mm_inv)[0]
    )
    if baseline_mean_coupling <= 0.0:
        raise ValueError("Baseline mean HFE coupling is non-positive.")
    return mean_coupling / baseline_mean_coupling


def build_hfe_rn_baseline_model_with_bands(
    radius_mm,
    counts,
    errors,
    th_u_result=None,
    mu_mm_inv=MU_HFE_2P5MEV_MM,
    radius_grid_mm=R_GRID,
    z_value=Z_BAND,
):
    """
    Build HFE (Th/U & Rn) from the Th/U baseline and an additive radon model.

    The radon term uses the baseline-design spreadsheet values for the non-IV
    and IV contributions. Its evolution is derived from the same transport
    kernel as the Th/U model, but without the bulk-HFE mass scaling; only the
    IV contribution carries the extra r_IV^2 source-strength scaling.
    """
    radius_mm = np.asarray(radius_mm, float)
    counts = np.asarray(counts, float)
    errors = np.asarray(errors, float)
    valid_mask = np.isfinite(radius_mm) & np.isfinite(counts)
    if valid_mask.sum() < 2:
        return None

    radius_fit = radius_mm[valid_mask]
    counts_fit = counts[valid_mask]
    errors_fit = errors[valid_mask]
    sigma = _safe_sigma(errors_fit)

    if th_u_result is None:
        th_u_fit = np.zeros_like(radius_fit)
        th_u_curve = np.zeros_like(radius_grid_mm, dtype=float)
        th_u_sigma_grid = np.zeros_like(radius_grid_mm, dtype=float)
    else:
        th_u_fit = np.interp(radius_fit, radius_grid_mm, th_u_result["curve"])
        th_u_curve = th_u_result["curve"]
        if th_u_result["lo"] is not None and th_u_result["hi"] is not None:
            th_u_sigma_grid = 0.5 * (th_u_result["hi"] - th_u_result["lo"])
        else:
            th_u_sigma_grid = np.zeros_like(radius_grid_mm, dtype=float)

    baseline_inputs = _load_hfe_rn_baseline_split()
    baseline_radius_mm = baseline_inputs["radius_mm"]
    background_other = baseline_inputs["other_background"]
    background_other_error = baseline_inputs["other_error"]
    background_iv = baseline_inputs["iv_background"]
    background_iv_error = baseline_inputs["iv_error"]

    transport_ratio_grid = _radon_transport_ratio(
        radius_grid_mm,
        baseline_radius_mm,
        mu_mm_inv,
    )
    radon_curve = transport_ratio_grid * (
        background_other
        + background_iv * (radius_grid_mm / baseline_radius_mm) ** 2
    )
    total_curve = th_u_curve + radon_curve

    iv_shape_grid = (radius_grid_mm / baseline_radius_mm) ** 2
    radon_sigma_grid = transport_ratio_grid * np.sqrt(
        background_other_error**2 + (background_iv_error * iv_shape_grid) ** 2
    )
    total_sigma_grid = np.sqrt(np.maximum(radon_sigma_grid**2 + th_u_sigma_grid**2, 0.0))
    lower_band = np.clip(total_curve - total_sigma_grid, 1e-300, np.inf)
    upper_band = total_curve + total_sigma_grid

    return {
        "params": (background_other, background_iv, mu_mm_inv),
        "cov": None,
        "curve": total_curve,
        "lo": lower_band,
        "hi": upper_band,
        "mu": mu_mm_inv,
        "background_other": background_other,
        "background_iv": background_iv,
        "rn_curve": radon_curve,
        "baseline_radius_mm": baseline_radius_mm,
    }


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
        [f_grid, norm_factor * derivative_mu],
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


def fit_hfe_fixed_mu_with_bands(
    radius_mm,
    counts,
    errors,
    mu_mm_inv=MU_HFE_2P5MEV_MM,
    radius_grid_mm=R_GRID,
    z_value=Z_BAND,
):
    """Fit B = B0 + A*F(r; mu_fixed) with a fixed HFE attenuation coefficient."""
    radius_mm = np.asarray(radius_mm, float)
    counts = np.asarray(counts, float)
    errors = np.asarray(errors, float)
    valid_mask = np.isfinite(radius_mm) & np.isfinite(counts)
    if valid_mask.sum() < 2:
        return None

    radius_fit = radius_mm[valid_mask]
    counts_fit = counts[valid_mask]
    errors_fit = errors[valid_mask]

    cumulative_shape_grid = _analytic_f_on_grid(mu_mm_inv, radius_grid_mm)
    model_basis = np.interp(radius_fit, radius_grid_mm, cumulative_shape_grid)

    sigma = _safe_sigma(errors_fit)
    if sigma is None:
        weights = np.ones_like(counts_fit)
    else:
        weights = 1.0 / sigma**2

    design = np.column_stack([np.ones_like(model_basis), model_basis])
    weighted_design = design * np.sqrt(weights)[:, np.newaxis]
    weighted_counts = counts_fit * np.sqrt(weights)
    params, _, _, _ = np.linalg.lstsq(weighted_design, weighted_counts, rcond=None)
    background_offset, norm_factor = params

    if background_offset < 0.0:
        background_offset = 0.0
        denom = np.sum(weights * model_basis**2)
        if denom <= 0.0:
            return None
        norm_factor = float(np.sum(weights * model_basis * counts_fit) / denom)
        norm_factor = max(norm_factor, 0.0)
        variance = (1.0 / denom) if sigma is not None else None
        curve = norm_factor * cumulative_shape_grid
        lower_band, upper_band = _confidence_band_single(
            curve,
            cumulative_shape_grid,
            variance,
            z_value,
        )
        covariance = np.array([[variance]]) if variance is not None else None
        return {
            "params": (background_offset, norm_factor, mu_mm_inv),
            "cov": covariance,
            "curve": curve,
            "lo": lower_band,
            "hi": upper_band,
            "mu": mu_mm_inv,
            "b0": background_offset,
        }

    normal_matrix = weighted_design.T @ weighted_design
    if np.linalg.matrix_rank(normal_matrix) < normal_matrix.shape[0]:
        return None
    covariance = np.linalg.inv(normal_matrix) if sigma is not None else None

    curve = background_offset + norm_factor * cumulative_shape_grid
    if covariance is None:
        lower_band = upper_band = None
    else:
        jac_b0 = np.ones_like(curve)
        jac_norm = cumulative_shape_grid
        lower_band, upper_band = _confidence_band(
            curve,
            [jac_b0, jac_norm],
            covariance,
            z_value,
        )

    return {
        "params": (background_offset, norm_factor, mu_mm_inv),
        "cov": covariance,
        "curve": curve,
        "lo": lower_band,
        "hi": upper_band,
        "mu": mu_mm_inv,
        "b0": background_offset,
    }


def main():
    """Run the analysis and plot the results."""
    summary_frame = load_data(XLSX_PATH)
    hfe_components = [HFE_NO_RN_LABEL, HFE_RN_LABEL, HFE_PO_LABEL]
    summary_frame = summary_frame[
        summary_frame["Component"].isin(hfe_components)
    ].copy()

    series_by_component = {}
    for component in hfe_components:
        component_data = summary_frame[summary_frame["Component"] == component]
        if component_data.empty:
            continue
        series_by_component[component] = {
            "radius_mm": component_data["r_mm"].to_numpy(),
            "counts": component_data["y"].to_numpy(),
            "errors": component_data["e"].fillna(0).to_numpy(),
        }

    if not series_by_component:
        raise ValueError("No HFE data found in the Summary sheet.")

    baseline_radius = BASELINE_RADIUS_MM
    if baseline_radius is None:
        baseline_radius = float(summary_frame["r_mm"].max())
    baseline_mass = float(hfe_mass_tonnes_from_radius_mm(baseline_radius))

    components_order = [
        component
        for component in hfe_components
        if component in series_by_component
    ]
    default_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    color_map = {
        component: default_colors[color_index % len(default_colors)]
        for color_index, component in enumerate(components_order)
    }
    fit_curves = {}
    bands = {}
    fit_results = {}

    for component in [HFE_NO_RN_LABEL, HFE_PO_LABEL]:
        if component not in series_by_component:
            continue
        component_series = series_by_component[component]
        result = fit_hfe_fixed_mu_with_bands(
            component_series["radius_mm"],
            component_series["counts"],
            component_series["errors"],
        )
        if result is not None:
            b0, _, mu_mm_inv = result["params"]
            fit_results[component] = result
            fit_curves[component] = result["curve"]
            bands[component] = (result["lo"], result["hi"])
            print(f"{component}: mu fixed at {mu_mm_inv:.4e} 1/mm, B0 = {b0:.4e}")

    if HFE_RN_LABEL in series_by_component:
        component_series = series_by_component[HFE_RN_LABEL]
        th_u_result = fit_results.get(HFE_NO_RN_LABEL)
        result = build_hfe_rn_baseline_model_with_bands(
            component_series["radius_mm"],
            component_series["counts"],
            component_series["errors"],
            th_u_result,
        )
        if result is not None:
            fit_curves[HFE_RN_LABEL] = result["curve"]
            bands[HFE_RN_LABEL] = (result["lo"], result["hi"])
            fit_results[HFE_RN_LABEL] = result
            print(
                f"{HFE_RN_LABEL}: mu fixed at {result['mu']:.4e} 1/mm "
                f"B_other = {result['background_other']:.4e}, "
                f"B_IV = {result['background_iv']:.4e} "
                f"(baseline-normalized transport evolution from R={result['baseline_radius_mm']:.0f} mm)"
            )

    _, axis = plt.subplots(figsize=(10, 7))
    mass_grid_t = hfe_mass_tonnes_from_radius_mm(R_GRID)

    for component in components_order:
        component_series = series_by_component[component]
        axis.errorbar(
            hfe_mass_tonnes_from_radius_mm(component_series["radius_mm"]),
            component_series["counts"],
            yerr=component_series["errors"],
            fmt="o",
            ms=4,
            label=LEGEND_LABELS.get(component, component),
            color=color_map[component],
        )

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

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path)
    plt.close()


if __name__ == "__main__":
    main()
