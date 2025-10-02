#!/usr/bin/env python3
"""
Background vs IV radius analysis from 'Summary' sheet.

"""

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# --- Config ---------------------------------------------------------------
XLSX_PATH = ("/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/"
             "nickel-cryostats-analysis/budget/bkgd_vs_hfe-shield.xlsx")
R_GRID = np.linspace(950, 1800, 600)  # mm

# TPC geometry for analytic model (cm)
R_TPC = 56.665
H_TPC = 59.15

# --- Plot style -----------------------------------------------------------
mpl.rcParams.update({
    "figure.dpi": 160,
    "savefig.dpi": 300,
    "axes.spines.top": True,
    "axes.spines.right": True,
    "axes.linewidth": 0.8,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 6,
    "xtick.minor.size": 3,
    "ytick.major.size": 6,
    "ytick.minor.size": 3,
    "font.size": 12,
    "legend.frameon": True,
})

# --- Data I/O -------------------------------------------------------------
def load_data(filepath):
    """Load data from Excel file and return processed DataFrame."""
    data_frame = pd.read_excel(filepath, sheet_name="Summary", header=None)
    for i, (_, row) in enumerate(data_frame.iterrows()):
        vals = [str(v).strip().lower() for v in row.values]
        if all(h in vals for h in ["component", "iv radius (mm)", "background", "error"]):
            data_frame = data_frame.iloc[i+1:, :4].copy()
            data_frame.columns = ["Component", "r_mm", "y", "e"]
            break
    data_frame["Component"] = data_frame["Component"].ffill().astype(str).str.strip()
    for column in ["r_mm", "y", "e"]:
        data_frame[column] = pd.to_numeric(data_frame[column], errors="coerce")
    mapping = {
        "hfe (no rn-222)": "HFE (no Rn-222)",
        "hfe": "HFE",
        "iv": "IV",
        "ov": "OV",
        "water": "Water",
        "water (theoretical)": "Water (theoretical)",
    }
    data_frame["Component"] = (data_frame["Component"].str.lower()
                               .map(mapping).fillna(data_frame["Component"]))
    return data_frame.dropna(subset=["r_mm", "y"]).reset_index(drop=True)

# --- Fitting utilities ---
def fit_exp_curve(radius, y_vals, errors, relerr_floor=1e-3):
    """
    Fit y = A * exp(k r) in log-space with WLS:
      ln y = b0 + b1 r, weights = 1/σ_ln^2 with σ_ln ≈ e/y.
    Computes σ_ln inline (previously _safe_rel_err).
    Returns (mu, y_fit_grid) with μ = -k in 1/mm.
    """
    radius = np.asarray(radius, float)
    y_vals = np.asarray(y_vals, float)
    errors = np.asarray(errors, float)

    # Inline relative errors for log-space weighting (σ_ln ≈ e / y)
    sigma_rel = np.full_like(y_vals, relerr_floor, dtype=float)
    mask0 = (y_vals > 0) & np.isfinite(y_vals)
    sigma_rel[mask0] = np.maximum(errors[mask0] / y_vals[mask0], relerr_floor)

    mask = mask0 & np.isfinite(radius) & np.isfinite(y_vals)
    if mask.sum() < 2:
        return None

    radius_masked = radius[mask]
    ln_y_vals = np.log(y_vals[mask])
    weights = 1.0 / (sigma_rel[mask] ** 2)

    # Weighted normal equations without building a big diagonal matrix
    design_matrix = np.vstack([np.ones_like(radius_masked), radius_masked]).T
    weighted_design = design_matrix * weights[:, None]
    normal_matrix = design_matrix.T @ weighted_design
    normal_vector = design_matrix.T @ (weights * ln_y_vals)
    b0, b1 = np.linalg.solve(normal_matrix, normal_vector)

    mu = -b1
    y_fit = np.exp(b0 + b1 * R_GRID)
    return mu, y_fit

# --- Analytic HFE (single function) ---------------------------------------
def analytic_hfe_curve(radius_mm, y_vals, errors, mu_min=0.02, mu_max=0.5, n_mu=2000):
    """
    Single-entry analytic HFE fitter:
      - builds the cumulative model with explicit 4π r^2 and f_solid = 0.5*(R_TPC/r)^2
        (no path-length factor beyond exp(-μ Δr));
      - scans μ in [mu_min, mu_max] (cm^-1);
      - for each μ, computes the optimal normalization C(μ) by weighted least squares
        at the data points; evaluates χ²(μ);
      - returns best μ (in 1/mm) and the best-fit curve on R_GRID (in counts/yr units).

    Inputs are data radii radius_mm (mm), y_vals (counts/yr), errors (errors in counts/yr).
    """
    radius_mm = np.asarray(radius_mm, float)
    y_vals = np.asarray(y_vals, float)
    errors = np.asarray(errors, float)

    mask = np.isfinite(radius_mm) & np.isfinite(y_vals) & (y_vals > 0)
    if mask.sum() < 2:
        return None

    # Grids in cm
    r_cm_grid = R_GRID / 10.0
    r_cm_pts = radius_mm[mask] / 10.0
    y_pts = y_vals[mask]
    # small floor for weights to avoid division by zero
    e_pts = np.where((errors[mask] > 0) & np.isfinite(errors[mask]),
                     errors[mask], 0.01 * y_vals[mask])

    mu_space = np.linspace(mu_min, mu_max, n_mu)  # cm^-1
    best_chi2 = np.inf
    best_mu_cm = None
    best_normalization = None

    # Precompute static geometric factors on the grid
    f_solid_grid = np.where(r_cm_grid >= R_TPC, 0.5 * (R_TPC / r_cm_grid) ** 2, 0.0)
    dr = r_cm_grid[1] - r_cm_grid[0]

    for mu_cm in mu_space:
        # build cumulative shape on grid (unnormalized)
        dist_grid = np.maximum(r_cm_grid - R_TPC, 0.0)
        atten_grid = np.exp(-mu_cm * dist_grid)
        integrand = 4.0 * np.pi * (r_cm_grid ** 2) * f_solid_grid * atten_grid
        model_grid = np.cumsum(integrand) * dr  # shape vs r on R_GRID

        # interpolate to the data radii
        model_at_pts = np.interp(r_cm_pts, r_cm_grid, model_grid)

        # optimal normalization C(μ): minimize Σ ((C m_i - y_i)/σ_i)^2
        weights = 1.0 / (e_pts**2)
        numerator = np.sum(weights * model_at_pts * y_pts)
        denominator = np.sum(weights * model_at_pts * model_at_pts)
        if denominator <= 0 or not np.isfinite(denominator):
            continue
        normalization = numerator / denominator

        residuals = (normalization * model_at_pts - y_pts) / e_pts
        chi2 = np.sum(residuals**2)

        if chi2 < best_chi2:
            best_chi2 = chi2
            best_mu_cm = mu_cm
            best_normalization = normalization
            best_model_grid = model_grid  # keep the unscaled shape

    if best_mu_cm is None:
        return None

    # Best-fit curve on the plotting grid
    if best_model_grid is None:
        return None
    y_fit_grid = best_normalization * best_model_grid
    mu_best_mm = best_mu_cm / 10.0  # convert cm^-1 → mm^-1
    return mu_best_mm, y_fit_grid

# --- Main ---------------------------------------------------------------
data_frame = load_data(XLSX_PATH)

series = {}
for comp in data_frame["Component"].unique():
    d = data_frame[data_frame["Component"] == comp]
    series[comp] = {"r": d["r_mm"].to_numpy(),
                    "y": d["y"].to_numpy(),
                    "e": d["e"].fillna(0).to_numpy()}

comps_order = [c for c in ["HFE (no Rn-222)", "HFE", "IV", "OV", "Water", "Water (theoretical)"]
               if c in series]
palette = plt.get_cmap("tab10").colors
markers = ["o", "s", "^", "D", "P", "v", "X"]
color_map = {c: palette[i % len(palette)] for i, c in enumerate(comps_order)}
marker_map = {c: markers[i % len(markers)] for i, c in enumerate(comps_order)}

fit_results = {}

# Analytic HFE
if "HFE (no Rn-222)" in series:
    d = series["HFE (no Rn-222)"]
    res = analytic_hfe_curve(d["r"], d["y"], d["e"])
    if res is not None:
        mu_mm, y_fit = res
        fit_results["HFE (no Rn-222)"] = {"mu": mu_mm, "y_fit": y_fit}

# Exponentials (skip Water MC)
for comp in ["HFE", "IV", "OV", "Water (theoretical)"]:
    if comp in series:
        radius = series[comp]["r"]
        y_vals = series[comp]["y"]
        errors = series[comp]["e"]
        res = fit_exp_curve(radius, y_vals, errors)
        if res is not None:
            mu, y_fit = res
            fit_results[comp] = {"mu": mu, "y_fit": y_fit}

# --- Plot ---------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7.0, 4.8), constrained_layout=True)

# Data points
for comp in comps_order:
    d = series[comp]
    ax.errorbar(
        d["r"], d["y"], yerr=d["e"],
        fmt=marker_map[comp],
        ms=3.0,
        mew=0.8,
        mec=color_map[comp],
        mfc=color_map[comp],
        elinewidth=0.6,
        capsize=1.5,
        capthick=0.6,
        color=color_map[comp],
        label=comp,
        zorder=4
    )

# Fits
for comp, res in fit_results.items():
    color = color_map.get(comp, "0.3")
    y_fit = res["y_fit"]
    ax.semilogy(R_GRID, y_fit, "-", lw=1.0, color=color, alpha=0.95, zorder=3, label="_nolegend_")

# Axes
ax.set_xlabel("Inner vessel radius (mm)")
ax.set_ylabel("Background (counts/yr)")
ax.set_yscale("log")
ax.set_xlim(R_GRID.min(), R_GRID.max())
ax.set_ylim(1e-6, 2e-1)
ax.grid(True, which="both", linestyle=":", alpha=0.35)

# Legend
leg = ax.legend(loc="best", ncol=2, handlelength=1.4, columnspacing=0.8,
                framealpha=0.9, fontsize=10)
leg.get_frame().set_linewidth(0.8)

# Optional: print μ
for comp, res in fit_results.items():
    print(f"{comp}: μ = {res['mu']:.4f} 1/mm")

ax.set_title("Background Contribution vs IV Radius", pad=8)
plt.show()
