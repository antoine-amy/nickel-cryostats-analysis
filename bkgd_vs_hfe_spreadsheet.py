#!/usr/bin/env python3
"""
Background vs IV radius analysis from 'Summary' sheet, with confidence bands,
plus printed intersection points against the HFE(no Rn-222) curve.

• HFE (no Rn-222): analytic geometry-based model (explicit 4π r^2 and
  f_solid(r) = 0.5 * (R_TPC / r)^2), NO path-length multiplier.
  We fit normalization C and attenuation μ, and propagate their covariance.

• Others (HFE, IV, OV, Water theoretical):
  y(r) = A * exp(k r). We fit A,k with weights from the y-errors,
  then compute bands via the delta method.

Bands shown are for the MEAN fit (confidence bands). For ~95% bands, set Z_BAND=1.96.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# --- Config ---
XLSX_PATH = ("/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/"
             "nickel-cryostats-analysis/budget/bkgd_vs_hfe-shield.xlsx")
R_GRID = np.linspace(950, 1800, 600)  # mm

# TPC geometry for analytic model (cm)
R_TPC = 56.665
H_TPC = 59.15  # unused in analytic model (kept for reference)

Z_BAND = 1.0  # 1σ confidence band; set 1.96 for ~95%

# Font sizes
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14

# ---------- Data loading ----------
def load_data(filepath):
    """Load data from Excel file and return processed DataFrame."""
    data_df = pd.read_excel(filepath, sheet_name="Summary", header=None)
    # find header row
    for i, (_, row) in enumerate(data_df.iterrows()):
        vals = [str(v).strip().lower() for v in row.values]
        if all(h in vals for h in ["component", "iv radius (mm)", "background", "error"]):
            data_df = data_df.iloc[i+1:, :4].copy()
            data_df.columns = ["Component", "r_mm", "y", "e"]
            break

    data_df["Component"] = data_df["Component"].ffill().astype(str).str.strip()
    for col in ["r_mm", "y", "e"]:
        data_df[col] = pd.to_numeric(data_df[col], errors="coerce")

    mapping = {
        "hfe (no rn-222)": "HFE (no Rn-222)",
        "hfe": "HFE",
        "iv": "IV",
        "ov": "OV",
        "water": "Water",
        "water (theoretical)": "Water (theoretical)",
    }
    data_df["Component"] = (data_df["Component"].str.lower()
                           .map(mapping).fillna(data_df["Component"]))
    return data_df.dropna(subset=["r_mm", "y"]).reset_index(drop=True)

def _safe_sigma(e: np.ndarray):
    """Return sigma array for curve_fit; if all zeros/NaN, return None (unweighted)."""
    e = np.asarray(e, float)
    ok = np.isfinite(e) & (e > 0)
    if not np.any(ok):
        return None
    min_pos = np.nanmin(e[ok])
    return np.where((~np.isfinite(e)) | (e <= 0), min_pos, e)

# ---------- Analytic HFE(no Rn-222) model ----------
def _analytic_F_on_grid(mu_mm_inv, r_grid_mm=R_GRID):
    """Unnormalized cumulative shape F(r; μ) on R_GRID (mm); model uses cm internally."""
    r_cm_grid = r_grid_mm / 10.0
    mu_cm_inv = mu_mm_inv * 10.0  # mm^-1 -> cm^-1
    f_solid = np.where(r_cm_grid >= R_TPC, 0.5 * (R_TPC / r_cm_grid)**2, 0.0)
    dist = np.maximum(r_cm_grid - R_TPC, 0.0)
    atten = np.exp(-mu_cm_inv * dist)
    integrand = 4.0 * np.pi * (r_cm_grid**2) * f_solid * atten
    dr = r_cm_grid[1] - r_cm_grid[0]
    F = np.cumsum(integrand) * dr
    return F

def fit_hfe_no_rn_with_bands(r_mm, y, e, xgrid=R_GRID, z=Z_BAND):
    r_mm = np.asarray(r_mm, float); y = np.asarray(y, float); e = np.asarray(e, float)
    mask = np.isfinite(r_mm) & np.isfinite(y)
    r_fit, y_fit, e_fit = r_mm[mask], y[mask], e[mask]
    if r_fit.size < 2:
        return None

    def model(r_mm, C, mu_mm):
        F = _analytic_F_on_grid(mu_mm)             # on base grid
        return C * np.interp(r_mm, R_GRID, F)      # interp to requested r

    # p0: quick guess
    mu0 = 0.01  # 1/mm
    F0 = _analytic_F_on_grid(mu0)
    C0 = (y_fit[-1] / np.interp(r_fit[-1], R_GRID, F0)) if F0[-1] > 0 else 1.0
    p0 = (float(C0), float(mu0))

    sigma = _safe_sigma(e_fit)
    popt, pcov = curve_fit(model, r_fit, y_fit, p0=p0, sigma=sigma,
                           absolute_sigma=True, maxfev=20000)
    C, mu = popt

    F_grid = _analytic_F_on_grid(mu)
    curve = C * F_grid

    # Jacobian wrt [C, μ] on xgrid: ∂y/∂C = F ; ∂y/∂μ = C * ∂F/∂μ
    h = max(1e-6, 1e-6 * abs(mu))
    dF_dmu = (_analytic_F_on_grid(mu + h) - _analytic_F_on_grid(mu - h)) / (2.0 * h)

    Jc = F_grid
    Jm = C * dF_dmu
    var = (pcov[0, 0] * Jc**2 + 2.0 * pcov[0, 1] * Jc * Jm + pcov[1, 1] * Jm**2)
    se = np.sqrt(np.maximum(var, 0.0))
    lo = np.clip(curve - z * se, 1e-300, np.inf)
    hi = curve + z * se

    return {"params": (C, mu), "cov": pcov, "curve": curve, "lo": lo, "hi": hi}

# ---------- Exponential y = A * exp(k r) for others ----------
def fit_attenuation_with_bands(r, y, e, xgrid=R_GRID, z=Z_BAND):
    r = np.asarray(r, float); y = np.asarray(y, float); e = np.asarray(e, float)
    mask = np.isfinite(r) & np.isfinite(y) & (y > 0)
    if mask.sum() < 2:
        return None

    r_fit, y_fit, e_fit = r[mask], y[mask], e[mask]
    # initial guess via log-linear
    k0, lnA0 = np.polyfit(r_fit, np.log(y_fit), 1)
    p0 = (float(np.exp(lnA0)), float(k0))

    def f(x, A, k):
        return A * np.exp(k * x)

    sigma = _safe_sigma(e_fit)
    popt, pcov = curve_fit(f, r_fit, y_fit, p0=p0, sigma=sigma,
                           absolute_sigma=True, maxfev=10000)
    A, k = popt
    y_grid = f(xgrid, A, k)

    # Jacobian wrt [A, k]
    J0 = np.exp(k * xgrid)                 # ∂y/∂A
    J1 = A * xgrid * np.exp(k * xgrid)     # ∂y/∂k
    var = (pcov[0, 0] * J0**2 + 2.0 * pcov[0, 1] * J0 * J1 + pcov[1, 1] * J1**2)
    se = np.sqrt(np.maximum(var, 0.0))
    lo = np.clip(y_grid - z * se, 1e-300, np.inf)
    hi = y_grid + z * se

    return {"params": (A, k), "cov": pcov, "curve": y_grid, "lo": lo, "hi": hi, "mu": -k}

# ---------- Utilities: curve intersections ----------
def _interp_log(xi, x, y):
    """Log-linear interpolate strictly positive y(x) at xi."""
    y = np.clip(y, 1e-300, np.inf)
    return np.exp(np.interp(xi, x, np.log(y)))

def find_intersections(x, y1, y2):
    """
    Return (xs, ys) where curves y1(x) and y2(x) cross.
    xs, ys are numpy arrays (could be multiple crossings).
    """
    y1 = np.asarray(y1, float); y2 = np.asarray(y2, float)
    d = y1 - y2
    idx = np.where(np.sign(d[:-1]) * np.sign(d[1:]) <= 0)[0]
    xs, ys = [], []
    for i in idx:
        x0, x1 = x[i], x[i+1]
        d0, d1 = d[i], d[i+1]
        if d1 == d0:
            t = 0.0
        else:
            t = d0 / (d0 - d1)  # linear interpolation of zero crossing
        t = np.clip(t, 0.0, 1.0)
        xi = x0 + t * (x1 - x0)
        y1i = _interp_log(xi, x, y1)
        y2i = _interp_log(xi, x, y2)
        yi = 0.5 * (y1i + y2i)
        xs.append(xi); ys.append(yi)
    return np.array(xs), np.array(ys)

# ---------- Main ----------
df = load_data(XLSX_PATH)

series = {}
for comp in df["Component"].unique():
    d = df[df["Component"] == comp]
    series[comp] = {
        "r": d["r_mm"].to_numpy(),
        "y": d["y"].to_numpy(),
        "e": d["e"].fillna(0).to_numpy(),
    }

comps_order = [c for c in [
    "HFE (no Rn-222)", "HFE", "IV", "OV",
    "Water", "Water (theoretical)", "Transition Box"
] if c in series]
default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
color_map = {comp: default_colors[i % len(default_colors)] for i, comp in enumerate(comps_order)}

fit_curves, bands, mus = {}, {}, {}

# HFE (no Rn-222): analytic fit + band
if "HFE (no Rn-222)" in series:
    d = series["HFE (no Rn-222)"]
    res = fit_hfe_no_rn_with_bands(d["r"], d["y"], d["e"])
    if res is not None:
        C, mu_mm = res["params"]
        fit_curves["HFE (no Rn-222)"] = res["curve"]
        bands["HFE (no Rn-222)"] = (res["lo"], res["hi"])
        mus["HFE (no Rn-222)"] = mu_mm
        print(f"HFE (no Rn-222): μ = {mu_mm:.4f} 1/mm, C = {C:.3g}")

# Others: exponential fit + band
for comp in ["HFE", "IV", "OV", "Water (theoretical)", "Transition Box"]:
    if comp in series:
        d = series[comp]
        res = fit_attenuation_with_bands(d["r"], d["y"], d["e"])
        if res is None:
            continue
        A, k = res["params"]; mu = -k
        fit_curves[comp] = res["curve"]
        bands[comp] = (res["lo"], res["hi"])
        mus[comp] = mu
        print(f"{comp}: μ = {mu:.4f} 1/mm (A = {A:.3g}, k = {k:.4g})")

# ---------- Crossings vs HFE (no Rn-222) ----------
baseline = "HFE (no Rn-222)"
if baseline in fit_curves:
    for comp in ["Water (theoretical)", "OV", "IV", "Transition Box"]:
        if comp in fit_curves:
            xs, ys = find_intersections(R_GRID, fit_curves[comp], fit_curves[baseline])
            if xs.size:
                for j, (xi, yi) in enumerate(zip(xs, ys), 1):
                    print(f"{comp} crosses {baseline} at r ≈ {xi:.1f} mm, y ≈ {yi:.3g} counts/yr (#{j})")
            else:
                print(f"{comp} does not cross {baseline} on [{R_GRID.min():.0f}, {R_GRID.max():.0f}] mm.")

# ---------- Plot ----------
fig, ax = plt.subplots(figsize=(12, 8))

# data points with error bars (legend shows only these)
for comp in comps_order:
    d = series[comp]
    ax.errorbar(d["r"], d["y"], yerr=d["e"], fmt='o', ms=4, label=comp, color=color_map[comp])

# fit curves + shaded confidence bands (mean fit)
for comp, curve in fit_curves.items():
    col = color_map.get(comp, None)
    ax.semilogy(R_GRID, curve, '--', linewidth=1.2, alpha=0.9, color=col, label='_nolegend_')
    if comp in bands:
        lo, hi = bands[comp]
        lo = np.clip(lo, 1e-300, np.inf)  # strictly positive for log axis
        ax.fill_between(R_GRID, lo, hi, color=col, alpha=0.15, linewidth=0, label='_nolegend_')

ax.set_xlabel("Inner vessel radius (mm)", fontsize=FS_LABEL)
ax.set_ylabel("Background (counts/yr)", fontsize=FS_LABEL)
ax.set_yscale("log")
ax.set_ylim(1e-5, 2e-1)
ax.tick_params(axis='both', which='major', labelsize=FS_TICK)
ax.grid(True, which="both", linestyle=':', alpha=0.4)
ax.legend(fontsize=FS_LEGEND)

plt.tight_layout()
plt.savefig("bkgd_vs_radius_bands.png", dpi=250)
plt.show()
print("Saved figure -> bkgd_vs_radius_bands.png")