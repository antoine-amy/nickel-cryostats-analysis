#!/usr/bin/env python3
"""
Background vs IV radius from 'Summary' (already without Rn-222).

Robust to:
- merged header rows (pandas 'Unnamed' columns)
- continuation rows with blank 'Component'
- extra tables below the summary (e.g., "Isotope / Hit Efficiency")
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation

# ---------------- Config ----------------
# Use your original file path OR the uploaded one below.
XLSX_PATH  = "/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/nickel-cryostats-analysis/background/ReducedHFE_bkgd.xlsx"
# XLSX_PATH = "/mnt/data/ReducedHFE_bkgd.xlsx"   # ← works with the uploaded file
SHEET_NAME = "Summary"

# HFE analytic fit constants (geometry in cm)
R_TPC_CM = 56.665
H_TPC_CM = 59.15
NORM_RADIUS_MM = 1691.0  # normalization point for HFE fit (in mm)

# Grids for plotting fits
R_GRID_MM = np.linspace(950.0, 1800.0, 600)  # mm
R_GRID_CM = R_GRID_MM / 10.0                 # cm (for HFE geometry math)

MU_SCAN = np.linspace(0.02, 0.50, 2000)  # cm^-1 (HFE)
MU_TABULATED = 0.04 * 1.72  # 0.0688 cm^-1 (for info only)

# HFE mass params (vs mm on x-axis)
DENSITY_T_PER_M3 = 1.72
VOL_OFFSET_M3    = 2.2 - 0.58
HARDWARE_MASS_T  = 0.240

COLORS = {"HFE": "#1f77b4", "IV": "#ff7f0e", "OV": "#2ca02c", "Water": "#9467bd"}
COMPONENTS = {"HFE", "IV", "OV", "Water"}  # what to keep from the sheet

# ---------------- Helpers ----------------
def fsolid(r_cm):
    r = np.asarray(r_cm, float)
    f = np.zeros_like(r)
    mask = r >= R_TPC_CM
    if np.any(mask):
        rm = r[mask]
        cap = 0.5 * (1 - np.sqrt(1 - (R_TPC_CM / rm) ** 2))
        bar = (R_TPC_CM * (2 * H_TPC_CM)) / (4 * np.pi * rm ** 2)
        f[mask] = np.minimum(cap + bar, 0.5)
    return f

def fpath(r_cm):
    d = np.clip(r_cm - R_TPC_CM, 0, None)
    return 1 + ((np.pi / 2) - 1) * np.clip(d / H_TPC_CM, 0, 1)

def hfe_curve(mu_cm_inv, r_grid_cm, r_pts_mm, y_pts):
    """Geometry-corrected HFE cumulative term, normalized at NORM_RADIUS_MM."""
    dr = r_grid_cm[1] - r_grid_cm[0]
    delta = np.clip(r_grid_cm - R_TPC_CM, 0, None)
    atten = np.exp(-mu_cm_inv * fpath(r_grid_cm) * delta)
    integrand = 4 * np.pi * r_grid_cm ** 2 * fsolid(r_grid_cm) * atten
    cum = np.cumsum(integrand) * dr
    # normalize at 1691 mm (nearest HFE point)
    idx0 = int(np.argmin(np.abs(r_grid_cm - NORM_RADIUS_MM / 10.0)))
    r_pts_cm = np.asarray(r_pts_mm, float) / 10.0
    idx_ref = int(np.argmin(np.abs(r_pts_cm - NORM_RADIUS_MM / 10.0)))
    y_ref = y_pts[idx_ref]
    return cum * (y_ref / cum[idx0])

def fit_mu_hfe(r_pts_mm, y_pts):
    best_i, best_rss = 0, float("inf")
    for i, mu in enumerate(MU_SCAN):
        model = hfe_curve(mu, R_GRID_CM, r_pts_mm, y_pts)
        model_at_pts = np.interp(r_pts_mm, R_GRID_MM, model)
        rss = np.sum((model_at_pts - y_pts) ** 2)
        if rss < best_rss:
            best_rss, best_i = rss, i
    return float(MU_SCAN[best_i])

def loglin_fit_mm(r_mm, y, r0_mm):
    """Return (k, b) for ln(y) = k*(r_mm - r0_mm) + b using finite positive y."""
    r = np.asarray(r_mm, float)
    y = np.asarray(y, float)
    mask = np.isfinite(r) & np.isfinite(y) & (y > 0)
    if np.count_nonzero(mask) < 2:
        return None
    x = r[mask] - float(r0_mm)
    k, b = np.polyfit(x, np.log(y[mask]), 1)
    return float(k), float(b)

def eval_loglin_mm(kb, r_mm, r0_mm):
    if kb is None:
        return None
    k, b = kb
    return np.exp(b + k * (np.asarray(r_mm, float) - float(r0_mm)))

def load_summary_resilient(xlsx_path, sheet_name="Summary"):
    """
    Read 'Summary' even if the first row is merged and extra tables exist below.
    Returns a tidy DataFrame with columns: Component, r_mm, y, e
    """
    raw = pd.read_excel(xlsx_path, sheet_name=sheet_name, header=None)

    # Find the header row that contains our four column names
    wanted = ["Component", "IV Radius (mm)", "Background", "Error"]
    header_row = None
    for i, row in raw.iterrows():
        vals = [str(v).strip() for v in row.values]
        if all(h in vals for h in wanted):
            header_row = i
            break
    if header_row is None:
        raise KeyError("Could not find header row with columns: " + ", ".join(wanted))

    # Slice the data block under the header; take first 4 columns to avoid noise
    df = raw.iloc[header_row + 1 :, :4].copy()
    df.columns = wanted

    # Stop if another header-like row appears (defensive)
    mask_headerish = df["Component"].astype(str).str.strip().isin(["Component", "IV Radius (mm)"])
    if mask_headerish.any():
        first_repeat = np.flatnonzero(mask_headerish.values)[0]
        df = df.iloc[:first_repeat].copy()

    # Forward-fill components and keep only the ones we care about
    df["Component"] = df["Component"].ffill().astype(str).str.strip()
    df = df[df["Component"].isin(COMPONENTS)]

    # Coerce to numeric and drop rows without numbers
    df["r_mm"] = pd.to_numeric(df["IV Radius (mm)"], errors="coerce")
    df["y"]    = pd.to_numeric(df["Background"],     errors="coerce")
    df["e"]    = pd.to_numeric(df["Error"],          errors="coerce")
    df = df.dropna(subset=["r_mm", "y"])

    # Clean up and sort
    df = df[["Component", "r_mm", "y", "e"]].sort_values(["Component", "r_mm"]).reset_index(drop=True)
    return df

# ---------------- Load Summary (resilient) ----------------
df = load_summary_resilient(XLSX_PATH, SHEET_NAME)

# Split by component
series = {}
for comp in COMPONENTS:
    sub = df[df["Component"] == comp]
    if not sub.empty:
        series[comp] = {
            "r_mm": sub["r_mm"].to_numpy(float),
            "y":    sub["y"].to_numpy(float),
            "e":    np.nan_to_num(sub["e"].to_numpy(float), nan=0.0, posinf=0.0, neginf=0.0),
        }

if "HFE" not in series:
    raise RuntimeError("HFE rows are required for the analytic fit.")

# ---------------- Fits ----------------
# HFE analytic
mu_hfe = fit_mu_hfe(series["HFE"]["r_mm"], series["HFE"]["y"])
hfe_fit_curve = hfe_curve(mu_hfe, R_GRID_CM, series["HFE"]["r_mm"], series["HFE"]["y"])
print(f"Best-fit μ_HFE = {mu_hfe:.5f} cm⁻¹ "
      f"(≈ {mu_hfe / MU_TABULATED:.2f} × tabulated 0.06880 cm⁻¹)")

# IV/OV/Water log-linear fits in mm
iv_fit = ov_fit = water_fit = None
if "IV" in series:
    r0_iv = series["IV"]["r_mm"][0]
    iv_fit = loglin_fit_mm(series["IV"]["r_mm"], series["IV"]["y"], r0_mm=r0_iv)
    if iv_fit:
        k_iv, _ = iv_fit
        print(f"IV fit: k = {k_iv:+.5e} mm⁻¹  (μ_IV ≡ -k = {-k_iv:+.5e} mm⁻¹, r0 = {r0_iv:.1f} mm)")
if "OV" in series:
    r0_ov = series["OV"]["r_mm"][0]
    ov_fit = loglin_fit_mm(series["OV"]["r_mm"], series["OV"]["y"], r0_mm=r0_ov)
    if ov_fit:
        k_ov, _ = ov_fit
        print(f"OV fit: k = {k_ov:+.5e} mm⁻¹  (μ_OV ≡ -k = {-k_ov:+.5e} mm⁻¹, r0 = {r0_ov:.1f} mm)")
if "Water" in series:
    r0_w = series["Water"]["r_mm"][0]
    water_fit = loglin_fit_mm(series["Water"]["r_mm"], series["Water"]["y"], r0_mm=r0_w)
    if water_fit:
        k_w, _ = water_fit
        print(f"Water fit: k = {k_w:+.5e} mm⁻¹  (μ_W ≡ -k = {-k_w:+.5e} mm⁻¹, r0 = {r0_w:.1f} mm)")

iv_curve     = eval_loglin_mm(iv_fit,     R_GRID_MM, r0_mm=series["IV"]["r_mm"][0])     if "IV" in series and iv_fit     else None
ov_curve     = eval_loglin_mm(ov_fit,     R_GRID_MM, r0_mm=series["OV"]["r_mm"][0])     if "OV" in series and ov_fit     else None
water_curve  = eval_loglin_mm(water_fit,  R_GRID_MM, r0_mm=series["Water"]["r_mm"][0])  if "Water" in series and water_fit else None

# ---------------- Plot 1: Background vs radius (mm) ----------------
plt.rc("font", size=12)
fig, ax = plt.subplots(figsize=(13, 9))

# HFE: analytic fit + data points
ax.semilogy(R_GRID_MM, hfe_fit_curve, "-", color=COLORS["HFE"], lw=3, alpha=0.9, label="HFE (analytic fit)")
ax.errorbar(series["HFE"]["r_mm"], series["HFE"]["y"], yerr=series["HFE"]["e"],
            fmt="o", ms=5, capsize=4, label="HFE (Summary)",
            color=COLORS["HFE"], markerfacecolor="white",
            markeredgecolor=COLORS["HFE"], markeredgewidth=1.2)

# IV / OV / Water: points + log-linear fits
for comp, marker, fit_curve in [("IV", "s", iv_curve), ("OV", "^", ov_curve), ("Water", "D", water_curve)]:
    if comp in series:
        ax.errorbar(series[comp]["r_mm"], series[comp]["y"], yerr=series[comp]["e"],
                    fmt=marker, ms=5, capsize=4, label=f"{comp} (Summary)",
                    color=COLORS[comp], markerfacecolor="white",
                    markeredgecolor=COLORS[comp], markeredgewidth=1.2)
        if fit_curve is not None:
            ax.semilogy(R_GRID_MM, fit_curve, "-", color=COLORS[comp], lw=2, alpha=0.85, label=f"{comp} (fit)")

ax.set_xlabel("Inner vessel radius r (mm)", fontsize=14, fontweight="bold")
ax.set_ylabel("Background (counts/yr)",      fontsize=14, fontweight="bold")
ax.set_yscale("log")
ax.set_ylim(1e-6, 2e-1)
ax.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.3)
ax.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
ax.minorticks_on()
ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=False))
ax.legend(ncol=2, loc="upper left", fontsize=11, fancybox=True, shadow=True)

plt.title("Cryostat: HFE (analytic fit) + IV/OV/Water (log-linear fits) vs IV Radius (mm)",
          fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.show()

# ---------------- Plot 2: HFE mass vs radius (mm on x-axis) ----------------
fig2, axm = plt.subplots(figsize=(13, 5))
r_m = R_GRID_MM / 1000.0  # mm → m
vol = (4/3) * np.pi * r_m**3
hfe_vol = vol - VOL_OFFSET_M3
mass_hfe_t = hfe_vol * DENSITY_T_PER_M3 - HARDWARE_MASS_T
axm.plot(R_GRID_MM, mass_hfe_t, ":", color=COLORS["HFE"], lw=2, alpha=0.9)
axm.set_xlabel("Inner vessel radius r (mm)", fontsize=14, fontweight="bold")
axm.set_ylabel("HFE mass (t)",              fontsize=14, fontweight="bold")
axm.grid(which="major", linestyle="-", linewidth=0.8, alpha=0.3)
axm.grid(which="minor", linestyle=":", linewidth=0.5, alpha=0.2)
axm.minorticks_on()
plt.title("HFE Mass vs IV Radius", fontsize=16, fontweight="bold", pad=20)
plt.tight_layout()
plt.show()