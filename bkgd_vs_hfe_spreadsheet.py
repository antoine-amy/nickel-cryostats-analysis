#!/usr/bin/env python3
"""
Background vs IV radius analysis from 'Summary' sheet.

• HFE (no Rn-222): analytic geometry-based fit (internal model + fit combined)
• Others (HFE, IV, OV, Water, Water (theoretical)): individual attenuation-law fits
   y(r) = A * exp(k r)  →  attenuation coefficient μ = -k  [1/mm]
   (Legend shows data only; fits are dashed without legend entries)
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Config ---
XLSX_PATH = ("/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/"
             "nickel-cryostats-analysis/budget/bkgd_vs_hfe-shield.xlsx")
R_GRID = np.linspace(950, 1800, 600)  # mm

# TPC geometry for analytic model (cm)
R_TPC = 56.665
H_TPC = 59.15
NORM_RADIUS = 169.1  # cm  (reference for normalization)

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

    # standardize names
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

def fit_hfe_no_rn_analytic(r_mm, y):
    """
    Combined analytic HFE model + fit for HFE (no Rn-222).

    Returns:
        curve_grid : fitted curve evaluated on R_GRID (counts/yr)
        mu_mm_inv  : fitted μ in 1/mm (converted from cm^-1)
    """
    # --- analytic model (in cm) ---
    def analytic_model(r_cm, mu_cm_inv):
        # solid angle factor
        f_solid = np.where(
            r_cm >= R_TPC,
            np.minimum(
                0.5 * (1 - np.sqrt(1 - (R_TPC / r_cm) ** 2))
                + R_TPC * 2 * H_TPC / (4 * np.pi * r_cm ** 2),
                0.5,
            ),
            0.0,
        )
        # path length factor
        distance = np.maximum(r_cm - R_TPC, 0.0)
        f_path = 1.0 + ((np.pi / 2) - 1.0) * np.minimum(distance / H_TPC, 1.0)
        atten = np.exp(-mu_cm_inv * f_path * distance)

        dr = r_cm[1] - r_cm[0]
        integrand = 4 * np.pi * r_cm**2 * f_solid * atten
        return np.cumsum(integrand) * dr

    r_cm_data = r_mm / 10.0
    r_cm_grid = R_GRID / 10.0

    # choose reference from data (closest to NORM_RADIUS)
    idx_ref = np.argmin(np.abs(r_cm_data - NORM_RADIUS))
    y_ref = y[idx_ref]

    mu_grid = np.linspace(0.02, 0.5, 500)  # cm^-1
    best_mu_cm, best_err, best_curve = None, np.inf, None

    for mu_cm in mu_grid:
        model = analytic_model(r_cm_grid, mu_cm)
        idx_norm = np.argmin(np.abs(r_cm_grid - NORM_RADIUS))
        if model[idx_norm] <= 0:
            continue
        model = model * (y_ref / model[idx_norm])  # normalize
        model_at_pts = np.interp(r_mm, R_GRID, model)
        err = np.sum((model_at_pts - y) ** 2)
        if err < best_err:
            best_err = err
            best_mu_cm = mu_cm
            best_curve = model

    mu_mm_inv = best_mu_cm / 10.0 if best_mu_cm is not None else None  # convert cm^-1 → mm^-1
    return best_curve, mu_mm_inv

def fit_attenuation_individual(r, y):
    """
    Fit y = A * exp(k r) for one component, return μ = -k (1/mm) and curve on R_GRID.
    """
    mask = (r > 0) & (y > 0) & np.isfinite(r) & np.isfinite(y)
    if mask.sum() < 2:
        return None, None
    k, ln_a = np.polyfit(r[mask], np.log(y[mask]), 1)
    mu = -k  # attenuation coefficient (1/mm)
    a_coeff = float(np.exp(ln_a))
    fitted_curve = a_coeff * np.exp(k * R_GRID)
    return mu, fitted_curve

# --- Load & organize ---
df = load_data(XLSX_PATH)

series = {}
for comp in df["Component"].unique():
    d = df[df["Component"] == comp]
    series[comp] = {
        "r": d["r_mm"].to_numpy(),
        "y": d["y"].to_numpy(),
        "e": d["e"].fillna(0).to_numpy(),
    }

# plotting order / colors
comps_order = [c for c in ["HFE (no Rn-222)", "HFE", "IV", "OV", "Water",
                        "Water (theoretical)"] if c in series]
default_colors = plt.rcParams['axes.prop_cycle'].by_key().get('color', [])
color_map = {comp: default_colors[i % len(default_colors)] for i, comp in enumerate(comps_order)}

# --- Fits ---
fit_curves = {}
mus = {}  # attenuation coefficients per component (1/mm)

# Analytic fit for HFE (no Rn-222)
if "HFE (no Rn-222)" in series:
    d = series["HFE (no Rn-222)"]
    curve, mu_mm = fit_hfe_no_rn_analytic(d["r"], d["y"])
    fit_curves["HFE (no Rn-222)"] = curve
    mus["HFE (no Rn-222)"] = mu_mm
    if mu_mm is not None:
        print(f"HFE (no Rn-222): μ = {mu_mm:.4f} 1/mm")

# Individual attenuation fits for each other component
for comp in ["HFE", "IV", "OV", "Water", "Water (theoretical)"]:
    if comp in series:
        comp_data = series[comp]
        mu_coeff, curve_data = fit_attenuation_individual(comp_data["r"], comp_data["y"])
        if mu_coeff is None:
            continue
        mus[comp] = mu_coeff
        fit_curves[comp] = curve_data
        print(f"{comp}: μ = {mu_coeff:.4f} 1/mm")

# --- Plot ---
fig, ax = plt.subplots(figsize=(12, 8))

# data points (legend shows only these)
for comp in comps_order:
    comp_data = series[comp]
    ax.errorbar(comp_data["r"], comp_data["y"], yerr=comp_data["e"], fmt='o', ms=4,
                label=comp, color=color_map[comp])

# fit curves (no legend entry)
for comp, curve in fit_curves.items():
    if curve is None:
        continue
    ax.semilogy(R_GRID, curve, '--', linewidth=1.2, alpha=0.7,
                color=color_map.get(comp, None), label='_nolegend_')

ax.set_xlabel("Inner vessel radius (mm)")
ax.set_ylabel("Background (counts/yr)")
ax.set_yscale("log")
ax.set_ylim(1e-6, 2e-1)
ax.grid(True, which="both", linestyle=':', alpha=0.4)
ax.legend()
ax.set_title("Background Contribution vs IV Radius")

# Note box in the upper-right corner inside axes
NOTE_TEXT = ("Note (water theoretical): μ_HFE taken at 2.5 MeV. MC studies indicate\n"
             "stronger attenuation in HFE; therefore, with less HFE (smaller IV radius),\n"
             "the increase in background is expected to be weaker than this theory curve.")
ax.text(0.48, 0.98, NOTE_TEXT, transform=ax.transAxes,
        ha='left', va='top', fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "#f7f7f7", "edgecolor": "#cccccc"})

plt.tight_layout()
plt.show()
