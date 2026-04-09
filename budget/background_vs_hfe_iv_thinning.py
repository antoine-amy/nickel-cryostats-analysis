#!/usr/bin/env python3
"""
IV background vs HFE mass: fixed-thickness vs thickness-optimised vessel.

Two curves are shown for the inner vessel (IV):
  - Fixed thickness (spreadsheet total):
        B_fixed(R) = A * exp(k * R)         [fitted from spreadsheet totals]

  - Thin vessel (extra thickness optimisation):
        B_thin(R)  = B_fixed(R) * (R / R_IV^0)
    The spreadsheet fixed-thickness totals already include the IV mass change
    with radius. Under the ASME-inspired t ∝ R scaling, the thin-vessel mass
    scales as m ∝ R^3 instead of the fixed-thickness m ∝ R^2 behaviour, so the
    extra correction relative to the spreadsheet curve is only (R/R_IV^0).

The confidence band of the thin curve is the fixed band scaled by the same
deterministic factor (R/R_IV^0).
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

BUDGET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUDGET_DIR.parent
for _p in (str(PROJECT_ROOT), str(BUDGET_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from hfe_volume_to_iv_radius import (
    BASELINE_TARGET_MASS_KG_1,
    calibration_loss_tonnes_for_target,
    hfe_mass_tonnes,
)

CALIBRATION_LOSS_TONNES = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_1)


def hfe_mass_tonnes_from_radius_mm(radius_mm):
    radius_m = np.asarray(radius_mm, float) / 1000.0
    return hfe_mass_tonnes(radius_m, CALIBRATION_LOSS_TONNES)


# --- Config ---
XLSX_PATH = BUDGET_DIR / "Summary_bkgd_vs_hfe-shield.xlsx"
R_GRID = np.linspace(950, 1800, 600)   # mm
Z_BAND = 1.0

R_IV0_MM = 1691.0   # baseline IV outer radius
X_AXIS_MAX_TONNES = 35.0
TOTAL_INTRINSIC_BACKGROUND = 0.55

DESIGN_CONFIGS = {
    "Baseline":    1691.00,
    "Recommended": 1308.37,
    "Aggressive":  1121.80,
}

FS_LABEL  = 18
FS_TICK   = 14
FS_LEGEND = 14


def _safe_sigma(errors):
    errors = np.asarray(errors, float)
    valid = np.isfinite(errors) & (errors > 0)
    if not np.any(valid):
        return None
    min_pos = np.nanmin(errors[valid])
    return np.where((~np.isfinite(errors)) | (errors <= 0), min_pos, errors)


def _confidence_band(curve, jac_a, jac_k, cov, z):
    variance = (
        cov[0, 0] * jac_a**2
        + 2.0 * cov[0, 1] * jac_a * jac_k
        + cov[1, 1] * jac_k**2
    )
    se = np.sqrt(np.maximum(variance, 0.0))
    return np.clip(curve - z * se, 1e-300, np.inf), curve + z * se


def load_iv_data(filepath):
    df = pd.read_excel(filepath, sheet_name="Summary", header=None)
    for i, (_, row) in enumerate(df.iterrows()):
        vals = [str(v).strip().lower() for v in row.values]
        if all(h in vals for h in ["component", "iv radius (mm)", "background", "error"]):
            df = df.iloc[i + 1:, :4].copy()
            df.columns = ["Component", "r_mm", "y", "e"]
            break
    df["Component"] = df["Component"].ffill().astype(str).str.strip()
    for col in ["r_mm", "y", "e"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["Component"] = df["Component"].str.lower().map({"iv": "IV"}).fillna(df["Component"])
    df = df.dropna(subset=["r_mm", "y"])
    return df[df["Component"] == "IV"].copy()


def fit_iv(iv_data):
    r = iv_data["r_mm"].to_numpy()
    y = iv_data["y"].to_numpy()
    e = iv_data["e"].fillna(0).to_numpy()

    valid = np.isfinite(r) & np.isfinite(y) & (y > 0)
    r, y, e = r[valid], y[valid], e[valid]

    k0, lnA0 = np.polyfit(r, np.log(y), 1)

    def model(x, A, k):
        return A * np.exp(k * x)

    sigma = _safe_sigma(e)
    params, cov = curve_fit(model, r, y, p0=(np.exp(lnA0), k0),
                            sigma=sigma, absolute_sigma=True, maxfev=10000)
    A, k = params
    curve = model(R_GRID, A, k)
    lo, hi = _confidence_band(
        curve,
        np.exp(k * R_GRID),
        A * R_GRID * np.exp(k * R_GRID),
        cov, Z_BAND,
    )
    return {"A": A, "k": k, "cov": cov, "curve": curve, "lo": lo, "hi": hi,
            "r_data": r, "y_data": y, "e_data": e}


def main():
    iv_data = load_iv_data(XLSX_PATH)
    fit = fit_iv(iv_data)

    A, k = fit["A"], fit["k"]
    B0 = A * np.exp(k * R_IV0_MM)
    print(f"IV fit:  A = {A:.4g},  k = {k:.4e} mm⁻¹,  mu = {-k:.4e} mm⁻¹")
    print(f"B_IV at baseline ({R_IV0_MM:.0f} mm) = {B0:.3e} cts/y/2t/FWHM")

    # Extra thin/fixed correction: the spreadsheet fixed-thickness totals
    # already include the vessel-mass decrease with radius.
    mass_factor = R_GRID / R_IV0_MM

    thin_curve = fit["curve"] * mass_factor
    thin_lo    = fit["lo"]    * mass_factor
    thin_hi    = fit["hi"]    * mass_factor

    # Print values at design configurations
    print(f"\n{'Config':<15} {'R (mm)':>8} {'HFE (t)':>8} "
          f"{'B_fixed':>12} {'B_thin':>12} {'thin/fixed':>12}")
    print("-" * 65)
    for label, r_mm in DESIGN_CONFIGS.items():
        mass_t   = float(hfe_mass_tonnes_from_radius_mm(r_mm))
        b_fixed  = float(A * np.exp(k * r_mm))
        mf       = r_mm / R_IV0_MM
        b_thin   = b_fixed * mf
        print(f"{label:<15} {r_mm:>8.1f} {mass_t:>8.2f} "
              f"{b_fixed:>12.3e} {b_thin:>12.3e} {mf:>12.3f}")

    # --- Plot ---
    mass_grid_t = hfe_mass_tonnes_from_radius_mm(R_GRID)
    colors = {"fixed": "C2", "thin": "C3"}

    _, axis = plt.subplots(figsize=(10, 7))

    # Fixed-thickness curve + band
    axis.semilogy(mass_grid_t, fit["curve"], "--", linewidth=1.2, alpha=0.9,
                  color=colors["fixed"], label="IV: fixed thickness")
    axis.fill_between(mass_grid_t,
                      np.clip(fit["lo"], 1e-300, np.inf), fit["hi"],
                      color=colors["fixed"], alpha=0.15, linewidth=0,
                      label="_nolegend_")

    # Thin-vessel curve + band
    axis.semilogy(mass_grid_t, thin_curve, "--", linewidth=1.2, alpha=0.9,
                  color=colors["thin"], label=r"IV: optimised thickness ($t \propto R$)")
    axis.fill_between(mass_grid_t,
                      np.clip(thin_lo, 1e-300, np.inf), thin_hi,
                      color=colors["thin"], alpha=0.15, linewidth=0,
                      label="_nolegend_")

    # Data points
    axis.errorbar(
        hfe_mass_tonnes_from_radius_mm(fit["r_data"]),
        fit["y_data"],
        yerr=fit["e_data"],
        fmt="o", ms=5, mew=1.1, elinewidth=1.4, capsize=4, capthick=1.2,
        color=colors["fixed"], zorder=3, label="_nolegend_",
    )

    # Total intrinsic background marker
    baseline_mass = float(hfe_mass_tonnes_from_radius_mm(R_IV0_MM))
    axis.plot([baseline_mass], [TOTAL_INTRINSIC_BACKGROUND],
              marker="s", ms=5, color="0.15", markerfacecolor="none",
              linestyle="None", label="_nolegend_")
    axis.annotate(
        "total intrinsic background",
        xy=(baseline_mass, TOTAL_INTRINSIC_BACKGROUND),
        xytext=(-10, -2), textcoords="offset points",
        ha="right", va="center", fontsize=FS_TICK - 2, color="0.15",
    )

    # Design vertical bars
    for label, r_mm in DESIGN_CONFIGS.items():
        mass_t = float(hfe_mass_tonnes_from_radius_mm(r_mm))
        axis.axvline(mass_t, color="black", linestyle=(0, (4, 2)), linewidth=1.5)
        axis.text(mass_t, 0.03, label,
                  rotation=90, ha="right", va="bottom",
                  transform=axis.get_xaxis_transform(),
                  fontsize=FS_TICK, color="black")

    axis.set_xlabel("HFE mass (tonnes)", fontsize=FS_LABEL)
    axis.set_ylabel("Background rate [cts/(y·2t·FWHM)]", fontsize=FS_LABEL)
    axis.set_yscale("log")
    axis.set_xlim(float(mass_grid_t.min()), X_AXIS_MAX_TONNES)
    axis.set_ylim(1e-5, 1.0)
    axis.tick_params(axis="both", which="major", labelsize=FS_TICK)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=FS_LEGEND, loc="upper left")

    plt.tight_layout()
    output_dir = BUDGET_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
