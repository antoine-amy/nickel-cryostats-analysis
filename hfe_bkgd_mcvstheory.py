#!/usr/bin/env python3
"""
HFE (no Rn-222): compare MC data + analytic fit to theoretical HFE attenuation
with the 4π r^2 factor and the solid-angle factor kept explicit, and NO path-length factor.

Model (explicit):
  dB/dr ∝ [ 4π r^2 ] * [ f_solid(r) ] * exp( -μ * (r - R_TPC) )
with f_solid(r) = 0.5 * (R_TPC / r)^2  (1/r^2 solid-angle model)

We fit μ to the MC points (with a single normalization at 169.1 cm),
and overlay the theory curve with μ_theory = 0.00592496 1/mm.
"""

import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# MC Data
# --------------------------
R_iv = np.array([1000, 1026, 1100, 1300, 1691], dtype=float)  # mm
B_obs = np.array([4.16e-03, 4.41e-03, 4.44e-03, 4.96e-03, 5.02e-03], dtype=float)
E_obs = np.array([2.89e-03, 3.09e-03, 3.08e-03, 3.46e-03, 3.48e-03], dtype=float)

# Geometry
R_TPC_cm = 56.665
R_TPC_mm = R_TPC_cm * 10.0
NORM_RADIUS_CM = 169.1  # reference for normalization

# Theory μ (2.5 MeV photon attenuation in HFE)
MU_THEORY = 0.00592496  # 1/mm

# Radius/thickness grids
R_GRID = np.linspace(R_TPC_mm, 2200, 4000)  # mm
T_GRID = R_GRID - R_TPC_mm                  # mm (thickness from TPC surface)
t_data = R_iv - R_TPC_mm                    # mm

# --------------------------
# Solid-angle (explicit 1/r^2 form) and integrand (no path factor)
# --------------------------
def f_solid_r2(r_cm: np.ndarray) -> np.ndarray:
    """Explicit solid-angle model: f_solid(r) = 1/2 * (R_TPC / r)^2, with r in cm."""
    return 0.5 * (R_TPC_cm / r_cm) ** 2

def analytic_integrand_explicit(r_cm: np.ndarray, mu_cm_inv: float) -> np.ndarray:
    """
    Explicit integrand:
      dB/dr ∝ [4π r^2] * [f_solid(r)] * exp(-μ * (r - R_TPC))
    r_cm in cm, mu_cm_inv in 1/cm. Returns arbitrary units.
    """
    r_cm = np.asarray(r_cm)
    four_pi_r2 = 4.0 * np.pi * (r_cm ** 2)
    fsolid = f_solid_r2(r_cm)
    dist = r_cm - R_TPC_cm  # r_cm grid starts at R_TPC_cm
    atten = np.exp(-mu_cm_inv * dist)
    return four_pi_r2 * fsolid * atten

def analytic_cumulative_explicit(r_cm_grid: np.ndarray, mu_cm_inv: float):
    integ = analytic_integrand_explicit(r_cm_grid, mu_cm_inv)
    dr = r_cm_grid[1] - r_cm_grid[0]
    return np.cumsum(integ) * dr, integ

# --------------------------
# Fit μ to MC data (keep explicit factors)
# --------------------------
def fit_mu_explicit(r_mm, y_data, R_grid_mm, norm_radius_cm):
    r_cm_grid = R_grid_mm / 10.0
    r_cm_data = r_mm / 10.0
    idx_ref = np.argmin(np.abs(r_cm_data - norm_radius_cm))
    y_ref = float(y_data[idx_ref])

    mu_scan = np.linspace(0.02, 0.5, 600)  # 1/cm
    best = {"mu_cm": None, "curve": None, "scale": None, "err": np.inf}
    idx_norm = np.argmin(np.abs(r_cm_grid - norm_radius_cm))

    for mu_cm in mu_scan:
        B_raw, _ = analytic_cumulative_explicit(r_cm_grid, mu_cm)
        scale = y_ref / B_raw[idx_norm]
        curve = B_raw * scale
        model_at_pts = np.interp(r_mm, R_grid_mm, curve)
        err = np.sum((model_at_pts - y_data) ** 2)
        if err < best["err"]:
            best.update(mu_cm=mu_cm, curve=curve, scale=scale, err=err)

    mu_mm = best["mu_cm"] / 10.0 if best["mu_cm"] else None  # report 1/mm
    return mu_mm, best["curve"], best["scale"]

# --------------------------
# Percentiles from integrand (numerical, over the provided grid)
# --------------------------
def percentile_thickness(integrand, r_grid_mm, p, r_ref_mm):
    """
    Find t where ∫_{r_ref}^{r_ref+t} integrand dr = p * ∫_{r_ref}^{∞} integrand dr
    """
    mask = r_grid_mm >= r_ref_mm
    r = r_grid_mm[mask]; w = integrand[mask]
    dr = r[1] - r[0]
    cum = np.cumsum(w) * dr
    tot = cum[-1]
    if tot <= 0:
        return np.nan
    target = p * tot
    i = np.searchsorted(cum, target)
    if i == 0:
        r_star = r[0]
    else:
        c1, c2 = cum[i-1], cum[i]
        r1, r2 = r[i-1], r[i]
        wlin = 0.0 if c2 == c1 else (target - c1) / (c2 - c1)
        r_star = r1 + np.clip(wlin, 0.0, 1.0) * (r2 - r1)
    return r_star - r_ref_mm

# --------------------------
# Main
# --------------------------
def main():
    # Fit μ (explicit 4π r^2 * f_solid; no path-length factor)
    mu_fit, B_fit, scale = fit_mu_explicit(R_iv, B_obs, R_GRID, NORM_RADIUS_CM)
    if mu_fit is None:
        raise RuntimeError("Fit failed (μ not found).")

    # Build integrands/curves (fit & theory)
    r_cm_grid = R_GRID / 10.0
    B_raw_fit, integ_fit = analytic_cumulative_explicit(r_cm_grid, mu_fit * 10.0)
    integ_fit *= scale

    B_raw_theory, integ_theory = analytic_cumulative_explicit(r_cm_grid, MU_THEORY * 10.0)
    # Normalize theory to match fit at the norm radius
    r_ref_norm = NORM_RADIUS_CM * 10.0
    B_ref = np.interp(r_ref_norm, R_GRID, B_fit)
    B_ref_th = np.interp(r_ref_norm, R_GRID, B_raw_theory)
    scale_th = B_ref / B_ref_th
    B_theory = B_raw_theory * scale_th
    integ_theory *= scale_th

    # Percentiles (from TPC surface) using the differential contribution
    t50 = percentile_thickness(integ_fit, R_GRID, 0.5, R_TPC_mm)
    t90 = percentile_thickness(integ_fit, R_GRID, 0.9, R_TPC_mm)

    print("\n=== MC vs Theory (explicit 4π r^2 * f_solid, no path-length factor) ===")
    print(f"μ_fit (MC)     = {mu_fit:.6f} 1/mm")
    print(f"μ_theory       = {MU_THEORY:.6f} 1/mm")
    print(f"t50 (MC fit)   = {t50:.1f} mm")
    print(f"t90 (MC fit)   = {t90:.1f} mm")

    # Plot background vs thickness
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.errorbar(t_data, B_obs, yerr=E_obs, fmt="o", label="MC data (HFE no Rn-222)")
    ax.semilogy(T_GRID, B_fit, label=f"MC fit (μ={mu_fit:.4g} 1/mm)  [explicit factors]")
    ax.semilogy(T_GRID, B_theory, "--", label=f"Theory (μ={MU_THEORY:.4g} 1/mm)  [explicit factors]")
    ax.set_xlabel("HFE thickness t = R_IV - R_TPC (mm)")
    ax.set_ylabel("Background B(t) [counts/yr]")
    ax.set_title("HFE self-shielding: MC vs Theory (explicit 4π r² × f_solid, no path)")
    ax.grid(True, which="both", linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Plot differential shell contribution
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.plot(T_GRID, integ_fit, label="MC fit: contribution density  [explicit]")
    ax.plot(T_GRID, integ_theory, "--", label="Theory: contribution density  [explicit]")
    ax.axvline(t50, linestyle=":", label=f"t50 = {t50:.0f} mm")
    ax.axvline(t90, linestyle=":", label=f"t90 = {t90:.0f} mm")
    ax.set_xlabel("HFE thickness t (mm)")
    ax.set_ylabel("Contribution per mm (arb. units)")
    ax.set_title("Shell-by-shell contribution to HFE background (explicit factors)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()