#!/usr/bin/env python3
"""
Cryostat + HFE background versus inner-vessel radius
–––––––––––––––––––––––––––––––––––––––––––––––––––––
Three analytic curves are shown:

   • "No-geometry" theory  – simple 1-D attenuation (4π r² shell, radial path).
   • "Geometry-corrected, tabulated μ" – adds solid-angle + path-length factors.
   • "Geometry-corrected, best-fit μ" – same model, μ fitted to MC points.

The corrected model still drives the μ-fit, but the uncorrected curve is
plotted for reference so you can see exactly how much geometry bends it down.
"""

import numpy as np
import matplotlib.pyplot as plt

# ============================================================================
# Constants and inputs
# ============================================================================
BASE_BG_HFE    = 5e-3             # counts/yr at INITIAL_IV_RAD (MC point)
MU_TABULATED   = 0.04 * 1.72      # 0.0688 cm⁻¹ for 2.615 MeV γ in HFE
INITIAL_IV_RAD = 169.1            # cm – reference radius for normalisation

# IV background (orange) — unchanged
A_IV   = 4.601024e-03             # counts/yr
MU_IV  = 0.006223                 # mm⁻¹
r0_iv  = min([1025, 1225, 1510, 1690])  # mm

# Cylindrical TPC geometry
R_TPC = 56.665   # cm
H_TPC = 59.15    # cm  (half-height)

# HFE mass axis parameters
DENSITY_T_PER_M3 = 1.72           # t m⁻³
VOL_OFFSET_M3    = 2.2 - 0.58     # m³ (cryostat – TPC) at R = 169.1 cm
HARDWARE_MASS_T  = 0.240          # t

# Monte-Carlo HFE & IV points
r_mc   = np.array([102.6, 110.0, 130.0, 169.1])        # cm
bg_mc  = np.array([4.40751e-3, 4.43671e-3, 4.96e-3, 5e-3])  # counts/yr

iv_radii = np.array([1025, 1225, 1510, 1690]) / 10  # cm
iv_bgmc  = [4.5e-3, 1.4e-3, 2.5e-4, 7e-5]

# Radius grid for integrations
r_cm = np.linspace(95, 170, 300)
dr   = r_cm[1] - r_cm[0]

# ============================================================================
# Geometry helpers
# ============================================================================
def fsolid(r_cm):
    """Fraction of 4π seen by the cylindrical TPC (≈ <5 % error)."""
    r = np.asarray(r_cm, dtype=float)
    f = np.zeros_like(r)
    mask = r >= R_TPC
    if not np.any(mask):
        return f
    rm = r[mask]
    cap_term    = 0.5 * (1.0 - np.sqrt(1.0 - (R_TPC / rm) ** 2))
    barrel_term = (R_TPC * (2 * H_TPC)) / (4.0 * np.pi * rm ** 2)
    f[mask] = np.minimum(cap_term + barrel_term, 0.5)
    return f

def fpath(r_cm):
    """〈chord〉 / (r – R_TPC): 1 at wall → π/2 once ≥ H_TPC away."""
    dr = np.clip(r_cm - R_TPC, 0, None)
    return 1 + ((np.pi/2) - 1) * np.clip(dr / H_TPC, 0, 1)

# ============================================================================
# Analytic HFE background curves
# ============================================================================
def bg_no_geometry(mu):
    """1-D attenuation: no solid-angle or path-length corrections."""
    delta_r  = np.clip(r_cm - R_TPC, 0, None)
    atten    = np.exp(-mu * delta_r)
    integrand = 4 * np.pi * r_cm**2 * atten
    cum_int   = np.cumsum(integrand) * dr
    idx0  = np.argmin(np.abs(r_cm - INITIAL_IV_RAD))
    norm  = BASE_BG_HFE / cum_int[idx0]
    return norm * cum_int

def bg_geometry(mu):
    """Geometry-corrected attenuation (solid angle + path length)."""
    delta_r  = np.clip(r_cm - R_TPC, 0, None)
    atten    = np.exp(-mu * fpath(r_cm) * delta_r)
    integrand = 4 * np.pi * r_cm**2 * fsolid(r_cm) * atten
    cum_int   = np.cumsum(integrand) * dr
    idx0  = np.argmin(np.abs(r_cm - INITIAL_IV_RAD))
    norm  = BASE_BG_HFE / cum_int[idx0]
    return norm * cum_int

# ============================================================================
# Fit μ_HFE to the MC points using the geometry-corrected model
# ============================================================================
mu_vals = np.linspace(0.02, 0.12, 600)
rss = [np.sum((np.interp(r_mc, r_cm, bg_geometry(mu)) - bg_mc) ** 2)
       for mu in mu_vals]
mu_best = mu_vals[np.argmin(rss)]
print(f"Best-fit μ_HFE = {mu_best:.5f} cm⁻¹  "
      f"(≈ {mu_best / MU_TABULATED:.2f} × tabulated)")

# ============================================================================
# Plot
# ============================================================================
fig, ax1 = plt.subplots(figsize=(10, 7))

# HFE curves
# ax1.semilogy(r_cm, bg_no_geometry(MU_TABULATED), ':',
#              color='tab:blue',
#              label=f'HFE BG (theoretical, no fix)')

# ax1.semilogy(r_cm, bg_geometry(MU_TABULATED), '-',
#              color='tab:blue',
#              label=f'HFE BG (theoretical)')

ax1.semilogy(r_cm, bg_geometry(mu_best), '--',
             color='tab:blue',
             label=f'HFE BG (fit)')

ax1.scatter(r_mc, bg_mc, marker='x', color='tab:blue', label='HFE BG (MC)')

# IV curves
r_mm = r_cm * 10
bg_iv = A_IV * np.exp(-MU_IV * (r_mm - r0_iv))
ax1.semilogy(r_cm, bg_iv, '--', color='tab:orange', label='IV BG (fit)')
ax1.scatter(iv_radii, iv_bgmc, marker='x',
            color='tab:orange', label='IV BG (MC)')

# Axis styling
ax1.set_xlabel('Inner-vessel radius r (cm)', fontsize=14)
ax1.set_ylabel('Background (counts/yr)',      fontsize=14)
ax1.tick_params(axis='both', which='major', labelsize=12)
ax1.grid(True, which='both', ls='--', alpha=0.3)

# Secondary y-axis: HFE mass
ax2 = ax1.twinx()
iv_radius_m  = r_cm / 100.0
total_vol_m3 = (4/3) * np.pi * iv_radius_m**3
hfe_vol_m3   = total_vol_m3 - VOL_OFFSET_M3
mass_hfe_t   = hfe_vol_m3 * DENSITY_T_PER_M3 - HARDWARE_MASS_T
ax2.plot(r_cm, mass_hfe_t, '-.', color='gray', label='HFE mass (t)')
ax2.set_ylabel('HFE mass (t)', fontsize=14)
ax2.tick_params(axis='y', which='major', labelsize=12)

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2,
           loc='lower center', ncol=2, fontsize=12)

plt.title('Cryostat & HFE Background vs IV Radius', fontsize=16)
plt.tight_layout()
plt.show()