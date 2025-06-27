#!/usr/bin/env python3
"""
Cryostat + HFE + Vessel backgrounds vs inner-vessel radius
–––––––––––––––––––––––––––––––––––––––––––––––––––––
Streamlined: best-fit curves, MC points with error bars for HFE, IV, and OV with consistent coloring.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter

# ============================================================================
# Constants and inputs
# ============================================================================
BASE_BG_HFE    = 5e-3             # counts/yr at INITIAL_IV_RAD (MC point)
MU_TABULATED   = 0.04 * 1.72      # 0.0688 cm⁻¹ for 2.615 MeV γ in HFE
INITIAL_IV_RAD = 169.1            # cm – reference radius for normalization

# Geometry (TPC)
R_TPC = 56.665   # cm radius
H_TPC = 59.15    # cm half-height

# IV and OV fit reference radii (mm)
r0_iv = 1026
r0_ov = 1026

# HFE mass axis parameters
DENSITY_T_PER_M3 = 1.72           # t/m³
VOL_OFFSET_M3    = 2.2 - 0.58     # m³ at R=169.1 cm
HARDWARE_MASS_T  = 0.240          # t

# Radius grid
r_cm = np.linspace(95, 170, 300)
dr   = r_cm[1] - r_cm[0]

# ============================================================================
# Monte-Carlo data + errors
# ============================================================================
r_mc       = np.array([102.6, 110.0, 130.0, 169.1])  # cm
bg_mc      = np.array([4.40751e-3, 4.43671e-3, 4.96e-3, 5e-3])
bg_mc_err  = np.array([5.11e-3,    5.09e-3,    5.72e-3,    5.75e-3])

iv_radii    = np.array([1026, 1226, 1510, 1691]) / 10  # cm
iv_bgmc     = np.array([3.98e-3, 1.07e-3, 1.84e-4, 5.62e-5])
iv_bgmc_err = np.array([2.97e-3, 6.95e-4, 1.09e-4, 3.45e-5])

ov_radii    = iv_radii                                # cm
ov_bgmc     = np.array([4.79e-3, 1.58e-3, 2.82e-4, 8.15e-5])
ov_bgmc_err = np.array([3.43e-3, 1.00e-3, 1.73e-4, 3.98e-5])

# ============================================================================
# Geometry helpers
# ============================================================================
def fsolid(r):
    r = np.asarray(r, float)
    f = np.zeros_like(r)
    mask = r >= R_TPC
    if np.any(mask):
        rm = r[mask]
        cap = 0.5 * (1 - np.sqrt(1 - (R_TPC/rm)**2))
        bar = (R_TPC * (2*H_TPC)) / (4*np.pi*rm**2)
        f[mask] = np.minimum(cap + bar, 0.5)
    return f

def fpath(r):
    d = np.clip(r - R_TPC, 0, None)
    return 1 + ((np.pi/2) - 1) * np.clip(d/H_TPC, 0, 1)

# ============================================================================
# Analytic HFE background (geometry-corrected)
# ============================================================================
def bg_geometry(mu):
    delta_r = np.clip(r_cm - R_TPC, 0, None)
    atten   = np.exp(-mu * fpath(r_cm) * delta_r)
    integrand = 4 * np.pi * r_cm**2 * fsolid(r_cm) * atten
    cum = np.cumsum(integrand) * dr
    idx0 = np.argmin(np.abs(r_cm - INITIAL_IV_RAD))
    return cum * (BASE_BG_HFE / cum[idx0])

# ============================================================================
# Fit μ_HFE to the MC points
# ============================================================================
mu_vals = np.linspace(0.02, 0.12, 600)
rss = [np.sum((np.interp(r_mc, r_cm, bg_geometry(mu)) - bg_mc)**2) for mu in mu_vals]
mu_best = mu_vals[np.argmin(rss)]
print(f"Best-fit μ_HFE = {mu_best:.5f} cm⁻¹ (~{mu_best / MU_TABULATED:.2f}× tabulated)")

# ============================================================================
# Fit IV and OV with log-linear model
# ============================================================================
r_mm_iv = iv_radii * 10
coef_iv = np.polyfit(r_mm_iv - r0_iv, np.log(iv_bgmc), 1)
mu_iv_fit = -coef_iv[0]
logA_iv   = coef_iv[1]

r_mm_ov = ov_radii * 10
coef_ov = np.polyfit(r_mm_ov - r0_ov, np.log(ov_bgmc), 1)
mu_ov_fit = -coef_ov[0]
logA_ov   = coef_ov[1]

# ============================================================================
# Plot
# ============================================================================
plt.rc('font', size=12)
fig, ax1 = plt.subplots(figsize=(12,8))

# Colors for components
HFE_COLOR = '#4A90A4'
IV_COLOR  = '#D2691E'
OV_COLOR  = '#228B22'

# Plot fits
ax1.semilogy(r_cm, bg_geometry(mu_best), '--', color=HFE_COLOR, label='HFE fit', lw=2)
bg_iv_fit = np.exp(logA_iv + coef_iv[0] * (r_cm*10 - r0_iv))
ax1.semilogy(r_cm, bg_iv_fit, '--', color=IV_COLOR, label='IV fit', lw=2)
bg_ov_fit = np.exp(logA_ov + coef_ov[0] * (r_cm*10 - r0_ov))
ax1.semilogy(r_cm, bg_ov_fit, '--', color=OV_COLOR, label='OV fit', lw=2)

# Plot MC data
ax1.errorbar(r_mc, bg_mc, yerr=bg_mc_err, fmt='o', ms=6, capsize=4,
             label='HFE MC', color=HFE_COLOR, markerfacecolor='white', markeredgecolor=HFE_COLOR)
ax1.errorbar(iv_radii, iv_bgmc, yerr=iv_bgmc_err, fmt='s', ms=6, capsize=4,
             label='IV MC', color=IV_COLOR, markerfacecolor='white', markeredgecolor=IV_COLOR)
ax1.errorbar(ov_radii, ov_bgmc, yerr=ov_bgmc_err, fmt='^', ms=6, capsize=4,
             label='OV MC', color=OV_COLOR, markerfacecolor='white', markeredgecolor=OV_COLOR)

# Axes styling
ax1.set_xlabel('Inner vessel radius r (cm)')
ax1.set_ylabel('Background (counts/yr)')
ax1.set_yscale('log')
ax1.set_ylim(1e-6, 0.1)
ax1.grid(which='both', linestyle='--', linewidth=0.5, alpha=0.7)
ax1.minorticks_on()
ax1.yaxis.set_major_formatter(ScalarFormatter(useMathText=True))
ax1.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

# Secondary axis: HFE mass
ax2 = ax1.twinx()
iv_m = r_cm / 100.0
vol = (4/3) * np.pi * iv_m**3
hfe_vol = vol - VOL_OFFSET_M3
mass_hfe = hfe_vol * DENSITY_T_PER_M3 - HARDWARE_MASS_T
ax2.plot(r_cm, mass_hfe, ':', color=HFE_COLOR, label='HFE mass (t)')
ax2.set_ylabel('HFE mass (t)')
ax2.minorticks_on()

# Legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, ncol=1, 
           bbox_to_anchor=(1.15, 1), loc='upper left')

plt.title('Cryostat & HFE Background vs IV Radius')
plt.tight_layout()
plt.show()
