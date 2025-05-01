#!/usr/bin/env python3
"""
Simplified MLI scaling analysis for nEXO.
Computes HFE and inner vessel (IV) backgrounds and HFE mass vs vessel radius.
"""
import numpy as np
import matplotlib.pyplot as plt

# Constants
# EXO-200 radiative fit
BASE_BG_HFE = 5e-3        # counts/year at INITIAL_IV_RAD
MU_HFE = 0.04 * 1.72      # cm^-1 (mass * density)
INITIAL_IV_RAD = 168.5    # cm

# IV background fit parameters
A_IV = 4.601024e-03       # counts/year
MU_IV = 0.006223          # mm^-1
r0_iv = min([1025, 1225, 1510, 1690])  # mm

# TPC geometry
R_TPC = 56.665            # cm
H_TPC = 59.15             # half-height cm
vol_tpc = np.pi * R_TPC**2 * (2 * H_TPC)

# HFE density
RHO_HFE = 1.72            # g/cm^3

# Radii array for calculation
r_cm = np.linspace(95, 170, 300)
dr = r_cm[1] - r_cm[0]

# HFE background via vectorized integral
atten = np.exp(-MU_HFE * np.clip(r_cm - R_TPC, 0, None))
integrand = 4 * np.pi * r_cm**2 * atten
cum_int = np.cumsum(integrand) * dr

# Normalization to BASE_BG_HFE at INITIAL_IV_RAD
idx0 = np.argmin(np.abs(r_cm - INITIAL_IV_RAD))
c_factor = BASE_BG_HFE / cum_int[idx0]
bg_hfe = c_factor * cum_int

# IV background
r_mm = r_cm * 10
bg_iv = A_IV * np.exp(-MU_IV * (r_mm - r0_iv))

# HFE mass: sphere minus TPC cylinder
vol_sphere = 4/3 * np.pi * r_cm**3
vol_hfe = np.maximum(vol_sphere - vol_tpc, 0)
mass_hfe = vol_hfe * RHO_HFE / 1e6  # convert g → tonne

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.loglog(r_cm, bg_hfe, '-', label='HFE BG')
ax1.loglog(r_cm, bg_iv, '--', label='IV BG')
ax1.scatter([x / 10 for x in [1025, 1225, 1510, 1690]],
            [4.5e-3, 1.4e-3, 2.5e-4, 7e-5],
            marker='^', label='Image data')
ax1.set_xlabel('IV radius (cm)')
ax1.set_ylabel('Background (counts/year)')
ax1.grid(True, which='both', ls='--', alpha=0.3)

ax2 = ax1.twinx()
ax2.plot(r_cm, mass_hfe, '-.', label='HFE mass (tonnes)')
ax2.set_ylabel('HFE mass (t)')
ax2.grid(False)

# Combined legend
lines, labels = ax1.get_legend_handles_labels()
l2, l2l = ax2.get_legend_handles_labels()
ax1.legend(lines + l2, labels + l2l, loc='upper center', ncol=3)

plt.title('Simplified MLI: BG & HFE Mass vs IV Radius')
plt.tight_layout()
plt.show()
