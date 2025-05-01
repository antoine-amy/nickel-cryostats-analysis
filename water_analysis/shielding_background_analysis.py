#!/usr/bin/env python3
"""Analysis of shielding and background rates for the nickel cryostat design.

This script analyzes the relationship between inner vessel radius and background rates,
and HFE thickness versus required radon concentration for shielding calculations.
"""
import numpy as np
import matplotlib.pyplot as plt

# --- Shared constants ---
# TODO: Verify the SNO+ sensitivity value with Andrea or presenter
R0, B0, MU, MU_ERR, REQ = 1691, 2.76e-8, 0.00685, 0.0005, 1e-5
THICK_ORIG, RN_ORIG, MU_HFE, RHO, SNO = 76, 9e-9, 0.03945, 1.72, 2e-9

# 1) Inner vessel radius vs. background rate
r = np.linspace(1000, 1700, 100)
br = B0 * np.exp(-MU * (r - R0))
br_up = B0 * np.exp(-(MU - MU_ERR) * (r - R0))
br_dn = B0 * np.exp(-(MU + MU_ERR) * (r - R0))
plt.figure(figsize=(9, 7))
plt.semilogy(r, br, label='Expected')
plt.fill_between(r, br_dn, br_up, alpha=0.3, label='±1σ')
plt.axhline(REQ, color='r', ls='--', label=f'Req {REQ:.1e}')
plt.axvline(R0, color='g', ls=':', label=f'Baseline {R0} mm')
plt.xlabel('Radius (mm)')
plt.ylabel('Background rate (counts/year)')
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
print(f'Minimum radius: {R0 - np.log(REQ/B0)/MU:.1f} mm')

# 2) HFE thickness vs. required radon concentration
t = np.linspace(10, 80, 100)
trans = np.exp(-MU_HFE * RHO * t)
trans0 = np.exp(-MU_HFE * RHO * THICK_ORIG)
rr = RN_ORIG * (trans0 / trans)
plt.figure(figsize=(9, 7))
plt.semilogy(t, rr, label='Required radon')
plt.axhline(SNO, color='r', ls='--', label=f'SNO+ {SNO:.1e} Bq/kg')
max_t = t[np.where(rr >= SNO)[0][-1]].item()
plt.axvline(max_t, color='k', ls=':', label=f'Max {max_t:.0f} cm')
plt.xlabel('HFE thickness (cm)')
plt.ylabel('Radon concentration (Bq/kg)')
plt.grid(True, which='both', ls='--', alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
val25 = RN_ORIG * (trans0 / np.exp(-MU_HFE * RHO * 25))
print(f'At 25 cm: {val25:.2e} Bq/kg →', 'Feasible' if val25 >= SNO else 'Undetectable')
