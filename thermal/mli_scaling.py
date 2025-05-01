#!/usr/bin/env python3
"""
Analysis of MLI (Multi-Layer Insulation) scaling from EXO-200 to nEXO.

This script calculates the required number of MLI layers and analyzes radiative heat loads
and background rates for the nEXO cryostat design based on EXO-200 parameters.
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# Constants
SB = 5.67e-8            # Stefan-Boltzmann (W/m²K⁴)
# EXO-200
A200, Q200, L200, eps200 = 0.9, 90, 5, 0.035
T_o, T_i = 300, 170      # K
# nEXO
R_NEXO, Q_NEXO, C_NEXO = 1.691, 500, 3000
A_NEXO = 4 * math.pi * R_NEXO**2
# Background specs
nom_layers, nom_mass = 5, 6.5  # layers, kg
act_U = [5740e-12, 2500e-12]
act_Th = [1800e-12, 1540e-12]
eff_U, eff_Th = 7e-9, 9.72e-9
SEC_Y = 3.15576e7
spec_U, spec_Th = 1.24e4, 4050  # Bq/g

# EXO-200 radiative
EPS_EFF200 = eps200 / (L200 + 1)
Q_RAD200 = EPS_EFF200 * SB * A200 * (T_o**4 - T_i**4)
F_RAD200 = Q_RAD200 / Q200

# nEXO target radiative
Q_RAD_NEXO = F_RAD200 * Q_NEXO

# Calculate required layers
RADIATIVE_FACTOR = SB * A_NEXO * (T_o**4 - T_i**4)
n_layers = math.ceil(eps200 / (Q_RAD_NEXO / RADIATIVE_FACTOR) - 1)

# Actual nEXO performance
eps_eff_nexo = eps200 / (n_layers + 1)
Q_nexo_rad = eps_eff_nexo * RADIATIVE_FACTOR
f_nexo_rad = Q_nexo_rad / Q_NEXO
T_stab = 0.1 * (C_NEXO / Q200) * (Q_nexo_rad + (Q_NEXO - Q_RAD_NEXO)) / C_NEXO

print(f"EXO-200 radiative fraction: {F_RAD200:.1%}")
print(f"nEXO required layers: {n_layers}")
print(f"nEXO radiative load: {Q_nexo_rad:.1f} W ({f_nexo_rad:.1%})")
print(f"Est. temp stability: {T_stab:.4f} K")

# ----------------------------
# Plot radiative load vs. layers
# ----------------------------
plt.figure(figsize=(9, 7))
plt.rcParams.update({'font.size': 12})

# main curve
lyr = np.arange(1, n_layers * 2 + 1)
Qr = eps200 / (lyr + 1) * RADIATIVE_FACTOR
plt.plot(lyr, Qr, 'b-', linewidth=2, label='Aluminized PET (ε ≤ 0.035)')

# target line
plt.axhline(
    y=Q_RAD_NEXO,
    color='g',
    linestyle='--',
    linewidth=2,
    label=f'Target Radiative Budget from EXO-200 scaling ({Q_RAD_NEXO:.2f} W, {F_RAD200*100:.2f}% of total heat)'
)

# recommended layers
plt.axvline(
    x=n_layers,
    color='r',
    linestyle=':',
    linewidth=2,
    label=f'Min Required Layers ({n_layers})'
)

plt.xlabel('Number of MLI Layers', fontsize=14)
plt.ylabel('Radiative Heat Load (W)', fontsize=14)
plt.title(
    'nEXO: Radiative Heat Load vs. Number of MLI Layers\n'
    'Inner Cryostat Vessel radius: 1691 mm',
    fontsize=16
)
plt.grid(True)
plt.legend(fontsize=12)
plt.yscale('log')
plt.xlim(0, 50)
plt.tight_layout()
plt.show()

# ----------------------------
# Plot background vs. layers
# ----------------------------
plt.figure(figsize=(9, 7))
plt.rcParams.update({'font.size': 12})

lyr_bg = np.arange(1, max(n_layers, 40) + 1)
mass = nom_mass * lyr_bg / nom_layers

# pre-computed lists from above
bkg_A = []
for m in mass:
    bgA = (m * 1e3 * act_U[0] * spec_U * eff_U * SEC_Y +
           m * 1e3 * act_Th[0] * spec_Th * eff_Th * SEC_Y)
    bkg_A.append(bgA)

bkg_B = []
for m in mass:
    bgB = (m * 1e3 * act_U[1] * spec_U * eff_U * SEC_Y +
           m * 1e3 * act_Th[1] * spec_Th * eff_Th * SEC_Y)
    bkg_B.append(bgB)

plt.plot(
    lyr_bg,
    bkg_A,
    'b-',
    linewidth=2,
    label='Type A (U-238: 5.74×10⁻⁹ g/g, Th-232: 1.80×10⁻⁹ g/g)'
)
plt.plot(
    lyr_bg,
    bkg_B,
    'r-',
    linewidth=2,
    label='Type B (U-238: 2.50×10⁻⁹ g/g, Th-232: 1.54×10⁻⁹ g/g)'
)

plt.axvline(
    x=nom_layers,
    color='k',
    linestyle=':',
    linewidth=2,
    label='5 Layers (EXO-200 baseline)'
)
plt.axvline(
    x=n_layers,
    color='r',
    linestyle='--',
    linewidth=2,
    label=f'Recommended Layers ({n_layers})'
)

plt.xlabel('Number of MLI Layers', fontsize=14)
plt.ylabel('Background Contribution (counts/year)', fontsize=14)
plt.title(
    'Background Contribution vs. Number of MLI Layers\n'
    'Inner Cryostat Vessel radius: 1691 mm',
    fontsize=16
)
plt.grid(True)
plt.legend(fontsize=12)
plt.yscale('log')
plt.xlim(0, 40)
plt.tight_layout()
plt.show()