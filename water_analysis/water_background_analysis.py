#!/usr/bin/env python3
"""Analysis of water background radiation and hit efficiency calculations."""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle

# Constants
TANK_W, VESSEL_D = 12.3, 4.46  # Tank width and vessel diameter (m)
RHO, RN, BR, SOLID = 1000, 9e-9, 0.01545, 0.5  # density, Rn activity, gamma BR, solid angle
MU = 0.01  # combined attenuation coefficient

R = VESSEL_D / 2
BR_EFF = RN * BR * SOLID

# 1) Heatmap slice at z=0
N = 200
x = np.linspace(-TANK_W/2, TANK_W/2, N)
X, Y = np.meshgrid(x, x)
d = np.sqrt(X**2 + Y**2) - R
E = np.where(d > 0, BR_EFF * np.exp(-MU * RHO * d), np.nan)

plt.figure(figsize=(9, 7))
pc = plt.pcolormesh(X, Y, E, norm=LogNorm(), shading='auto')
plt.gca().add_patch(Circle((0, 0), R, fill=False, color='r'))
plt.colorbar(pc, label='Hit efficiency')
plt.title('Hit efficiency slice at z=0 m')
plt.xlabel('x (m)')
plt.ylabel('y (m)')
plt.tight_layout()
plt.show()

# 2) Radial dependence
r = np.linspace(R, TANK_W/2, 500)
d2 = r - R
eff2 = BR_EFF * np.exp(-MU * RHO * d2)
dr = r[1] - r[0]
vol_shell = 4 * np.pi * r**2 * dr
activity = vol_shell * RHO * RN * BR
contrib = activity * eff2
cum = np.cumsum(contrib)
perc = cum / cum[-1] * 100

fig, ax1 = plt.subplots()
ax1.plot(r, eff2, 'b', label='Efficiency')
ax1.set_xlabel('Radius (m)')
ax1.set_ylabel('Efficiency', color='b')

ax2 = ax1.twinx()
ax2.plot(r, perc, 'r', label='Cumulative %')
ax2.axhline(90, ls=':', color='k', label='90% threshold')
ax2.set_ylabel('Cumulative contribution (%)', color='r')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')

plt.title('Radial profile')
plt.tight_layout()
plt.show()

idx = np.searchsorted(perc, 90)
print(f"90% of contrib within {d2[idx]:.2f} m of vessel")
