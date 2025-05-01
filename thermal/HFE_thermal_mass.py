#!/usr/bin/env python3
"""Analysis of HFE thermal mass requirements for temperature stability.

This script calculates the minimum HFE mass needed to maintain temperature stability
given various components, heat loads, and cooling system response times.
"""

import numpy as np
import matplotlib.pyplot as plt

# Module-level constants (Pylint naming style)
BASE_CAPS = np.array([5000 * 350, 566 * 385, 644 * 710])  # J/K
BASE_CAP = BASE_CAPS.sum()
HEAT_LOAD = 500  # W
MAX_TEMP_RISE = 0.1  # K
RESPONSE_TIME_S = 3600  # s (1 h)
HFE_SPECIFIC_HEAT = 1213.36  # J/kg/K
HFE_COOL_SPECIFIC_HEAT = 1250  # J/kg/K for cooldown
COOLING_POWER = 3000  # W

# 1) Minimum HFE mass calculation
ENERGY_INPUT = HEAT_LOAD * RESPONSE_TIME_S  # J
REQUIRED_CAPACITY = ENERGY_INPUT / MAX_TEMP_RISE  # J/K
minimum_hfe_mass = max(0, (REQUIRED_CAPACITY - BASE_CAP) / HFE_SPECIFIC_HEAT)
print(f"Minimum HFE mass: {minimum_hfe_mass/1000:.1f} tonnes")

# 2) Thermal stats at minimum mass
total_capacity = BASE_CAP + minimum_hfe_mass * HFE_COOL_SPECIFIC_HEAT
cooling_energy = total_capacity * (300 - 165)  # J for ΔT=135 K
cooling_time_h = cooling_energy / COOLING_POWER / 3600
print(f"Total heat capacity: {total_capacity/1e3:.1f} kJ/K")
print(f"Cooldown energy: {cooling_energy/1e9:.2f} GJ")
print(f"Cooldown time: {cooling_time_h:.1f} h")

# 3) Contribution breakdown
COMPONENT_NAMES = ['LXe', 'Copper TPC', 'IV Carbon', 'HFE']
capacities = np.append(BASE_CAPS, minimum_hfe_mass * HFE_COOL_SPECIFIC_HEAT)
for name, cap in zip(COMPONENT_NAMES, capacities):
    percentage = cap / total_capacity * 100
    print(f"{name}: {cap/1e3:.1f} kJ/K ({percentage:.1f} %) ")

# 4) Visualization: temperature rise vs HFE mass
hfe_masses = np.linspace(0, minimum_hfe_mass * 2, 100)  # kg
RESPONSE_TIMES_H = np.array([0.5, 1, 2, 4])  # hours

plt.figure(figsize=(9, 7))
for response_time_h in RESPONSE_TIMES_H:
    response_time_s = response_time_h * 3600
    temperature_rise = (
        HEAT_LOAD * response_time_s /
        (BASE_CAP + hfe_masses * HFE_COOL_SPECIFIC_HEAT)
    )
    plt.plot(
        hfe_masses / 1000,
        temperature_rise,
        label=f"{int(response_time_h)}h response time",
        linewidth=2
    )

# Add stability and minimum lines
plt.axhline(
    y=MAX_TEMP_RISE,
    color='r',
    linestyle='--',
    linewidth=2,
    label=f"Stability limit ({MAX_TEMP_RISE} K)"
)
plt.axvline(
    x=minimum_hfe_mass / 1000,
    color='g',
    linestyle='--',
    linewidth=2,
    label=f"Minimum HFE mass ({minimum_hfe_mass/1000:.1f} tonnes)"
)

# Grid and legend
plt.grid(True, alpha=0.3)
plt.xlabel('HFE mass (tonnes)')
plt.ylabel('Temperature rise (K)')
plt.legend(fontsize=16)
plt.tight_layout()
plt.show()
