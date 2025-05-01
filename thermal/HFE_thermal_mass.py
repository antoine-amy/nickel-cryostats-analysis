import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class Component:
    mass: float  # kg
    specific_heat: float  # J/kg/K
    name: str


def calculate_minimum_hfe_mass(
    base_components: List[Component],
    heat_load: float,  # Watts
    max_temp_rise: float,  # Kelvin
    response_time: float,  # seconds - time for cooling system response
    hfe_specific_heat: float,  # J/kg/K
) -> float:
    """Calculate minimum HFE mass needed to maintain temperature stability"""

    # Calculate base heat capacity
    base_capacity = sum(comp.mass * comp.specific_heat for comp in base_components)

    # Calculate required total heat capacity for given temp rise
    # Use response time instead of 24h for more realistic scenario
    energy_input = heat_load * response_time  # Joules
    required_total_capacity = energy_input / max_temp_rise  # J/K

    # Calculate required HFE mass
    required_hfe_capacity = required_total_capacity - base_capacity
    min_hfe_mass = required_hfe_capacity / hfe_specific_heat

    return max(0, min_hfe_mass)


# Define base components
base_components = [
    Component(5000, 350, "LXe"),
    Component(566, 385, "Copper TPC"),
    Component(644, 710, "IV Carbon Composite"),
]

# Calculate for stability requirement
# Assume 1-hour response time for cooling system adjustment
min_hfe_mass = calculate_minimum_hfe_mass(
    base_components=base_components,
    heat_load=500,  # 500W steady state
    max_temp_rise=0.1,  # 0.1K stability
    response_time=3600,  # 1 hour response time
    hfe_specific_heat=1213.36,
)

# Create visualization
hfe_masses = np.linspace(0, min_hfe_mass * 2, 100)
response_times = np.array([0.5, 1, 2, 4])  # hours

plt.figure(figsize=(12, 9))

for hours in response_times:
    temp_rises = []
    for mass in hfe_masses:
        total_capacity = sum(comp.mass * comp.specific_heat for comp in base_components)
        total_capacity += mass * 1250
        temp_rise = (500 * hours * 3600) / total_capacity
        temp_rises.append(temp_rise)
    plt.plot(
        hfe_masses / 1000, temp_rises, label=f"{hours}h response time", linewidth=2
    )

plt.axhline(y=0.1, color="r", linestyle="--", label="Stability limit (0.1K)")
plt.axvline(
    x=min_hfe_mass / 1000,
    color="g",
    linestyle="--",
    label=f"Minimum HFE mass ({min_hfe_mass/1000:.1f} tonnes)",
)

plt.grid(True, alpha=0.3)
plt.xlabel("HFE Mass (tonnes)", fontsize=20)
plt.ylabel("Temperature Rise (K)", fontsize=20)
# plt.title("Temperature Rise vs HFE Mass for Different Response Times\n(500W steady-state heat load)",fontsize=14)
plt.legend(fontsize=16)

# Print analysis
print(f"\nMinimum HFE requirement: {min_hfe_mass/1000:.1f} tonnes")

# Calculate thermal properties at minimum mass
min_config = base_components + [Component(min_hfe_mass, 1250, "HFE-7200")]
total_capacity = sum(comp.mass * comp.specific_heat for comp in min_config)
cooldown_energy = total_capacity * (300 - 165)  # J for cooling from 300K to 165K
cooldown_time = cooldown_energy / 3000  # seconds with 3kW cooling

print(f"\nAt minimum configuration:")
print(f"Total heat capacity: {total_capacity/1000:.1f} kJ/K")
print(f"Cooldown energy requirement: {cooldown_energy/1e9:.2f} GJ")
print(f"Cooldown time with 3kW cooling: {cooldown_time/3600:.1f} hours")

# Show contribution breakdown
print("\nHeat capacity contributions:")
for comp in min_config:
    contribution = comp.mass * comp.specific_heat
    percentage = (contribution / total_capacity) * 100
    print(f"{comp.name}: {contribution/1000:.1f} kJ/K ({percentage:.1f}%)")

plt.tight_layout()
plt.show()
