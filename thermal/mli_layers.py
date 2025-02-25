import numpy as np
import matplotlib.pyplot as plt


def calculate_mli_heat_load(
    T_hot: float,  # K
    T_cold: float,  # K
    radius: float,  # m
    n_layers: int,  # number of MLI layers
    emissivity: float,  # effective emissivity of MLI
) -> float:
    """
    Calculate heat load through MLI using modified Stefan-Boltzmann equation
    Returns heat load in Watts
    """
    # Stefan-Boltzmann constant
    sigma = 5.67e-8  # W/(m²·K⁴)

    # Surface area of sphere
    area = 4 * np.pi * radius**2

    # Modified Stefan-Boltzmann equation for MLI
    # Q = σ·A·ε*(T_h⁴ - T_c⁴)/(N + 1)
    # where N is number of layers and ε is effective emissivity

    heat_load = sigma * area * emissivity * (T_hot**4 - T_cold**4) / (n_layers + 1)

    return heat_load


# Parameters
T_hot = 300  # K (room temperature)
T_cold = 165  # K (HFE temperature)
radius = 2.23  # m (assuming 4.46m diameter from your data)
emissivity = 0.03  # typical for aluminized mylar

# Calculate heat load vs number of layers
n_layers = np.arange(1, 51)
heat_loads = [
    calculate_mli_heat_load(T_hot, T_cold, radius, n, emissivity) for n in n_layers
]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(n_layers, heat_loads, "b-", linewidth=2)
plt.grid(True, alpha=0.3)
plt.xlabel("Number of MLI Layers", fontsize=12)
plt.ylabel("Heat Load (W)", fontsize=12)
plt.title("Heat Load vs Number of MLI Layers", fontsize=14)

# Add target line for 500W heat load
plt.axhline(y=500, color="r", linestyle="--", label="Target Heat Load (500W)")

# Find minimum layers needed for 500W target
min_layers = np.where(np.array(heat_loads) < 500)[0][0] + 1

plt.axvline(
    x=min_layers, color="g", linestyle="--", label=f"Minimum Layers ({min_layers})"
)

plt.legend()

# Print analysis
print(f"\nMinimum MLI layers needed: {min_layers}")
print(f"Heat load at minimum layers: {heat_loads[min_layers-1]:.1f} W")
print(f"\nHeat load per layer at operating point:")
print(f"Initial layer: {heat_loads[0]/1:.1f} W")
print(f"At minimum: {heat_loads[min_layers-1]/min_layers:.1f} W per layer")

# Calculate material requirements
layer_thickness = 0.025  # mm (typical for aluminized mylar)
total_thickness = layer_thickness * min_layers
material_density = 1.4  # g/cm³ (typical for mylar)
surface_area = 4 * np.pi * radius**2
material_mass = surface_area * (total_thickness / 1000) * material_density

print(f"\nMaterial requirements:")
print(f"Total MLI thickness: {total_thickness:.2f} mm")
print(f"Total material mass: {material_mass:.1f} kg")

plt.tight_layout()
plt.show()
