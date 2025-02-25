import numpy as np
import matplotlib.pyplot as plt

# Constants (corrected for water density)
mu = 0.03945  # Mass attenuation coefficient (cm²/g)
rho = 1.72  # HFE density (g/cm³)
original_thickness = 76  # Original HFE thickness (cm)
original_Rn = 9e-9  # Original radon concentration (Bq/kg water)
SNO_plus_sensitivity = (
    2e-9  # 2 mBq/m³ → 2e-6 Bq/m³ → 2e-9 Bq/kg (since 1 m³ water = 1000 kg)
)

# Cette valeur n'est pas bonne. Cf conv avec claude, et les slides qui ne sont pas vraiment compréhensible. Demander a Andrea peut_être? Ou le gars qui a fait la pres?


# Gamma transmission function
def transmission(thickness):
    return np.exp(-mu * rho * thickness)


# Required Rn concentration (Bq/kg) to maintain background
def required_radon(thickness):
    T_original = transmission(original_thickness)
    T_new = transmission(thickness)
    return original_Rn * (T_original / T_new)


# Generate HFE thickness range (cm)
thicknesses = np.linspace(10, 80, 100)
required_Rn = [required_radon(t) for t in thicknesses]

# Find maximum allowable HFE thickness where required Rn ≥ SNO+ sensitivity
feasible_thickness = thicknesses[
    np.where(np.array(required_Rn) >= SNO_plus_sensitivity)[0][-1]
]

# Plot
plt.figure(figsize=(10, 6))
plt.semilogy(thicknesses, required_Rn, lw=2, label="Required Radon Concentration")
plt.axhline(
    SNO_plus_sensitivity,
    color="red",
    ls="--",
    label=f"SNO+ Sensitivity ({SNO_plus_sensitivity:.1e} Bq/kg)",
)
plt.axvline(
    feasible_thickness,
    color="grey",
    ls=":",
    label=f"Max Verifiable Thickness ({feasible_thickness:.0f} cm)",
)

plt.xlabel("HFE Thickness (cm)")
plt.ylabel("Radon Concentration (Bq/kg)")
plt.title("HFE Thickness vs. Required Radon (Corrected for Water Density)")
plt.grid(True, which="both", linestyle="--")
plt.legend()
plt.show()

print(f"At 25 cm HFE (proposed design):\nRequired Rn = {required_radon(25):.2e} Bq/kg")
print(
    f"SNO+ can measure down to {SNO_plus_sensitivity:.1e} Bq/kg → {'Feasible' if required_radon(25) >= SNO_plus_sensitivity else 'Undetectable'}"
)
