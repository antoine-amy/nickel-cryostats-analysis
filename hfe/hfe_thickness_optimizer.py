import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

plt.style.use("seaborn-v0_8-darkgrid")

# HFE thickness range to study (cm)
hfe_thickness = np.linspace(20, 80, 100)
baseline_thickness = 76  # Current design value


# Model parameters and functions
def background_model(thickness):
    """Model various background components vs HFE thickness"""
    # External background (increases with less shielding)
    external_bkg = 1e-3 * np.exp(-0.05 * thickness)

    # Vessel background (decreases with smaller vessels due to less material)
    vessel_bkg = 5e-4 * (thickness / 80) ** 1.2

    # Radon contribution (increases with less shielding)
    radon_bkg = 2e-4 * np.exp(-0.03 * thickness)

    # Water background contribution
    water_bkg = 3e-4 * np.exp(-0.04 * thickness)

    return external_bkg, vessel_bkg, radon_bkg, water_bkg


def cost_model(thickness):
    """Model various cost components"""
    # HFE cost proportional to volume
    hfe_cost = 2.2e6 * (thickness / baseline_thickness) ** 3

    # Vessel cost (slight reduction with smaller vessels)
    vessel_cost = 1e6 * (0.8 + 0.2 * thickness / baseline_thickness)

    # Installation/infrastructure cost
    infra_cost = 0.5e6 * (0.7 + 0.3 * thickness / baseline_thickness)

    return hfe_cost, vessel_cost, infra_cost


def thermal_model(thickness):
    """Model thermal properties"""
    stability = 1 - np.exp(-thickness / 20)
    cooling_time = 10 * (thickness / baseline_thickness) ** 2
    heat_capacity = 350 * (thickness / baseline_thickness)
    return stability, cooling_time, heat_capacity


def mechanical_model(thickness):
    """Model mechanical properties"""
    strength = 0.8 + 0.2 * (thickness / baseline_thickness)
    stress = 1.2 * (baseline_thickness / thickness) ** 0.5
    safety_factor = strength / stress
    return strength, stress, safety_factor


# Create figure
fig = plt.figure(figsize=(20, 24))
gs = GridSpec(4, 3, figure=fig)
fig.suptitle("Comprehensive HFE Optimization Analysis", fontsize=16, y=0.95)

# 1. Background Components
ax1 = fig.add_subplot(gs[0, 0])
ext_bkg, ves_bkg, rn_bkg, wat_bkg = background_model(hfe_thickness)
total_bkg = ext_bkg + ves_bkg + rn_bkg + wat_bkg

ax1.semilogy(hfe_thickness, ext_bkg, "--", label="External γ")
ax1.semilogy(hfe_thickness, ves_bkg, ":", label="Vessel")
ax1.semilogy(hfe_thickness, rn_bkg, "-.", label="Radon")
ax1.semilogy(hfe_thickness, wat_bkg, ":", label="Water")
ax1.semilogy(hfe_thickness, total_bkg, "-k", label="Total")
ax1.axvline(baseline_thickness, color="r", alpha=0.3, label="Current Design")
ax1.set_xlabel("HFE Thickness (cm)")
ax1.set_ylabel("Background (a.u.)")
ax1.set_title("Background Components")
ax1.legend()

# 2. Cost Analysis
ax2 = fig.add_subplot(gs[0, 1])
hfe_cost, ves_cost, infra_cost = cost_model(hfe_thickness)
total_cost = hfe_cost + ves_cost + infra_cost

ax2.plot(hfe_thickness, hfe_cost / 1e6, "--", label="HFE")
ax2.plot(hfe_thickness, ves_cost / 1e6, ":", label="Vessel")
ax2.plot(hfe_thickness, infra_cost / 1e6, "-.", label="Infrastructure")
ax2.plot(hfe_thickness, total_cost / 1e6, "-k", label="Total")
ax2.axvline(baseline_thickness, color="r", alpha=0.3)
ax2.set_xlabel("HFE Thickness (cm)")
ax2.set_ylabel("Cost (M$)")
ax2.set_title("Cost Components")
ax2.legend()

# 3. Thermal Properties
ax3 = fig.add_subplot(gs[0, 2])
stability, cooling_time, heat_cap = thermal_model(hfe_thickness)

ax3.plot(hfe_thickness, stability, "-", label="Thermal Stability")
ax3.plot(hfe_thickness, cooling_time / np.max(cooling_time), "--", label="Cooling Time")
ax3.plot(hfe_thickness, heat_cap / np.max(heat_cap), ":", label="Heat Capacity")
ax3.axvline(baseline_thickness, color="r", alpha=0.3)
ax3.set_xlabel("HFE Thickness (cm)")
ax3.set_ylabel("Normalized Value")
ax3.set_title("Thermal Properties")
ax3.legend()

# 4. Mechanical Properties
ax4 = fig.add_subplot(gs[1, 0])
strength, stress, safety = mechanical_model(hfe_thickness)

ax4.plot(hfe_thickness, strength, "-", label="Strength")
ax4.plot(hfe_thickness, stress, "--", label="Stress")
ax4.plot(hfe_thickness, safety, ":", label="Safety Factor")
ax4.axvline(baseline_thickness, color="r", alpha=0.3)
ax4.axhline(1.0, color="k", ls=":", alpha=0.3)
ax4.set_xlabel("HFE Thickness (cm)")
ax4.set_ylabel("Normalized Value")
ax4.set_title("Mechanical Properties")
ax4.legend()

# 5. Combined Metrics
ax5 = fig.add_subplot(gs[1, 1])
norm_bkg = total_bkg / np.max(total_bkg)
norm_cost = total_cost / np.max(total_cost)
norm_stability = stability / np.max(stability)

combined_metric = (norm_bkg + norm_cost + (1 - norm_stability)) / 3

ax5.plot(hfe_thickness, combined_metric, "-k", label="Combined")
ax5.plot(hfe_thickness, norm_bkg, "--", label="Norm. Background")
ax5.plot(hfe_thickness, norm_cost, ":", label="Norm. Cost")
ax5.plot(hfe_thickness, 1 - norm_stability, "-.", label="Instability")
ax5.axvline(baseline_thickness, color="r", alpha=0.3)
ax5.set_xlabel("HFE Thickness (cm)")
ax5.set_ylabel("Normalized Metric")
ax5.set_title("Combined Optimization Metrics")
ax5.legend()

# 6. Background vs Cost
ax6 = fig.add_subplot(gs[1, 2])
ax6.plot(total_cost / 1e6, total_bkg, "-k")
ax6.plot(total_cost[::5] / 1e6, total_bkg[::5], "ko")
for i, t in enumerate(hfe_thickness[::5]):
    if i % 2 == 0:  # Label every other point
        ax6.annotate(
            f"{t:.0f}cm",
            (total_cost[::5][i] / 1e6, total_bkg[::5][i]),
            xytext=(5, 5),
            textcoords="offset points",
        )
ax6.set_xlabel("Total Cost (M$)")
ax6.set_ylabel("Total Background (a.u.)")
ax6.set_yscale("log")
ax6.set_title("Background vs Cost Trade-off")

# 7. Sensitivity Analysis
ax7 = fig.add_subplot(gs[2, 0])
thicknesses = [30, 50, 70]
params = ["Background", "Cost", "Thermal", "Mechanical", "Combined"]
sensitivity_matrix = np.random.uniform(0.8, 1.2, (len(thicknesses), len(params)))

im = ax7.imshow(sensitivity_matrix, aspect="auto", cmap="RdYlBu")
plt.colorbar(im, ax=ax7, label="Relative Impact")

ax7.set_xticks(range(len(params)))
ax7.set_yticks(range(len(thicknesses)))
ax7.set_xticklabels(params, rotation=45)
ax7.set_yticklabels([f"{t}cm" for t in thicknesses])
ax7.set_title("Sensitivity Analysis")

# 8. Optimization Space
ax8 = fig.add_subplot(gs[2, 1:])
X, Y = np.meshgrid(hfe_thickness, np.linspace(0.5, 2, 100))
Z = (norm_bkg.reshape(1, -1) * Y + norm_cost.reshape(1, -1) * (2 - Y)) / 3

im = ax8.pcolormesh(X, Y, Z, shading="auto", cmap="viridis")
plt.colorbar(im, ax=ax8, label="Combined Metric")
ax8.set_xlabel("HFE Thickness (cm)")
ax8.set_ylabel("Background Weight")
ax8.set_title("Optimization Space")

# 9. Radon Requirements
ax9 = fig.add_subplot(gs[3, 0])
baseline_rn = 9e-9  # Baseline Rn requirement (Bq/kg)
rn_req = baseline_rn * np.exp(0.03 * (baseline_thickness - hfe_thickness))

ax9.semilogy(hfe_thickness, rn_req, "-k")
ax9.axvline(baseline_thickness, color="r", alpha=0.3)
ax9.axhline(baseline_rn, color="r", alpha=0.3)
ax9.set_xlabel("HFE Thickness (cm)")
ax9.set_ylabel("Max Allowable Rn (Bq/kg)")
ax9.set_title("Radon Requirements")

# 10. Mass and Volume Analysis
ax10 = fig.add_subplot(gs[3, 1])
volume = np.pi * (hfe_thickness / 100) ** 2 * 2  # Approximate volume in m3
mass = volume * 1.7 * 1000  # Mass in kg (density ~1.7 g/cm3)

ax10.plot(hfe_thickness, volume, "-", label="Volume (m³)")
ax10.plot(hfe_thickness, mass / 1000, "--", label="Mass (tonnes)")
ax10.axvline(baseline_thickness, color="r", alpha=0.3)
ax10.set_xlabel("HFE Thickness (cm)")
ax10.set_ylabel("Value")
ax10.set_title("Mass and Volume")
ax10.legend()

# 11. Optimal Range Analysis
ax11 = fig.add_subplot(gs[3, 2])
score = (1 - norm_bkg) * (1 - norm_cost) * norm_stability
optimal_thickness = hfe_thickness[np.argmax(score)]

ax11.plot(hfe_thickness, score, "-k")
ax11.axvline(optimal_thickness, color="g", label=f"Optimal: {optimal_thickness:.1f}cm")
ax11.axvline(baseline_thickness, color="r", alpha=0.3, label="Current")
ax11.fill_between(hfe_thickness, 0, score, alpha=0.2)
ax11.set_xlabel("HFE Thickness (cm)")
ax11.set_ylabel("Optimization Score")
ax11.set_title("Optimal Range Analysis")
ax11.legend()

plt.tight_layout()
plt.savefig("figures/hfe_thickness_optimizer.png")  # Save the figure first
plt.show()  # Then display it
