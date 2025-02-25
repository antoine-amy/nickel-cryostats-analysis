import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.stats import norm
import matplotlib.colors as mcolors

# Constants (unchanged)
WATER_TANK_HEIGHT = 13.3  # m
WATER_TANK_WIDTH = 12.3  # m (assuming cubic tank)
OUTER_VESSEL_DIAMETER = 4.46  # m (spherical vessel)

# Physics parameters with solid angle correction
WATER_DENSITY = 1000  # kg/m³
RADON_ACTIVITY = 9e-9  # Bq/kg (Rn-222 only)
GAMMA_2447_BRANCHING_RATIO = 0.01545  # 1.545% for the 2447.69 keV gamma
SOLID_ANGLE_FACTOR = 0.5  # 50% of radiation directed towards TPC
HIT_EFFICIENCY = 2e-9 * SOLID_ANGLE_FACTOR  # counts/decay (corrected for solid angle)
ATTENUATION_COEFF = 0.1 * 0.1  # m²/kg


def calculate_volumes():
    """Calculate volumes of components with spherical vessel and cubic tank."""
    tank_volume = WATER_TANK_WIDTH**2 * WATER_TANK_HEIGHT
    vessel_radius = OUTER_VESSEL_DIAMETER / 2
    vessel_volume = (4 / 3) * np.pi * vessel_radius**3
    water_volume = tank_volume - vessel_volume
    return tank_volume, vessel_volume, water_volume


def calculate_distance_from_vessel(x, y, z):
    """3D distance from point (x,y,z) to spherical vessel surface."""
    vessel_radius = OUTER_VESSEL_DIAMETER / 2
    r = np.sqrt(x**2 + y**2 + z**2)
    return np.where(r > vessel_radius, r - vessel_radius, 0)


def calculate_attenuation(distance):
    """Exponential attenuation for beta/gamma radiation."""
    return np.exp(-ATTENUATION_COEFF * WATER_DENSITY * distance)


def create_background_heatmap(z_slice=0):
    """3D heatmap sliced at given z-coordinate, including solid angle effect."""
    n_points = 200
    x = np.linspace(-WATER_TANK_WIDTH / 2, WATER_TANK_WIDTH / 2, n_points)
    y = np.linspace(-WATER_TANK_WIDTH / 2, WATER_TANK_WIDTH / 2, n_points)
    X, Y = np.meshgrid(x, y)
    Z = np.full_like(X, z_slice)

    R = calculate_distance_from_vessel(X, Y, Z)
    attenuation = calculate_attenuation(R)
    efficiency = np.where(R > 0, HIT_EFFICIENCY * attenuation, np.nan)

    fig, ax = plt.subplots(figsize=(12, 10))
    cmap = plt.cm.viridis.copy()
    cmap.set_bad("white")

    im = ax.pcolormesh(
        X, Y, efficiency, cmap=cmap, shading="auto", norm=mcolors.LogNorm()
    )

    vessel_radius = OUTER_VESSEL_DIAMETER / 2
    vessel = plt.Circle((0, 0), vessel_radius, fill=False, color="red", linewidth=2)
    ax.add_patch(vessel)
    ax.set_title(f"Hit Efficiency at z={z_slice} m (with 50% solid angle)")
    plt.colorbar(im, label="Hit Efficiency (counts/decay)")
    plt.show()


def calculate_background_rate():
    """Vectorized 3D calculation of background rate with solid angle correction."""
    _, _, water_volume = calculate_volumes()
    water_mass = water_volume * WATER_DENSITY

    n = 50  # Grid resolution
    x = np.linspace(-WATER_TANK_WIDTH / 2, WATER_TANK_WIDTH / 2, n)
    y = np.linspace(-WATER_TANK_WIDTH / 2, WATER_TANK_WIDTH / 2, n)
    z = np.linspace(-WATER_TANK_HEIGHT / 2, WATER_TANK_HEIGHT / 2, n)
    X, Y, Z = np.meshgrid(x, y, z, indexing="ij")

    R = calculate_distance_from_vessel(X, Y, Z)
    in_water = (
        (R > 0)
        & (np.abs(X) <= WATER_TANK_WIDTH / 2)
        & (np.abs(Y) <= WATER_TANK_WIDTH / 2)
        & (np.abs(Z) <= WATER_TANK_HEIGHT / 2)
    )

    attenuation = calculate_attenuation(R[in_water])
    efficiencies = (
        HIT_EFFICIENCY * attenuation
    )  # Solid angle already included in HIT_EFFICIENCY

    voxel_volume = (x[1] - x[0]) * (y[1] - y[0]) * (z[1] - z[0])
    total_efficiency = np.sum(efficiencies) * voxel_volume
    avg_efficiency = total_efficiency / (np.sum(in_water) * voxel_volume)

    total_activity = water_mass * RADON_ACTIVITY * GAMMA_2447_BRANCHING_RATIO
    bg_rate = total_activity * avg_efficiency

    return {
        "water_mass": water_mass,
        "avg_hit_efficiency": avg_efficiency,
        "total_activity": total_activity,
        "background_rate": bg_rate,
    }


def plot_radial_dependence():
    """Radial profile of hit efficiency and cumulative contribution percentage."""
    r = np.linspace(OUTER_VESSEL_DIAMETER / 2, WATER_TANK_WIDTH / 2, 500)
    distances = r - OUTER_VESSEL_DIAMETER / 2
    attenuation = calculate_attenuation(distances)
    efficiencies = (
        HIT_EFFICIENCY * attenuation
    )  # Solid angle included in HIT_EFFICIENCY

    dr = r[1] - r[0]
    annular_vol = 4 * np.pi * r**2 * dr
    activity = annular_vol * WATER_DENSITY * RADON_ACTIVITY * GAMMA_2447_BRANCHING_RATIO
    differential_contrib = activity * efficiencies
    cumulative_contrib = np.cumsum(differential_contrib)
    cumulative_percentage = (cumulative_contrib / cumulative_contrib[-1]) * 100

    fig, ax1 = plt.subplots()

    ax1.plot(r, efficiencies, "b-", label="Hit Efficiency")
    ax1.set_xlabel("Radius (m)")
    ax1.set_ylabel("Efficiency (counts/decay)", color="b")
    ax1.tick_params("y", colors="b")

    ax2 = ax1.twinx()
    ax2.plot(r, cumulative_percentage, "r-", label="Cumulative Contribution")
    ax2.axhline(y=90, color="k", linestyle=":", label="90% threshold")
    ax2.set_ylabel("Cumulative Contribution (%)", color="r")
    ax2.tick_params("y", colors="r")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="center right")

    plt.title("Radial Dependence of Background (with 50% solid angle)")
    plt.show()

    r90_idx = np.searchsorted(cumulative_percentage, 90)
    r90_distance = distances[r90_idx]
    print(f"\n90% of contribution occurs within {r90_distance:.2f} m of vessel surface")


def main():
    results = calculate_background_rate()
    print("=== nEXO Background Calculation (with 50% solid angle) ===")
    print(f"Water Mass: {results['water_mass']:.2e} kg")
    print(f"Avg Hit Efficiency: {results['avg_hit_efficiency']:.2e} counts/decay")
    print(f"Total Activity: {results['total_activity']:.2e} Bq")
    print(f"Background Rate: {results['background_rate']:.2e} counts/s")
    print(f"Annual Background: {results['background_rate']*86400*365:.2e} counts/year")

    create_background_heatmap(z_slice=0)
    plot_radial_dependence()


if __name__ == "__main__":
    main()
