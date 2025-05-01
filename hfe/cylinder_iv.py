#!/usr/bin/env python3
"""
Monte Carlo sampling of TPC surface and HFE thickness/mass computations.
"""
import numpy as np

# Constants
SAMPLE_SIZE = 100000
R_TPC = 567.0        # mm
H_TPC = 1183.0       # mm
R_CRYO = 1685.0      # mm
RHO_HFE = 1718.3     # kg/m³
MU_GAMMA = 0.007     # mm⁻¹
SEED = 42


def sample_tpc_surface(radius, height, n=SAMPLE_SIZE):
    """Return distances from center for random points on TPC surfaces."""
    area_side = 2 * np.pi * radius * height
    area_caps = 2 * np.pi * radius**2
    n_side = int(n * area_side / (area_side + area_caps))
    n_caps = n - n_side

    # Side
    theta = np.random.rand(n_side) * 2 * np.pi
    z_side = (np.random.rand(n_side) - 0.5) * height
    x_side = radius * np.cos(theta)
    y_side = radius * np.sin(theta)

    # Caps
    r = radius * np.sqrt(np.random.rand(n_caps))
    theta2 = np.random.rand(n_caps) * 2 * np.pi
    x_cap = r * np.cos(theta2)
    y_cap = r * np.sin(theta2)
    z_cap = np.concatenate([
        np.full(n_caps // 2,  height / 2),
        np.full(n_caps - n_caps // 2, -height / 2)
    ])

    coords = np.vstack((
        np.concatenate((x_side, x_cap)),
        np.concatenate((y_side, y_cap)),
        np.concatenate((z_side, z_cap))
    ))
    return np.linalg.norm(coords, axis=0)


def effective_hfe_thickness(distances, cryo_radius, mu):
    """Compute effective, mean, and min HFE thickness from center distances."""
    d_hfe = cryo_radius - distances
    t_eff = -np.log(np.mean(np.exp(-mu * d_hfe))) / mu
    return t_eff, d_hfe.mean(), d_hfe.min(), d_hfe


def vessel_hfe_mass(radius, height, extra, density, shape='cyl'):
    """Compute HFE volume and mass for 'cyl' or 'sph' vessel shapes."""
    if shape == 'sph':
        R = extra
        V = 4/3 * np.pi * R**3 - np.pi * radius**2 * height
    else:
        R = radius + extra
        H = height + 2 * extra
        V = np.pi * (R**2 * H - radius**2 * height)
    mass = V * density / 1e9  # mm³ → m³
    return mass


if __name__ == '__main__':
    np.random.seed(SEED)
    dist = sample_tpc_surface(R_TPC, H_TPC)
    t_eff, t_mean, t_min, d_hfe = effective_hfe_thickness(dist, R_CRYO, MU_GAMMA)

    mass_sph = vessel_hfe_mass(R_TPC, H_TPC, R_CRYO, RHO_HFE, 'sph')
    mass_cyl = vessel_hfe_mass(R_TPC, H_TPC, t_eff,    RHO_HFE, 'cyl')

    print(f"Effective thickness: {t_eff:.1f} mm")
    print(f"Mean thickness:      {t_mean:.1f} mm")
    print(f"Minimum thickness:   {t_min:.1f} mm")
    print(f"HFE mass (sph):      {mass_sph/1000:.2f} t")
    print(f"HFE mass (cyl):      {mass_cyl/1000:.2f} t")
