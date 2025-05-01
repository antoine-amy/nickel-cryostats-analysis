import numpy as np
import matplotlib.pyplot as plt


def sample_tpc_surface(tpc_radius, tpc_height, n=100000):
    """
    Monte Carlo sample points on a cylindrical TPC.
    Returns distances from center for each surface point.
    """
    # Surface areas for weighting
    A_side = 2 * np.pi * tpc_radius * tpc_height
    A_caps = 2 * np.pi * tpc_radius**2
    # Sample counts
    n_side = int(n * A_side / (A_side + A_caps))
    n_caps = n - n_side
    # Side sampling
    θ = np.random.rand(n_side) * 2 * np.pi
    z_side = (np.random.rand(n_side) - 0.5) * tpc_height
    x_side = tpc_radius * np.cos(θ)
    y_side = tpc_radius * np.sin(θ)
    # Caps sampling
    r = tpc_radius * np.sqrt(np.random.rand(n_caps))
    θ2 = np.random.rand(n_caps) * 2 * np.pi
    x_cap = r * np.cos(θ2)
    y_cap = r * np.sin(θ2)
    z_cap = np.concatenate([
        np.full(n_caps//2, tpc_height/2),
        np.full(n_caps - n_caps//2, -tpc_height/2)
    ])
    # Combine
    x = np.concatenate([x_side, x_cap])
    y = np.concatenate([y_side, y_cap])
    z = np.concatenate([z_side, z_cap])
    return np.sqrt(x**2 + y**2 + z**2)


def effective_hfe_thickness(dist_center, cryo_radius, mu):
    """
    Compute effective and mean HFE thickness.
    Returns (t_eff, mean, t_min, all_thicknesses).
    """
    d_hfe = cryo_radius - dist_center
    atten = np.exp(-mu * d_hfe)
    t_eff = -np.log(np.mean(atten)) / mu
    return t_eff, np.mean(d_hfe), np.min(d_hfe), d_hfe


def vessel_hfe_mass(tpc_radius, tpc_height, extra_thick, density, shape='cyl'):
    """
    Compute HFE mass for cylindrical or spherical vessel.
    shape: 'cyl' or 'sph'
    extra_thick: thickness (cyl) or cryo_radius (sph).
    """
    if shape == 'sph':
        R = extra_thick
        V_hfe = (4/3)*np.pi*R**3 - np.pi*tpc_radius**2*tpc_height
    else:
        R = tpc_radius + extra_thick
        H = tpc_height + 2 * extra_thick
        V_hfe = np.pi*(R**2 * H - tpc_radius**2 * tpc_height)
    mass = (V_hfe / 1e9) * density  # mm³→m³ then ×density
    return V_hfe, mass


if __name__ == '__main__':
    # Parameters
    R_tpc, H_tpc = 567.0, 1183.0  # mm
    R_cryo = 1685.0  # mm
    rho_hfe = 1718.3  # kg/m³
    mu_gamma = 0.007  # mm^-1

    np.random.seed(42)
    dist_center = sample_tpc_surface(R_tpc, H_tpc, n=100000)
    t_eff, t_mean, t_min, d_hfe = effective_hfe_thickness(dist_center, R_cryo, mu_gamma)

    # Compute masses
    _, mass_sph = vessel_hfe_mass(R_tpc, H_tpc, R_cryo, rho_hfe, shape='sph')
    _, mass_cyl = vessel_hfe_mass(R_tpc, H_tpc, t_eff, rho_hfe, shape='cyl')

    # Print results
    print(f"Effective thickness: {t_eff:.1f} mm")
    print(f"Mean thickness:      {t_mean:.1f} mm")
    print(f"Minimum thickness:   {t_min:.1f} mm")
    print(f"HFE mass (sph):      {mass_sph/1000:.2f} t")
    print(f"HFE mass (cyl):      {mass_cyl/1000:.2f} t")

    # Plot distribution
    plt.figure(figsize=(8,4))
    plt.hist(d_hfe, bins=50)
    plt.axvline(t_mean, linestyle='--', label='Mean thickness')
    plt.axvline(t_eff, linestyle='-.', label='Effective thickness')
    plt.axvline(t_min, linestyle=':', label='Minimum thickness')
    plt.xlabel('HFE thickness (mm)')
    plt.ylabel('Sample count')
    plt.title('Monte Carlo HFE Thickness Distribution')
    plt.legend()
    plt.tight_layout()
    plt.show()
