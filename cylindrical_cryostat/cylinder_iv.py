#!/usr/bin/env python3
"""
Compute HFE metrics for:
  1) Baseline spherical cryostat
  2) Cylindrical cryostat with Monte Carlo–derived effective thickness
  3) Cylindrical cryostat with fixed 760 mm thickness

Print a simple summary without extra libraries.
"""
import numpy as np
import matplotlib.pyplot as plt

# ── Constants ────────────────────────────────────────────────────────────────
SEED = 45
SAMPLE_SIZE = 100_000
R_TPC = 638.5    # mm
H_TPC = 1277.0   # mm
R_CRYO = 1691.0   # mm
RHO_HFE = 1_730.3  # kg/m³
MU_GAMMA = 0.007    # mm⁻¹
EXTRA_FIXED = 760.0 # mm
EXTRA_FIXED_500 = 500.0 # mm

# ── Monte Carlo sampling of TPC surface points ───────────────────────────────
def sample_tpc_surface(radius, height, n=SAMPLE_SIZE, shape='cyl'):
    if shape == 'cyl':
        area_side = 2*np.pi*radius*height
        area_caps = 2*np.pi*radius**2
        n_side = int(n*area_side/(area_side+area_caps))
        n_caps = n - n_side

        # side
        theta = np.random.rand(n_side)*2*np.pi
        z_side = (np.random.rand(n_side)-0.5)*height
        x_side = radius*np.cos(theta)
        y_side = radius*np.sin(theta)

        # caps
        r = radius*np.sqrt(np.random.rand(n_caps))
        theta2 = np.random.rand(n_caps)*2*np.pi
        x_cap = r*np.cos(theta2)
        y_cap = r*np.sin(theta2)
        z_cap = np.concatenate([
            np.full(n_caps//2, height/2),
            np.full(n_caps - n_caps//2, -height/2)
        ])

        coords = np.vstack((
            np.concatenate((x_side, x_cap)),
            np.concatenate((y_side, y_cap)),
            np.concatenate((z_side, z_cap))
        ))
        return coords

    # spherical case
    # Generate points on a sphere using spherical coordinates
    phi = np.arccos(2*np.random.rand(n) - 1)  # polar angle
    theta = 2*np.pi*np.random.rand(n)  # azimuthal angle

    x = radius * np.sin(phi) * np.cos(theta)
    y = radius * np.sin(phi) * np.sin(theta)
    z = radius * np.cos(phi)

    return np.vstack((x, y, z))


def plot_sampling_points(coords, radius, height):
    plt.subplots(figsize=(10, 10))

    # Calculate r coordinates
    r = np.sqrt(coords[0]**2 + coords[1]**2)
    z = coords[2]

    # Separate side and cap points
    epsilon = 1e-3
    side_mask = np.abs(np.abs(z) - height/2) > epsilon
    cap_mask = ~side_mask

    # Plot TPC surface points with different colors
    plt.scatter(r[side_mask], z[side_mask], c='blue', alpha=0.3, s=10, label='Side points')
    plt.scatter(r[cap_mask], z[cap_mask], c='red', alpha=0.3, s=10, label='Cap points')

    # Plot TPC outline
    # Side
    plt.plot([radius, radius], [-height/2, height/2], 'k-', linewidth=2, label='TPC surface')
    # Top cap
    plt.plot([0, radius], [height/2, height/2], 'k-', linewidth=2)
    # Bottom cap
    plt.plot([0, radius], [-height/2, -height/2], 'k-', linewidth=2)
    # Radial line
    plt.plot([0, radius], [0, 0], 'k-', linewidth=2)

    # Plot vessel outline (760mm away from TPC)
    vessel_radius = radius + 760
    vessel_height = height + 2*760
    # Side
    plt.plot([vessel_radius, vessel_radius], [-vessel_height/2, vessel_height/2],
             'g--', linewidth=2, label='Vessel surface')
    # Top cap
    plt.plot([0, vessel_radius], [vessel_height/2, vessel_height/2], 'g--', linewidth=2)
    # Bottom cap
    plt.plot([0, vessel_radius], [-vessel_height/2, -vessel_height/2], 'g--', linewidth=2)

    # Add corner thickness annotation
    corner_thickness = np.sqrt(2) * 760
    plt.annotate(f'Corner thickness: {corner_thickness:.1f} mm',
                xy=(radius, height/2),
                xytext=(radius + 200, height/2 + 200),
                arrowprops={"facecolor": 'black', "shrink": 0.05})

    # Add point count annotations
    plt.annotate(f'Side points: {np.sum(side_mask)}',
                xy=(radius/2, 0),
                xytext=(radius/2, height/4))
    plt.annotate(f'Cap points: {np.sum(cap_mask)}',
                xy=(radius/2, height/2),
                xytext=(radius/2, height/2 + height/4))

    plt.xlabel('Radial position (mm)')
    plt.ylabel('Axial position (mm)')
    plt.title('Monte Carlo Sampling Points on TPC Surface (r-z plane)')
    plt.legend()
    plt.grid(True)
    plt.gca().set_aspect('equal')
    plt.show()


def calculate_cylindrical_thickness(radius, height, min_thickness):
    coords = sample_tpc_surface(radius, height)
    x = coords[0]
    y = coords[1]
    z = coords[2]
    r = np.sqrt(x**2 + y**2)
    epsilon = 1e-3
    side_mask = np.abs(np.abs(z) - height/2) > epsilon
    cap_mask = ~side_mask

    # Debug prints
    print("\nDebug information:")
    print(f"Total points: {len(z)}")
    print(f"Side points: {np.sum(side_mask)}")
    print(f"Cap points: {np.sum(cap_mask)}")
    print(f"Min radius: {np.min(r):.2f} mm")
    print(f"Max radius: {np.max(r):.2f} mm")
    print(f"Expected radius: {radius:.2f} mm")

    thickness = np.zeros_like(z)
    thickness[side_mask] = min_thickness

    # For cap points, thickness increases from min_thickness at r=0 to
    # sqrt(2)*min_thickness at r=radius
    # This is because at the corner (r=radius), we have both radial and axial thickness
    thickness[cap_mask] = np.sqrt(min_thickness**2 +
                                 (min_thickness * (r[cap_mask]/radius))**2)

    # More debug prints
    print(f"Min thickness: {np.min(thickness):.2f} mm")
    print(f"Max thickness: {np.max(thickness):.2f} mm")
    print(f"Mean thickness: {np.mean(thickness):.2f} mm")

    # Create visualization
    plot_sampling_points(coords, radius, height)

    return thickness.mean(), thickness.min(), thickness.max()


# ── Effective HFE thickness from MC distances ────────────────────────────────
def effective_hfe_thickness(coords, cryo_radius, mu, shape='cyl'):
    if shape == 'sph':
        # For spherical case, calculate distance from center to surface point
        distances = np.linalg.norm(coords, axis=0)
        # Thickness is cryo_radius - distances
        d_hfe = cryo_radius - distances
        t_eff = -np.log(np.mean(np.exp(-mu*d_hfe))) / mu
        return t_eff, d_hfe.mean(), d_hfe.min()

    # For cylindrical case, use the original calculation
    distances = np.linalg.norm(coords, axis=0)
    d_hfe = cryo_radius - distances
    t_eff = -np.log(np.mean(np.exp(-mu*d_hfe))) / mu
    return t_eff, d_hfe.mean(), d_hfe.min()


# ── Volume & mass for spherical or cylindrical vessels ───────────────────────
def vessel_hfe_mass(radius, height, extra, density, shape='cyl'):
    if shape == 'sph':
        radius_vessel = extra
        volume = 4/3*np.pi*radius_vessel**3 - np.pi*radius**2*height
    else:
        radius_vessel = radius + extra
        height_vessel = height + 2*extra
        volume = np.pi*(radius_vessel**2*height_vessel - radius**2*height)
    mass = volume * density / 1e9  # mm³→m³→kg
    return volume, mass


def calculate_transmission(thickness, mu):
    return np.exp(-mu * thickness) * 100  # Convert to percentage


# ── Main ────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    np.random.seed(SEED)

    # Calculate equivalent spherical radius for TPC
    v_tpc = np.pi * R_TPC**2 * H_TPC
    r_tpc_sph = (3*v_tpc/(4*np.pi))**(1/3)  # radius of sphere with same volume

    # Baseline spherical
    v_sph, mass_sph = vessel_hfe_mass(R_TPC, H_TPC, R_CRYO, RHO_HFE, 'sph')
    coords_sph = sample_tpc_surface(r_tpc_sph, r_tpc_sph, shape='sph')  # Use spherical radius
    t_eff_sph, t_mean_sph, t_min_sph = effective_hfe_thickness(coords_sph, R_CRYO, MU_GAMMA, 'sph')
    trans_sph = calculate_transmission(t_mean_sph, MU_GAMMA)

    # Cylindrical with MC-derived effective thickness
    coords_cyl = sample_tpc_surface(R_TPC, H_TPC, shape='cyl')
    t_eff_cyl, t_mean, t_min = effective_hfe_thickness(coords_cyl, R_CRYO, MU_GAMMA, 'cyl')
    v_cyl_eff, mass_cyl_eff = vessel_hfe_mass(R_TPC, H_TPC, t_eff_cyl, RHO_HFE, 'cyl')
    trans_cyl_eff = calculate_transmission(t_mean, MU_GAMMA)

    # Cylindrical with fixed 760 mm
    v_cyl_fix, mass_cyl_fix = vessel_hfe_mass(R_TPC, H_TPC, EXTRA_FIXED, RHO_HFE, 'cyl')
    t_mean_fix, t_min_fix, t_max_fix = calculate_cylindrical_thickness(R_TPC, H_TPC, EXTRA_FIXED)
    trans_cyl_fix = calculate_transmission(t_mean_fix, MU_GAMMA)

    # Cylindrical with fixed 500 mm
    v_cyl_fix_500, mass_cyl_fix_500 = vessel_hfe_mass(R_TPC, H_TPC, EXTRA_FIXED_500, RHO_HFE, 'cyl')
    t_mean_fix_500, t_min_fix_500, t_max_fix_500 = calculate_cylindrical_thickness(R_TPC, H_TPC, EXTRA_FIXED_500)
    trans_cyl_fix_500 = calculate_transmission(t_mean_fix_500, MU_GAMMA)

    # Print summary
    print("\nHFE Summary\n" + "-"*60)
    print("1) Baseline (spherical):")
    print(f"   Eff. thickness    = {t_eff_sph:.1f} mm")
    print(f"   Mean thickness    = {t_mean_sph:.1f} mm")
    print(f"   Min thickness     = {t_min_sph:.1f} mm")
    print(f"   Transmission      = {trans_sph:.2f}%")
    print(f"   HFE volume = {v_sph/1e9:.3f} m³")
    print(f"   HFE mass   = {mass_sph/1000:.2f} t\n")

    print("2) Cylindrical (effective thickness):")
    print(f"   Inner radius      = {R_TPC + t_eff_cyl:.1f} mm")
    print(f"   Inner height      = {H_TPC + 2*t_eff_cyl:.1f} mm")
    print(f"   Eff. thickness    = {t_eff_cyl:.1f} mm")
    print(f"   Mean thickness    = {t_mean:.1f} mm")
    print(f"   Min thickness     = {t_min:.1f} mm")
    print(f"   Transmission      = {trans_cyl_eff:.2f}%")
    print(f"   HFE volume        = {v_cyl_eff/1e9:.3f} m³")
    print(f"   HFE mass          = {mass_cyl_eff/1000:.2f} t\n")

    print("3) Cylindrical (fixed 760 mm):")
    print(f"   Inner radius      = {R_TPC + EXTRA_FIXED:.1f} mm")
    print(f"   Inner height      = {H_TPC + 2*EXTRA_FIXED:.1f} mm")
    print(f"   Min thickness     = {t_min_fix:.1f} mm")
    print(f"   Mean thickness    = {t_mean_fix:.1f} mm")
    print(f"   Max thickness     = {t_max_fix:.1f} mm")
    print(f"   Transmission      = {trans_cyl_fix:.2f}%")
    print(f"   HFE volume        = {v_cyl_fix/1e9:.3f} m³")
    print(f"   HFE mass          = {mass_cyl_fix/1000:.2f} t\n")

    print("4) Cylindrical (fixed 500 mm):")
    print(f"   Inner radius      = {R_TPC + EXTRA_FIXED_500:.1f} mm")
    print(f"   Inner height      = {H_TPC + 2*EXTRA_FIXED_500:.1f} mm")
    print(f"   Min thickness     = {t_min_fix_500:.1f} mm")
    print(f"   Mean thickness    = {t_mean_fix_500:.1f} mm")
    print(f"   Max thickness     = {t_max_fix_500:.1f} mm")
    print(f"   Transmission      = {trans_cyl_fix_500:.2f}%")
    print(f"   HFE volume        = {v_cyl_fix_500/1e9:.3f} m³")
    print(f"   HFE mass          = {mass_cyl_fix_500/1000:.2f} t\n")
