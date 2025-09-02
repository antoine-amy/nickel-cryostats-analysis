#!/usr/bin/env python3
"""
Analysis of U-238/Rn-222 and Th-232 water background:
radial hit-efficiency profile with shaded error bands and log-scaled y-axis.
"""

import numpy as np
import matplotlib.pyplot as plt

# Constants
TANK_SIZE = 12.3           # m
VESSEL_DIAM = 4.46         # m
RHO_WATER = 1000           # kg/m³
MU_MASS = 0.01             # m²/kg
EFF0_U = 2e-10             # counts/ROI/2 t/decay
EFF0_TH = 1.2579e-9        # counts/ROI/2 t/decay
BRANCHING_U = 1.0
BRANCHING_TH = 0.3594

# Derived constant
VESSEL_R = VESSEL_DIAM / 2  # m


def main():
    """Compute and plot radial hit-efficiency profiles with error bands."""
    n_pts = 200
    radii = np.linspace(VESSEL_R, TANK_SIZE / 2, n_pts)

    # geometry factor Ω/2π = 1 − sqrt(1 − (R/r)²)
    ratio = VESSEL_R / radii
    inside = np.clip(1.0 - ratio**2, 0.0, 1.0)
    geom = 1.0 - np.sqrt(inside)

    # attenuation factor
    dist = radii - VESSEL_R
    attn = np.exp(-MU_MASS * RHO_WATER * dist)

    # central hit efficiencies
    eff_u = EFF0_U * geom * attn
    eff_th = EFF0_TH * geom * attn

    # error on EFF0
    sigma_u = np.sqrt(EFF0_U * 1e10 * BRANCHING_U) / 1e10
    sigma_th = np.sqrt(EFF0_TH * 1e10 * BRANCHING_TH) / 1e10

    # propagated error bands
    err_u = sigma_u * geom * attn
    err_th = sigma_th * geom * attn

    # plot
    fig, ax = plt.subplots(figsize=(10, 7))
    line_u, = ax.plot(radii, eff_u, label='U-238 / Rn-222')
    ax.fill_between(radii,
                    eff_u - err_u,
                    eff_u + err_u,
                    color=line_u.get_color(),
                    alpha=0.3)

    line_th, = ax.plot(radii, eff_th, label='Th-232')
    ax.fill_between(radii,
                    eff_th - err_th,
                    eff_th + err_th,
                    color=line_th.get_color(),
                    alpha=0.3)

    ax.set_yscale('log')
    ax.set_xlabel('Radius (m)')
    ax.set_ylabel('Hit efficiency (counts/ROI/2 t/decay)')
    ax.set_title('Radial hit-efficiency with shaded error bands')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # 3D spherical integration for mean efficiency and 90% thickness
    dr = radii[1] - radii[0]
    shell_volumes = 4 * np.pi * radii**2 * dr
    counts_shell = shell_volumes * geom * attn

    mean_attn = np.sum(counts_shell) / np.sum(shell_volumes)
    mean_eff_u = EFF0_U * mean_attn
    mean_eff_th = EFF0_TH * mean_attn

    print(
        f'Mean hit efficiency (U-238 / Rn-222): '
        f'{mean_eff_u:.2e} counts/ROI/2 t/decay'
    )
    print(
        f'Mean hit efficiency (Th-232): '
        f'{mean_eff_th:.2e} counts/ROI/2 t/decay'
    )

    cum = np.cumsum(counts_shell)
    perc = cum / cum[-1] * 100.0
    idx_90 = np.searchsorted(perc, 90.0)
    thick_90 = dist[idx_90]

    print(
        f'90% of counts originate within {thick_90:.2f} m '
        f'({thick_90*1000:.0f} mm) of vessel surface'
    )


if __name__ == '__main__':
    main()
