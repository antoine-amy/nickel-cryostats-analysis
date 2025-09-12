#!/usr/bin/env python3
"""
Water-background hit efficiency with cylindrical tank geometry.

- Tank: cylinder with diameter 12.3 m (radius 6.15 m) and height 12.8 m.
- Vessel: sphere of diameter 4.46 m (radius 2.23 m) centered in the tank.

What this script does:
1) Plots a radial hit-efficiency profile in the equatorial plane (z = 0)
   with 1σ shaded bands (log Y). This is a visual slice only.
2) Computes MEAN hit efficiencies (±1σ) using full cylindrical integration
   over (r,z) with volume weight 2π r Δr Δz, excluding the spherical vessel.
   The same geom(s) and attenuation(s) formulas are used as in the original.

Geometry/physics model (same as spherical version, but volume is cylindrical):
- Distance from vessel center: s = sqrt(r^2 + z^2)
- Geometry factor to a spherical vessel of radius Rv:
    geom(s) = 1 - sqrt( max(1 - (Rv/s)^2, 0) ),  for s >= Rv, else 0
- Attenuation in water along inward path: attn(s) = exp(-μρ (s - Rv)), for s >= Rv
- Hit efficiency at a point: eff(s) = EFF0 * geom(s) * attn(s)

Notes:
- The 1σ on EFF0 is derived as in your original code (assuming 1e10 parents):
    sigma = sqrt(EFF0 * 1e10 * branching) / 1e10
- The “90% thickness” is computed from the cylindrical volume by sorting cells
  by path length (s - Rv) and finding the distance enclosing 90% of counts.
"""

import numpy as np
import matplotlib.pyplot as plt

# ---------------- Constants ----------------
TANK_DIAM = 12.3          # m
TANK_HEIGHT = 12.8        # m
VESSEL_DIAM = 4.46        # m
RHO_WATER = 1000          # kg/m^3
MU_MASS = 0.01            # m^2/kg  (mass attenuation coefficient)
EFF0_U = 2e-10            # counts/ROI/2 t/decay
EFF0_TH = 1.2579e-9       # counts/ROI/2 t/decay
BRANCHING_U = 1.0
BRANCHING_TH = 0.3594

# Derived geometry
R_CYL = TANK_DIAM / 2.0   # m
H_HALF = TANK_HEIGHT / 2.0
R_V = VESSEL_DIAM / 2.0

# ---------------- Slice (for plotting only) ----------------
def equatorial_slice(n_pts=200):
    """Return r (midplane) and efficiencies/1σ along z=0 for plotting."""
    r = np.linspace(R_V, R_CYL, n_pts)        # equatorial line, z=0
    s = r                                     # at z=0, s = r
    geom = 1.0 - np.sqrt(np.clip(1.0 - (R_V / s) ** 2, 0.0, 1.0))
    dist = s - R_V
    attn = np.exp(-MU_MASS * RHO_WATER * dist)

    eff_u = EFF0_U * geom * attn
    eff_th = EFF0_TH * geom * attn

    # 1σ on EFF0 (same assumption as original)
    sigma_u = np.sqrt(EFF0_U * 1e10 * BRANCHING_U) / 1e10
    sigma_th = np.sqrt(EFF0_TH * 1e10 * BRANCHING_TH) / 1e10

    err_u = sigma_u * geom * attn
    err_th = sigma_th * geom * attn
    return r, eff_u, eff_th, err_u, err_th

# ---------------- Cylindrical integration (for MEANS) ----------------
def cylindrical_mean(n_r=400, n_z=400):
    """
    Compute mean attenuation <geom*attn> over cylindrical water volume,
    excluding the spherical vessel.
    """
    # Grid (cell centers)
    r_edges = np.linspace(0.0, R_CYL, n_r + 1)
    z_edges = np.linspace(-H_HALF, H_HALF, n_z + 1)
    r = 0.5 * (r_edges[:-1] + r_edges[1:])   # (n_r,)
    z = 0.5 * (z_edges[:-1] + z_edges[1:])   # (n_z,)
    dr = r_edges[1:] - r_edges[:-1]
    dz = z_edges[1:] - z_edges[:-1]

    # 2D grids
    R, Z = np.meshgrid(r, z, indexing='xy')         # shape (n_z, n_r)
    dA = (2.0 * np.pi) * R * dr[np.newaxis, :] * dz[:, np.newaxis]  # volume weight

    s = np.sqrt(R**2 + Z**2)
    mask = s >= R_V  # exclude interior of vessel

    geom = np.zeros_like(s)
    with np.errstate(invalid='ignore', divide='ignore'):
        geom[mask] = 1.0 - np.sqrt(np.clip(1.0 - (R_V / s[mask])**2, 0.0, 1.0))

    dist = np.zeros_like(s)
    dist[mask] = s[mask] - R_V
    attn = np.ones_like(s)
    attn[mask] = np.exp(-MU_MASS * RHO_WATER * dist[mask])

    g_times_a = geom * attn

    # Mean <geom*attn> over water volume
    num = np.sum(dA[mask] * g_times_a[mask])
    den = np.sum(dA[mask])
    mean_attn = num / den

    # For 90% thickness: flatten contributions vs. distance from surface
    contrib = (dA * g_times_a)
    dflat = dist[mask].ravel()
    cflat = contrib[mask].ravel()
    order = np.argsort(dflat)
    d_sorted = dflat[order]
    c_sorted = cflat[order]
    c_cum = np.cumsum(c_sorted)
    c_tot = c_cum[-1]
    idx90 = np.searchsorted(c_cum, 0.9 * c_tot)
    thick90 = float(d_sorted[min(idx90, len(d_sorted) - 1)])

    return mean_attn, thick90

def main():
    # ---- Plot equatorial slice (z=0) ----
    r, eff_u, eff_th, err_u, err_th = equatorial_slice(n_pts=400)

    fig, ax = plt.subplots(figsize=(10, 7))
    lu, = ax.plot(r, eff_u, label='U-238 / Rn-222 (equatorial slice)')
    ax.fill_between(r, np.clip(eff_u - err_u, 1e-300, None), eff_u + err_u,
                    color=lu.get_color(), alpha=0.3)

    lt, = ax.plot(r, eff_th, label='Th-232 (equatorial slice)')
    ax.fill_between(r, np.clip(eff_th - err_th, 1e-300, None), eff_th + err_th,
                    color=lt.get_color(), alpha=0.3)

    ax.set_yscale('log')
    ax.set_xlabel('Radius in midplane r (m)  [z = 0 slice]')
    ax.set_ylabel('Hit efficiency (counts/ROI/2 t/decay)')
    ax.set_title('Water: equatorial radial hit-efficiency with 1σ bands (log Y)')
    ax.legend()
    ax.grid(True, which='both', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    # ---- Mean efficiencies from full cylindrical volume ----
    mean_attn, thick90 = cylindrical_mean(n_r=500, n_z=500)

    # 1σ on EFF0 (same assumption as original)
    sigma_u = np.sqrt(EFF0_U * 1e10 * BRANCHING_U) / 1e10
    sigma_th = np.sqrt(EFF0_TH * 1e10 * BRANCHING_TH) / 1e10

    mean_eff_u = EFF0_U * mean_attn
    mean_eff_th = EFF0_TH * mean_attn
    mean_err_u = sigma_u * mean_attn
    mean_err_th = sigma_th * mean_attn

    print(
        f"Mean hit efficiency (U-238 / Rn-222, cylindrical tank): "
        f"{mean_eff_u:.3e} ± {mean_err_u:.3e} counts/ROI/2 t/decay (1σ)"
    )
    print(
        f"Mean hit efficiency (Th-232, cylindrical tank): "
        f"{mean_eff_th:.3e} ± {mean_err_th:.3e} counts/ROI/2 t/decay (1σ)"
    )
    print(
        f"90% of counts originate within {thick90:.2f} m "
        f"({thick90*1000:.0f} mm) of the vessel surface (cylindrical volume)."
    )

if __name__ == "__main__":
    main()