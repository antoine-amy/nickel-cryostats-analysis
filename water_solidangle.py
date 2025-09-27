#!/usr/bin/env python3
"""
Geometric acceptance G(r) for nEXO: spherical shell around a cylindrical TPC.

Definition:
  For a point P on a sphere of radius r centered at C_s (shell center),
  consider only inward-going directions (toward the TPC center C_t).
  G_point = fraction of those directions that intersect the finite cylinder
            (TPC) of radius a and half-height H centered at C_t, axis z.
  G(r) = average of G_point over uniformly distributed points on the shell.

Notes:
  - This computes *pure geometry* (no attenuation).
  - "Inward hemisphere" normalization (2π sr), matching the derivation you used.
  - Supports small mis-centering: d = C_t - C_s (default 0).

Defaults (nEXO-ish):
  a = 0.56665 m            # TPC radius ~ 56.665 cm
  H = 0.5915 m             # TPC half-height ~ 59.15 cm
"""

import numpy as np
import matplotlib.pyplot as plt

# ----------------------------- Ray / Cylinder Intersections -----------------------------
def intersect_cylinder_finite(p, v, a, H):
    """
    Ray-cylinder intersection against a finite cylinder aligned with z-axis,
    centered at origin, radius a, z in [-H, +H].
    Ray: p + t v, with t > 0.

    Returns True if the ray hits the lateral surface or either cap.
    """
    px, py, pz = p
    vx, vy, vz = v

    hits = False
    t_candidates = []

    # --- Lateral surface: x^2 + y^2 = a^2
    A = vx*vx + vy*vy
    B = 2.0 * (px*vx + py*vy)
    C = px*px + py*py - a*a

    if A > 0.0:
        disc = B*B - 4.0*A*C
        if disc >= 0.0:
            sqrt_disc = np.sqrt(disc)
            t1 = (-B - sqrt_disc) / (2.0*A)
            t2 = (-B + sqrt_disc) / (2.0*A)
            if t1 > 1e-12: t_candidates.append(t1)
            if t2 > 1e-12: t_candidates.append(t2)

    # Filter lateral t by z-slab
    for t in t_candidates:
        z = pz + t * vz
        if -H <= z <= H:
            return True  # lateral hit valid

    # --- Caps (top/bottom disks): z = ±H
    if abs(vz) > 1e-15:
        # Top cap z = +H
        t_top = ( H - pz) / vz
        if t_top > 1e-12:
            x = px + t_top * vx
            y = py + t_top * vy
            if x*x + y*y <= a*a:
                return True
        # Bottom cap z = -H
        t_bot = (-H - pz) / vz
        if t_bot > 1e-12:
            x = px + t_bot * vx
            y = py + t_bot * vy
            if x*x + y*y <= a*a:
                return True

    return hits

# ----------------------------- Sampling Helpers -----------------------------
def sample_points_on_sphere(n, r, rng):
    """Uniform points on sphere radius r centered at origin."""
    # Method: normal vector normalized
    x = rng.normal(size=(n, 3))
    x /= np.linalg.norm(x, axis=1)[:, None]
    return r * x  # shape (n,3)

def local_basis_inward(axis):
    """
    Build an orthonormal basis (e1, e2, ez) with ez = normalized 'axis' (inward).
    """
    ez = axis / np.linalg.norm(axis)
    # Choose a helper vector not parallel to ez
    helper = np.array([0.0, 0.0, 1.0]) if abs(ez[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, ez);  n1 = np.linalg.norm(e1)
    if n1 < 1e-15:  # fallback if degenerate
        helper = np.array([1.0, 0.0, 0.0])
        e1 = np.cross(helper, ez); n1 = np.linalg.norm(e1)
    e1 /= n1
    e2 = np.cross(ez, e1)
    return e1, e2, ez

def sample_inward_hemisphere_dirs(n, e1, e2, ez, rng):
    """
    Sample 'n' directions uniformly on the inward hemisphere around ez.
    Construction: cos(psi) ~ U[0,1], phi ~ U[0,2π)
    v = sinψ cosφ e1 + sinψ sinφ e2 + cosψ ez
    """
    u = rng.random(n)              # cosψ
    cos_psi = u
    sin_psi = np.sqrt(1.0 - u*u)
    phi = 2.0 * np.pi * rng.random(n)
    c, s = np.cos(phi), np.sin(phi)
    # Combine components
    v = (sin_psi * c)[:,None]*e1[None,:] + (sin_psi * s)[:,None]*e2[None,:] + (cos_psi)[:,None]*ez[None,:]
    return v  # shape (n,3)

# ----------------------------- Core Estimator -----------------------------
def geometric_acceptance_r(r, a, H, n_pts=200, n_dirs=4000, offset=(0.0,0.0,0.0), rng=None):
    """
    Estimate G(r): shell-averaged inward-hemisphere geometric acceptance at radius r.

    Params
    ------
    r      : shell radius (m)
    a, H   : TPC radius (m), half-height (m)
    n_pts  : number of shell points sampled
    n_dirs : number of inward directions per point
    offset : vector (dx,dy,dz) = TPC_center - shell_center (m)
    rng    : numpy Generator

    Returns
    -------
    G_mean : mean acceptance over shell
    G_sem  : standard error on the mean
    """
    if rng is None:
        rng = np.random.default_rng()
    offset = np.asarray(offset, dtype=float)

    if r <= 0.0:
        raise ValueError("Shell radius r must be > 0.")

    # Sample shell points centered at shell origin (0,0,0); TPC is at 'offset'
    P = sample_points_on_sphere(n_pts, r, rng)  # (n_pts,3)

    acc = np.zeros(n_pts, dtype=float)

    for i in range(n_pts):
        p = P[i]
        # Inward axis points from P to TPC center
        axis = offset - p
        e1, e2, ez = local_basis_inward(axis)

        # Sample inward-hemisphere directions
        V = sample_inward_hemisphere_dirs(n_dirs, e1, e2, ez, rng)  # (n_dirs,3)

        # Shift to TPC-centered coords for intersection
        p_tpc = p - offset

        # Test intersections; vectorized in chunks to limit memory
        hits = 0
        chunk = 8000
        for j in range(0, n_dirs, chunk):
            v = V[j:j+chunk]
            # Loop (kept simple; could be numba-accelerated if needed)
            for k in range(v.shape[0]):
                if intersect_cylinder_finite(p_tpc, v[k], a, H):
                    hits += 1

        acc[i] = hits / float(n_dirs)

    G_mean = acc.mean()
    G_sem  = acc.std(ddof=1) / np.sqrt(n_pts) if n_pts > 1 else 0.0
    return G_mean, G_sem

def geometric_acceptance_curve(radii, a, H, n_pts=200, n_dirs=4000, offset=(0,0,0), rng=None):
    """Compute G(r) and SEM for an array of radii."""
    if rng is None:
        rng = np.random.default_rng(12345)
    G = np.zeros_like(radii, dtype=float)
    E = np.zeros_like(radii, dtype=float)
    for i, r in enumerate(radii):
        G[i], E[i] = geometric_acceptance_r(r, a, H, n_pts=n_pts, n_dirs=n_dirs, offset=offset, rng=rng)
    return G, E

# ----------------------------- Demo / Plot -----------------------------
if __name__ == "__main__":
    # nEXO-ish TPC geometry
    a = 0.56665   # m, TPC radius
    H = 0.5915    # m, TPC half-height
    # Radii to scan (from just outside TPC to several meters)
    radii = np.linspace(0.8, 6.0, 18)  # meters

    # Sampling settings (tune for precision/speed)
    n_pts  = 180     # shell points per radius
    n_dirs = 4000    # directions per point
    rng = np.random.default_rng(2025)

    # Mis-centering (set to small values to test robustness)
    offset = (0.00, 0.00, 0.00)  # meters

    G, E = geometric_acceptance_curve(radii, a, H, n_pts=n_pts, n_dirs=n_dirs, offset=offset, rng=rng)

    # Normalize to an OV-like reference radius for easy comparison
    r_ref = 2.23  # m (example OV radius ~ 2.23 m)
    # Interpolate G at r_ref
    G_ref = np.interp(r_ref, radii, G)
    ratio = G / G_ref

    # Far-field ~ 1/r^2 scaling normalized at r_ref
    ratio_power2 = (r_ref / radii)**2

    # Plot
    fig, ax = plt.subplots()
    ax.errorbar(radii, ratio, yerr=E / G_ref, fmt="o", capsize=3, label="MC:  G(r)/G(r_ref)")
    ax.plot(radii, ratio_power2, linestyle="--", label="Far-field $(r_{ref}/r)^2$")

    ax.set_xlabel("Shell radius r (m)")
    ax.set_ylabel("Geometric acceptance ratio  $G(r)/G(r_{ref})$")
    ax.set_title("nEXO geometric acceptance from spherical shell to cylindrical TPC")
    ax.grid(True)
    ax.legend()
    plt.tight_layout()
    plt.show()

    # Print a small table
    print("   r [m]    G(r)       SEM       G/G_ref    (r_ref/r)^2")
    for r, g, e, q, p2 in zip(radii, G, E, ratio, ratio_power2):
        print(f"{r:7.3f}  {g:8.5e}  {e:8.2e}   {q:8.5f}   {p2:8.5f}")