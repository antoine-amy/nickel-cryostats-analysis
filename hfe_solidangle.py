#!/usr/bin/env python3
"""
MC geometric acceptance for spherical shell -> finite cylinder,
plus analytic fits. Adds a near-field corrected fit:
  G(r) = A/r^2 + B/r^3   (recommended)
optionally:
  G(r) = A/r^2 + B/r^3 + C/r^4
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Geometry (nEXO-ish drift dimensions)
# -----------------------------
a = 0.56665  # m
h = 0.5915   # m

# Radii to scan (m)
radii = np.linspace(0.95, 1.70, 22)

# MC sampling controls
N_POS = 600_000   # <-- (you had 6000_000 which is 6 million; keep if you can afford it)
SEED = 2025
USE_INWARD_HEMISPHERE = True

# ----------------------------- Vectorized ray-cylinder intersection -----------------------------
def rays_hit_finite_cylinder(P: np.ndarray, V: np.ndarray, a: float, h: float, eps: float = 1e-14) -> np.ndarray:
    px, py, pz = P[:, 0], P[:, 1], P[:, 2]
    vx, vy, vz = V[:, 0], V[:, 1], V[:, 2]
    hit = np.zeros(P.shape[0], dtype=bool)

    A = vx*vx + vy*vy
    B = 2.0*(px*vx + py*vy)
    C = px*px + py*py - a*a

    side_mask = A > eps
    if np.any(side_mask):
        As = A[side_mask]; Bs = B[side_mask]; Cs = C[side_mask]
        disc = Bs*Bs - 4.0*As*Cs
        disc_mask = disc >= 0.0
        idx_side = np.where(side_mask)[0]
        idx = idx_side[disc_mask]
        if idx.size:
            sqrt_disc = np.sqrt(np.maximum(disc[disc_mask], 0.0))
            As2 = A[idx]; Bs2 = B[idx]
            t1 = (-Bs2 - sqrt_disc) / (2.0*As2)
            t2 = (-Bs2 + sqrt_disc) / (2.0*As2)

            t = np.where((t1 > 0.0) & (t2 > 0.0), np.minimum(t1, t2),
                         np.where(t1 > 0.0, t1,
                                  np.where(t2 > 0.0, t2, np.inf)))
            z_at = pz[idx] + t * vz[idx]
            hit[idx] |= (t < np.inf) & (z_at >= -h) & (z_at <= +h)

    cap_mask = np.abs(vz) > eps
    if np.any(cap_mask):
        idx = np.where(cap_mask)[0]
        vz2 = vz[idx]
        t_top = (h - pz[idx]) / vz2
        x_top = px[idx] + t_top * vx[idx]
        y_top = py[idx] + t_top * vy[idx]
        top_hit = (t_top > 0.0) & (x_top*x_top + y_top*y_top <= a*a)

        t_bot = (-h - pz[idx]) / vz2
        x_bot = px[idx] + t_bot * vx[idx]
        y_bot = py[idx] + t_bot * vy[idx]
        bot_hit = (t_bot > 0.0) & (x_bot*x_bot + y_bot*y_bot <= a*a)
        hit[idx] |= (top_hit | bot_hit)

    return hit

# ----------------------------- Sampling helpers -----------------------------
def sample_points_on_sphere(n: int, r: float, rng: np.random.Generator) -> np.ndarray:
    X = rng.normal(size=(n, 3))
    X /= np.linalg.norm(X, axis=1)[:, None]
    return r * X

def sample_dirs_full_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(-1.0, 1.0, size=n)
    phi = rng.uniform(0.0, 2.0*np.pi, size=n)
    s = np.sqrt(np.maximum(0.0, 1.0 - u*u))
    return np.column_stack([s*np.cos(phi), s*np.sin(phi), u])

def sample_dirs_inward_hemisphere(P: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    N = P.shape[0]
    ez = -P / np.linalg.norm(P, axis=1)[:, None]
    helper = np.zeros_like(ez)
    use_z = np.abs(ez[:, 2]) < 0.9
    helper[use_z] = np.array([0.0, 0.0, 1.0])
    helper[~use_z] = np.array([1.0, 0.0, 0.0])
    e1 = np.cross(helper, ez)
    e1 /= np.linalg.norm(e1, axis=1)[:, None]
    e2 = np.cross(ez, e1)

    cos_psi = rng.random(N)
    sin_psi = np.sqrt(np.maximum(0.0, 1.0 - cos_psi*cos_psi))
    phi = rng.uniform(0.0, 2.0*np.pi, size=N)
    c, s = np.cos(phi), np.sin(phi)
    return (sin_psi*c)[:, None]*e1 + (sin_psi*s)[:, None]*e2 + (cos_psi)[:, None]*ez

# ----------------------------- MC estimator -----------------------------
def mc_G_inward_or_full(r: float, a: float, h: float, rng: np.random.Generator,
                        n_pos: int = 100_000, inward: bool = True) -> float:
    P = sample_points_on_sphere(n_pos, r, rng)
    V = sample_dirs_inward_hemisphere(P, rng) if inward else sample_dirs_full_sphere(n_pos, rng)
    hit = rays_hit_finite_cylinder(P, V, a=a, h=h)
    return float(np.count_nonzero(hit)) / float(n_pos)

# ----------------------------- Candidate analytic models -----------------------------
def G_model_k_a2_over_r2(r: np.ndarray, k: float, a: float) -> np.ndarray:
    return k * (a / r)**2

def G_model_k1k2_over_r2(r: np.ndarray
                         , k1: float, k2: float, a: float, h: float) -> np.ndarray:
    return (k1*a*a + k2*a*h) / (r*r)

def G_model_r2_r3(r: np.ndarray, A: float, B: float) -> np.ndarray:
    """Near-field corrected: G(r) = A/r^2 + B/r^3"""
    return A/(r*r) + B/(r*r*r)

def G_model_r2_r3_r4(r: np.ndarray, A: float, B: float, C: float) -> np.ndarray:
    return A/(r*r) + B/(r*r*r) + C/(r*r*r*r)

def fit_linear_in_basis(r: np.ndarray, y: np.ndarray, basis_cols: list[np.ndarray]) -> np.ndarray:
    """Least squares fit y ≈ sum_i beta_i * basis_i"""
    A = np.column_stack(basis_cols)
    beta, *_ = np.linalg.lstsq(A, y, rcond=None)
    return beta

def fit_r2_r3(r: np.ndarray, G: np.ndarray) -> tuple[float, float]:
    beta = fit_linear_in_basis(r, G, [1.0/(r*r), 1.0/(r*r*r)])
    return float(beta[0]), float(beta[1])

def fit_r2_r3_r4(r: np.ndarray, G: np.ndarray) -> tuple[float, float, float]:
    beta = fit_linear_in_basis(r, G, [1.0/(r*r), 1.0/(r*r*r), 1.0/(r*r*r*r)])
    return float(beta[0]), float(beta[1]), float(beta[2])

def fit_k_for_a2_over_r2(r: np.ndarray, G: np.ndarray, a: float, rmin_fit: float = 1.6) -> float:
    m = r >= rmin_fit
    x = (a / r[m])**2
    y = G[m]
    return float(np.dot(x, y) / np.dot(x, x))

def fit_k1k2_for_over_r2(r: np.ndarray, G: np.ndarray, a: float, h: float, rmin_fit: float = 1.6) -> tuple[float, float]:
    m = r >= rmin_fit
    X1 = (a*a) / (r[m]**2)
    X2 = (a*h) / (r[m]**2)
    Y = G[m]
    beta = fit_linear_in_basis(r[m], Y, [X1, X2])
    return float(beta[0]), float(beta[1])

# ----------------------------- Main -----------------------------
if __name__ == "__main__":
    rng = np.random.default_rng(SEED)

    # --- MC curve
    G_mc = np.zeros_like(radii, dtype=float)
    for i, r in enumerate(radii):
        G_mc[i] = mc_G_inward_or_full(r, a, h, rng, n_pos=N_POS, inward=USE_INWARD_HEMISPHERE)

    norm_label = "G = Ω/(2π) (inward hemi)" if USE_INWARD_HEMISPHERE else "f = Ω/(4π) (full sphere)"

    # Existing fits
    k_best = fit_k_for_a2_over_r2(radii, G_mc, a=a, rmin_fit=1.6)
    G_k_best = G_model_k_a2_over_r2(radii, k=k_best, a=a)

    k1_best, k2_best = fit_k1k2_for_over_r2(radii, G_mc, a=a, h=h, rmin_fit=1.6)
    G_k1k2_best = G_model_k1k2_over_r2(radii, k1_best, k2_best, a=a, h=h)

    G_proj_theory = (0.5*a*a + a*h) / (2.0 * radii*radii)

    # --- NEW: better near-field fit over full range (0.95–1.70 m)
    A23, B23 = fit_r2_r3(radii, G_mc)
    G_r2r3 = G_model_r2_r3(radii, A23, B23)

    # Optional: 3-parameter version
    A234, B234, C234 = fit_r2_r3_r4(radii, G_mc)
    G_r2r3r4 = G_model_r2_r3_r4(radii, A234, B234, C234)

    # --- Plots
    fig, ax = plt.subplots()
    ax.plot(radii, G_mc, "o", label=f"MC ({norm_label})")
    ax.plot(radii, G_k_best, "-", lw=2.0, label=f"best k*(a/r)^2, k={k_best:.3g} (fit r>=1.6m)")
    ax.plot(radii, G_k1k2_best, "-", lw=2.0, label=f"best (k1 a^2 + k2 a h)/r^2 (fit r>=1.6m)")
    ax.plot(radii, G_proj_theory, ":", lw=2.5, label="proj-area theory")
    ax.plot(radii, G_r2r3, "-", lw=2.5, label=r"NEW fit: $A/r^2 + B/r^3$")
    ax.plot(radii, G_r2r3r4, "--", lw=2.0, label=r"optional: $A/r^2+B/r^3+C/r^4$")
    ax.set_xlabel("Shell radius r [m]")
    ax.set_ylabel("Geometric acceptance G(r)")
    ax.set_title("MC vs analytic fits (including near-field correction)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()

    # Ratio plot
    fig2, ax2 = plt.subplots()
    ax2.plot(radii, np.ones_like(radii), "k-", lw=1.0, label="MC/MC")
    ax2.plot(radii, G_k_best / G_mc, lw=2.0, label="k*(a/r)^2 / MC")
    ax2.plot(radii, G_k1k2_best / G_mc, lw=2.0, label="2-term / MC")
    ax2.plot(radii, G_proj_theory / G_mc, ":", lw=2.5, label="proj-theory / MC")
    ax2.plot(radii, G_r2r3 / G_mc, lw=2.5, label=r"NEW: $(A/r^2+B/r^3)$/MC")
    ax2.plot(radii, G_r2r3r4 / G_mc, "--", lw=2.0, label=r"optional 3-par / MC")
    ax2.set_xlabel("Shell radius r [m]")
    ax2.set_ylabel("Analytic / MC")
    ax2.set_title("Diagnostic ratio")
    ax2.set_ylim(0.9, 1.1)
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9, ncol=2)
    plt.tight_layout()
    plt.show()

    # Print coefficients for use in your analytic model
    print(f"Normalization: {norm_label}")
    print(f"Far-field k (fit r>=1.6m): k = {k_best:.6g}")
    print(f"Far-field k1,k2 (fit r>=1.6m): k1 = {k1_best:.6g}, k2 = {k2_best:.6g}")
    print("\nNEW near-field fit over full range:")
    print(f"  G(r) = A/r^2 + B/r^3")
    print(f"  A = {A23:.8g}   [m^2]")
    print(f"  B = {B23:.8g}   [m^3]")
    print("\nOptional 3-par fit:")
    print(f"  G(r) = A/r^2 + B/r^3 + C/r^4")
    print(f"  A = {A234:.8g} [m^2],  B = {B234:.8g} [m^3],  C = {C234:.8g} [m^4]")