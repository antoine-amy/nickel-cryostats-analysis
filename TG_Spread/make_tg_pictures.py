#!/usr/bin/env python3
"""
Generate illustrative figures for the TG-spread note.

Outputs:
  phi_z.png, Phi_z.png,
  lambda_demo_z1.png, lambda_demo_z2.png,
  lambda_log.png
"""
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      300,
    "axes.grid":       True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTDIR = pathlib.Path(".").resolve()

def savefig(figure, fname):
    """tight-layout + save + message"""
    figure.tight_layout()
    figure.savefig(OUTDIR / fname)
    plt.close(figure)
    print("  ✔", fname)

# 1.  φ(z)
z = np.linspace(-4, 4, 400)
fig1 = plt.figure()
plt.plot(z, norm.pdf(z))
plt.title(r"Standard Normal PDF  $\phi(z)$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\phi(z)$")
savefig(fig1, "phi_z.png")

# 2.  Φ(z)
fig2 = plt.figure()
plt.plot(z, norm.cdf(z))
plt.title(r"Standard Normal CDF  $\Phi(z)$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\Phi(z)$")
savefig(fig2, "Pphi_z.png")

# 3.  Mills ratio λ(z)  (log scale)  ── ★ title fixed ★
z_pos = np.linspace(0.01, 6.0, 500)
lam = norm.pdf(z_pos) / norm.cdf(z_pos)

fig3 = plt.figure()
plt.semilogy(z_pos, lam)
plt.title(r"Mills Ratio  $\lambda(z)=\phi(z)/\Phi(z)$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\lambda(z)$")
savefig(fig3, "lambda_log.png")

# 4 & 5.  Geometric λ demos
def lambda_demo(z0, fname):
    """Generate geometric lambda demonstration plot."""
    x = np.linspace(-4, 4, 400)
    pdf = norm.pdf(x)
    left = x <= z0

    demo_fig = plt.figure()
    plt.plot(x, pdf, label=r'$\phi(z)$')
    plt.fill_between(x[left], pdf[left], alpha=0.3,
                     label=rf'Area $\Phi({z0})$')
    plt.axvline(z0, ls='--', color='k', label=fr'$z_0={z0}$')

    fz   = norm.pdf(z0)
    lam0 = fz / norm.cdf(z0)
    plt.scatter([z0], [fz], zorder=3)
    plt.text(z0+0.1, fz, r'$\phi(z_0)$', va='bottom')
    plt.text(z0+0.1, fz*0.55,
             rf'$\lambda(z_0) = {lam0:.3f}$', va='top')

    plt.title(fr'Geometric interpretation of $\lambda(z)$ at $z_0={z0}$')
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\phi(z)$")
    plt.legend()
    savefig(demo_fig, fname)

lambda_demo(1.0, "lambda_demo_z1.png")
lambda_demo(2.0, "lambda_demo_z2.png")

print("\nAll TG-spread figures saved to:", OUTDIR)
