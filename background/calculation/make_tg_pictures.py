#!/usr/bin/env python3
"""
Generate illustrative figures for the TG-spread note.

Outputs:
  phi_z.png, Phi_z.png,
  lambda_demo_z1.png, lambda_demo_z2.png,
  lambda_log.png
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import pathlib

# ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi":      300,
    "axes.grid":       True,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

OUTDIR = pathlib.Path(".").resolve()

def savefig(fig, fname):
    """tight-layout + save + message"""
    fig.tight_layout()
    fig.savefig(OUTDIR / fname)
    plt.close(fig)
    print("  ✔", fname)

# 1.  φ(z)
z = np.linspace(-4, 4, 400)
fig = plt.figure()
plt.plot(z, norm.pdf(z))
plt.title(r"Standard Normal PDF  $\phi(z)$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\phi(z)$")
savefig(fig, "phi_z.png")

# 2.  Φ(z)
fig = plt.figure()
plt.plot(z, norm.cdf(z))
plt.title(r"Standard Normal CDF  $\Phi(z)$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\Phi(z)$")
savefig(fig, "Pphi_z.png")

# 3.  Mills ratio λ(z)  (log scale)  ── ★ title fixed ★
z_pos = np.linspace(0.01, 6.0, 500)
lam   = norm.pdf(z_pos) / norm.sf(z_pos)

fig = plt.figure()
plt.semilogy(z_pos, lam)
plt.title(r"Mills Ratio  $\lambda(z)=\phi(z)/(1-\Phi(z))$")
plt.xlabel(r"$z$")
plt.ylabel(r"$\lambda(z)$")
savefig(fig, "lambda_log.png")

# 4 & 5.  Geometric λ demos
def lambda_demo(z0, fname):
    x = np.linspace(-4, 4, 400)
    pdf = norm.pdf(x)
    tail = x >= z0

    fig = plt.figure()
    plt.plot(x, pdf, label=r'$\phi(z)$')
    plt.fill_between(x[tail], pdf[tail], alpha=0.3,
                     label=r'Tail $1-\Phi(z_0)$')
    plt.axvline(z0, ls='--', color='k', label=fr'$z_0={z0}$')

    fz = norm.pdf(z0)
    lam0 = fz / (1 - norm.cdf(z0))
    plt.scatter([z0], [fz], zorder=3)
    plt.text(z0+0.1, fz, r'$\phi(z_0)$', va='bottom')
    plt.text(z0+0.1, fz*0.55,
             rf'$\lambda(z_0) = {lam0:.2f}$', va='top')

    plt.title(fr'Geometric interpretation of $\lambda(z)$ at $z_0={z0}$')
    plt.xlabel(r"$z$")
    plt.ylabel(r"$\phi(z)$")
    plt.legend()
    savefig(fig, fname)

lambda_demo(1.0, "lambda_demo_z1.png")
lambda_demo(2.0, "lambda_demo_z2.png")

print("\nAll TG-spread figures saved to:", OUTDIR)