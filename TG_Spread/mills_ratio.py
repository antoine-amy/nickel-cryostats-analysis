"""
Mills ratio visualization and demonstration.

This module provides functions to visualize the Mills ratio and its
geometric interpretation for truncated Gaussian distributions.
"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# helper function to shade tail and annotate lambda for a given z0
def plot_lambda_demo(z0):
    """Plot lambda demonstration for given z0."""
    x = np.linspace(-4, 4, 400)
    pdf = norm.pdf(x)

    plt.figure()
    plt.plot(x, pdf, label=r'$\phi(z)$')
    # shade right tail beyond z0
    tail_mask = x >= z0
    plt.fill_between(x[tail_mask], pdf[tail_mask], alpha=0.3, label=r'Tail prob. $\;1-\Phi(z_0)$')
    plt.axvline(z0, linestyle='--', label=fr'$z_0={z0}$')
    # annotate density value at z0
    fz = norm.pdf(z0)
    plt.scatter([z0], [fz], s=60)
    plt.text(z0+0.1, fz, r'$\phi(z_0)$', va='bottom')
    # compute lambda
    lambda_z = fz / (1 - norm.cdf(z0))
    plt.text(z0+0.1, fz*0.5, fr'$\lambda = {lambda_z:.2f}$', va='top')
    plt.title(rf'Geometric meaning of $\lambda(z)$ at $z_0={z0}$')
    plt.xlabel('z')
    plt.ylabel(r'$\phi(z)$')
    plt.grid(True)
    plt.legend()

# Plot for z0 = 1
plot_lambda_demo(1)

# Plot for z0 = 2
plot_lambda_demo(2)

# Comprehensive lambda curve (positive domain) with log y-axis
z_pos = np.linspace(0.01, 6, 500)
lam = norm.pdf(z_pos) / norm.sf(z_pos)  # sf = 1 - cdf

plt.figure()
plt.semilogy(z_pos, lam)
plt.title(r'Mills Ratio $\lambda(z)=\phi(z)/[1-\Phi(z)]$ (log‑scale)')
plt.xlabel('z (positive domain)')
plt.ylabel(r'$\lambda(z)$')
plt.grid(True)

# Plot of mean shift factor u*lambda vs z (assuming u=1 for illustration)
plt.figure()
mean_shift = lam  # u=1
plt.plot(z_pos, mean_shift)
plt.title(r'Mean shift factor $u\,\lambda(z)$ (here $u=1$)')
plt.xlabel('z')
plt.ylabel(r'$\lambda(z)$')
plt.grid(True)
plt.show()
