#!/usr/bin/env python3
"""
Compute mean transmission ⟨T⟩, mean attenuation ⟨A⟩, and mean HFE thickness
using the piecewise integral form only, and visualize:

  1) Trapezoidal integration of T(θ)·sinθ (scaled ×10³)
  2) Combined plot of r(θ), Δr(θ) and R_IC marker

All figures use figsize=(10, 7).
"""

import numpy as np
import matplotlib.pyplot as plt

# ---- Geometry & material parameters (mm, mm⁻¹) ----
A    = 638.5    # cylinder radius
H    = 1277.0   # cylinder full height
R_IC = 1691.0   # enclosing inner cryostat radius
MU   = 0.007    # attenuation coefficient for HFE-7200 (2.5 MeV gamma)

# ---- θ grid (avoid singular endpoints) ----
EPS   = 1e-6
N     = 20000
theta = np.linspace(EPS, np.pi - EPS, N)

# ---- Intersection r(θ) and Δr(θ) ----
r_side   = A / np.sin(theta)
r_cap    = (H/2) / np.abs(np.cos(theta))
r_theta  = np.minimum(r_side, r_cap)
delta_r  = R_IC - r_theta

# Print min and max of delta_r
print(f"\nMin Δr(θ): {np.min(delta_r):.2f} mm")
print(f"Max Δr(θ): {np.max(delta_r):.2f} mm\n")

# ---- Transmission T(θ) and integrand f(θ) ----
T_theta = np.exp(-MU * delta_r)
f_theta = T_theta * np.sin(theta)

# ---- Piecewise method for mean values ----
theta_c      = np.arctan2(2*A, H)
mask_cap1    = theta < theta_c
mask_side    = (theta >= theta_c) & (theta <= np.pi - theta_c)

# cap & side contributions to transmission
delta_cap1 = R_IC - (H/2)/np.abs(np.cos(theta[mask_cap1]))
I_cap1     = np.trapezoid(np.exp(-MU * delta_cap1) * np.sin(theta[mask_cap1]),
                         x=theta[mask_cap1])

delta_side = R_IC - A/np.sin(theta[mask_side])
I_side     = np.trapezoid(np.exp(-MU * delta_side) * np.sin(theta[mask_side]),
                         x=theta[mask_side])

I_piece            = I_cap1 + 0.5*I_side
mean_transmission  = float(I_piece)
mean_attenuation   = 1.0 - mean_transmission

# cap & side contributions to thickness
Ith_cap1 = np.trapezoid(delta_cap1 * np.sin(theta[mask_cap1]),
                        x=theta[mask_cap1])
Ith_side = np.trapezoid(delta_side * np.sin(theta[mask_side]),
                        x=theta[mask_side])

mean_thickness = float(Ith_cap1 + 0.5*Ith_side)

# ---- Effective thickness from mean transmission ----
#   exp(-MU * δ_eff) = ⟨T⟩  =>  δ_eff = -ln(⟨T⟩)/MU
effective_thickness = -np.log(mean_transmission) / MU

# ---- Print results ----
print(f"{'⟨T⟩':>10s} {'⟨A⟩':>12s} {'⟨Δr⟩ (mm)':>14s} {'δ_eff (mm)':>14s}")
print("-" * 60)
print(f"{mean_transmission:10.6f} {mean_attenuation:12.6f} {mean_thickness:14.2f} {effective_thickness:14.2f}")

# ---- Plot 1: Trapezoidal integration of f(θ) ----
plt.figure(figsize=(10, 7))
plt.plot(theta, f_theta * 1e3,
         label=r'Trapezoidal Integration $(\times10^3)$',
         linewidth=2)
plt.fill_between(theta, f_theta * 1e3, alpha=0.3)
plt.xlabel(r'$\theta$ (rad)')
plt.ylabel(r'Transmission $(\times10^3)$')
plt.grid(True)
plt.legend()
plt.show()

# ---- Plot 2: Combined r(θ), Δr(θ) and R_IC ----
plt.figure(figsize=(10, 7))

# r(θ)
plt.plot(theta, r_theta,
         label=r'$r(\theta)$',
         color='orange', linewidth=2)

# Δr(θ)
plt.plot(theta, delta_r,
         label=r'$\Delta r(\theta)$',
         color='cornflowerblue', linewidth=2)

# R_IC horizontal marker
plt.axhline(R_IC,
            linestyle='--', color='gray',
            label=r'$R_{\mathrm{IC}}$')

# critical angle marker
plt.axvline(theta_c,
            linestyle='-.', color='black',
            label=r'$\theta_c = \arctan\frac{2A}{H}$')

plt.xlabel(r'$\theta$ (rad)')
plt.ylabel('Distance (mm)')
plt.grid(True)
plt.legend(loc='upper right')
plt.show()
