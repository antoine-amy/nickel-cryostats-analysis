"""
Module for analyzing Gaussian vs truncated half-normal distributions.

This module compares a Gaussian distribution with a truncated half-normal
distribution, calculating and visualizing their statistical properties.
"""

import numpy as np
import matplotlib.pyplot as plt

# Given value
SIGMA = 7.43e-4  # mBq/kg
MU = 0.0         # central value

# x-axis range: from negative to positive
x = np.linspace(-4*SIGMA, 5*SIGMA, 1000)

# Original Gaussian PDF
pdf = (1 / (SIGMA * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - MU) / SIGMA)**2)

# Truncated (half-normal) PDF: x >= 0, renormalize by dividing by 0.5
trunc_pdf = np.where(x >= 0, pdf / 0.5, 0)

# Compute truncated statistics
mean_trunc = float(np.trapezoid(x * trunc_pdf, x))
var_trunc = np.trapezoid((x - mean_trunc)**2 * trunc_pdf, x)
std_trunc = float(np.sqrt(var_trunc))

# Plot
plt.figure(figsize=(12, 5))
plt.plot(x, pdf, color='C0', label=f'Gaussian (μ=0, σ={SIGMA:.2e})')
plt.plot(x, trunc_pdf, color='C1',
         label=f'Truncated half-normal (μ={mean_trunc:.2e}, σ={std_trunc:.2e})',
         linewidth=1.5)

# Vertical lines: mu, sigma, truncated mean - use different dash styles for mu vs sigma
plt.axvline(MU, linestyle='--', color='C0', label='μ = 0')
plt.axvline(SIGMA, linestyle=':', color='C0', label=f'σ = {SIGMA:.2e}')
plt.axvline(mean_trunc, linestyle='--', color='C1', label=f'μ = {mean_trunc:.2e}')
plt.axvline(std_trunc, linestyle=':', color='C1', label=f'σ = {std_trunc:.2e}')

plt.xlabel('Activity (mBq/kg)')
plt.ylabel('Probability Density')
plt.title('Gaussian vs Truncated Half-Normal for 0 ± 7.43×10⁻⁴ mBq/kg')
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True)
plt.tight_layout()
plt.show()
