"""
Background illustration script for cryostat analysis.

This module creates visualizations of background distributions
with truncated Gaussian spread calculations.
"""

import math
import numpy as np
import matplotlib.pyplot as plt

# Parameters
MASS = 5.54e3                  # kg
ACTIVITY = 0.0                 # Bq/kg
ACTIVITY_ERR = 7.43e-7         # Bq/kg
EFFICIENCY = 2.00e-10          # counts per decay
EFFICIENCY_ERR = math.sqrt(EFFICIENCY / 1e10)  # simulation stat error
SECONDS_PER_YEAR = 86400 * 365.25

# Compute u (counts/sec)
u_sec = math.hypot(MASS * EFFICIENCY * ACTIVITY_ERR,
                   MASS * ACTIVITY   * EFFICIENCY_ERR)
u_year = u_sec * SECONDS_PER_YEAR

# Compute TG-spread mean and spread (counts/sec)
T = MASS * ACTIVITY * EFFICIENCY
z = T / u_sec if u_sec > 0 else 0
pdf = math.exp(-0.5 * z*z) / math.sqrt(2*math.pi)
cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))
mean_sec = (T + u_sec * (pdf / cdf)) if cdf > 0 else 0
term_b = 1 + math.erf(z / math.sqrt(2))
a_over_b = (math.exp(-0.5*z*z) / term_b) if term_b > 0 else 0
inner = 1 - (z * a_over_b) / math.sqrt(8*math.pi) - (a_over_b**2) / (8*math.pi)
spread_sec = u_sec * math.sqrt(inner) if inner > 0 else 0

# Convert to counts/year
bg_mean = mean_sec * SECONDS_PER_YEAR
bg_spread = spread_sec * SECONDS_PER_YEAR

# Prepare half-normal PDF for plotting
x = np.linspace(0, 4 * u_year, 500)
pdf_year = (np.sqrt(2) / (u_year * np.sqrt(np.pi))) * np.exp(-0.5 * (x / u_year)**2)

# Plot
plt.figure()
plt.plot(x, pdf_year, label='Truncated background PDF')
plt.fill_between(x, pdf_year, where=[xi <= bg_mean for xi in x], alpha=0.3)
plt.axvline(bg_mean, linestyle='--', label=f'Mean = {bg_mean:.2e}')
plt.axvline(u_year, linestyle=':', label=f'u = {u_year:.2e}')
plt.axvline(bg_mean + bg_spread, linestyle='-.', label=f'Mean + spread = {bg_mean + bg_spread:.2e}')

# Annotate spread
plt.annotate(
    'spread',
    xy=(bg_mean + bg_spread, max(pdf_year)*0.6),
    xytext=(bg_mean, max(pdf_year)*0.6),
    arrowprops={'arrowstyle': '->'}
)

plt.xlabel('Background (counts/year/2t/ROI)')
plt.ylabel('Probability Density')
plt.title('Background Distribution with u, Mean, and Spread')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
