# =============================================================================
# 90% UL to 1σ Gaussian Approximation. Values from U-238 radioactivity measurements from Nickel entry R-207-1-1-1-1.
# =============================================================================
import numpy as np
import matplotlib.pyplot as plt

# Given values
ul = 1.22e-3       # 90% UL in mBq/kg
sigma = ul / 1.644854  # Convert UL to 1σ using z0.95 ≈ 1.645

# x-axis range: from -3σ to 1.5×UL
x = np.linspace(-3*sigma, 1.5*ul, 1000)
# Gaussian PDF centered at 0 with width sigma
pdf = np.exp(-0.5 * (x / sigma)**2) / (sigma * np.sqrt(2 * np.pi))

plt.figure()
plt.plot(x, pdf, label='Gaussian PDF (μ=0, σ)')
# Vertical lines for UL and sigma
plt.axvline(ul, linestyle='--', label=f'90% UL = {ul:.2e} mBq/kg')
plt.axvline(sigma, linestyle=':', label=f'σ = {sigma:.2e} mBq/kg')

# Annotation showing conversion
plt.annotate(
    r'$\sigma = \frac{\rm UL}{1.645}$',
    xy=(sigma, plt.ylim()[1]*0.7),
    xytext=(ul, plt.ylim()[1]*0.7),
    arrowprops={'arrowstyle': '->'}
)

plt.xlabel('Activity (mBq/kg)')
plt.ylabel('Probability Density')
plt.title('From 90% UL to 0 ± 1σ Gaussian Approximation')
plt.legend()
plt.tight_layout()
plt.show()