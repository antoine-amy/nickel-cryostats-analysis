#!/usr/bin/env python3
"""
Plot water‐background contribution versus HFE shielding thickness,
including error bands from uncertainties in μ and B₀.
"""

import numpy as np
import matplotlib.pyplot as plt


def main():
    """Compute and plot background vs. shielding thickness."""
    # Constants
    MU = 0.00685         # attenuation coefficient (1/mm)
    MU_ERR = 0.0005      # uncertainty in MU
    R0 = 1691.0          # reference radius (mm)
    B0 = 5.450e-05       # central background (counts/year)
    B0_ERR = 4.390e-05   # uncertainty on B0

    # Shielding thickness array (mm)
    r = np.linspace(1000.0, 1700.0, 100)
    delta = r - R0

    # Central and μ‐error curves
    br = B0 * np.exp(-MU * delta)
    br_up_mu = B0 * np.exp(-(MU - MU_ERR) * delta)
    br_dn_mu = B0 * np.exp(-(MU + MU_ERR) * delta)

    # B0 error envelopes at central μ
    br_up_b0 = (B0 + B0_ERR) * np.exp(-MU * delta)
    br_dn_b0 = (B0 - B0_ERR) * np.exp(-MU * delta)

    # Total envelope combining both sources of uncertainty
    upper = np.maximum(br_up_mu, br_up_b0)
    lower = np.minimum(br_dn_mu, br_dn_b0)

    # Monte Carlo data points
    mc_r = [1026.0, 1691.0]  # mm
    mc_bg = [6.14e-6, 1.23e-6]  # counts/year
    mc_err = [1.22e-2, 4.98e-4]  # counts/year

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(r, br, label='Theoretical B(r)')
    ax.fill_between(r, lower, upper, alpha=0.3,
                    label='Theoretical uncertainty')
    ax.errorbar(mc_r, mc_bg, yerr=mc_err, fmt='o', color='red', 
                capsize=5, label='Monte Carlo data')

    ax.set_xlabel('Shielding thickness r (mm)')
    ax.set_ylabel('Background (counts/year)')
    ax.set_title('Background vs. HFE Shielding Thickness')
    ax.set_yscale('log')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
