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
    B0 = 7.391e-05       # central background (counts/year)
    B0_ERR = 5.858e-05   # uncertainty on B0

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

    # Plotting
    fig, ax = plt.subplots()
    ax.plot(r, br, color='C0', label='Central B(r)')
    ax.fill_between(r, br_dn_mu, br_up_mu, color='C0', alpha=0.3,
                    label='μ uncertainty')
    ax.fill_between(r, br_dn_b0, br_up_b0, color='C1', alpha=0.3,
                    label='B₀ uncertainty')
    ax.fill_between(r, lower, upper, color='gray', alpha=0.2,
                    label='Total uncertainty')

    ax.set_xlabel('Shielding thickness r (mm)')
    ax.set_ylabel('Background (counts/year)')
    ax.set_title('Background vs. HFE Shielding Thickness')
    ax.legend(loc='upper right')
    ax.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    main()
