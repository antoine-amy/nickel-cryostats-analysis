"""Plot and fit IV hit efficiencies versus inner-vessel radius.

Includes uncertainty bars and an exponential attenuation fit to June 2025 data
for Th232 and U238, plus a combined coefficient.
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Branching ratios (still used for error estimates)
BR = {'Th232': 0.3594, 'U238': 1.0}

# Data dictionaries as before...
th232_info = {
    'Jan. 2025 Results': {
        1026: {'eff': 2.5158e-07, 'ng': 1e9},
        1226: {'eff': 6.00198e-08, 'ng': 1e9},
        1510: {'eff': 6.1098e-09, 'ng': 1e9},
        1691: {'eff': 2.1564e-09, 'ng': 1e9},
    },
    'Feb. 2025 Results': {
        1026: {'eff': 2.6300892e-07, 'ng': 1e10},
        1226: {'eff': 6.591396e-08, 'ng': 1e10},
        1510: {'eff': 8.94906e-09,  'ng': 1e10},
        1691: {'eff': 2.19234e-09,  'ng': 1e10},
    },
    'Jun. 2025 Results': {
        1026: {'eff': 2.914734e-07, 'ng': 1e9},
        1226: {'eff': 6.70281e-08,  'ng': 1e10},
        1510: {'eff': 9.20064e-09,  'ng': 1e10},
        1691: {'eff': 2.62362e-09,  'ng': 1e10},
    },
    'Jun. 2025 Results (original geometry only)': {
        1691: {'eff': 2.19234e-09,  'ng': 1e10},
    },
    'MDB_2022': {
        1691: {'eff': 3.922e-09,    'ng': 1e9},
    },
    'MDB_2025': {
        1691: {'eff': 9.72e-09,     'ng': 1e9},
    },
}

u238_info = {
    'Jan. 2025 Results': {
        1026: {'eff': 1.7e-07, 'ng': 1e9},
        1226: {'eff': 3.3e-08, 'ng': 1e9},
        1510: {'eff': 1.0e-09, 'ng': 1e9},
        1691: {'eff': 3e-09,   'ng': 1e9},
    },
    'Feb. 2025 Results': {
        1026: {'eff': 1.65e-07, 'ng': 1e10},
        1226: {'eff': 2.87e-08, 'ng': 1e10},
        1510: {'eff': 3.1e-09,  'ng': 1e10},
        1691: {'eff': 8e-10,    'ng': 1e10},
    },
    'Jun. 2025 Results': {
        1026: {'eff': 1.73e-07, 'ng': 1e9},
        1226: {'eff': 3.12e-08, 'ng': 1e10},
        1510: {'eff': 2.7e-09,  'ng': 1e10},
        1691: {'eff': 1e-09,    'ng': 1e10},
    },
    'Jun. 2025 Results (original geometry only)': {
        1691: {'eff': 1e-09,    'ng': 1e10},
    },
    'MDB_2022': {
        1691: {'eff': 0.0,      'ng': 1e9},
    },
    'MDB_2025': {
        1691: {'eff': 9e-09,    'ng': 1e9},
    },
}

def compute_error(eff, ng, br):
    """Return Poisson-based uncertainty on counts = eff * ng * br."""
    return np.sqrt(eff * br / ng) if eff > 0 else 1.0 / ng

def plot_with_errors(ax, data_info, nuc_label):
    """Plot efficiency points with error bars on the given axes."""
    for label, points in data_info.items():
        xs = sorted(points.keys())
        ys = [points[x]['eff'] for x in xs]
        errs = [compute_error(points[x]['eff'], points[x]['ng'], BR[nuc_label]) for x in xs]
        fmt = 'x' if len(xs) == 1 else 'o-'
        ax.errorbar(xs, ys, yerr=errs, fmt=fmt, capsize=3, markersize=5, label=label)
    ax.set_xlim(1000, 1800)
    ax.set_yscale('log')
    ax.set_xlabel('Inner Vessel Radius (mm)')
    ax.set_ylabel(f'Hit Efficiency ({nuc_label})')
    ax.set_title(f'{nuc_label} Efficiencies vs Radius')
    ax.legend(fontsize='small')
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)

# Attenuation model: I(r) = I0 * exp(-mu * r)
def attenuation_model(r, i0, mu):
    """Exponential attenuation model I(r) = I0 * exp(-mu * r)."""
    return i0 * np.exp(-mu * r)

# ——— Plotting ———
fig, axs = plt.subplots(1, 2, figsize=(16, 5), sharey=True)
plot_with_errors(axs[0], th232_info, 'Th232')
plot_with_errors(axs[1], u238_info, 'U238')
plt.tight_layout()
plt.show()

# ——— Fit Jun. 2025 data and print μ with error ———
def fit_attenuation(data_info, nuc_label):
    """Fit attenuation model to a dataset and print mu ± sigma_mu."""
    pts = data_info['Jun. 2025 Results']
    xs = np.array(sorted(pts.keys()))
    ys = np.array([pts[x]['eff'] for x in xs])
    errs = np.array([compute_error(pts[x]['eff'], pts[x]['ng'], BR[nuc_label]) for x in xs])
    # initial guess: I0≈first point, μ≈1e-3
    p0 = [ys[0], 1e-3]
    result = curve_fit(attenuation_model, xs, ys,
                       sigma=errs, absolute_sigma=True, p0=p0)
    popt, pcov = result[0], result[1]
    _i0, mu = popt
    mu_err = np.sqrt(pcov[1, 1])
    print(f"{nuc_label} attenuation μ = {mu:.3e} ± {mu_err:.3e} mm⁻¹")
    return mu, mu_err

mu_th, mu_th_err = fit_attenuation(th232_info, 'Th232')
mu_u, mu_u_err   = fit_attenuation(u238_info,  'U238')

# ——— Combined coefficient (1/3 Th + 2/3 U) ———
mu_comb    = (1/3)*mu_th + (2/3)*mu_u
mu_comb_err = np.sqrt((1/3*mu_th_err)**2 + (2/3*mu_u_err)**2)
print(f"Combined μ = {mu_comb:.3e} ± {mu_comb_err:.3e} mm⁻¹")
