"""Plot comparison of Th-232 and U-238 background means and errors 
for various cryostat configurations."""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# ─── Input data ────────────────────────────────────────────────────────────────
means = {
    '2022 Budget':           {'Th-232': 3.922e-9,   'U-238': 0.0},
    '2025 Budget':           {'Th-232': 9.72e-9,    'U-238': 9.00e-9},
    'Lip Part\n(02/25, no supports/tubings)': {'Th-232': 2.15e-9, 'U-238': 0.80e-9},
    'Lip Part\n(03/25)':     {'Th-232': 2.15e-9,    'U-238': 0.50e-9},
    'Lower/Lip/Upper\n(04/25)':              {'Th-232': 4.53e-9, 'U-238': 0.70e-9},
    'Lip/Lower/Upper\n(04/25)':              {'Th-232': 5.13942e-9, 'U-238': 0.80e-9},
    'Lip/Lower/Upper\n(04/25; 2024 recon. code)': {'Th-232': 2.1564e-9, 'U-238': 0.80e-9},
    'Lip/Lower/Upper\n(04/25, no radius, no kill external γ)': {
        'Th-232': 3.98934e-9, 'U-238': 0.30e-9
    },
    'Upper Part\n(04/25)':     {'Th-232': 3.34242e-9, 'U-238': 0.70e-9},
    'Lower Part\n(04/25)':     {'Th-232': 2.37204e-9, 'U-238': 0.00e-9},
    'Lip Part\n(04/25)':       {'Th-232': 2.04858e-9, 'U-238': 0.30e-9},
    'Lip/Lower/Upper\n(05/25, \"source/add\" fix)': {'Th-232': 2.91114e-9, 'U-238': 1.10000e-9},
}

# ─── Constants ─────────────────────────────────────────────────────────────────
NG = 1e10               # Number of generated events
SCALE = 1e9             # Scale efficiencies to ×10⁻⁹ for plotting
SHOW_ONLY_APRIL_2025 = False  # If True, only show April 2025 parts and budgets

# ─── Label grouping ───────────────────────────────────────────────────────────
budget_labels = [lbl for lbl in means if 'Budget' in lbl]
result_labels = [lbl for lbl in means if lbl not in budget_labels]

if SHOW_ONLY_APRIL_2025:
    april_2025_labels = [
        'Upper Part\n(04/25)',
        'Lower Part\n(04/25)',
        'Lip Part\n(04/25)',
        'Lip/Lower/Upper\n(05/25, \"source/add\" fix)',
    ]
    result_labels = [lbl for lbl in result_labels if lbl in april_2025_labels]

plot_labels = budget_labels + result_labels

# ─── Error computation ─────────────────────────────────────────────────────────
def compute_errors(means_dict: dict) -> dict:
    error_dict = {}
    for config_label, vals in means_dict.items():
        if 'Budget' in config_label:
            # Use fixed uncertainties for budgets
            error_dict[config_label] = {'Th-232': 1.869e-9, 'U-238': 3.00e-9}
        else:
            # Poisson-based errors; force 1/NG if N_sel = 0
            errs = {}
            for isotope, eff in vals.items():
                n_selected = eff * NG
                if n_selected > 0:
                    errs[isotope] = np.sqrt(n_selected) / NG
                else:
                    errs[isotope] = 1.0 / NG
            error_dict[config_label] = errs
    return error_dict

errors = compute_errors(means)

# ─── Color mapping ────────────────────────────────────────────────────────────
part_colors = {'Upper': 'red', 'Lower': 'green', 'Lip': 'blue', 'All': 'black'}

def pick_color(config_label: str) -> str:
    if ('Budget' in config_label or
        'Lip/Lower/Upper' in config_label or
        'Lower/Lip/Upper' in config_label):
        return part_colors['All']
    for part in ('Upper', 'Lower', 'Lip'):
        if part in config_label:
            return part_colors[part]
    return 'black'

color_map = {lbl: pick_color(lbl) for lbl in means}

# ─── Plotting ─────────────────────────────────────────────────────────────────
fig, (ax_th, ax_u) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)

for ax, iso in zip((ax_th, ax_u), ['Th-232', 'U-238']):
    for idx, label in enumerate(plot_labels):
        val = means[label][iso] * SCALE
        err = errors[label][iso] * SCALE
        clr = color_map[label]
        ax.errorbar(
            idx, val,
            yerr=err,
            fmt='o',
            markersize=6,
            markerfacecolor=clr,
            markeredgecolor=clr,
            ecolor=clr,
            linestyle='None'
        )
    ax.set_ylabel(f'{iso} (×10⁻⁹)')
    ax.grid(True, axis='y', linestyle='--', alpha=0.5)
    ax.set_ylim(0)

# ─── X-axis formatting ────────────────────────────────────────────────────────
ax_u.set_xticks(range(len(plot_labels)))
ax_u.set_xticklabels(plot_labels, rotation=45, ha='right', fontsize=8)
ax_u.set_xlabel('Configuration')

# ─── Legend ──────────────────────────────────────────────────────────────────
legend_elems = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=8, label=part)
    for part, color in part_colors.items()
]
fig.legend(handles=legend_elems, loc='upper right', bbox_to_anchor=(0.98, 0.98),
           fontsize='small', title='Part type')

plt.tight_layout(pad=1.0, h_pad=0.5)
plt.show()
