#!/usr/bin/env python3
"""
Inner-cryostat vessel thickness and mass scaling with reduced HFE radius.

Shows t_IV(R_IV) and m_IV(R_IV) as functions of HFE mass on a shared x-axis
consistent with the other _only background plots.

Scaling laws (ASME membrane stress / buckling, t ∝ R at fixed pressure):
    t_IV(R) = t_IV^0 * (R / R_IV^0)
    m_IV(R) = m_IV^0 * (R / R_IV^0)^3    [spherical shell: m ∝ R^2 * t ∝ R^3]

Baseline values:
    R_IV^0 = 1691 mm  (outer radius of the inner vessel)
    t_IV^0 = 5 mm
    m_IV^0 = 1681.6 kg
"""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from hfe_volume_to_iv_radius import (
    BASELINE_TARGET_MASS_KG_1,
    calibration_loss_tonnes_for_target,
    hfe_mass_tonnes,
)

CALIBRATION_LOSS_TONNES = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_1)


def hfe_mass_tonnes_from_radius_mm(radius_mm):
    """Convert IV radius (mm) to HFE mass (tonnes)."""
    radius_m = np.asarray(radius_mm, float) / 1000.0
    return hfe_mass_tonnes(radius_m, CALIBRATION_LOSS_TONNES)


# --- Baseline vessel parameters ---
R_IV0_MM  = 1691.0    # outer radius at baseline, mm
T_IV0_MM  = 5.0       # wall thickness at baseline, mm
M_IV0_KG  = 1681.6    # vessel mass at baseline, kg

# --- Grid (same as _only background plots) ---
R_GRID         = np.linspace(950, 1800, 600)   # mm
X_AXIS_MAX_T   = 35.0                          # tonnes

# Named configurations [label: R_IV in mm]
CONFIGS = {
    "Baseline":            1691.00,
    "Recommended (13 t)":  1308.37,
    "Aggressive (7 t)":    1121.80,
}
CONFIG_COLORS = {
    "Baseline":            "0.3",
    "Recommended (13 t)":  "C1",
    "Aggressive (7 t)":    "C3",
}

# Font sizes
FS_LABEL  = 18
FS_TICK   = 14
FS_LEGEND = 14


def t_iv(r_mm):
    """Inner-vessel wall thickness (mm) as a function of outer radius (mm)."""
    return T_IV0_MM * (np.asarray(r_mm, float) / R_IV0_MM)


def m_iv(r_mm):
    """Inner-vessel mass (kg) as a function of outer radius (mm)."""
    return M_IV0_KG * (np.asarray(r_mm, float) / R_IV0_MM) ** 3


def main():
    mass_grid_t = hfe_mass_tonnes_from_radius_mm(R_GRID)
    t_grid      = t_iv(R_GRID)
    m_grid      = m_iv(R_GRID)

    # --- Print verification table ---
    print(f"{'Configuration':<25} {'R_IV (mm)':>10} {'HFE (t)':>9}"
          f" {'t_IV (mm)':>10} {'m_IV (kg)':>10} {'m ratio':>8}")
    print("-" * 78)
    for label, r_mm in CONFIGS.items():
        mass_t = float(hfe_mass_tonnes_from_radius_mm(r_mm))
        t_val  = float(t_iv(r_mm))
        m_val  = float(m_iv(r_mm))
        print(f"{label:<25} {r_mm:>10.2f} {mass_t:>9.2f}"
              f" {t_val:>10.3f} {m_val:>10.1f} {m_val/M_IV0_KG:>8.3f}")

    # --- Plot ---
    fig, ax_t = plt.subplots(figsize=(10, 7))
    ax_m = ax_t.twinx()

    color_t = "C0"
    color_m = "C2"

    ax_t.plot(mass_grid_t, t_grid, color=color_t, linewidth=1.2, alpha=0.9,
              label=r"$t_\mathrm{IV}$ (mm)")
    ax_m.plot(mass_grid_t, m_grid, color=color_m, linewidth=1.2, alpha=0.9,
              linestyle="--", label=r"$m_\mathrm{IV}$ (kg)")

    # Highlight named configurations
    for label, r_mm in CONFIGS.items():
        mass_t = float(hfe_mass_tonnes_from_radius_mm(r_mm))
        t_val  = float(t_iv(r_mm))
        m_val  = float(m_iv(r_mm))
        col    = CONFIG_COLORS[label]

        ax_t.axvline(mass_t, color="black", linestyle=(0, (4, 2)), linewidth=1.5)
        ax_t.plot(mass_t, t_val, marker="o", color=color_t,
                  markeredgecolor="black", markeredgewidth=1.5, markersize=5, zorder=4)
        ax_m.plot(mass_t, m_val, marker="s", color=color_m,
                  markeredgecolor="black", markeredgewidth=1.5, markersize=5, zorder=4)

        # Annotation
        ax_t.text(
            mass_t, 0.02,
            label.split(" (")[0],
            transform=ax_t.get_xaxis_transform(),
            rotation=90, ha="right", va="bottom",
            fontsize=FS_TICK, color="black",
        )

    ax_t.set_xlabel("HFE mass (tonnes)", fontsize=FS_LABEL)
    ax_t.set_ylabel(r"Wall thickness $t_\mathrm{IV}$ (mm)",
                    fontsize=FS_LABEL, color=color_t)
    ax_m.set_ylabel(r"Vessel mass $m_\mathrm{IV}$ (kg)",
                    fontsize=FS_LABEL, color=color_m)

    ax_t.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax_t.tick_params(axis="y", colors=color_t)
    ax_m.tick_params(axis="y", labelsize=FS_TICK, colors=color_m)

    ax_t.set_xlim(float(mass_grid_t.min()), X_AXIS_MAX_T)
    ax_t.set_ylim(bottom=0)
    ax_m.set_ylim(bottom=0)

    ax_t.grid(True, which="both", linestyle=":", alpha=0.4)

    # Combined legend
    lines_t, labels_t = ax_t.get_legend_handles_labels()
    lines_m, labels_m = ax_m.get_legend_handles_labels()
    ax_t.legend(lines_t + lines_m, labels_t + labels_m,
                fontsize=FS_LEGEND, loc="upper left")

    plt.tight_layout()
    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path)
    plt.close()
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
