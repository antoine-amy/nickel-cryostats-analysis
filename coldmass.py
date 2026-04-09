#!/usr/bin/env python3
"""
Cold-mass / thermal-inertia vs IV radius (HFE shielding volume model)

What this script shows
----------------------
Primary (left) y-axis:
  Energy required for a +1 K temperature rise (MJ/K):
      E_1K = m * c_p(T) * ΔT   with ΔT = 1 K

Secondary (right) y-axis:
  Equivalent cold mass expressed in tonnes of LXe:
      m_eq(LXe) = E_1K / (c_p,LXe * ΔT)

Plotted items
-------------
- HFE: E_1K(R_IV) computed from the geometry-driven HFE mass vs IV radius.
- Horizontal references:
    * LXe (5 t) energy line  (colored)
    * Cryostat IV Ni (2 t) energy line  (different color)
- Vertical reference lines:
    * Baseline design R_IV = 1691 mm
    * Reduced HFE case R_IV = 1100 mm
- Annotated points:
    * 1691 mm (baseline), 1300 mm, 1100 mm on the ENERGY axis (MJ/K)
      plus the corresponding HFE mass (t) for context.

Notes on c_p
------------
For a 1 K increment, treating c_p(T) as constant is a first-order approximation.
If you have better cryogenic c_p values (at ~165 K), replace the constants below.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

# Font sizes (matching background_vs_hfe style)
FS_LABEL    = 18
FS_TICK     = 14
FS_LEGEND   = 14
FS_ANNOTATE = 12

# -----------------------------
# Geometry-driven HFE mass model (from your provided code)
# -----------------------------
DENSITY_T_PER_M3 = 1.73        # tonne per m^3 of HFE
VOL_OFFSET_M3    = 2.2 - 0.58  # m^3 correction term
HFE_LOSS_TONNES  = 0.430       # HFE lost to environment (baseline 1691 mm IV)

DEFAULT_IV_RADIUS_M = 1.691
DEFAULT_MIN_THICKNESS_M = 0.76
TPC_RADIUS_M = DEFAULT_IV_RADIUS_M - DEFAULT_MIN_THICKNESS_M  # implied

R_GRID_MM = np.linspace(950.0, 1800.0, 600)
BASELINE_RADIUS_MM = DEFAULT_IV_RADIUS_M * 1000.0  # 1691 mm
R_RECOMMENDED_MM = 1308.37
R_AGGRESSIVE_MM  = 1121.80
X_AXIS_MAX_TONNES = 35.0

def hfe_volume_m3(iv_radius_m: np.ndarray) -> np.ndarray:
    """Return HFE volume (m^3) for a given IV radius (m)."""
    iv_radius_m = np.asarray(iv_radius_m, dtype=float)
    total_volume = (4.0 / 3.0) * np.pi * iv_radius_m**3
    return total_volume - VOL_OFFSET_M3

def hfe_mass_tonnes(iv_radius_m: np.ndarray) -> np.ndarray:
    """Return HFE mass (tonnes) for a given IV radius (m)."""
    volume_m3 = hfe_volume_m3(iv_radius_m)
    return volume_m3 * DENSITY_T_PER_M3 - HFE_LOSS_TONNES

def min_hfe_thickness_cm(iv_radius_m: np.ndarray, tpc_radius_m: float = TPC_RADIUS_M) -> np.ndarray:
    """Return minimal HFE thickness (cm) for a given IV radius (m)."""
    iv_radius_m = np.asarray(iv_radius_m, dtype=float)
    return (iv_radius_m - tpc_radius_m) * 100.0

# -----------------------------
# Thermodynamics: Energy for a 1 K rise
# -----------------------------
DT_K = 1.0  # 1 K increment

# Specific heats c_p (J/kg/K). Replace with your preferred cryogenic values if available.
CP_HFE = 1.20e3   # HFE-7200 effective c_p (J/kg/K) [placeholder constant]
CP_LXE = 0.34e3   # LXe c_p near 165 K (J/kg/K) [placeholder constant]
CP_NI  = 0.50e3   # Ni (CVD Ni proxy) effective c_p (J/kg/K) [placeholder constant]

# Fixed masses for reference lines (tonnes)
M_LXE_T = 5.0
M_IV_NI_T = 2.0

TON_TO_KG = 1000.0

def energy_MJ_per_K(mass_t: np.ndarray, cp_J_per_kgK: float) -> np.ndarray:
    """
    Energy for ΔT = 1 K in MJ/K:
        E_1K = m * c_p * (1 K)
    """
    mass_kg = np.asarray(mass_t, dtype=float) * TON_TO_KG
    return (mass_kg * cp_J_per_kgK * DT_K) / 1.0e6  # MJ

def lxe_equiv_tons_from_energy(E_MJ_per_K: np.ndarray) -> np.ndarray:
    """
    Convert energy (MJ/K) to equivalent tonnes of LXe:
        m_eq(LXe) = E / (c_p,LXe * 1 K)
    """
    E_J = np.asarray(E_MJ_per_K, dtype=float) * 1.0e6
    m_kg = E_J / (CP_LXE * DT_K)
    return m_kg / TON_TO_KG

# -----------------------------
# Compute curves
# -----------------------------
r_grid_m = R_GRID_MM / 1000.0
m_hfe_t = hfe_mass_tonnes(r_grid_m)            # tonnes
E_hfe = energy_MJ_per_K(m_hfe_t, CP_HFE)       # MJ/K (since ΔT=1 K)

E_lxe = float(energy_MJ_per_K(M_LXE_T, CP_LXE))
E_ni  = float(energy_MJ_per_K(M_IV_NI_T, CP_NI))

# Helper for point values at specific radii
def interp_at(r_mm: float, x_mm: np.ndarray, y: np.ndarray) -> float:
    return float(np.interp(r_mm, x_mm, y))

# Values to annotate (energy at key designs)
DESIGN_CONFIGS = {
    "Recommended": R_RECOMMENDED_MM,
    "Aggressive":  R_AGGRESSIVE_MM,
    "Baseline":    BASELINE_RADIUS_MM,
}

m_designs = {label: interp_at(r, R_GRID_MM, m_hfe_t) for label, r in DESIGN_CONFIGS.items()}
E_designs = {label: interp_at(r, R_GRID_MM, E_hfe)   for label, r in DESIGN_CONFIGS.items()}

# -----------------------------
# Plot
# -----------------------------
fig, axE = plt.subplots(figsize=(10, 7))

# HFE energy curve
axE.plot(m_hfe_t, E_hfe, linestyle="-.", linewidth=1.2, alpha=0.9, label="HFE cold mass")

# Horizontal reference lines
axE.axhline(
    E_lxe,
    linestyle="-.",
    linewidth=1.2,
    alpha=0.9,
    color="C2",
    label="LXe reference (5 t)",
)
axE.axhline(
    E_ni,
    linestyle="-.",
    linewidth=1.2,
    alpha=0.9,
    color="C3",
    label="IV Ni vessel (2 t)",
)

# Vertical lines at all three design configs (black)
for label, r_mm in DESIGN_CONFIGS.items():
    m_t = m_designs[label]
    axE.axvline(m_t, color="black", linestyle=(0, (4, 2)), linewidth=1.5)
    axE.text(m_t, 0.50, label,
             rotation=90, ha="right", va="bottom",
             transform=axE.get_xaxis_transform(),
             fontsize=FS_TICK, color="black")

# Scatter points at design configs
key_m = np.array([m_designs[l] for l in DESIGN_CONFIGS])
key_E = np.array([E_designs[l] for l in DESIGN_CONFIGS])
axE.scatter(key_m, key_E, s=25, zorder=5)

for label in DESIGN_CONFIGS:
    m_t = m_designs[label]
    E_val = E_designs[label]
    axE.annotate(
        f"{E_val:.1f} MJ/K",
        xy=(m_t, E_val),
        xytext=(-8, 8),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=FS_ANNOTATE,
        color="C0",
    )

# Axis labels / formatting
axE.set_xlabel("HFE mass (tonnes)", fontsize=FS_LABEL)
axE.set_ylabel("Energy for 1 K rise (MJ/K)", fontsize=FS_LABEL)
axE.set_xlim(float(m_hfe_t.min()), X_AXIS_MAX_TONNES)
axE.tick_params(axis="both", which="major", labelsize=FS_TICK)
axE.grid(True, which="both", linestyle=":", alpha=0.4)

# Secondary y-axis: equivalent cold mass in tonnes of LXe
axM = axE.twinx()
ymin, ymax = axE.get_ylim()
axM.set_ylim(lxe_equiv_tons_from_energy(ymin), lxe_equiv_tons_from_energy(ymax))
axM.set_ylabel("Equivalent cold mass (t of LXe)", fontsize=FS_LABEL)
axM.tick_params(axis="y", labelsize=FS_TICK)

# Legend
axE.legend(fontsize=FS_LEGEND, frameon=True)

plt.tight_layout()
output_dir = Path(__file__).resolve().parent / "budget" / "figures"
output_dir.mkdir(parents=True, exist_ok=True)
output_path = output_dir / "coldmass.png"
plt.savefig(output_path)
plt.close()
print(f"Saved to {output_path}")
