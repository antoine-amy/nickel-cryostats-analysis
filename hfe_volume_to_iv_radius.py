#!/usr/bin/env python3
"""
Compute HFE properties from the inner-cryostat radius using a simplified
sphere-minus-cylinder geometry.

Model:
  HFE volume = sphere(inner cryostat) - cylindrical TPC envelope
  HFE mass   = density * volume - calibration_loss

The calibration_loss can be chosen so the baseline radius reproduces a target
baseline HFE mass (for example 31810 kg or 31593.29 kg).

Also reports:
  - minimum HFE thickness (corner-to-sphere)
  - mean radial thickness <Δr> from angular averaging
  - effective attenuation thickness δ_eff if mu_HFE is provided
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

# -----------------------------
# Geometry and material inputs
# -----------------------------
DENSITY_T_PER_M3 = 1.73  # HFE density used in your original script

# Baseline simplified geometry from your note
DEFAULT_RIC_M = 1.691     # inner cryostat radius [m]
TPC_RADIUS_M = 0.6385     # cylindrical TPC radius a_TPC [m]
TPC_HEIGHT_M = 1.277      # full TPC height H_TPC [m]

# One of these can be selected as the baseline target mass
BASELINE_TARGET_MASS_KG_1 = 31810.0
BASELINE_TARGET_MASS_KG_2 = 31593.29

# Optional attenuation coefficient for "effective thickness"
# Put in 1/m if you want delta_eff = -ln(<T>)/mu.
MU_HFE_INV_M: Optional[float] = None

# Plot config
R_GRID_MM = np.linspace(950.0, 1800.0, 600)
REFERENCE_RADIUS_MM = 1100.0
FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14


@dataclass
class HFESummary:
    ric_m: float
    hfe_volume_m3: float
    hfe_mass_tonnes: float
    hfe_mass_kg: float
    min_thickness_mm: float
    mean_thickness_mm: float
    effective_thickness_mm: Optional[float]
    avg_transmission: Optional[float]
    avg_attenuation: Optional[float]


# -----------------------------
# Geometry helpers
# -----------------------------
def sphere_volume_m3(radius_m: float | np.ndarray) -> float | np.ndarray:
    radius_m = np.asarray(radius_m, dtype=float)
    return (4.0 / 3.0) * np.pi * radius_m**3


def tpc_cylinder_volume_m3(
    tpc_radius_m: float = TPC_RADIUS_M,
    tpc_height_m: float = TPC_HEIGHT_M,
) -> float:
    return float(np.pi * tpc_radius_m**2 * tpc_height_m)


def hfe_volume_m3(
    ric_m: float | np.ndarray,
    tpc_radius_m: float = TPC_RADIUS_M,
    tpc_height_m: float = TPC_HEIGHT_M,
) -> float | np.ndarray:
    """
    Simplified HFE volume:
      volume inside spherical inner cryostat
      minus cylindrical TPC envelope volume.
    """
    return sphere_volume_m3(ric_m) - tpc_cylinder_volume_m3(tpc_radius_m, tpc_height_m)


def ideal_hfe_mass_tonnes(
    ric_m: float | np.ndarray,
    density_t_per_m3: float = DENSITY_T_PER_M3,
) -> float | np.ndarray:
    return density_t_per_m3 * hfe_volume_m3(ric_m)


def calibration_loss_tonnes_for_target(
    target_mass_kg: float,
    baseline_ric_m: float = DEFAULT_RIC_M,
    density_t_per_m3: float = DENSITY_T_PER_M3,
) -> float:
    ideal_baseline_t = float(ideal_hfe_mass_tonnes(baseline_ric_m, density_t_per_m3))
    return ideal_baseline_t - target_mass_kg / 1000.0


def hfe_mass_tonnes(
    ric_m: float | np.ndarray,
    calibration_loss_tonnes: float,
    density_t_per_m3: float = DENSITY_T_PER_M3,
) -> float | np.ndarray:
    return ideal_hfe_mass_tonnes(ric_m, density_t_per_m3) - calibration_loss_tonnes


# -----------------------------
# Thickness helpers
# -----------------------------
def tpc_corner_radius_m(
    tpc_radius_m: float = TPC_RADIUS_M,
    tpc_height_m: float = TPC_HEIGHT_M,
) -> float:
    return math.sqrt(tpc_radius_m**2 + (tpc_height_m / 2.0) ** 2)


def min_hfe_thickness_mm(
    ric_m: float | np.ndarray,
    tpc_radius_m: float = TPC_RADIUS_M,
    tpc_height_m: float = TPC_HEIGHT_M,
) -> float | np.ndarray:
    """
    Minimum HFE thickness occurs at the cylindrical TPC corner.
    """
    ric_m = np.asarray(ric_m, dtype=float)
    rmax = tpc_corner_radius_m(tpc_radius_m, tpc_height_m)
    return (ric_m - rmax) * 1000.0


def r_intersection_m(
    theta_rad: np.ndarray,
    tpc_radius_m: float = TPC_RADIUS_M,
    tpc_height_m: float = TPC_HEIGHT_M,
) -> np.ndarray:
    """
    Radial distance from origin to the TPC boundary along polar angle theta.
    Matches the geometry in your note.
    """
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)

    with np.errstate(divide="ignore", invalid="ignore"):
        r_side = tpc_radius_m / np.where(np.abs(sin_theta) > 0, np.abs(sin_theta), np.nan)
        r_cap = (tpc_height_m / 2.0) / np.where(np.abs(cos_theta) > 0, np.abs(cos_theta), np.nan)

    return np.minimum(r_side, r_cap)


def mean_hfe_thickness_mm(
    ric_m: float,
    n_theta: int = 20000,
    tpc_radius_m: float = TPC_RADIUS_M,
    tpc_height_m: float = TPC_HEIGHT_M,
) -> float:
    theta = np.linspace(0.0, np.pi, n_theta)
    r_theta = r_intersection_m(theta, tpc_radius_m, tpc_height_m)
    delta_r = ric_m - r_theta
    mean_delta_r = 0.5 * np.trapz(delta_r * np.sin(theta), theta)
    return float(mean_delta_r * 1000.0)


def attenuation_summary(
    ric_m: float,
    mu_hfe_inv_m: float,
    n_theta: int = 20000,
    tpc_radius_m: float = TPC_RADIUS_M,
    tpc_height_m: float = TPC_HEIGHT_M,
) -> tuple[float, float, float]:
    """
    Returns:
      <T>, <A>, delta_eff [mm]
    """
    theta = np.linspace(0.0, np.pi, n_theta)
    r_theta = r_intersection_m(theta, tpc_radius_m, tpc_height_m)
    delta_r = ric_m - r_theta
    transmission = np.exp(-mu_hfe_inv_m * delta_r)
    avg_T = 0.5 * np.trapz(transmission * np.sin(theta), theta)
    avg_A = 1.0 - avg_T
    delta_eff_m = -math.log(avg_T) / mu_hfe_inv_m
    return float(avg_T), float(avg_A), float(delta_eff_m * 1000.0)


# -----------------------------
# Reporting
# -----------------------------
def summarize_from_radius(
    ric_m: float,
    calibration_loss_tonnes: float,
    mu_hfe_inv_m: Optional[float] = MU_HFE_INV_M,
) -> HFESummary:
    volume_m3 = float(hfe_volume_m3(ric_m))
    mass_t = float(hfe_mass_tonnes(ric_m, calibration_loss_tonnes))
    dmin_mm = float(min_hfe_thickness_mm(ric_m))
    dmean_mm = mean_hfe_thickness_mm(ric_m)

    if mu_hfe_inv_m is None:
        avg_T = avg_A = deff_mm = None
    else:
        avg_T, avg_A, deff_mm = attenuation_summary(ric_m, mu_hfe_inv_m)

    return HFESummary(
        ric_m=ric_m,
        hfe_volume_m3=volume_m3,
        hfe_mass_tonnes=mass_t,
        hfe_mass_kg=1000.0 * mass_t,
        min_thickness_mm=dmin_mm,
        mean_thickness_mm=dmean_mm,
        effective_thickness_mm=deff_mm,
        avg_transmission=avg_T,
        avg_attenuation=avg_A,
    )


def print_summary(summary: HFESummary, label: str | None = None) -> None:
    if label:
        print(label)
    print(f"Inner-cryostat radius:   {summary.ric_m:.3f} m")
    print(f"HFE volume:              {summary.hfe_volume_m3:.4f} m^3")
    print(f"HFE mass:                {summary.hfe_mass_tonnes:.4f} t ({summary.hfe_mass_kg:.1f} kg)")
    print(f"Minimum HFE thickness:   {summary.min_thickness_mm:.2f} mm")
    print(f"Mean HFE thickness:      {summary.mean_thickness_mm:.2f} mm")
    if summary.effective_thickness_mm is not None:
        print(f"Effective thickness:     {summary.effective_thickness_mm:.2f} mm")
        print(f"Average transmission:    {summary.avg_transmission:.6e}")
        print(f"Average attenuation:     {summary.avg_attenuation:.6f}")


# -----------------------------
# Plot
# -----------------------------
def plot_hfe_vs_iv_radius(
    calibration_loss_tonnes: float,
    radius_grid_mm: np.ndarray = R_GRID_MM,
) -> None:
    radius_grid_mm = np.asarray(radius_grid_mm, dtype=float)
    radius_grid_m = radius_grid_mm / 1000.0
    mass_t = hfe_mass_tonnes(radius_grid_m, calibration_loss_tonnes)

    baseline_mass_t = float(hfe_mass_tonnes(DEFAULT_RIC_M, calibration_loss_tonnes))
    reference_mass_t = float(hfe_mass_tonnes(REFERENCE_RADIUS_MM / 1000.0, calibration_loss_tonnes))
    mass_pct = 100.0 * mass_t / baseline_mass_t
    reference_mass_pct = 100.0 * reference_mass_t / baseline_mass_t

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(radius_grid_mm, mass_pct, "--", linewidth=1.5, label="HFE mass")

    ax.set_xlabel("Inner cryostat radius (mm)", fontsize=FS_LABEL)
    ax.set_ylabel("HFE mass (% of baseline)", fontsize=FS_LABEL)
    ax.tick_params(axis="both", which="major", labelsize=FS_TICK)
    ax.grid(True, which="both", linestyle=":", alpha=0.4)

    ax.axvline(DEFAULT_RIC_M * 1000.0, color="0.6", linestyle="--", linewidth=1.0)
    ax.text(
        DEFAULT_RIC_M * 1000.0,
        0.92,
        "Baseline (1691 mm)",
        rotation=90,
        va="top",
        ha="right",
        transform=ax.get_xaxis_transform(),
        fontsize=FS_TICK,
        color="0.3",
    )

    ax.axvline(REFERENCE_RADIUS_MM, color="0.6", linestyle="--", linewidth=1.0)
    ax.text(
        REFERENCE_RADIUS_MM,
        0.85,
        "IV = 1100 mm",
        rotation=90,
        va="top",
        ha="right",
        transform=ax.get_xaxis_transform(),
        fontsize=FS_TICK,
        color="0.3",
    )

    ax.annotate(
        f"{reference_mass_t:.2f} t",
        xy=(REFERENCE_RADIUS_MM, reference_mass_pct),
        xytext=(8, -5),
        textcoords="offset points",
        ha="left",
        va="center",
        fontsize=FS_TICK,
    )

    ax.legend(fontsize=FS_LEGEND, loc="upper left")
    plt.tight_layout()
    plt.show()


def main() -> None:
    # Choose which baseline mass you want to reproduce
    target_mass_kg = BASELINE_TARGET_MASS_KG_1
    # target_mass_kg = BASELINE_TARGET_MASS_KG_2

    cal_loss_t = calibration_loss_tonnes_for_target(target_mass_kg)

    print(f"Chosen baseline target mass: {target_mass_kg:.2f} kg")
    print(f"Derived calibration loss:    {1000.0 * cal_loss_t:.2f} kg")
    print()

    baseline = summarize_from_radius(DEFAULT_RIC_M, cal_loss_t, MU_HFE_INV_M)
    ref_1100 = summarize_from_radius(REFERENCE_RADIUS_MM / 1000.0, cal_loss_t, MU_HFE_INV_M)

    print_summary(baseline, label="Baseline radius")
    print()
    print_summary(ref_1100, label="IV = 1100 mm")
    print()

    # Also print both possible calibration losses for comparison
    cal1 = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_1)
    cal2 = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_2)
    print(f"Calibration loss for 31810.00 kg  baseline: {1000.0 * cal1:.2f} kg")
    print(f"Calibration loss for 31593.29 kg baseline: {1000.0 * cal2:.2f} kg")

    plot_hfe_vs_iv_radius(cal_loss_t)


if __name__ == "__main__":
    main()