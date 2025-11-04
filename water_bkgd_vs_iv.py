#!/usr/bin/env python3
"""
Water-shield background vs IV radius — with OV-ε uncertainty propagated.

OV→water model
  ε_water(r_IV, r_OV) = ε_OV(r_OV) · S_HFE(r_IV) · F
    ε_OV(r_OV) = ε_OV(r_OV_ref) · (r_OV_ref / r_OV)^2
    F = (2π / V_water) ∬ (r_OV/s)^2 · exp[-μ_water (s − r_OV)] · r dr dz,  s = √(r^2+z^2)
    S_HFE(r_IV) = exp[− μ_HFE · (r_IV − R_IV_ref)]  # ε decreases as r_IV increases

Uncertainty (TG-spread method):
  • OV efficiency 1σ at the reference radius: σ_ε0,i = sqrt( ε_OV0,i · branch_i / N_OV )
  • Propagate multiplicatively to water: ε_i = ε_OV0,i · (r_OV_ref/r_OV)^2 · F · S_HFE
  • Activity inputs include (A_i ± σ_A,i)
  • Convert to counts/y with a truncated Gaussian for the positive-only rate:
        t_i   = M_water · A_i · ε_i
        mean  = TG_mean(t_i, σ_A,i) · sec_per_year
        σ_i   = TG_spread(t_i, σ_A,i, σ_ε,i) · sec_per_year
  • Total band: σ_tot = sqrt( Σ σ_i^2 )

We include U-238, Rn-222 (same ε as U), and Th-232 (branch_Th = 0.3594).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np


SECONDS_PER_YEAR = 86400.0 * 365.25
SQRT_TWO = math.sqrt(2.0)
SQRT_TWO_PI = math.sqrt(2.0 * math.pi)
NOV_EVENTS = 1.0e10


# ---------- styling ----------
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["DejaVu Serif", "Times New Roman", "Times"],
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.alpha": 0.35,
    "grid.linestyle": ":",
    "lines.linewidth": 2.2,
    "errorbar.capsize": 4,
})


@dataclass(frozen=True)
class GeometryConfig:
    """Nominal detector geometry in metres and millimetres."""

    r_ov_ref_mm: float
    r_ov_mm: float
    r_tank_m: float
    h_tank_m: float
    r_iv_ref_mm: float


@dataclass(frozen=True)
class MaterialProperties:
    """Material properties relevant to transport and density."""

    rho_water_kg_per_m3: float
    mu_water_mm: float
    mu_hfe_mm: float


@dataclass(frozen=True)
class IsotopeParams:
    """Baseline detection and activity inputs for one isotope."""

    name: str
    ov_ref_efficiency: float
    activity_bq_per_kg: float
    activity_error_bq_per_kg: float
    branch_fraction: float


@dataclass(frozen=True)
class IntegrationGrid:
    """Simple (nr, nz) grid descriptor for cylindrical integration."""

    radial_bins: int
    axial_bins: int


@dataclass(frozen=True)
class MeshData:
    """Reusable cylindrical mesh components."""

    radius_mesh: np.ndarray
    distance: np.ndarray
    mask: np.ndarray
    dr: float
    dz: float
    r_tank_mm: float
    h_tank_mm: float


@dataclass(frozen=True)
class PlotData:
    """Inputs required to render the background plot."""

    radii_mm: np.ndarray
    means: np.ndarray
    sigmas: np.ndarray
    mc_points: Tuple[np.ndarray, np.ndarray, np.ndarray]
    delta90_mm: float


@dataclass(frozen=True)
class IsotopeResult:
    """Intermediate per-isotope quantities for reporting."""

    name: str
    efficiency: float
    efficiency_error: float
    mean_counts: float
    spread_counts: float


def cylindrical_mesh(geometry: GeometryConfig,
                     grid: IntegrationGrid) -> MeshData:
    """Build radius/height meshes, distance mask, and cell spacings."""
    r_tank_mm = geometry.r_tank_m * 1000.0
    h_tank_mm = geometry.h_tank_m * 1000.0
    dr = r_tank_mm / grid.radial_bins
    dz = h_tank_mm / grid.axial_bins

    radius_centres = (np.arange(grid.radial_bins) + 0.5) * dr
    height_centres = (-h_tank_mm / 2.0) + (np.arange(grid.axial_bins) + 0.5) * dz
    radius_mesh, height_mesh = np.meshgrid(radius_centres, height_centres, indexing="xy")
    distance = np.sqrt(radius_mesh * radius_mesh + height_mesh * height_mesh)
    mask = distance >= geometry.r_ov_mm
    return MeshData(radius_mesh, distance, mask, dr, dz, r_tank_mm, h_tank_mm)


def water_volumes_mm(r_ov_mm: float,
                     r_tank_mm: float,
                     h_tank_mm: float) -> Tuple[float, float, float]:
    """Return tank, OV, and water volumes in mm^3 for the given geometry."""
    tank_volume = math.pi * (r_tank_mm ** 2) * h_tank_mm
    ov_volume = (4.0 / 3.0) * math.pi * (r_ov_mm ** 3)
    water_volume = tank_volume - ov_volume
    if water_volume <= 0.0:
        msg = "Non-positive water volume; check geometry."
        raise ValueError(msg)
    return tank_volume, ov_volume, water_volume


def water_mass_kg(geometry: GeometryConfig,
                  materials: MaterialProperties) -> float:
    """Return the total water mass in kilograms."""
    tank_volume_m3 = math.pi * (geometry.r_tank_m ** 2) * geometry.h_tank_m
    ov_volume_m3 = (4.0 / 3.0) * math.pi * (geometry.r_ov_mm / 1000.0) ** 3
    return (tank_volume_m3 - ov_volume_m3) * materials.rho_water_kg_per_m3


def ov_efficiency_uncertainty(efficiency: float,
                              isotope: IsotopeParams) -> float:
    """Return the OV-derived efficiency uncertainty scaled to the water geometry."""
    if efficiency <= 0.0 or isotope.ov_ref_efficiency <= 0.0:
        if isotope.branch_fraction <= 0.0:
            return 0.0
        return (1.14 * isotope.branch_fraction) / NOV_EVENTS
    scale = isotope.branch_fraction / (NOV_EVENTS * isotope.ov_ref_efficiency)
    return efficiency * math.sqrt(scale)


def truncated_gaussian_mean(mass_kg: float,
                            activity_bq_per_kg: float,
                            activity_error_bq_per_kg: float,
                            efficiency: float) -> float:
    """Return the truncated-Gaussian mean counts per year."""
    t_counts_per_sec = mass_kg * activity_bq_per_kg * efficiency
    u = abs(mass_kg * efficiency * activity_error_bq_per_kg)
    if u < 1e-15:
        mean_counts_per_sec = max(0.0, t_counts_per_sec)
    else:
        z = t_counts_per_sec / u if u > 0.0 else 0.0
        pdf = math.exp(-0.5 * z * z) / SQRT_TWO_PI
        cdf = 0.5 * (1.0 + math.erf(z / SQRT_TWO))
        if cdf <= 0.0:
            mean_counts_per_sec = max(0.0, t_counts_per_sec)
        else:
            mean_counts_per_sec = t_counts_per_sec + u * pdf / cdf
    return mean_counts_per_sec * SECONDS_PER_YEAR


def truncated_gaussian_spread(mass_kg: float,
                              activity_bq_per_kg: float,
                              activity_error_bq_per_kg: float,
                              efficiency: float,
                              efficiency_error: float) -> float:
    """Return the TG-spread (1σ) counts per year."""
    t_counts_per_sec = mass_kg * activity_bq_per_kg * efficiency
    term_activity = mass_kg * efficiency * activity_error_bq_per_kg
    term_efficiency = mass_kg * activity_bq_per_kg * efficiency_error
    u = math.hypot(term_activity, term_efficiency)
    if u < 1e-15:
        return 0.0
    z = t_counts_per_sec / u if u > 0.0 else 0.0
    pdf = math.exp(-0.5 * z * z) / SQRT_TWO_PI
    cdf = 0.5 * (1.0 + math.erf(z / SQRT_TWO))
    if cdf <= 0.0:
        return 0.0
    lam = pdf / cdf
    inner = 1.0 - z * lam - (lam * lam)
    if inner <= 0.0:
        return 0.0
    return u * math.sqrt(inner) * SECONDS_PER_YEAR



def mean_factor(mu_water_mm: float,
                geometry: GeometryConfig,
                grid: IntegrationGrid) -> float:
    """Return the average attenuation/geometry factor F for the water volume."""
    mesh = cylindrical_mesh(geometry, grid)
    _, _, water_volume = water_volumes_mm(
        geometry.r_ov_mm, mesh.r_tank_mm, mesh.h_tank_mm)

    with np.errstate(divide="ignore", invalid="ignore"):
        geometry_factor = np.zeros_like(mesh.distance)
        geometry_factor[mesh.mask] = (
            (geometry.r_ov_mm ** 2) / (mesh.distance[mesh.mask] ** 2)
        )

    attenuation = np.ones_like(mesh.distance)
    attenuation[mesh.mask] = np.exp(
        -mu_water_mm * (mesh.distance[mesh.mask] - geometry.r_ov_mm))

    volume_element = (2.0 * math.pi) * mesh.radius_mesh * mesh.dr * mesh.dz
    integral = float(np.sum(geometry_factor * attenuation * volume_element))
    return integral / water_volume


def delta90_water_thickness(materials: MaterialProperties,
                            geometry: GeometryConfig,
                            grid: IntegrationGrid) -> float:
    """Return the 90%-containment path length (Δ90) for attenuated contributions."""
    mesh = cylindrical_mesh(geometry, grid)

    with np.errstate(divide="ignore", invalid="ignore"):
        geometry_factor = np.zeros_like(mesh.distance)
        geometry_factor[mesh.mask] = (
            (geometry.r_ov_mm ** 2) / (mesh.distance[mesh.mask] ** 2)
        )

    attenuation = np.ones_like(mesh.distance)
    attenuation[mesh.mask] = np.exp(
        -materials.mu_water_mm * (mesh.distance[mesh.mask] - geometry.r_ov_mm))
    volume_element = (2.0 * math.pi) * mesh.radius_mesh * mesh.dr * mesh.dz

    contributions = (geometry_factor * attenuation * volume_element)[mesh.mask]
    delta = (mesh.distance - geometry.r_ov_mm)[mesh.mask]
    if contributions.size == 0:
        return 0.0

    order = np.argsort(delta)
    cumulative = np.cumsum(contributions[order])
    target = 0.9 * cumulative[-1]
    index = np.searchsorted(cumulative, target, side="left")
    return float(delta[order][index])


def hfe_screening_factor(mu_hfe_mm: float, r_iv_mm: float, r_iv_ref_mm: float) -> float:
    """Return the HFE self-screening term."""
    return math.exp(-mu_hfe_mm * (r_iv_mm - r_iv_ref_mm))


def water_detection_efficiency(isotope: IsotopeParams,
                               r_iv_mm: float,
                               geometry: GeometryConfig,
                               materials: MaterialProperties,
                               grid: IntegrationGrid) -> Tuple[float, float]:
    """Return (ε_water, σ_ε_water) propagated from the OV baseline."""
    geometric_scale = (geometry.r_ov_ref_mm / geometry.r_ov_mm) ** 2
    attenuation = mean_factor(materials.mu_water_mm, geometry, grid)
    screening = hfe_screening_factor(materials.mu_hfe_mm, r_iv_mm, geometry.r_iv_ref_mm)
    scale = geometric_scale * attenuation * screening

    efficiency = isotope.ov_ref_efficiency * scale
    efficiency_error = ov_efficiency_uncertainty(efficiency, isotope)
    return efficiency, efficiency_error


def model_background(r_iv_mm: float,
                     geometry: GeometryConfig,
                     materials: MaterialProperties,
                     isotopes: Sequence[IsotopeParams],
                     grid: IntegrationGrid) -> Tuple[float, float, Tuple[IsotopeResult, ...]]:
    """Return total TG mean and spread for the supplied IV radius."""
    mass_kg = water_mass_kg(geometry, materials)

    total_mean = 0.0
    total_spread_sq = 0.0
    results: list[IsotopeResult] = []
    for isotope in isotopes:
        efficiency, efficiency_error = water_detection_efficiency(
            isotope, r_iv_mm, geometry, materials, grid)
        mean_counts = truncated_gaussian_mean(
            mass_kg,
            isotope.activity_bq_per_kg,
            isotope.activity_error_bq_per_kg,
            efficiency,
        )
        spread_counts = truncated_gaussian_spread(
            mass_kg,
            isotope.activity_bq_per_kg,
            isotope.activity_error_bq_per_kg,
            efficiency,
            efficiency_error,
        )
        results.append(IsotopeResult(
            name=isotope.name,
            efficiency=efficiency,
            efficiency_error=efficiency_error,
            mean_counts=mean_counts,
            spread_counts=spread_counts,
        ))
        total_mean += mean_counts
        total_spread_sq += spread_counts ** 2

    return total_mean, math.sqrt(total_spread_sq), tuple(results)


def baseline_geometry() -> GeometryConfig:
    """Default SNOLAB water shield geometry."""
    return GeometryConfig(
        r_ov_ref_mm=2230.0,
        r_ov_mm=2230.0,
        r_tank_m=12.3 / 2.0,
        h_tank_m=13.3,
        r_iv_ref_mm=1691.0,
    )


def baseline_materials() -> MaterialProperties:
    """Baseline material properties."""
    return MaterialProperties(
        rho_water_kg_per_m3=1000.0,
        mu_water_mm=0.0045,
        mu_hfe_mm=0.00592496,
    )


def baseline_isotopes() -> Tuple[IsotopeParams, ...]:
    """Return the trio of isotopes considered in the background estimate."""
    return (
        IsotopeParams("U-238", 2.00e-10, 1.98e-06, 8.776e-07, 1.0),
        IsotopeParams("Rn-222", 2.00e-10, 0.0, 0.0, 1.0),
        IsotopeParams("Th-232", 1.26e-09, 3.93e-07, 3.084e-08, 0.3594),
    )


def default_radii(start_mm: float = 1000.0,
                  stop_mm: float = 1700.0,
                  samples: int = 81) -> np.ndarray:
    """Return evenly spaced IV radii used for the sweep."""
    return np.linspace(start_mm, stop_mm, samples)


def compute_background_curve(radii_mm: Iterable[float],
                             geometry: GeometryConfig,
                             materials: MaterialProperties,
                             isotopes: Sequence[IsotopeParams],
                             grid: IntegrationGrid) -> Tuple[np.ndarray, np.ndarray]:
    """Evaluate the mean and σ curves for the supplied radii."""
    radii_array = np.asarray(list(radii_mm), dtype=float)
    means = np.empty_like(radii_array)
    sigmas = np.empty_like(radii_array)

    for index, radius in enumerate(radii_array):
        mean_counts, sigma_counts, _ = model_background(
            radius, geometry, materials, isotopes, grid)
        means[index] = mean_counts
        sigmas[index] = sigma_counts

    return means, sigmas


def reference_mc_points() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the available MC comparison points (radius, mean, σ)."""
    mc_radii = np.array([1026.0, 1691.0], dtype=float)
    mc_background = np.array([1.65E-2, 1.56E-4], dtype=float)
    mc_uncertainty = np.array([9.51E-3, 7.55E-4], dtype=float)
    return mc_radii, mc_background, mc_uncertainty


def plot_background_curve(plot_data: PlotData,
                          geometry: GeometryConfig) -> None:
    """Render the background sweep with MC comparison points."""
    upper = plot_data.means + plot_data.sigmas
    lower = np.maximum(plot_data.means - plot_data.sigmas, 1e-30)
    mc_radii, mc_background, mc_uncertainty = plot_data.mc_points

    fig, ax = plt.subplots(figsize=(6.4, 4.0), dpi=180)

    ax.fill_between(plot_data.radii_mm, lower, upper, color="0.8", alpha=0.6)
    ax.plot(
        plot_data.radii_mm,
        plot_data.means,
        "-o",
        markevery=6,
        mec="black",
        mfc="white",
        ms=4.5,
        lw=2.2,
        color="black",
        label="OV derived",
    )

    ax.errorbar(
        mc_radii,
        mc_background,
        yerr=mc_uncertainty,
        fmt="D",
        mfc="#ff8c00",
        mec="#ff8c00",
        ecolor="#ff8c00",
        elinewidth=1.1,
        ms=5.0,
        label="MC",
    )

    ax.axvline(geometry.r_iv_ref_mm, ls=":", lw=1.2, color="black")

    ax.set_xlabel("IV radius [mm]")
    ax.set_ylabel("Background [counts/yr/ROI/2t]")
    ax.set_yscale("log")
    ax.set_xlim(plot_data.radii_mm.min(), plot_data.radii_mm.max())

    ax.minorticks_on()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    ax.text(
        0.02,
        0.04,
        rf"$\Delta_{{90}}$ (baseline) $= {plot_data.delta90_mm / 1000.0:.2f}\,\mathrm{{m}}$",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=10,
    )

    ax.legend(loc="upper right", frameon=True, framealpha=0.95, edgecolor="0.85")
    fig.tight_layout()
    plt.show()


def print_reference_backgrounds(geometry: GeometryConfig,
                                materials: MaterialProperties,
                                isotopes: Sequence[IsotopeParams],
                                sample_radii_mm: Sequence[float],
                                grid: IntegrationGrid) -> None:
    """Print background means, bands, and hit efficiencies for selected radii."""
    print("\nBackground summary (default OV radius):")
    mass_kg = water_mass_kg(geometry, materials)
    print(f"  Water mass: {mass_kg:,.1f} kg")
    for radius in sample_radii_mm:
        mean_counts, sigma_counts, results = model_background(
            radius, geometry, materials, isotopes, grid)
        print(
            f"  IV radius {radius:6.1f} mm -> "
            f"{mean_counts:9.3e} ± {sigma_counts:9.3e} counts/yr/ROI/2t"
        )
        joined = ", ".join(f"{res.name}: {res.efficiency:8.3e}" for res in results)
        print(f"    Hit ε: {joined}")


def main() -> None:
    """Run the sweep, report key radii, and generate the plot."""
    geometry = baseline_geometry()
    materials = baseline_materials()
    isotopes = baseline_isotopes()
    sweep_grid = IntegrationGrid(radial_bins=160, axial_bins=200)
    delta_grid = IntegrationGrid(radial_bins=240, axial_bins=320)

    radii_mm = default_radii()
    delta90_mm = delta90_water_thickness(materials, geometry, delta_grid)
    print(f"Δ90 (baseline) = {delta90_mm / 1000.0:.3f} m")

    means, sigmas = compute_background_curve(
        radii_mm,
        geometry,
        materials,
        isotopes,
        sweep_grid,
    )

    print_reference_backgrounds(
        geometry,
        materials,
        isotopes,
        sample_radii_mm=(geometry.r_iv_ref_mm, 1300.0, 1026.0),
        grid=sweep_grid,
    )

    plot_data = PlotData(
        radii_mm=radii_mm,
        means=means,
        sigmas=sigmas,
        mc_points=reference_mc_points(),
        delta90_mm=delta90_mm,
    )

    plot_background_curve(
        plot_data,
        geometry,
    )


if __name__ == "__main__":
    main()
