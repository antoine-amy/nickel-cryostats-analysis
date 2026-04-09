#!/usr/bin/env python3
"""
Semi-analytical water background vs OV radius for the fixed IV = 1226 mm design.

This keeps the water-only model structure used in background_vs_hfe_water_only.py
but promotes the OV radius to the scanned variable:

  ε_water(r_OV) = ε_OV(r_OV_ref) * (r_OV_ref / r_OV)^2 * F_water(r_OV) * S_HFE(IV fixed)

where F_water(r_OV) is the water attenuation / geometry average over the tank
volume outside the spherical OV.

Two semi-analytical totals are shown:
  - Th/U only
  - Th/U + Rn
"""

from __future__ import annotations

from pathlib import Path
import math
import sys

import matplotlib.pyplot as plt
import numpy as np
from openpyxl import load_workbook


BUDGET_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BUDGET_DIR.parent
for _path in (str(PROJECT_ROOT), str(BUDGET_DIR)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from hfe_volume_to_iv_radius import (
    BASELINE_TARGET_MASS_KG_1,
    calibration_loss_tonnes_for_target,
    hfe_mass_tonnes,
)
from background_vs_hfe_water_only import (
    MU_HFE_2P5MEV_MM,
    WATER_MODEL_AXIAL_BINS,
    WATER_MODEL_RADIAL_BINS,
    _ov_efficiency_uncertainty,
    _truncated_gaussian_mean,
    _truncated_gaussian_spread,
    build_water_semi_analytical_curve_with_bands,
)


CALIBRATION_LOSS_TONNES = calibration_loss_tonnes_for_target(BASELINE_TARGET_MASS_KG_1)


def hfe_mass_tonnes_from_radius_mm(radius_mm):
    radius_m = np.asarray(radius_mm, float) / 1000.0
    return hfe_mass_tonnes(radius_m, CALIBRATION_LOSS_TONNES)


XLSX_PATH = BUDGET_DIR / "Summary_bkgd_vs_hfe-shield.xlsx"
IV_RADIUS_MM = 1226.0
BASELINE_OV_RADIUS_MM = 2230.0
R_OV_GRID = np.linspace(1700.0, 2300.0, 500)
Z_BAND = 1.0
TOTAL_INTRINSIC_BACKGROUND = 0.55

FS_LABEL = 18
FS_TICK = 14
FS_LEGEND = 14


def load_water_model_inputs(filepath: Path) -> dict[str, object]:
    workbook = load_workbook(filepath, data_only=True, read_only=True)
    try:
        water_sheet = workbook["Water"]
        ov_sheet = workbook["OV"]

        tank_diameter_m = float(water_sheet["G21"].value)
        tank_height_m = float(water_sheet["H21"].value)
        density_water_kg_per_m3 = float(water_sheet["E18"].value)
        branch_th = float(water_sheet["B21"].value)
        branch_u = float(water_sheet["E21"].value)
        rn_activity_bq_per_kg = float(water_sheet["D57"].value)
        rn_activity_error_bq_per_kg = float(water_sheet["E57"].value)

        inputs = {
            "tank_radius_mm": 1000.0 * tank_diameter_m / 2.0,
            "tank_height_mm": 1000.0 * tank_height_m,
            "density_water_kg_per_m3": density_water_kg_per_m3,
            "mu_water_mm": float(water_sheet["I21"].value),
            "mu_hfe_mm": MU_HFE_2P5MEV_MM,
            "r_iv_ref_mm": float(water_sheet["H25"].value),
            "r_ov_ref_mm": float(water_sheet["B14"].value),
            "generated_ov_events": float(ov_sheet["A16"].value),
            "isotopes": (
                {
                    "name": "Th232",
                    "activity_bq_per_kg": float(water_sheet["F14"].value),
                    "activity_error_bq_per_kg": float(water_sheet["G14"].value),
                    "ov_ref_efficiency": float(ov_sheet["I10"].value),
                    "branch_fraction": branch_th,
                },
                {
                    "name": "U238",
                    "activity_bq_per_kg": float(water_sheet["F15"].value),
                    "activity_error_bq_per_kg": float(water_sheet["G15"].value),
                    "ov_ref_efficiency": float(ov_sheet["I11"].value),
                    "branch_fraction": branch_u,
                },
                {
                    "name": "Rn222",
                    "activity_bq_per_kg": rn_activity_bq_per_kg,
                    "activity_error_bq_per_kg": rn_activity_error_bq_per_kg,
                    "ov_ref_efficiency": float(ov_sheet["I11"].value),
                    "branch_fraction": branch_u,
                },
            ),
        }
    finally:
        workbook.close()

    return inputs


def water_mass_kg_from_ov_radius(r_ov_mm: float, inputs: dict[str, object]) -> float:
    tank_radius_m = float(inputs["tank_radius_mm"]) / 1000.0
    tank_height_m = float(inputs["tank_height_mm"]) / 1000.0
    ov_volume_m3 = (4.0 / 3.0) * math.pi * (r_ov_mm / 1000.0) ** 3
    tank_volume_m3 = math.pi * tank_radius_m**2 * tank_height_m
    return float(inputs["density_water_kg_per_m3"]) * (tank_volume_m3 - ov_volume_m3)


def water_mean_factor_from_ov_radius(r_ov_mm: float, inputs: dict[str, object]) -> float:
    tank_radius_mm = float(inputs["tank_radius_mm"])
    tank_height_mm = float(inputs["tank_height_mm"])
    mu_water_mm = float(inputs["mu_water_mm"])

    radial_step = tank_radius_mm / WATER_MODEL_RADIAL_BINS
    axial_step = tank_height_mm / WATER_MODEL_AXIAL_BINS
    radial_centres = (np.arange(WATER_MODEL_RADIAL_BINS) + 0.5) * radial_step
    axial_centres = (
        -tank_height_mm / 2.0
        + (np.arange(WATER_MODEL_AXIAL_BINS) + 0.5) * axial_step
    )
    radius_mesh, height_mesh = np.meshgrid(radial_centres, axial_centres, indexing="xy")
    distance = np.sqrt(radius_mesh**2 + height_mesh**2)
    mask = distance >= r_ov_mm

    geometry_factor = np.zeros_like(distance)
    geometry_factor[mask] = (r_ov_mm**2) / (distance[mask] ** 2)

    attenuation = np.ones_like(distance)
    attenuation[mask] = np.exp(-mu_water_mm * (distance[mask] - r_ov_mm))

    volume_element = 2.0 * math.pi * radius_mesh * radial_step * axial_step
    integral = float(np.sum(geometry_factor * attenuation * volume_element))
    water_volume = (
        math.pi * tank_radius_mm**2 * tank_height_mm
        - (4.0 / 3.0) * math.pi * r_ov_mm**3
    )
    return integral / water_volume


def evaluate_water_model_vs_ov_radius(
    ov_radius_mm,
    inputs: dict[str, object],
    include_rn: bool,
) -> dict[str, np.ndarray]:
    ov_radius_mm = np.asarray(ov_radius_mm, float)
    screening = np.exp(
        -float(inputs["mu_hfe_mm"]) * (IV_RADIUS_MM - float(inputs["r_iv_ref_mm"]))
    )

    total_curve = np.zeros_like(ov_radius_mm)
    total_sigma_sq = np.zeros_like(ov_radius_mm)
    mean_factor = np.zeros_like(ov_radius_mm)
    water_mass_kg = np.zeros_like(ov_radius_mm)

    for index, radius in enumerate(ov_radius_mm):
        mean_factor[index] = water_mean_factor_from_ov_radius(float(radius), inputs)
        water_mass_kg[index] = water_mass_kg_from_ov_radius(float(radius), inputs)
        geometric_scale = (float(inputs["r_ov_ref_mm"]) / float(radius)) ** 2

        for isotope in inputs["isotopes"]:
            if (not include_rn) and isotope["name"] == "Rn222":
                continue

            efficiency = (
                isotope["ov_ref_efficiency"]
                * geometric_scale
                * screening
                * mean_factor[index]
            )
            efficiency_error = _ov_efficiency_uncertainty(
                efficiency,
                isotope["branch_fraction"],
                float(inputs["generated_ov_events"]),
                isotope["ov_ref_efficiency"],
            )
            total_curve[index] += _truncated_gaussian_mean(
                water_mass_kg[index],
                isotope["activity_bq_per_kg"],
                isotope["activity_error_bq_per_kg"],
                efficiency,
            )
            sigma = _truncated_gaussian_spread(
                water_mass_kg[index],
                isotope["activity_bq_per_kg"],
                isotope["activity_error_bq_per_kg"],
                efficiency,
                efficiency_error,
            )
            total_sigma_sq[index] += sigma**2

    total_sigma = np.sqrt(total_sigma_sq)
    lower_band = np.clip(total_curve - Z_BAND * total_sigma, 1e-300, np.inf)
    upper_band = total_curve + Z_BAND * total_sigma
    return {
        "curve": total_curve,
        "sigma": total_sigma,
        "lo": lower_band,
        "hi": upper_band,
        "mean_factor": mean_factor,
        "water_mass_kg": water_mass_kg,
    }


def main() -> None:
    inputs = load_water_model_inputs(XLSX_PATH)
    model_no_rn = evaluate_water_model_vs_ov_radius(R_OV_GRID, inputs, include_rn=False)
    model_with_rn = evaluate_water_model_vs_ov_radius(R_OV_GRID, inputs, include_rn=True)

    baseline_no_rn = evaluate_water_model_vs_ov_radius(
        np.array([BASELINE_OV_RADIUS_MM]),
        inputs,
        include_rn=False,
    )
    baseline_with_rn = evaluate_water_model_vs_ov_radius(
        np.array([BASELINE_OV_RADIUS_MM]),
        inputs,
        include_rn=True,
    )
    reference_no_rn = build_water_semi_analytical_curve_with_bands(
        radius_grid_mm=np.array([IV_RADIUS_MM]),
        include_rn=False,
    )
    reference_with_rn = build_water_semi_analytical_curve_with_bands(
        radius_grid_mm=np.array([IV_RADIUS_MM]),
        include_rn=True,
    )

    if not (
        np.isclose(baseline_no_rn["curve"][0], reference_no_rn["curve"][0], rtol=0.0, atol=1e-15)
        and np.isclose(baseline_no_rn["sigma"][0], reference_no_rn["sigma"][0], rtol=0.0, atol=1e-15)
    ):
        raise ValueError("OV-radius water model disagrees with water_only at the baseline no-Rn point.")
    if not (
        np.isclose(baseline_with_rn["curve"][0], reference_with_rn["curve"][0], rtol=0.0, atol=1e-15)
        and np.isclose(baseline_with_rn["sigma"][0], reference_with_rn["sigma"][0], rtol=0.0, atol=1e-15)
    ):
        raise ValueError("OV-radius water model disagrees with water_only at the baseline with-Rn point.")

    hfe_mass_t = float(hfe_mass_tonnes_from_radius_mm(IV_RADIUS_MM))
    hfe_mass_label_t = int(np.rint(hfe_mass_t))

    probe_radii = np.array([1765.0, BASELINE_OV_RADIUS_MM])
    probe_no_rn = evaluate_water_model_vs_ov_radius(probe_radii, inputs, include_rn=False)
    probe_with_rn = evaluate_water_model_vs_ov_radius(probe_radii, inputs, include_rn=True)
    print(f"IV radius fixed at {IV_RADIUS_MM:.0f} mm -> HFE mass = {hfe_mass_t:.2f} t")
    print(
        "Consistency check against water_only baseline: "
        f"no Rn = {baseline_no_rn['curve'][0]:.6e} +/- {baseline_no_rn['sigma'][0]:.6e}, "
        f"with Rn = {baseline_with_rn['curve'][0]:.6e} +/- {baseline_with_rn['sigma'][0]:.6e} [OK]"
    )
    for radius, value, sigma, value_rn, sigma_rn in zip(
        probe_radii,
        probe_no_rn["curve"],
        probe_no_rn["sigma"],
        probe_with_rn["curve"],
        probe_with_rn["sigma"],
    ):
        print(
            f"OV radius {radius:7.1f} mm -> "
            f"Water Th/U = {value:.6e} +/- {sigma:.6e}, "
            f"Water Th/U+Rn = {value_rn:.6e} +/- {sigma_rn:.6e}"
        )

    fig, axis = plt.subplots(figsize=(10, 7))
    curve_specs = [
        ("C1", "Water (semi-analytical): Th/U", model_no_rn),
        ("C2", "Water (semi-analytical): Th/U + Rn", model_with_rn),
    ]
    for color, label, model in curve_specs:
        axis.semilogy(
            R_OV_GRID,
            model["curve"],
            "--",
            linewidth=1.4,
            alpha=0.95,
            color=color,
            label=label,
        )
        axis.fill_between(
            R_OV_GRID,
            model["lo"],
            model["hi"],
            color=color,
            alpha=0.15,
            linewidth=0,
            label="_nolegend_",
        )

    axis.plot(
        [BASELINE_OV_RADIUS_MM],
        [TOTAL_INTRINSIC_BACKGROUND],
        marker="s",
        ms=6,
        color="0.15",
        markerfacecolor="none",
        linestyle="None",
        label="_nolegend_",
    )
    axis.annotate(
        "total intrinsic background",
        xy=(BASELINE_OV_RADIUS_MM, TOTAL_INTRINSIC_BACKGROUND),
        xytext=(-10, -2),
        textcoords="offset points",
        ha="right",
        va="center",
        fontsize=FS_TICK - 2,
        color="0.15",
    )

    axis.axvline(
        BASELINE_OV_RADIUS_MM,
        color="0.3",
        linestyle=(0, (4, 2)),
        linewidth=1.5,
    )
    axis.text(
        BASELINE_OV_RADIUS_MM,
        0.03,
        "Baseline Design",
        rotation=90,
        ha="right",
        va="bottom",
        transform=axis.get_xaxis_transform(),
        fontsize=FS_TICK,
        color="0.3",
    )
    axis.text(
        0.03,
        0.03,
        f"HFE mass = {hfe_mass_label_t:d} t",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=FS_TICK - 1,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.85, "edgecolor": "0.8"},
    )

    axis.set_xlabel("OV radius (mm)", fontsize=FS_LABEL)
    axis.set_ylabel("Background rate [cts/(y·2t·FWHM)]", fontsize=FS_LABEL)
    axis.set_yscale("log")
    axis.set_xlim(float(R_OV_GRID.min()), float(R_OV_GRID.max()))
    axis.set_ylim(1e-5, 1.0)
    axis.tick_params(axis="both", which="major", labelsize=FS_TICK)
    axis.grid(True, which="both", linestyle=":", alpha=0.4)
    axis.legend(fontsize=FS_LEGEND, loc="upper left")

    plt.tight_layout()
    output_dir = BUDGET_DIR / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{Path(__file__).stem}.png"
    plt.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"\nSaved to {output_path}")


if __name__ == "__main__":
    main()
