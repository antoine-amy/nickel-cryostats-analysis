#!/usr/bin/env python3
"""
Mean water hit efficiency with ONLY OV-baseline uncertainty.

Model (mm units for geometry, mm^-1 for attenuations):

  ε_water = ε_OV · S_HFE(r_IV) · F
  with
    F = (2π / V_water) ∬ g(s) · exp[-μ_water (s - r_OV)] · r dr dz
    s = sqrt(r^2 + z^2),
    g(s) = (s_OV / s)^2 for s ≥ r_OV, else 0,

  V_water = π r_tank^2 h_tank − (4/3) π r_OV^3

  HFE scaling:
    t_HFE = max(0, T_HFE − (R_IV_ref − r_IV))
    S_HFE = exp[ − μ_HFE · (t_HFE − T_HFE) ]

Uncertainty:
  σ(ε_water) = ε_water · [ σ(ε_OV) / ε_OV ]
"""

from dataclasses import dataclass
import math
from typing import Tuple

import numpy as np


@dataclass
class Inputs:
    """Deterministic inputs (match the spreadsheet)."""
    eps0: float          # OV baseline efficiency (per parent decay)
    r_ov_mm: float       # OV radius (mm)
    r_iv_mm: float       # IV radius (mm)
    r_tank_m: float      # TANK RADIUS (m)  ← pass radius (not diameter)
    h_tank_m: float      # tank height (m)
    mu_water_mm: float   # water linear attenuation (1/mm)
    mu_hfe_mm: float     # HFE linear attenuation (1/mm)
    t_hfe_mm: float      # default HFE thickness (mm)
    r_iv_ref_mm: float   # default/reference IV radius (mm)
    s_ov_mm: float = 2230.0  # OV anchor for g(s)
    nr: int = 160        # radial bins
    nz: int = 160        # vertical bins


def _mean_factor_f(
    mu_water_mm: float,
    r_ov_mm: float,
    r_tank_mm: float,
    h_tank_mm: float,
    s_ov_mm: float,
    nr: int,
    nz: int,
) -> float:
    """Volume-averaged integral factor F (dimensionless), mm-units."""
    v_tank = math.pi * (r_tank_mm**2) * h_tank_mm
    v_ov = (4.0 / 3.0) * math.pi * (r_ov_mm**3)
    v_water = v_tank - v_ov
    if v_water <= 0.0:
        raise ValueError("Non-positive water volume; check geometry.")

    # Midpoint grid
    dr = r_tank_mm / nr
    dz = h_tank_mm / nz
    r_cent = (np.arange(nr) + 0.5) * dr
    z_cent = (-h_tank_mm / 2.0) + (np.arange(nz) + 0.5) * dz

    R, Z = np.meshgrid(r_cent, z_cent, indexing="xy")  # shapes (nz, nr)

    S = np.sqrt(R * R + Z * Z)
    mask = S >= r_ov_mm

    # g(s) = (s_OV/s)^2 on water
    G = np.zeros_like(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        G[mask] = (s_ov_mm * s_ov_mm) / (S[mask] * S[mask])

    # Attenuation
    Attn = np.ones_like(S)
    Attn[mask] = np.exp(-mu_water_mm * (S[mask] - r_ov_mm))

    integrand = G * Attn
    dV = (2.0 * math.pi) * R * dr * dz
    iint = float(np.sum(integrand * dV))
    return iint / v_water


def _s_hfe(mu_hfe_mm: float, r_iv_mm: float, r_iv_ref_mm: float, t_hfe_mm: float) -> float:
    """S_HFE = exp[-μ_HFE · (t_HFE − T_HFE)], with t_HFE = max(0, T_HFE − (R_IV_ref − r_IV))."""
    t_hfe = max(0.0, t_hfe_mm - (r_iv_ref_mm - r_iv_mm))
    delta = t_hfe - t_hfe_mm
    return math.exp(-mu_hfe_mm * delta)


def water_efficiency_and_sigma(inp: Inputs, sigma_eps0: float) -> Tuple[float, float]:
    """
    Return (ε_water, σ_ε_water) using ONLY OV baseline uncertainty.
    """
    r_tank_mm = inp.r_tank_m * 1000.0
    h_tank_mm = inp.h_tank_m * 1000.0

    factor = _mean_factor_f(
        mu_water_mm=inp.mu_water_mm,
        r_ov_mm=inp.r_ov_mm,
        r_tank_mm=r_tank_mm,
        h_tank_mm=h_tank_mm,
        s_ov_mm=inp.s_ov_mm,
        nr=inp.nr,
        nz=inp.nz,
    )
    s_hfe = _s_hfe(inp.mu_hfe_mm, inp.r_iv_mm, inp.r_iv_ref_mm, inp.t_hfe_mm)

    eps_water = inp.eps0 * factor * s_hfe
    if inp.eps0 <= 0.0:
        raise ValueError("eps0 must be positive to propagate its relative uncertainty.")
    sigma = abs(eps_water) * (sigma_eps0 / inp.eps0)
    return eps_water, sigma


# ---------------- Example ----------------
if __name__ == "__main__":
    # Geometry / materials
    R_OV_MM = 2230.0
    R_IV_MM = 1691.0
    R_TANK_M = 12.3 / 2.0
    H_TANK_M = 12.8
    MU_WATER_MM = 0.0045    # ~2.5 MeV gamma
    MU_HFE_MM = 0.00592496
    T_HFE_MM = 760.0
    R_IV_REF_MM = 1691.0

    # OV baselines (per parent decay) and their 1σ
    EPS0_U = 2.00e-10
    EPS0_TH = 1.2579e-09
    SIG_EPS0_U = math.sqrt((EPS0_U * 1)/1e10)
    SIG_EPS0_TH = math.sqrt((EPS0_U * 0.3594)/1e10)

    common = dict(
        r_ov_mm=R_OV_MM,
        r_iv_mm=R_IV_MM,
        r_tank_m=R_TANK_M,
        h_tank_m=H_TANK_M,
        mu_water_mm=MU_WATER_MM,
        mu_hfe_mm=MU_HFE_MM,
        t_hfe_mm=T_HFE_MM,
        r_iv_ref_mm=R_IV_REF_MM,
    )

    inp_u = Inputs(EPS0_U, R_OV_MM, R_IV_MM, R_TANK_M, H_TANK_M,
                   MU_WATER_MM, MU_HFE_MM, T_HFE_MM, R_IV_REF_MM)
    inp_th = Inputs(EPS0_TH, R_OV_MM, R_IV_MM, R_TANK_M, H_TANK_M,
                    MU_WATER_MM, MU_HFE_MM, T_HFE_MM, R_IV_REF_MM)

    eps_u, sig_u = water_efficiency_and_sigma(inp_u, SIG_EPS0_U)
    eps_th, sig_th = water_efficiency_and_sigma(inp_th, SIG_EPS0_TH)

    print(f"U-238  ε_water = {eps_u:.6e} ± {sig_u:.2e}  (rel {sig_u/eps_u:.2%})")
    print(f"Th-232 ε_water = {eps_th:.6e} ± {sig_th:.2e} (rel {sig_th/eps_th:.2%})")
