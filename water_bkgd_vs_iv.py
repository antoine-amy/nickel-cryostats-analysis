#!/usr/bin/env python3
"""
Water background from U-238, Th-232, Rn-222 using the defined hit-efficiency model:

  ε_water(r_IV) = ε_OV · S_HFE(r_IV) · F
    F = (2π/V_water) ∬ g(s)·exp[-μ_water (s - r_OV)] r dr dz ,
    g(s)=(s_OV/s)^2 for s≥r_OV else 0

Uncertainty (theory case): propagate ONLY the OV-baseline error
  σ[ε_water] = ε_water · sqrt( tau / (ε_OV · N_OV) )

Compute TG-mean & TG-spread and plot vs IV radius.
All geometry in mm; attenuations in mm^-1 (tank dims entered in meters).
"""

from __future__ import annotations
import math
from typing import Tuple
import numpy as np
import matplotlib.pyplot as plt

SECONDS_PER_YEAR = 86400.0 * 365.25
_TINY = 1e-15

# ----------------------------- efficiency model -----------------------------
def mean_factor_F(mu_water_mm: float,
                  r_OV_mm: float,
                  r_tank_mm: float,
                  h_tank_mm: float,
                  s_OV_mm: float,
                  nr: int = 120,
                  nz: int = 120) -> float:
    """Dimensionless volume-averaged factor F (mm units)."""
    V_tank = math.pi * (r_tank_mm**2) * h_tank_mm
    V_OV   = (4.0/3.0) * math.pi * (r_OV_mm**3)
    V_water = V_tank - V_OV
    if V_water <= 0.0:
        raise ValueError("Non-positive water volume; check geometry.")

    # Midpoint grid in cylindrical (r, z)
    dr = r_tank_mm / nr
    dz = h_tank_mm / nz
    r_cent = (np.arange(nr) + 0.5) * dr
    z_cent = (-h_tank_mm/2.0) + (np.arange(nz) + 0.5) * dz
    R, Z = np.meshgrid(r_cent, z_cent, indexing="xy")  # (nz, nr)

    S = np.sqrt(R*R + Z*Z)
    mask = (S >= r_OV_mm)

    # g(s) and attenuation
    G = np.zeros_like(S)
    with np.errstate(divide="ignore", invalid="ignore"):
        G[mask] = (s_OV_mm*s_OV_mm) / (S[mask]*S[mask])
    Attn = np.ones_like(S)
    Attn[mask] = np.exp(-mu_water_mm * (S[mask] - r_OV_mm))

    integrand = G * Attn
    dV = (2.0 * math.pi) * R * dr * dz
    IINT = float(np.sum(integrand * dV))
    return IINT / V_water


def S_HFE(mu_hfe_mm: float, r_IV_mm: float, R_IV_ref_mm: float, T_HFE_mm: float) -> float:
    """
    S_HFE = exp[-μ_HFE · (t_HFE − T_HFE)],
    t_HFE = max(0, T_HFE − (R_IV_ref − r_IV)).
    (S_HFE = 1 at r_IV = R_IV_ref)
    """
    t_hfe = max(0.0, T_HFE_mm - (R_IV_ref_mm - r_IV_mm))
    return math.exp(-mu_hfe_mm * (t_hfe - T_HFE_mm))


def epsilon_water(eps0_OV: float,
                  r_IV_mm: float,
                  *,
                  mu_water_mm: float,
                  mu_hfe_mm: float,
                  r_OV_mm: float,
                  r_tank_m: float,
                  h_tank_m: float,
                  T_HFE_mm: float,
                  R_IV_ref_mm: float,
                  s_OV_mm: float = 2230.0,
                  nr: int = 120,
                  nz: int = 120) -> float:
    """Mean water hit efficiency from OV baseline at a given IV radius."""
    r_tank_mm = r_tank_m * 1000.0
    h_tank_mm = h_tank_m * 1000.0
    F = mean_factor_F(mu_water_mm, r_OV_mm, r_tank_mm, h_tank_mm, s_OV_mm, nr, nz)
    S = S_HFE(mu_hfe_mm, r_IV_mm, R_IV_ref_mm, T_HFE_mm)
    return eps0_OV * F * S


def sigma_epsilon_water(eps_water: float, eps0_OV: float, tau: float, N_OV: float) -> float:
    """
    OV-baseline-only uncertainty propagation:
      σ(ε_w) = ε_w * sqrt( tau / (ε_OV * N_OV) )
    """
    if eps0_OV <= 0.0 or N_OV <= 0.0:
        return 0.0
    return eps_water * math.sqrt(tau / (eps0_OV * N_OV))


# ----------------------------- TG for background -----------------------------
def tg_mean_spread_counts_per_year(mass_kg: float,
                                   activity_bq_per_kg: float,
                                   activity_err_bq_per_kg: float,
                                   eff_per_decay: float,
                                   eff_err_per_decay: float) -> Tuple[float, float]:
    """
    Truncated-Gaussian mean and 'TG-spread' (1σ) for a non-negative rate, in counts/year.
    Uses the λ = φ/Φ form:  μ_TG = t + u λ,  σ_TG^2 = u^2 (1 - z λ - λ^2).
    """
    t = mass_kg * activity_bq_per_kg * eff_per_decay
    u = math.hypot(mass_kg * eff_per_decay * activity_err_bq_per_kg,
                   mass_kg * activity_bq_per_kg * eff_err_per_decay)

    if u < _TINY:
        return max(0.0, t) * SECONDS_PER_YEAR, 0.0

    z = t / u
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    mean_sec = (t + u * (pdf / max(cdf, _TINY))) if cdf > _TINY else 0.0
    mean_sec = max(0.0, mean_sec)

    lam = pdf / max(cdf, _TINY)             # λ = φ/Φ
    inner = 1.0 - z * lam - lam * lam       # σ_TG^2 / u^2
    inner = max(inner, 0.0)                 # guard tiny negatives from rounding
    spread_sec = u * math.sqrt(inner)

    return mean_sec * SECONDS_PER_YEAR, spread_sec * SECONDS_PER_YEAR


# ----------------------------- drivers -----------------------------
def compute_background_at_radius(r_IV_mm: float,
                                 *,
                                 # Geometry / materials
                                 r_OV_mm: float,
                                 r_tank_m: float,
                                 h_tank_m: float,
                                 rho_water_kg_per_m3: float,
                                 mu_water_mm: float,
                                 mu_hfe_mm: float,
                                 T_HFE_mm: float,
                                 R_IV_ref_mm: float,
                                 # OV baselines & MC settings
                                 eps0_U: float,
                                 tau_U: float,
                                 N_OV: float,
                                 # Activities (Bq/kg) and 1σ
                                 act_U: float,  act_U_err: float,
                                 act_Rn: float, act_Rn_err: float) -> Tuple[float, float]:
    """
    Returns (total_mean_cnt/yr, total_spread_cnt/yr) at a given r_IV.
    """
    # Water mass (kg): tank cylinder minus OV sphere
    V_tank_m3 = math.pi * (r_tank_m**2) * h_tank_m
    V_OV_m3 = (4.0/3.0) * math.pi * (r_OV_mm/1000.0)**3
    m_water = (V_tank_m3 - V_OV_m3) * rho_water_kg_per_m3

    # Efficiencies ε_w(r_IV)
    eps_w_U  = epsilon_water(eps0_U,  r_IV_mm, mu_water_mm=mu_water_mm, mu_hfe_mm=mu_hfe_mm,
                             r_OV_mm=r_OV_mm, r_tank_m=r_tank_m, h_tank_m=h_tank_m,
                             T_HFE_mm=T_HFE_mm, R_IV_ref_mm=R_IV_ref_mm)
    # Rn uses same ε as U for detection
    eps_w_Rn = eps_w_U

    # Efficiency 1σ from OV baseline (only source by spec)
    sig_eps_U  = sigma_epsilon_water(eps_w_U,  eps0_U,  tau_U,  N_OV)
    sig_eps_Rn = sig_eps_U

    # TG backgrounds (counts/year)
    mean_U,  spread_U  = tg_mean_spread_counts_per_year(m_water, act_U,  act_U_err,  eps_w_U,  sig_eps_U)
    mean_Rn, spread_Rn = tg_mean_spread_counts_per_year(m_water, act_Rn, act_Rn_err, eps_w_Rn, sig_eps_Rn)

    total_mean = mean_U + mean_Rn
    total_spread = math.sqrt(spread_U**2 + spread_Rn**2)
    return total_mean, total_spread




# ----------------------------- main -----------------------------
def main() -> None:
    # ---------------- Inputs (adjust here as needed) ----------------
    # Geometry / materials
    r_OV_mm   = 2230.0
    r_tank_m  = 12.3 / 2.0   # tank radius (m)
    h_tank_m  = 12.8         # tank height (m)
    rho_water = 1000.0       # kg/m^3
    mu_water_mm = 0.0045     # ~2.5 MeV γ
    mu_hfe_mm   = 0.00592496
    T_HFE_mm    = 760.0      # 76 cm
    R_IV_ref_mm = 1691.0

    # OV baselines and branching
    eps0_U  = 2.00e-10
    tau_U = 1.0
    N_OV = 1e10

    # Activities (Bq/kg) ±1σ
    act_U,  act_U_err  = 76e-9, 13e-9
    act_Rn, act_Rn_err = 63e-9, 22e-9

    # Quick check at spreadsheet small-IV point
    r_IV_sheet_mm = 1026.0
    F = mean_factor_F(mu_water_mm, r_OV_mm, r_tank_m*1000.0, h_tank_m*1000.0, 2230.0, 120, 120)
    S = S_HFE(mu_hfe_mm, r_IV_sheet_mm, R_IV_ref_mm, T_HFE_mm)
    eps_water_U  = eps0_U  * F * S
    print(f"[Check @ r_IV={r_IV_sheet_mm:.0f} mm]  F={F:.6e}  S_HFE={S:.6e}")
    print(f"  ε_water(U)  = {eps_water_U:.6e}")

    # Background at reference IV (total)
    mean_ref, spread_ref = compute_background_at_radius(
        R_IV_ref_mm,
        r_OV_mm=r_OV_mm, r_tank_m=r_tank_m, h_tank_m=h_tank_m,
        rho_water_kg_per_m3=rho_water, mu_water_mm=mu_water_mm, mu_hfe_mm=mu_hfe_mm,
        T_HFE_mm=T_HFE_mm, R_IV_ref_mm=R_IV_ref_mm,
        eps0_U=eps0_U, tau_U=tau_U, N_OV=N_OV,
        act_U=act_U, act_U_err=act_U_err,
        act_Rn=act_Rn, act_Rn_err=act_Rn_err
    )
    print("Water background at reference IV radius (TOTAL):")
    print(f"  Total mean  : {mean_ref:.3e} counts/year")
    print(f"  Total 1σ TG : {spread_ref:.3e} counts/year\n")

    # Sweep IV radii and plot
    r_vals = np.linspace(1000.0, 1700.0, 71)  # mm

    # Total (U + Rn only)
    means_total = np.zeros_like(r_vals)
    spreads_total = np.zeros_like(r_vals)

    for i, r_iv in enumerate(r_vals):
        m_tot, s_tot = compute_background_at_radius(
            r_iv,
            r_OV_mm=r_OV_mm, r_tank_m=r_tank_m, h_tank_m=h_tank_m,
            rho_water_kg_per_m3=rho_water, mu_water_mm=mu_water_mm, mu_hfe_mm=mu_hfe_mm,
            T_HFE_mm=T_HFE_mm, R_IV_ref_mm=R_IV_ref_mm,
            eps0_U=eps0_U, tau_U=tau_U, N_OV=N_OV,
            act_U=act_U, act_U_err=act_U_err,
            act_Rn=act_Rn, act_Rn_err=act_Rn_err
        )
        means_total[i] = m_tot
        spreads_total[i] = s_tot

    upper_total = means_total + spreads_total
    lower_total = np.maximum(means_total - spreads_total, 1e-20)

    # Hard-coded comparison points (MC points updated)
    mc_r   = [1026.0, 1691.0]
    mc_bg  = [8.10e-4, 1.23e-6]
    mc_err = [4.43e-4, 4.07e-5]

    fig, ax = plt.subplots(figsize=(10, 7))

    # Total theory curve (U + Rn only)
    ax.plot(r_vals, means_total, linestyle="--", label="Theoretical background")
    ax.fill_between(r_vals, lower_total, upper_total, alpha=0.15)

    # MC points
    ax.errorbar(mc_r, mc_bg, yerr=mc_err, fmt="o", color="red", capsize=5,
                label="Monte Carlo")

    ax.set_xlabel("IV radius (mm)")
    ax.set_ylabel("Background (counts/year)")
    ax.set_title("Water Background vs IV Radius")
    ax.set_yscale("log")
    ax.grid(True, which="both", ls=":")
    ax.legend(loc="upper right")


    fig.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()