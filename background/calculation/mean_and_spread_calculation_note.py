#!/usr/bin/env python3
"""
Background contribution and TG-spread calculation
=================================================
Implements Eq. (3)–(4) of *nX_RadBackground_21.pdf* using the symbols
exactly as defined in the note.
    μ      – expected counts s⁻¹ before truncation  (m · a · ε)
    σ      – propagated standard deviation of μ
    z      – normal-score  μ/σ
    φ(z)   – standard-normal PDF
    Φ(z)   – standard-normal CDF
    λ(z)   – φ(z)/Φ(z)
    B      – truncated-Gaussian mean     (Eq. 3)
    σ_B    – truncated-Gaussian spread   (Eq. 4)
All results are returned in *counts year⁻¹*.
"""

from __future__ import annotations
import math
from typing import List, Tuple, Dict

SECONDS_PER_YEAR = 86_400 * 365.25
EPS: float = 1e-15  # numerical safety threshold

# ──────────────────────────────────
# Standard-normal helpers
# ──────────────────────────────────

def phi(z: float) -> float:
    """Standard-normal probability density φ(z)."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def Phi(z: float) -> float:
    """Standard-normal cumulative distribution Φ(z)."""
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

# ──────────────────────────────────
# Truncated Gaussian moments (note Eq. 3–4)
# ──────────────────────────────────

def truncated_gaussian_mean(mu: float, sigma: float) -> float:
    """Return B (Eq. 3) – mean of a Gaussian truncated at 0 (counts s⁻¹)."""
    if sigma < EPS:  # zero uncertainty ⇒ no truncation
        return max(0.0, mu)

    z = mu / sigma
    Phi_z = Phi(z)
    if Phi_z < EPS:  # practically all mass below zero ⇒ mean ≈ 0
        return 0.0

    lam = phi(z) / Phi_z  # λ(z)
    return mu + sigma * lam

def truncated_gaussian_spread(mu: float, sigma: float) -> float:
    """Return σ_B (Eq. 4) – TG-spread of a Gaussian truncated at 0 (counts s⁻¹)."""
    if sigma < EPS:
        return 0.0

    z = mu / sigma
    Phi_z = Phi(z)
    if Phi_z < EPS:
        return 0.0

    lam = phi(z) / Phi_z  # λ(z)
    inner = 1.0 - z * lam - lam * lam
    return sigma * math.sqrt(max(inner, 0.0))

# ──────────────────────────────────
# Statistical efficiency uncertainty (binomial, 1 σ)
# ──────────────────────────────────

def efficiency_stat_error(mu_eff: float, n_decays: int, branching: float = 1.0) -> float:
    """Eq. σ_ε = √(ε N β)/N for ε>0; 0 when ε == 0 (pure UL handled elsewhere)."""
    return math.sqrt(mu_eff * n_decays * branching) / n_decays if mu_eff > 0 else 0.0

# ──────────────────────────────────
# High-level driver
# ──────────────────────────────────
Component = Tuple[str, float, float, float, float | None, float, float]
ResultDict = Dict[str, Dict[str, float]]

def calc_component(mass: float,
                   activity: float, activity_err: float,
                   efficiency: float, efficiency_err: float) -> Tuple[float, float]:
    """Compute ⟨BG⟩ and TG-spread for one isotope (counts y⁻¹)."""
    # Step 1 – propagate uncertainties to μ, σ  (counts s⁻¹)
    mu     = mass * activity * efficiency
    sigma  = math.hypot(mass * efficiency * activity_err,
                        mass * activity   * efficiency_err)

    # Step 2 – truncate Gaussian at zero
    B_sec     = truncated_gaussian_mean(mu, sigma)
    sigmaB_sec = truncated_gaussian_spread(mu, sigma)

    # Step 3 – convert to counts y⁻¹
    return B_sec * SECONDS_PER_YEAR, sigmaB_sec * SECONDS_PER_YEAR

def run_calculation(mass: float, components: List[Component]) -> ResultDict:
    """Return dict with per-isotope and total background / TG-spread."""
    results: ResultDict = {}

    for (name, act, act_err, eff, eff_err, n_decays, branch) in components:
        # Derive σ_ε from counting statistics if not provided
        if eff_err is None:
            eff_err = efficiency_stat_error(eff, n_decays, branch)

        mean_y, spread_y = calc_component(mass, act, act_err, eff, eff_err)
        results[name] = {
            "mean":   mean_y,
            "spread": spread_y,
            "eff_err": eff_err,
            "mu":     eff,
        }

    # Quadrature sum for the total
    total_mean   = sum(v["mean"]   for v in results.values())
    total_spread = math.sqrt(sum(v["spread"]**2 for v in results.values()))
    results["total"] = {"mean": total_mean, "spread": total_spread}

    return results

# ──────────────────────────────────
# Example utilisation – runs when executed directly
# ──────────────────────────────────
if __name__ == "__main__":
    COMPONENT_MASS = 1681.6  # kg

    # Tuple format:
    # (name, activity [Bq/kg], σ_activity, ε, σ_ε (None ⇒ statistical), N_decays, branching)
    iv = [
        ("th232", 2.65e-7, 1.09e-7, 2.19e-9, None, 1e10, 0.3594),
        ("u238",  0.0,      7.42e-7, 8.00e-10, None, 1e10, 1.0),
    ]

    res = run_calculation(COMPONENT_MASS, iv)

    # Nicely formatted console output
    print("\n--- Background results (truncated Gaussian) ---")
    print(f"Component mass: {COMPONENT_MASS:.1f} kg\n")
    print(f"{'Isotope':<8} {'μ (eff)':>12} {'σ_ε':>12} {'⟨BG⟩ [cnt/y]':>16} {'± TG':>10}")

    for name, vals in res.items():
        if name == "total":
            continue
        print(f"{name:<8} {vals['mu']:12.3e} {vals['eff_err']:12.3e} {vals['mean']:16.3e} {vals['spread']:10.3e}")

    total = res["total"]
    print("\nTotal background: "
          f"{total['mean']:.3e} ± {total['spread']:.3e} counts/y (quadrature)")
