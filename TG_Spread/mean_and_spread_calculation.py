#!/usr/bin/env python3
"""Background contribution with one-sided truncated Gaussian (TG) moments.

Formulas (lower truncation at 0):
  B      = μ + σ * λ(z)
  σ_B    = σ * sqrt(1 - z*λ - λ^2)
where z = μ/σ and λ(z) = φ(z)/Φ(z) (standard-normal PDF/CDF ratio).
"""

import math

# ── Constants ──────────────────────────────────────────────────────────────
SECONDS_PER_YEAR = 86_400 * 365.25
EPS = 1e-15
UL_FACTOR = 1.14  # one-sided 68% UL for zero-efficiency case
USE_FULL_TG_FOR_ZERO_EFFICIENCY = False  # toggle

# ── Helpers: standard normal pieces ────────────────────────────────────────
def std_norm_pdf(z: float) -> float:
    """Standard-normal PDF φ(z)."""
    return math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)

def std_norm_cdf(z: float) -> float:
    """Standard-normal CDF Φ(z), numerically stable via erfc."""
    return 0.5 * math.erfc(-z / math.sqrt(2.0))

def mills_ratio_lower(z: float) -> float:
    """
    Lower-tail inverse Mills ratio λ(z) = φ(z) / Φ(z).
    Uses an asymptotic expansion for large negative z to avoid underflow.
    """
    if z < -6.0:
        t = -z  # t > 0
        # Asymptotic: λ ~ t + 1/t + 2/t^3 + 5/t^5  (accurate to ~1e-6 by z≈-6)
        return t + 1.0 / t + 2.0 / (t**3) + 5.0 / (t**5)
    # Otherwise compute directly
    phi = std_norm_pdf(z)
    cdf_phi = std_norm_cdf(z)
    # Guard against division by zero in extreme tails
    return phi / max(cdf_phi, 1e-300)

# ── TG moments at truncation 0 ────────────────────────────────────────────
def tg_mean(mu: float, sigma: float) -> float:
    """Truncated-Gaussian mean B (counts s⁻¹) for truncation at 0."""
    if sigma < EPS:
        return max(0.0, mu)
    z = mu / sigma
    lam = mills_ratio_lower(z)
    return mu + sigma * lam

def tg_spread(mu: float, sigma: float) -> float:
    """Truncated-Gaussian spread σ_B (counts s⁻¹) for truncation at 0."""
    if sigma < EPS:
        return 0.0
    z = mu / sigma
    lam = mills_ratio_lower(z)
    inner = 1.0 - z * lam - lam * lam
    # Clamp small negative due to rounding
    return sigma * math.sqrt(inner) if inner > 0.0 else 0.0

# ── Binomial/statistical efficiency error ─────────────────────────────────
def efficiency_stat_error(epsilon: float, num_decays: float, branching: float = 1.0) -> float:
    """
    σ_ε from counting stats: sqrt(ε * b / N).
    If ε == 0, optionally return 1σ UL ≈ 1.14 * b / N when toggle is ON; else 0.
    """
    if num_decays <= 0:
        return 0.0
    if epsilon > 0.0:
        return math.sqrt(epsilon * branching / num_decays)
    return (UL_FACTOR * branching / num_decays) if USE_FULL_TG_FOR_ZERO_EFFICIENCY else 0.0

# ── Inputs ───────────────────────────────
COMPONENT_MASS = 1681.6  # kg

# Tuple layout:
# (name, acti [Bq/kg], acti_err, effi μ, effi_err (None:statistical), N_decays, branching)

water = [
    ("u238", 7.6e-8, 1.3e-8, 0.0, None, 1e10, 1.0),
    ("th232", 3.1e-9, 0.9e-9, 3.59e-11, None, 1e10, 0.3594),
    ("Rn-222", 6.3e-8, 2.2e-8, 0.0, None, 1e10, 1.0),
]

iv_BB25 = [
    ("th232", 2.646e-7, 1.181e-7, 9.720e-9, None, 1e9, 0.3594),
    ("u238", 0.0, 7.43e-7, 9e-9, None, 1e9, 1.0),
]

iv = [
    ("th232", 2.65e-7, 1.09e-7, 2.19e-9, None, 1e10, 0.3594),
    ("u238", 0.0, 7.42e-7, 8.0e-10, None, 1e10, 1.0),
]

components = iv  # choose which dataset to run

# ── Calculation ────────────────────────────────
results = {}
for name, act, act_err, eff, eff_err, n_decays, branch in components:
    sigma_eff = eff_err if eff_err is not None else efficiency_stat_error(eff, n_decays, branch)
    MU_SEC = COMPONENT_MASS * act * eff
    sigma_sec = math.hypot(COMPONENT_MASS * eff * act_err,
                           COMPONENT_MASS * act * sigma_eff)
    mean_sec = tg_mean(MU_SEC, sigma_sec)
    spread_sec = tg_spread(MU_SEC, sigma_sec)
    results[name] = {
        "mu": eff,
        "eff_err": sigma_eff,
        "mean": mean_sec * SECONDS_PER_YEAR,
        "spread": spread_sec * SECONDS_PER_YEAR,
    }

total_mean = sum(v["mean"] for v in results.values())
total_spread = math.sqrt(sum(v["spread"] ** 2 for v in results.values()))

# ── Output ────────────────────────────────────────────────────────────────
print("\n--- Background results (TG with zero-ε UL toggle) ---")
print(f"UL mode: {'ON' if USE_FULL_TG_FOR_ZERO_EFFICIENCY else 'OFF'}\n")
print(f"{'Isotope':<8} {'μ (eff)':>12} {'σ_ε':>12} {'⟨BG⟩ [cnt/y]':>16} {'± TG':>10}")
for key, val in results.items():
    print(f"{key:<8} {val['mu']:12.3e} {val['eff_err']:12.3e} "
          f"{val['mean']:16.3e} {val['spread']:10.3e}")

print(f"\nTotal: {total_mean:.3e} ± {total_spread:.3e} counts/y")
