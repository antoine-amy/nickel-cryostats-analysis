#!/usr/bin/env python3
"""
Background contribution and TG-spread calculation
with optional statistical efficiency upper limits.

Antoine Amy — Updated July 2025
"""
import math

SECONDS_PER_YEAR = 86400 * 365.25
EPS = 1e-15

USE_FULL_TG_FOR_ZERO_EFFICIENCY = False  # ⇦ Toggle this to switch behavior

def efficiency_error(mu: float, n_decays: int, branching: float = 1.0) -> float:
    """
    Return 1σ statistical error on efficiency per parent decay.
    If mu = 0 and upper-limit mode is enabled, return one-sided 68% upper limit.
    """
    if mu == 0.0:
        return (1.14 * branching / n_decays) if USE_FULL_TG_FOR_ZERO_EFFICIENCY else 0.0
    return math.sqrt(mu * n_decays * branching) / n_decays


def calculate_bg_contribution(mass, activity, activity_err,
                              efficiency, efficiency_err):
    """Return (truncated mean, TG-spread) in counts / y / 2t / FWHM."""
    t = mass * activity * efficiency
    u = math.hypot(mass * efficiency * activity_err,
                   mass * activity   * efficiency_err)

    if u < EPS:                       # no uncertainty ⇒ no truncation
        return max(0.0, t) * SECONDS_PER_YEAR, 0.0

    z   = t / u
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # truncated mean
    mean_sec = t + u * (pdf / cdf) if cdf > EPS else 0.0
    mean_sec = max(0.0, mean_sec)

    # Excel TG-spread
    term_b   = 1.0 + math.erf(z / math.sqrt(2.0))
    a_over_b = (math.exp(-0.5 * z * z) / term_b) if term_b > EPS else 0.0
    inner    = 1.0 - (z * a_over_b) / math.sqrt(8.0 * math.pi) \
                     - (a_over_b ** 2) / (8.0 * math.pi)
    spread_sec = u * math.sqrt(inner) if inner > 0 else 0.0

    # convert to counts / year
    return mean_sec * SECONDS_PER_YEAR, spread_sec * SECONDS_PER_YEAR


def run_calculation(mass, component_list):
    """Compute background for each component and a quadrature total."""
    results = {}
    for (name, act, act_err,
         eff, eff_err,
         n_decays, branch) in component_list:

        # derive statistical σ if requested
        if eff_err is None:
            eff_err = efficiency_error(eff, n_decays, branch)

        mean, spread = calculate_bg_contribution(
            mass, act, act_err, eff, eff_err
        )
        results[name] = {
            'mean':     mean,
            'spread':   spread,
            'eff_err':  eff_err,
            'mu':       eff
        }

    total_mean   = sum(v['mean'] for v in results.values())
    total_spread = math.sqrt(sum(v['spread']**2 for v in results.values()))
    results['total'] = {'mean': total_mean, 'spread': total_spread}
    return results


# ─── Main block ───────────────────────────────────────────────────────────
if __name__ == '__main__':

    COMPONENT_MASS = 1681.6  # kg

    # Tuple layout:
    # (name, activity [Bq/kg], activity_err, efficiency μ, efficiency_err (None ⇒ statistical), N_decays, branching)
    water = [

        # U-238 - (branching = 1)
        ('u238', 7.6e-8, 1.3e-8, 0.0, None, 1e10, 1.0),

        # Th-232 — efficiency scaled by 208Tl γ branch (35.94 %)
        ('th232', 3.1e-9, 0.9e-9, 3.59e-11, None, 1e10, 0.3594),

        # Rn-222 — (branching = 1)
        ('Rn-222', 6.3e-8, 2.2e-8, 0.0, None, 1e10, 1.0),
    ]

    iv_BB25 = [

        # Th-232 — efficiency scaled by 208Tl γ branch (35.94 %)
        ('th232', 2.646e-7, 1.181e-7, 9.720e-9, None, 1e9, 0.3594),
        
        # U-238 - (branching = 1)
        ('u238', 0.0, 7.43e-7, 9e-9, None, 1e9, 1.0),

    ]

    iv = [

        # Th-232 — efficiency scaled by 208Tl γ branch (35.94 %)
        ('th232', 2.65e-7, 1.09e-7, 2.19e-9, None, 1e10, 0.3594),
        
        # U-238 - (branching = 1)
        ('u238', 0.0, 7.42e-7, 8.0e-10, None, 1e10, 1.0),

    ]

    components=iv

    # ---------- calculation ----------
    calc_results = run_calculation(COMPONENT_MASS, components)

    # ---------- nicely formatted output ----------
    print("\n--- Final Calculation Results (Excel TG-spread) ---")
    print(f"Component mass: {COMPONENT_MASS:.1f} kg")
    print(f"Mode: {'TG-spread with 1σ UL for zero-efficiency' if USE_FULL_TG_FOR_ZERO_EFFICIENCY else 'Zero background if efficiency = 0'}\n")

    header = ("Isotope", "μ (efficiency)", "σ_μ (stat)",
              "⟨BG⟩ [cnt/y]", "± TG_spread")
    print(f"{header[0]:<8s} {header[1]:>13s} {header[2]:>13s} {header[3]:>14s} {header[4]:>14s}")

    for comp in components:
        COMP_NAME = comp[0]
        res  = calc_results[COMP_NAME]
        ul_tag = " (UL)" if res['mu'] == 0 and USE_FULL_TG_FOR_ZERO_EFFICIENCY else ""
        print(f"{COMP_NAME:<8s} {res['mu']:13.3e} {res['eff_err']:13.3e}"
              f" {res['mean']:14.3e} {res['spread']:14.3e}{ul_tag}")

    total = calc_results['total']
    print("\nTotal background: "
          f"{total['mean']:.3e} ± {total['spread']:.3e} counts/y (quadrature)")