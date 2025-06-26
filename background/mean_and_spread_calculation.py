#!/usr/bin/env python3
"""
Background contribution and TG-spread calculation
with automatic statistical efficiency errors.

Antoine Amy — June 2025
"""
import math

SECONDS_PER_YEAR = 86400 * 365.25
EPS = 1e-15


# ─── Helper: binomial 1 σ error on parent-level efficiency ────────────────
def efficiency_error(mu: float, n_decays: int, branching: float = 1.0) -> float:
    """
    σ = sqrt(mu * branching / N)

    Parameters
    ----------
    mu : float
        Efficiency per PARENT decay.
    n_decays : int
        Total number of parent decays simulated.
    branching : float, optional
        Branching ratio used to scale the conditional efficiency
        back to parent-level, by default 1.0.

    Returns
    -------
    float
        1 σ statistical error on `mu`.
    """
    return math.sqrt(mu * branching / n_decays)


# ─── Core TG-spread machinery (unchanged) ─────────────────────────────────
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

    COMPONENT_MASS = 31_810.0  # kg

    # Tuple layout:
    # (name, activity [Bq/kg], activity_err,
    #  efficiency μ, efficiency_err (None ⇒ statistical),
    #  N_decays, branching)
    components = [

        # U-238 — γ comes directly from parent (branching = 1)
        ('u238', 0.0, 3.233e-8,
         1.770e-7, None,
         1_000_000_000, 1.0),

        # Th-232 — efficiency scaled by 208Tl γ branch (35.94 %)
        ('th232', 0.0, 3.257e-9,
         2.074e-7, None,
         1_000_000_000, 0.3594),
    ]

    # ---------- calculation ----------
    calc_results = run_calculation(COMPONENT_MASS, components)

    # ---------- nicely formatted output ----------
    print("\n--- Final Calculation Results (Excel TG-spread) ---")
    print(f"Component mass: {COMPONENT_MASS:.1f} kg\n")

    header = ("Isotope", "μ (efficiency)", "σ_μ (stat)",
              "⟨BG⟩ [cnt/y]", "± TG_spread")
    print(f"{header[0]:<8s} {header[1]:>13s} {header[2]:>13s} {header[3]:>14s} {header[4]:>14s}")

    for comp in components:           # preserve original order
        COMP_NAME = comp[0]
        res  = calc_results[COMP_NAME]
        print(f"{COMP_NAME:<8s} {res['mu']:13.3e} {res['eff_err']:13.3e}"
              f" {res['mean']:14.3e} {res['spread']:14.3e}")

    total = calc_results['total']
    print("\nTotal background: "
          f"{total['mean']:.3e} ± {total['spread']:.3e} counts/y (quadrature)")
