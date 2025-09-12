#!/usr/bin/env python3
"""
Compute water background contributions for U-238, Th-232, and Rn-222
using TG‐spread (truncated Gaussian) error propagation.
"""

import math

SECONDS_PER_YEAR = 86400 * 365.25
EPS = 1e-15


def calculate_bg_contribution(mass, activity, activity_err,
                              efficiency, efficiency_err):
    """
    Return (truncated mean, TG‐spread) in counts/year.

    Parameters
    ----------
    mass : float
        Component mass (kg).
    activity : float
        Specific activity (Bq/kg).
    activity_err : float
        1σ uncertainty on activity (Bq/kg).
    efficiency : float
        Hit efficiency (counts/decay).
    efficiency_err : float
        1σ uncertainty on efficiency (counts/decay).
    """
    # nominal rate (counts/s)
    t = mass * activity * efficiency
    # total uncertainty (counts/s)
    u = math.hypot(mass * efficiency * activity_err,
                   mass * activity   * efficiency_err)

    if u < EPS:
        # no uncertainty ⇒ just return t
        return max(0.0, t) * SECONDS_PER_YEAR, 0.0

    z = t / u
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2.0 * math.pi)
    cdf = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))

    # truncated‐mean correction
    mean_sec = t + u * (pdf / cdf) if cdf > EPS else 0.0
    mean_sec = max(0.0, mean_sec)

    # TG‐spread (Excel definition)
    term_b = 1.0 + math.erf(z / math.sqrt(2.0))
    a_over_b = (math.exp(-0.5 * z * z) / term_b) if term_b > EPS else 0.0
    inner = (1.0
             - (z * a_over_b) / math.sqrt(8.0 * math.pi)
             - (a_over_b**2) / (8.0 * math.pi))
    spread_sec = u * math.sqrt(inner) if inner > 0 else 0.0

    # convert to counts/year
    return mean_sec * SECONDS_PER_YEAR, spread_sec * SECONDS_PER_YEAR


def main():
    # Water mass (kg)
    water_mass = 1_747_092.874453

    # Mean hit efficiencies (counts/decay)
    eff_u_rn = 6.831e-13
    eff_th = 4.296e-12

    # Compute efficiency errors assuming N_decays = 1e10
    N_DECAYS = 1e10
    br_u = 1.0
    br_th = 0.3594
    eff_err_u = math.sqrt(eff_u_rn * N_DECAYS * br_u) / N_DECAYS
    eff_err_th = math.sqrt(eff_th   * N_DECAYS * br_th) / N_DECAYS

    # Activities (Bq/kg) and 1σ uncertainties
    act_u = 76e-9
    act_u_err = 13e-9

    act_th = 3.1e-9
    act_th_err = 0.9e-9

    act_rn = 63e-9
    act_rn_err = 22e-9

    # Calculate contributions
    mean_u,  spread_u  = calculate_bg_contribution(
        water_mass, act_u,  act_u_err,  eff_u_rn, eff_err_u
    )
    mean_th, spread_th = calculate_bg_contribution(
        water_mass, act_th, act_th_err, eff_th,   eff_err_th
    )
    mean_rn, spread_rn = calculate_bg_contribution(
        water_mass, act_rn, act_rn_err, eff_u_rn, eff_err_u
    )

    # Total in quadrature
    total_mean   = mean_u + mean_th + mean_rn
    total_spread = math.sqrt(spread_u**2 + spread_th**2 + spread_rn**2)

    # Output
    print(f"{'Component':<8s} {'Mean [cnt/y]':>15s}   ± TG‑spread")
    print(f"{'-'*40}")
    print(f"{'U-238':<8s} {mean_u:15.3e}   ± {spread_u:.3e}")
    print(f"{'Th-232':<8s} {mean_th:15.3e}   ± {spread_th:.3e}")
    print(f"{'Rn-222':<8s} {mean_rn:15.3e}   ± {spread_rn:.3e}")
    print(f"{'-'*40}")
    print(f"{'Total':<8s} {total_mean:15.3e}   ± {total_spread:.3e}")


if __name__ == "__main__":
    main()
