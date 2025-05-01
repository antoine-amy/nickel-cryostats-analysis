#!/usr/bin/env python3
"""
Background contribution and TG spread calculation.
"""
import math

SECONDS_PER_YEAR = 86400 * 365.25
EPS = 1e-15


def calculate_bg_contribution(mass, activity, activity_err, efficiency, efficiency_err):
    """Calculate truncated mean and Excel TG spread uncertainty per year."""
    t = mass * activity * efficiency
    u = math.hypot(mass * efficiency * activity_err,
                   mass * activity * efficiency_err)
    if u < EPS:
        return max(0.0, t) * SECONDS_PER_YEAR, 0.0

    z = t / u
    pdf = math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)
    cdf = 0.5 * (1 + math.erf(z / math.sqrt(2)))

    # Truncated mean
    mean_sec = 0.0
    if cdf > EPS:
        mean_sec = max(0.0, t + u * (pdf / cdf))

    # Excel TG Spread
    term_b = 1 + math.erf(z / math.sqrt(2))
    a_over_b = (math.exp(-0.5 * z * z) / term_b) if term_b > EPS else 0.0
    inner = 1 - (z * a_over_b) / math.sqrt(8 * math.pi) \
            - (a_over_b * a_over_b) / (8 * math.pi)
    spread_sec = u * math.sqrt(inner) if inner > 0 else 0.0

    return mean_sec * SECONDS_PER_YEAR, spread_sec * SECONDS_PER_YEAR


def run_calculation(mass, components):
    """Compute background for each component and totals."""
    results = {}
    for name, activity, activity_err, eff, eff_err in components:
        mean, spread = calculate_bg_contribution(
            mass, activity, activity_err, eff, eff_err
        )
        results[name] = {'mean': mean, 'spread': spread}

    total_mean = sum(v['mean'] for v in results.values())
    total_spread = math.sqrt(sum(v['spread']**2 for v in results.values()))
    results['total'] = {'mean': total_mean, 'spread': total_spread}

    return results


if __name__ == '__main__':
    # Input values
    COMPONENT_MASS = 31810.0  # kg
    components = [
        ('u238', 0.0, 3.233e-8, 1.770e-7, 1.330e-8),
        ('th232', 0.0, 3.257e-9, 2.074e-7, 8.633e-9),
    ]
    expected = {
        'u238': (4.584e-3, 5.630e-3),
        'th232': (5.410e-4, 6.644e-4),
    }

    # Run
    results = run_calculation(COMPONENT_MASS, components)

    # Output
    print("\n--- Final Calculation Results (Using Excel TG Spread formula) ---")
    print(f"Component Mass: {COMPONENT_MASS} kg")

    print("\nInput Data:")
    for name, activity, activity_err, eff, eff_err in components:
        LABEL = 'U-238' if name == 'u238' else 'Th-232'
        print(f"  {LABEL} Activity: {activity:.2e} +/- {activity_err:.2e} Bq/kg")
        print(f"  {LABEL} Efficiency: {eff:.3e} +/- {eff_err:.3e} [counts/decay/2t/FWHM]")

    print("\nCalculated Background Contributions [counts/y/2t/FWHM]")
    print("  (Using Truncated Mean and Excel 'TG Spread' Uncertainty)")
    print(f"  U-238 : {results['u238']['mean']:.5e} +/- {results['u238']['spread']:.5e}")
    print(f"  Th-232: {results['th232']['mean']:.5e} +/- {results['th232']['spread']:.5e}")
    total = results['total']
    print(f"  Total : {total['mean']:.5e} +/- {total['spread']:.5e} (Note: quadrature sum error)")

    print("\nComparison with expected 'good values' [counts/y/2t/FWHM]")
    print(f"  Calculated U-238: {results['u238']['mean']:.3e} +/- {results['u238']['spread']:.3e}")
    print(f"  Expected U-238:   {expected['u238'][0]:.3e} +/- {expected['u238'][1]:.3e}")
    print(
        f"  Calculated Th-232: {results['th232']['mean']:.3e} +/- {results['th232']['spread']:.3e}")
    print(f"  Expected Th-232:   {expected['th232'][0]:.3e} +/- {expected['th232'][1]:.3e}")
