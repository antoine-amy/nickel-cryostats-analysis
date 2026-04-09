#!/usr/bin/env python3
"""
nEXO cryostat: max vessel mass vs HFE thickness
with TG-Spread uncertainties (in tonnes)

Antoine Amy — July 2025
"""
import math
import numpy as np
import matplotlib.pyplot as plt

# ── Reference & attenuation ──────────────────────────────────────────────
t0 = 0.76                         # m, reference HFE thickness
MU_GAMMA = 0.00674403            # mm⁻¹, HFE-7200 at 165 K

# ── Activities (mBq/kg) and 1σ errors ────────────────────────────────────
ACT_Th232, ACT_Th232_ERR = 2.65e-4, 1.09e-4  # mBq/kg
ACT_U238,  ACT_U238_ERR  = 0.0,     7.42e-4  # mBq/kg

# ── Hit-efficiencies at 76 cm sphere→cyl ────────────────────────────────
hit_sph = {
    'IV': {'Th232': 9.720e-9, 'U238': 9.0e-9},
    'OV': {'Th232': 7.20e-9,  'U238': 0.0}
}
TRANS_SCALE = 0.0033 / 0.0013

# ── Background budgets (counts/year) ────────────────────────────────────
BUDGET_IV = 1.014e-3 + 1.241e-2
BUDGET_OV = 1.730e-3 + 1.994e-2

# ── Time conversion ──────────────────────────────────────────────────────
SECONDS_PER_YEAR = 365.25 * 86400.0

# ── Binomial σ on efficiency ─────────────────────────────────────────────
def eff_sigma(mu, n_decays, branch=1.0):
    return math.sqrt(mu * n_decays * branch) / n_decays

# ── TG-Spread helper (counts/sec → counts/yr) ────────────────────────────
def calculate_bg_contribution(mass, activity, activity_err,
                              efficiency, efficiency_err):
    # counts/sec
    t = mass * activity * efficiency
    # total 1σ uncertainty in counts/sec
    u = math.hypot(mass * efficiency * activity_err,
                   mass * activity   * efficiency_err)
    EPS = 1e-15
    if u < EPS:
        mean_yr = max(0.0, t) * SECONDS_PER_YEAR
        return mean_yr, 0.0

    z   = t / u
    pdf = math.exp(-0.5*z*z) / math.sqrt(2.0*math.pi)
    cdf = 0.5*(1.0 + math.erf(z/math.sqrt(2.0)))

    mean_sec = (t + u*(pdf/cdf)) if cdf > EPS else 0.0
    mean_sec = max(0.0, mean_sec)

    term_b   = 1.0 + math.erf(z/math.sqrt(2.0))
    a_over_b = (math.exp(-0.5*z*z)/term_b) if term_b > EPS else 0.0
    inner    = 1.0 - (z*a_over_b)/math.sqrt(8.0*math.pi) - (a_over_b**2)/(8.0*math.pi)
    spread_sec = u*math.sqrt(inner) if inner > 0 else 0.0

    return mean_sec*SECONDS_PER_YEAR, spread_sec*SECONDS_PER_YEAR

# ── HFE thickness grid ──────────────────────────────────────────────────
t_values = np.linspace(0.10, t0, 100)

# ── Arrays for max mass and its error (in tonnes) ────────────────────────
M_iv_mean_t = np.zeros_like(t_values)
M_iv_err_t  = np.zeros_like(t_values)
M_ov_mean_t = np.zeros_like(t_values)
M_ov_err_t  = np.zeros_like(t_values)

# ── Loop over thicknesses ────────────────────────────────────────────────
N_DECAYS, BR_TL208 = 1e10, 0.3594
for i, t in enumerate(t_values):
    Δt    = t0 - t
    atten = math.exp(MU_GAMMA * Δt * 1000.0)

    # scaled efficiencies & errors
    μ_IV_Th = hit_sph['IV']['Th232'] * TRANS_SCALE * atten
    μ_IV_U  = hit_sph['IV']['U238']  * TRANS_SCALE * atten
    μ_OV_Th = hit_sph['OV']['Th232'] * TRANS_SCALE * atten
    μ_OV_U  = hit_sph['OV']['U238']  * TRANS_SCALE * atten

    σμ_IV_Th = eff_sigma(μ_IV_Th, N_DECAYS, BR_TL208)
    σμ_IV_U  = eff_sigma(μ_IV_U,  N_DECAYS)
    σμ_OV_Th = eff_sigma(μ_OV_Th, N_DECAYS, BR_TL208)
    σμ_OV_U  = eff_sigma(μ_OV_U,  N_DECAYS)

    # activities in Bq/kg
    A_Th = ACT_Th232 * 1e-3
    A_U  = ACT_U238  * 1e-3
    σA_Th = ACT_Th232_ERR * 1e-3
    σA_U  = ACT_U238_ERR  * 1e-3

    # compute per-kg background mean & spread
    iv_th_mean, iv_th_spread = calculate_bg_contribution(
        1.0, A_Th, σA_Th, μ_IV_Th, σμ_IV_Th
    )
    iv_u_mean, iv_u_spread   = calculate_bg_contribution(
        1.0, A_U,  σA_U,  μ_IV_U,  σμ_IV_U
    )
    iv_mean  = iv_th_mean + iv_u_mean
    iv_spread = math.hypot(iv_th_spread, iv_u_spread)

    ov_th_mean, ov_th_spread = calculate_bg_contribution(
        1.0, A_Th, σA_Th, μ_OV_Th, σμ_OV_Th
    )
    ov_u_mean, ov_u_spread   = calculate_bg_contribution(
        1.0, A_U,  σA_U,  μ_OV_U,  σμ_OV_U
    )
    ov_mean  = ov_th_mean + ov_u_mean
    ov_spread = math.hypot(ov_th_spread, ov_u_spread)

    # invert to find max mass [kg], propagate error, then convert to tonnes
    if iv_mean > 0:
        M_iv_kg   = BUDGET_IV / iv_mean
        σM_iv_kg = BUDGET_IV * iv_spread / (iv_mean**2)
        M_iv_mean_t[i] = M_iv_kg / 1e3
        M_iv_err_t[i]  = σM_iv_kg / 1e3
    else:
        M_iv_mean_t[i] = np.nan
        M_iv_err_t[i]  = np.nan

    if ov_mean > 0:
        M_ov_kg   = BUDGET_OV / ov_mean
        σM_ov_kg = BUDGET_OV * ov_spread / (ov_mean**2)
        M_ov_mean_t[i] = M_ov_kg / 1e3
        M_ov_err_t[i]  = σM_ov_kg / 1e3
    else:
        M_ov_mean_t[i] = np.nan
        M_ov_err_t[i]  = np.nan

# ── Print for 76 cm HFE ────────────────────────────────────────────────
target = 0.76  # m
idx = np.argmin(np.abs(t_values - target))
print(f"At {t_values[idx]*100:4.1f} cm HFE:")
print(f"  IV max mass = {M_iv_mean_t[idx]:6.3f} t ± {M_iv_err_t[idx]:6.3f} t")
print(f"  OV max mass = {M_ov_mean_t[idx]:6.3f} t ± {M_ov_err_t[idx]:6.3f} t")

# ── Print for 180 mm HFE ────────────────────────────────────────────────
target = 0.18  # m
idx = np.argmin(np.abs(t_values - target))
print(f"\nAt {t_values[idx]*100:4.1f} cm HFE:")
print(f"  IV max mass = {M_iv_mean_t[idx]:6.3f} t ± {M_iv_err_t[idx]:6.3f} t")
print(f"  OV max mass = {M_ov_mean_t[idx]:6.3f} t ± {M_ov_err_t[idx]:6.3f} t")

# ── Print for 370 mm HFE ────────────────────────────────────────────────
target = 0.37  # m
idx = np.argmin(np.abs(t_values - target))
print(f"\nAt {t_values[idx]*100:4.1f} cm HFE:")
print(f"  IV max mass = {M_iv_mean_t[idx]:6.3f} t ± {M_iv_err_t[idx]:6.3f} t")
print(f"  OV max mass = {M_ov_mean_t[idx]:6.3f} t ± {M_ov_err_t[idx]:6.3f} t")

# ── Plot ────────────────────────────────────────────────────────────────
plt.figure(figsize=(10,6))
plt.plot(t_values*100, M_iv_mean_t, label='IV mean')
plt.fill_between(t_values*100,
                 M_iv_mean_t - M_iv_err_t,
                 M_iv_mean_t + M_iv_err_t,
                 alpha=0.3)
plt.plot(t_values*100, M_ov_mean_t, label='OV mean')
plt.fill_between(t_values*100,
                 M_ov_mean_t - M_ov_err_t,
                 M_ov_mean_t + M_ov_err_t,
                 alpha=0.3)

plt.xlabel('HFE thickness [cm]', fontsize=18)
plt.ylabel('Max Ni mass [tonnes]', fontsize=18)
#plt.title('Max vessel mass vs. HFE thickness (with TG-Spread)')
plt.yscale('log')
plt.legend(fontsize=14)
plt.grid(True, which='both', ls='--', alpha=0.5)
plt.tight_layout()
plt.show()
