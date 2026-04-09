#!/usr/bin/env python3
"""
Merged SNO/SNO+ water radioactivity summary (Bq/kg) with comparable mean-level errors,
plus SNO+ (2022) AV-water values.

For the SNO+ (2022) internal-water Th-232 upper limit, a spreadsheet proxy is used:
    mean = 0
    sigma = UL95 / 1.645
"""

from __future__ import annotations

import numpy as np

# Specific activities (natural chains)
SA_U  = 1.235e4  # Bq/g
SA_Th = 4.06e3   # Bq/g


def g_per_g_to_bq_per_kg(x_g_per_g: np.ndarray, SA: float) -> np.ndarray:
    """Convert concentration in g/g to activity in Bq/kg."""
    return x_g_per_g * SA * 1e3


def mean_std_sem(x: np.ndarray) -> tuple[float, float, float]:
    """
    Return (mean, sample std, standard error of mean) for x.
    Uses ddof=1 for sample std. SEM = std / sqrt(N).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float(np.mean(x)), float("nan"), float("nan")
    s = float(np.std(x, ddof=1))
    return float(np.mean(x)), s, s / np.sqrt(n)


def to_bqkg(g: float, SA: float) -> float:
    return g * 1e3 * SA


def fmt(x: float) -> str:
    return f"{x:.3e}"


def symmetrize_asym_error(stat: float, plus: float, minus: float) -> float:
    return float(np.hypot(stat, 0.5 * (plus + minus)))


def ul95_to_sigma_one_sided(limit: float) -> float:
    return float(limit) / 1.645


print("=== Merged SNO / SNO+ Water Radioactivity (Bq/kg) — comparable mean-level errors ===\n")

# -----------------------------------------------------------------------------
# 1) SNO+ AV water — arXiv:1812.05552 (Table V)
# -----------------------------------------------------------------------------

AV_U_x1e_14  = np.array([19.0, 48.5, 3.6, 8.7, 19.4, 53.5, 67.5])  # U in 1e-14 g/g
AV_Th_x1e_15 = np.array([5.9, 34.5, 2.7, 8.3, 9.4, 29.0, 67.1])    # Th in 1e-15 g/g

AV_U_periods_x1e_14 = np.array([
    AV_U_x1e_14[0],
    0.5 * (AV_U_x1e_14[1] + AV_U_x1e_14[2]),
    AV_U_x1e_14[3],
    AV_U_x1e_14[4],
    AV_U_x1e_14[5],
    AV_U_x1e_14[6],
])
AV_Th_periods_x1e_15 = np.array([
    AV_Th_x1e_15[0],
    0.5 * (AV_Th_x1e_15[1] + AV_Th_x1e_15[2]),
    AV_Th_x1e_15[3],
    AV_Th_x1e_15[4],
    AV_Th_x1e_15[5],
    AV_Th_x1e_15[6],
])

AV_U  = g_per_g_to_bq_per_kg(AV_U_periods_x1e_14 * 1e-14, SA_U)
AV_Th = g_per_g_to_bq_per_kg(AV_Th_periods_x1e_15 * 1e-15, SA_Th)

AV_U_mean,  AV_U_std,  AV_U_sem  = mean_std_sem(AV_U)
AV_Th_mean, AV_Th_std, AV_Th_sem = mean_std_sem(AV_Th)

print("1) SNO+ AV water — arXiv:1812.05552 (Table V)")
print(f"   U-238 chain (mean over periods):  {fmt(AV_U_mean)} ± {fmt(AV_U_sem)}   [SEM, N={AV_U.size}]")
print(f"     period scatter (std):           {fmt(AV_U_std)}")
print(f"   Th-232 chain (mean over periods): {fmt(AV_Th_mean)} ± {fmt(AV_Th_sem)}   [SEM, N={AV_Th.size}]")
print(f"     period scatter (std):           {fmt(AV_Th_std)}")
print("   Rn-222: not reported separately; included in U-chain (via Bi-214).\n")

# -----------------------------------------------------------------------------
# 2) SNO+ Water shielding — arXiv:1812.05552 (Table V)
# -----------------------------------------------------------------------------

WS_U_x1e_13  = np.array([2.2, 1.7, 0.6, 2.3, 1.2])       # U in 1e-13 g/g (Periods 1–5)
WS_Th_x1e_14 = np.array([9.9, 9.3, 10.6, 8.6, 10.0])     # Th in 1e-14 g/g (Periods 1–5)

WS_U  = g_per_g_to_bq_per_kg(WS_U_x1e_13  * 1e-13, SA_U)
WS_Th = g_per_g_to_bq_per_kg(WS_Th_x1e_14 * 1e-14, SA_Th)

WS_U_mean,  WS_U_std,  WS_U_sem  = mean_std_sem(WS_U)
WS_Th_mean, WS_Th_std, WS_Th_sem = mean_std_sem(WS_Th)

print("2) SNO+ Water shielding — arXiv:1812.05552 (Table V)")
print(f"   U-238 chain (mean over periods):  {fmt(WS_U_mean)} ± {fmt(WS_U_sem)}   [SEM, N={WS_U.size}]")
print(f"     period scatter (std):           {fmt(WS_U_std)}")
print(f"   Th-232 chain (mean over periods): {fmt(WS_Th_mean)} ± {fmt(WS_Th_sem)}   [SEM, N={WS_Th.size}]")
print(f"     period scatter (std):           {fmt(WS_Th_std)}")
print("   Rn-222: not reported separately; included in U-chain (via Bi-214).\n")

# -----------------------------------------------------------------------------
# 2b) SNO+ (2022) AV water — arXiv:2205.06400v2
# -----------------------------------------------------------------------------

SNO2022_AV_U_mean_g   = 5.78e-15
SNO2022_AV_U_stat_g   = 0.7e-15
SNO2022_AV_U_plus_g   = 1.5e-15
SNO2022_AV_U_minus_g  = 1.3e-15

SNO2022_AV_U_mean  = to_bqkg(SNO2022_AV_U_mean_g, SA_U)
SNO2022_AV_U_sigma = to_bqkg(
    symmetrize_asym_error(
        SNO2022_AV_U_stat_g,
        SNO2022_AV_U_plus_g,
        SNO2022_AV_U_minus_g
    ),
    SA_U
)

SNO2022_AV_Th_limit_g = 4.8e-16
SNO2022_AV_Th_limit   = to_bqkg(SNO2022_AV_Th_limit_g, SA_Th)
SNO2022_AV_Th_mean    = 0.0
SNO2022_AV_Th_sigma   = ul95_to_sigma_one_sided(SNO2022_AV_Th_limit)

print("2b) SNO+ (2022) AV water — arXiv:2205.06400v2")
print(f"   U-238 chain:  {fmt(SNO2022_AV_U_mean)} ± {fmt(SNO2022_AV_U_sigma)}")
print(f"   Th-232 proxy: {fmt(SNO2022_AV_Th_mean)} ± {fmt(SNO2022_AV_Th_sigma)}   [from one-sided 95% C.L. limit]")
print("   Rn-222: not reported separately; radon ingress contributes to the Bi-214-like component.\n")

# -----------------------------------------------------------------------------
# 3) SNO Phase III H2O (light-water shield) — arXiv:1107.2901
# -----------------------------------------------------------------------------

H2O_U_mean_g   = 35.0e-14
H2O_U_plus_g   = 9.9e-14
H2O_U_minus_g  = 5.4e-14

H2O_Th_mean_g  = 30.0e-15
H2O_Th_plus_g  = 9.2e-15
H2O_Th_minus_g = 19.4e-15

H2O_U_mean      = to_bqkg(H2O_U_mean_g, SA_U)
H2O_U_err_plus  = to_bqkg(H2O_U_plus_g, SA_U)
H2O_U_err_minus = to_bqkg(H2O_U_minus_g, SA_U)
H2O_U_sigma     = 0.5 * (H2O_U_err_plus + H2O_U_err_minus)

H2O_Th_mean      = to_bqkg(H2O_Th_mean_g, SA_Th)
H2O_Th_err_plus  = to_bqkg(H2O_Th_plus_g, SA_Th)
H2O_Th_err_minus = to_bqkg(H2O_Th_minus_g, SA_Th)
H2O_Th_sigma     = 0.5 * (H2O_Th_err_plus + H2O_Th_err_minus)

print("3) SNO Phase III H2O — arXiv:1107.2901 (Sec. VII, in situ, symmetrized)")
print(f"   U-238 chain:  {fmt(H2O_U_mean)} ± {fmt(H2O_U_sigma)}")
print(f"     original asym: +{fmt(H2O_U_err_plus)} / -{fmt(H2O_U_err_minus)}")
print(f"   Th-232 chain: {fmt(H2O_Th_mean)} ± {fmt(H2O_Th_sigma)}")
print(f"     original asym: +{fmt(H2O_Th_err_plus)} / -{fmt(H2O_Th_err_minus)}")
print("   Rn-222: no independent H2O value; U-chain reflects radon-supported equilibrium.\n")

# -----------------------------------------------------------------------------
# 4) SNO Phase III D2O (heavy water) — arXiv:1107.2901
# -----------------------------------------------------------------------------

D2O_U_mean_g  = 6.14e-15
D2O_U_err_g   = 1.01e-15
D2O_Th_mean_g = 0.77e-15
D2O_Th_err_g  = 0.21e-15

D2O_U_mean   = to_bqkg(D2O_U_mean_g, SA_U)
D2O_U_sigma  = to_bqkg(D2O_U_err_g,  SA_U)
D2O_Th_mean  = to_bqkg(D2O_Th_mean_g, SA_Th)
D2O_Th_sigma = to_bqkg(D2O_Th_err_g,  SA_Th)

print("4) SNO Phase III D2O — arXiv:1107.2901 (weighted means)")
print(f"   U-238 chain:  {fmt(D2O_U_mean)} ± {fmt(D2O_U_sigma)}")
print(f"   Th-232 chain: {fmt(D2O_Th_mean)} ± {fmt(D2O_Th_sigma)}")
print("   Rn-222: reported via separate assays but used to infer U-equivalent; not quoted as standalone activity.\n")

# -----------------------------------------------------------------------------
# 5) JUNO H2O water shield — arXiv:2510.17082
# -----------------------------------------------------------------------------

JUNO_Rn_mean = 6.10e-07
JUNO_Rn_err  = 5.00e-07

print("5) JUNO H2O Water Shield — arXiv:2510.17082")
print(f"   Rn-222: {fmt(JUNO_Rn_mean)} ± {fmt(JUNO_Rn_err)}")
print("   U/Th: N/A in this summary.\n")