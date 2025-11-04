#!/usr/bin/env python3
"""
Merged SNO/SNO+ water radioactivity summary (Bq/kg) with SYMMETRIC errors.

Datasets (with arXiv IDs):
1) SNO+  AV water (inside acrylic vessel) — arXiv:1812.05552 (2019), Table V
2) SNO+  Water shielding (outer water)    — arXiv:1812.05552 (2019), Table V
3) SNO   Phase III light-water shield     — arXiv:1107.2901  (2011), Sec. VII (in situ, asymmetric)
   -> Converted to symmetric: sigma = 0.5 * (sigma_plus + sigma_minus) after unit conversion.
4) SNO   Phase III heavy water (D2O)      — arXiv:1107.2901  (2011), weighted means (symmetric)

Specific activities for conversion:
  1 g U  = 1.235e4 Bq
  1 g Th = 4.06e3 Bq

Rn-222 is not reported separately for H2O; U-chain reflects radon-supported equilibrium.
"""

import numpy as np

# Specific activities (natural chains)
SA_U  = 1.235e4  # Bq/g
SA_Th = 4.06e3   # Bq/g

def g_per_g_to_bq_per_kg(x_g_per_g: np.ndarray, SA: float) -> np.ndarray:
    return x_g_per_g * SA * 1e3

def mean_std(x: np.ndarray) -> tuple[float, float]:
    return float(np.mean(x)), float(np.std(x, ddof=1))

def fmt(x: float) -> str:
    return f"{x:.3e}"

print("=== Merged SNO / SNO+ Water Radioactivity (Bq/kg) — symmetric errors ===\n")

# 1) SNO+ AV water — Table V (1812.05552): mean ± between-period std
AV_U_x1e_14  = np.array([19.0, 48.5, 3.6, 8.7, 19.4, 53.5, 67.5])  # U in 1e-14 g/g
AV_Th_x1e_15 = np.array([5.9, 34.5, 2.7, 8.3, 9.4, 29.0, 67.1])    # Th in 1e-15 g/g

AV_U  = g_per_g_to_bq_per_kg(AV_U_x1e_14 * 1e-14, SA_U)
AV_Th = g_per_g_to_bq_per_kg(AV_Th_x1e_15 * 1e-15, SA_Th)

AV_U_mean,  AV_U_std  = mean_std(AV_U)
AV_Th_mean, AV_Th_std = mean_std(AV_Th)

print("1) SNO+ AV water — arXiv:1812.05552 (Table V)")
print(f"   U-238 chain:  {fmt(AV_U_mean)} ± {fmt(AV_U_std)}")
print(f"   Th-232 chain: {fmt(AV_Th_mean)} ± {fmt(AV_Th_std)}")
print("   Rn-222: not reported separately; included in U-chain (via Bi-214).\n")

# 2) SNO+ Water shielding — Table V (1812.05552): mean ± between-period std
WS_U_x1e_13  = np.array([2.2, 1.7, 0.6, 2.3, 1.2])       # U in 1e-13 g/g
WS_Th_x1e_14 = np.array([9.9, 9.3, 10.6, 8.6, 10.0])     # Th in 1e-14 g/g

WS_U  = g_per_g_to_bq_per_kg(WS_U_x1e_13  * 1e-13, SA_U)
WS_Th = g_per_g_to_bq_per_kg(WS_Th_x1e_14 * 1e-14, SA_Th)

WS_U_mean,  WS_U_std  = mean_std(WS_U)
WS_Th_mean, WS_Th_std = mean_std(WS_Th)

print("2) SNO+ Water shielding — arXiv:1812.05552 (Table V)")
print(f"   U-238 chain:  {fmt(WS_U_mean)} ± {fmt(WS_U_std)}")
print(f"   Th-232 chain: {fmt(WS_Th_mean)} ± {fmt(WS_Th_std)}")
print("   Rn-222: not reported separately; included in U-chain (via Bi-214).\n")

# 3) SNO Phase III H2O (light-water shield) — 1107.2901 (Sec. VII, in situ, asymmetric)
H2O_U_mean_g   = 35.0e-14
H2O_U_plus_g   = 9.9e-14
H2O_U_minus_g  = 5.4e-14

H2O_Th_mean_g  = 30.0e-15
H2O_Th_plus_g  = 9.2e-15
H2O_Th_minus_g = 19.4e-15

def to_bqkg(g, SA): return g * 1e3 * SA

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

# 4) SNO Phase III D2O (heavy water) — 1107.2901 (weighted means, symmetric)
D2O_U_mean_g  = 6.14e-15
D2O_U_err_g   = 1.01e-15
D2O_Th_mean_g = 0.77e-15
D2O_Th_err_g  = 0.21e-15

D2O_U_mean  = to_bqkg(D2O_U_mean_g, SA_U);  D2O_U_sigma  = to_bqkg(D2O_U_err_g,  SA_U)
D2O_Th_mean = to_bqkg(D2O_Th_mean_g, SA_Th); D2O_Th_sigma = to_bqkg(D2O_Th_err_g, SA_Th)

print("4) SNO Phase III D2O — arXiv:1107.2901 (weighted means)")
print(f"   U-238 chain:  {fmt(D2O_U_mean)} ± {fmt(D2O_U_sigma)}")
print(f"   Th-232 chain: {fmt(D2O_Th_mean)} ± {fmt(D2O_Th_sigma)}")
print("   Rn-222: reported via separate assays but used to infer U-equivalent; not quoted as standalone activity.")
