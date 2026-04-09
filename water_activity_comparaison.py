import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Constants
SA_U  = 1.235e4  # Bq/g
SA_Th = 4.06e3   # Bq/g

def g_per_g_to_bq_per_kg(x_g_per_g, SA):
    return np.asarray(x_g_per_g, dtype=float) * SA * 1e3

def mean_std_sem(x):
    x = np.asarray(x, dtype=float)
    n = x.size
    if n < 2:
        return float(np.mean(x)), float("nan"), float("nan")
    s = float(np.std(x, ddof=1))
    return float(np.mean(x)), s, s / np.sqrt(n)

def to_bqkg(g, SA):
    return float(g) * 1e3 * SA

def symmetrize_asym_error(stat, plus, minus):
    return float(np.hypot(stat, 0.5 * (plus + minus)))

def ul95_to_sigma_one_sided(limit):
    return float(limit) / 1.645

# --- SNO+ AV water (Table V central values; Period 2 split averaged) ---
AV_U_x1e_14  = np.array([19.0, 48.5, 3.6, 8.7, 19.4, 53.5, 67.5])
AV_Th_x1e_15 = np.array([5.9, 34.5, 2.7, 8.3, 9.4, 29.0, 67.1])

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

AV_U_mean, AV_U_std, AV_U_sem     = mean_std_sem(AV_U)
AV_Th_mean, AV_Th_std, AV_Th_sem  = mean_std_sem(AV_Th)

# --- SNO+ Water Shield (Table V central values; SEM from period scatter) ---
WS_U_x1e_13  = np.array([2.2, 1.7, 0.6, 2.3, 1.2])
WS_Th_x1e_14 = np.array([9.9, 9.3, 10.6, 8.6, 10.0])

WS_U  = g_per_g_to_bq_per_kg(WS_U_x1e_13  * 1e-13, SA_U)
WS_Th = g_per_g_to_bq_per_kg(WS_Th_x1e_14 * 1e-14, SA_Th)

WS_U_mean, WS_U_std, WS_U_sem     = mean_std_sem(WS_U)
WS_Th_mean, WS_Th_std, WS_Th_sem  = mean_std_sem(WS_Th)

# --- SNO+ (2022) AV water from arXiv:2205.06400v2 ---
# U-chain from 214Bi: (5.78 ± 0.7 +1.5/-1.3) × 10^-15 gU/g
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

# Th-chain from 208Tl: < 4.8 × 10^-16 gTh/g (95% C.L.)
# Spreadsheet proxy: mean = 0, sigma = UL95 / 1.645
SNO2022_AV_Th_limit_g = 4.8e-16
SNO2022_AV_Th_limit   = to_bqkg(SNO2022_AV_Th_limit_g, SA_Th)
SNO2022_AV_Th_mean    = 0.0
SNO2022_AV_Th_sigma   = ul95_to_sigma_one_sided(SNO2022_AV_Th_limit)

# --- SNO Phase III H2O (in situ; asymmetric -> symmetrized after conversion) ---
H2O_U_mean_g   = 35.0e-14
H2O_U_plus_g   = 9.9e-14
H2O_U_minus_g  = 5.4e-14

H2O_Th_mean_g  = 30.0e-15
H2O_Th_plus_g  = 9.2e-15
H2O_Th_minus_g = 19.4e-15

H2O_U_mean      = to_bqkg(H2O_U_mean_g, SA_U)
H2O_U_sigma     = 0.5 * (to_bqkg(H2O_U_plus_g, SA_U) + to_bqkg(H2O_U_minus_g, SA_U))

H2O_Th_mean     = to_bqkg(H2O_Th_mean_g, SA_Th)
H2O_Th_sigma    = 0.5 * (to_bqkg(H2O_Th_plus_g, SA_Th) + to_bqkg(H2O_Th_minus_g, SA_Th))

# --- SNO Phase III D2O (weighted means; symmetric) ---
D2O_U_mean_g  = 6.14e-15
D2O_U_err_g   = 1.01e-15
D2O_Th_mean_g = 0.77e-15
D2O_Th_err_g  = 0.21e-15

D2O_U_mean   = to_bqkg(D2O_U_mean_g, SA_U)
D2O_U_sigma  = to_bqkg(D2O_U_err_g,  SA_U)
D2O_Th_mean  = to_bqkg(D2O_Th_mean_g, SA_Th)
D2O_Th_sigma = to_bqkg(D2O_Th_err_g,  SA_Th)

# --- JUNO (Rn only as provided) ---
JUNO_Rn_mean = 6.10e-07
JUNO_Rn_err  = 5.00e-07

# Build summary table for plotting / spreadsheet export
summary_rows = [
    ("SNO Phase III", "AV D2O",           "Th232", D2O_Th_mean,         D2O_Th_sigma),
    ("SNO Phase III", "AV D2O",           "U238",  D2O_U_mean,          D2O_U_sigma),

    ("SNO Phase III", "H2O Water Shield", "Th232", H2O_Th_mean,         H2O_Th_sigma),
    ("SNO Phase III", "H2O Water Shield", "U238",  H2O_U_mean,          H2O_U_sigma),

    ("SNO+",          "AV Water",         "Th232", AV_Th_mean,          AV_Th_sem),
    ("SNO+",          "AV Water",         "U238",  AV_U_mean,           AV_U_sem),

    ("SNO+",          "H2O Water Shield", "Th232", WS_Th_mean,          WS_Th_sem),
    ("SNO+",          "H2O Water Shield", "U238",  WS_U_mean,           WS_U_sem),

    ("SNO+ (2022)",   "AV Water",         "Th232", SNO2022_AV_Th_mean,  SNO2022_AV_Th_sigma),
    ("SNO+ (2022)",   "AV Water",         "U238",  SNO2022_AV_U_mean,   SNO2022_AV_U_sigma),

    ("JUNO",          "H2O Water Shield", "Rn222", JUNO_Rn_mean,        JUNO_Rn_err),
]

df = pd.DataFrame(summary_rows, columns=["Source", "Component", "Isotope", "Activity", "Error"])
df["Category"] = df["Source"] + " | " + df["Component"]

# Keep true spreadsheet values in Activity/Error, but use a positive plotting proxy on log scale
df["PlotActivity"] = df["Activity"]
mask_zero = (df["PlotActivity"] <= 0) | ~np.isfinite(df["PlotActivity"])
df.loc[mask_zero, "PlotActivity"] = df.loc[mask_zero, "Error"]

print(df.to_string(index=False))

# Grouped bar plot (log scale). Omit missing isotopes per category automatically.
categories = df["Category"].unique().tolist()
isotopes = ["Th232", "U238", "Rn222"]

A = np.full((len(isotopes), len(categories)), np.nan)
E = np.full((len(isotopes), len(categories)), np.nan)

for i, iso in enumerate(isotopes):
    sub = df[df["Isotope"] == iso]
    for j, cat in enumerate(categories):
        r = sub[sub["Category"] == cat]
        if not r.empty:
            A[i, j] = r["PlotActivity"].values[0]
            E[i, j] = r["Error"].values[0]

x = np.arange(len(categories))
width = 0.24
offsets = [-width, 0.0, width]

fig, ax = plt.subplots(figsize=(11.5, 7.0))

for i, iso in enumerate(isotopes):
    y = A[i]
    yerr = E[i]
    valid = np.isfinite(y) & (y > 0)
    ax.bar((x + offsets[i])[valid], y[valid], width=width, yerr=yerr[valid], capsize=3, label=iso)

ax.set_yscale("log")
ax.set_ylabel("Activity (Bq/kg)")
ax.set_title("Water radioactivity by source/component")
ax.set_xticks(x)
ax.set_xticklabels(categories, rotation=25, ha="right")
ax.grid(True, which="both", linestyle=":", linewidth=0.8)
ax.legend(title="Isotope", ncols=1, loc="upper left")

note = (
    "Error bars: SNO+ = SEM of period-to-period scatter (Period 2 z>0/z<0 averaged first for AV water).\n"
    "SNO Phase III = reported uncertainties (H2O symmetrized).\n"
    "SNO+ (2022) Th232 uses a spreadsheet proxy from the 95% C.L. upper limit: mean = 0, sigma = UL95 / 1.645.\n"
    "For visibility on the log plot only, zero-mean entries are plotted at their 1σ value."
)
fig.text(0.01, 0.02, note, ha="left", va="bottom", fontsize=9, wrap=True)

fig.tight_layout(rect=[0, 0.10, 1, 0.98])
plt.show()