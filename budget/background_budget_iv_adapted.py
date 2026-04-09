#!/usr/bin/env python3
"""
Intrinsic background budget by component with HFE-based coloring.
"""

import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Import semi-analytical HFE transport model from hfe_only script
sys.path.insert(0, str(Path(__file__).resolve().parent))
from background_vs_hfe_hfe_only import _analytic_f_on_grid


# ── Hardcoded inputs ─────────────────────────────────────────────────────
INPUT_FILE = Path(
    "/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/"
    "nickel-cryostats-analysis/budget/"
    "Summary_D-047_v86_250113-233135_2025-09-11.xlsx"
)
MIN_COUNT = 1e-4
IV_RADIUS_MM = None  # set to None to disable HFE scaling

BASELINE_RADIUS_MM = 1691.0
MU = 0.00674403  # HFE-7200 at 165 K (approx. LXe temperature), 1/mm

# HFE mass geometry (from hfe_volume_to_iv_radius.py)
HFE_DENSITY_T_PER_M3 = 1.73
TPC_RADIUS_M = 0.6385
TPC_HEIGHT_M = 1.277
BASELINE_HFE_MASS_KG = 31810.0  # target mass at baseline radius

def _hfe_mass_tonnes(iv_radius_mm: float) -> float:
    """HFE mass in tonnes for a given inner cryostat radius."""
    r_m = iv_radius_mm / 1000.0
    sphere_vol = (4.0 / 3.0) * np.pi * r_m ** 3
    tpc_vol = np.pi * TPC_RADIUS_M ** 2 * TPC_HEIGHT_M
    ideal_baseline = HFE_DENSITY_T_PER_M3 * (
        (4.0 / 3.0) * np.pi * (BASELINE_RADIUS_MM / 1000.0) ** 3 - tpc_vol
    )
    cal_loss = ideal_baseline - BASELINE_HFE_MASS_KG / 1000.0
    return HFE_DENSITY_T_PER_M3 * (sphere_vol - tpc_vol) - cal_loss

OUTSIDE_HFE = {
    "Cryopit concrete and shotcrete",
    "Outer Vessel Support",
    "Outer Cryostat",
    "Inner Vessel Support",
    "Inner Cryostat MLI",
    "Inner Cryostat",
    "CRE Transition Enclosures",
    "CRE Transition Boards",
    "PRE Transition Enclosures",
    "PRE Transition Boards",
    "OD: PMTs, PMT cable, and PMT mounts",
    "OD: Tank",
    "Heat Transfer Fluid",
}

# HFE volume components use the semi-analytical F(r)/F(baseline) scaling
# instead of the simple exponential used for fully external components.
HFE_VOLUME = {"Heat Transfer Fluid"}

# Pre-tabulate F at baseline once (R_GRID is module-level in hfe_only)
from background_vs_hfe_hfe_only import R_GRID as _HFE_R_GRID
_F_GRID = _analytic_f_on_grid(MU, _HFE_R_GRID)
_F_BASELINE = float(np.interp(BASELINE_RADIUS_MM, _HFE_R_GRID, _F_GRID))


def _hfe_volume_scale(iv_radius_mm: float) -> float:
    """Return F(r)/F(baseline) — the semi-analytical scaling for HFE self-background."""
    f_r = float(np.interp(iv_radius_mm, _HFE_R_GRID, _F_GRID))
    return f_r / _F_BASELINE

COMPONENT_PREFIX_RENAMES = [
    ("Outer Cryostat Support", "Outer Vessel Support"),
    ("Inner Cryostat Support", "Inner Vessel Support"),
    ("Outer Cryostat (", "Outer Cryostat"),
    ("Inner Cryostat (", "Inner Cryostat"),
    ("Outer Cryostat Liner", "Outer Cryostat Liner"),
    ("Inner Cryostat Liner", "Inner Cryostat Liner"),
    ("Outer Cryostat Feedthrough", "Outer Cryostat Feedthrough"),
    ("Inner Cryostat Feedthrough", "Inner Cryostat Feedthrough"),
    ("Inactive LXe", "Skin LXe"),
    ("Active LXe", "TPC LXe"),
]


def sqrt_sum_sq(series: pd.Series) -> float:
    """Return sqrt(sum(x^2)) for a numeric Series."""
    arr = series.to_numpy(dtype=float)
    return float(np.sqrt(np.sum(arr ** 2)))


# ── Plot style (LaTeX-ready) ─────────────────────────────────────────────
plt.rc("text", usetex=True)
plt.rcParams["text.latex.preamble"] = "\n".join(
    [
        r"\usepackage{tgheros}",
        r"\usepackage{sansmath}",
        r"\sansmath",
        r"\usepackage{siunitx}",
        r"\sisetup{detect-all}",
    ]
)
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 18,
        "axes.labelsize": 16,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "lines.linewidth": 2.0,
        "lines.markersize": 10,
        "errorbar.capsize": 4,
    }
)

if not INPUT_FILE.exists():
    raise FileNotFoundError(INPUT_FILE)

# ── Load + rename ────────────────────────────────────────────────────────
df = pd.read_excel(
    INPUT_FILE,
    sheet_name="Summary",
    usecols="A:I",
    engine="openpyxl",
    skipfooter=1,
).rename(columns={"Background [counts/y/2t/FWHM]": "TG Mean", "Error": "TG Spread"})

df["TG Mean"] = pd.to_numeric(df["TG Mean"], errors="coerce")
df["TG Spread"] = pd.to_numeric(df["TG Spread"], errors="coerce")

# ── Normalize component names ────────────────────────────────────────────
c = df["Component"].astype(str).str.strip()
for prefix, repl in COMPONENT_PREFIX_RENAMES:
    c = c.where(~c.str.startswith(prefix, na=False), repl)
df["Component"] = c

# ── Intrinsic tagging/cleanup ────────────────────────────────────────────
df.loc[
    df["Isotope"].astype(str).str.startswith("bb2n", na=False),
    "Category",
] = "Intrinsic Radioactivity"

intrinsic = df["Category"].astype(str).str.startswith("Intrinsic", na=False)
df.loc[intrinsic & df["Component"].astype(str).str.contains("LXe", na=False), "Component"] = "LXe"

df = df.loc[~df["Isotope"].isin(["bb0n", "Cs-137"]) & intrinsic].copy()

def build_grouped(frame: pd.DataFrame, r_mm: Optional[float]):
    """Return grouped DataFrame, external HFE scale, and HFE-volume scale."""
    ext_scale = None
    vol_scale = None
    working = frame.copy()
    if r_mm is not None:
        ext_scale = float(np.exp(-MU * (r_mm - BASELINE_RADIUS_MM)))
        vol_scale = _hfe_volume_scale(r_mm)
        ext_only = working["Component"].isin(OUTSIDE_HFE - HFE_VOLUME)
        hfe_vol = working["Component"].isin(HFE_VOLUME)
        working.loc[ext_only, ["TG Mean", "TG Spread"]] *= ext_scale
        working.loc[hfe_vol, ["TG Mean", "TG Spread"]] *= vol_scale

    grouped_frame = (
        working.groupby("Component", dropna=False)
        .agg(
            TG_Mean=("TG Mean", "sum"),
            TG_Spread=("TG Spread", sqrt_sum_sq),
        )
        .sort_values("TG_Mean")
    )
    grouped_frame = grouped_frame[grouped_frame["TG_Mean"] >= MIN_COUNT]
    return grouped_frame, ext_scale, vol_scale


def print_budget(
    grouped_frame: pd.DataFrame,
    r_mm: float,
    ext_scale: Optional[float],
    vol_scale: Optional[float],
) -> None:
    """Print a formatted table of background contributions."""
    scale_str = ""
    if ext_scale is not None:
        scale_str = (
            f", ext. scale = {ext_scale:.4f}"
            f", HFE vol. scale = {vol_scale:.4f}"
        )
    print(f"\n=== Inner Vessel r = {r_mm:.0f} mm{scale_str} ===")
    col_w = 40
    print(f"{'Component':<{col_w}} {'Mean [cts/y/2t/FWHM]':>22}  {'Spread':>12}")
    print("-" * (col_w + 38))
    total_mean = 0.0
    total_spread_sq = 0.0
    for comp, row in grouped_frame.sort_values("TG_Mean", ascending=False).iterrows():
        if comp in HFE_VOLUME:
            tag = " [hfe-vol]"
        elif comp in OUTSIDE_HFE:
            tag = " [ext]"
        else:
            tag = ""
        label = f"{comp}{tag}"
        print(f"{label:<{col_w}} {row['TG_Mean']:>22.4e}  {row['TG_Spread']:>12.4e}")
        total_mean += row["TG_Mean"]
        total_spread_sq += row["TG_Spread"] ** 2
    print("-" * (col_w + 38))
    print(f"{'TOTAL':<{col_w}} {total_mean:>22.4e}  {np.sqrt(total_spread_sq):>12.4e}")


def plot_budget(
    grouped_frame: pd.DataFrame,
    r_mm: Optional[float],
    ext_scale: Optional[float],
    vol_scale: Optional[float],
    hfe_mass_t: Optional[float] = None,
    baseline_frame: Optional[pd.DataFrame] = None,
):
    """Plot a single budget figure for the requested IV radius."""
    labels = grouped_frame.index.astype(str).tolist()
    y = np.arange(len(labels))

    fig, ax = plt.subplots(figsize=(12, 8))

    for yi, label, (val, err) in zip(
        y, labels, grouped_frame[["TG_Mean", "TG_Spread"]].to_numpy()
    ):
        if label in HFE_VOLUME:
            color = "darkorange"
        elif label in OUTSIDE_HFE:
            color = "darkred"
        else:
            color = "darkgreen"
        # grey line from baseline value to current value
        if baseline_frame is not None and label in baseline_frame.index:
            base_val = float(baseline_frame.loc[label, "TG_Mean"])
            if base_val >= MIN_COUNT and base_val != float(val):
                ax.plot(
                    [base_val, float(val)], [yi, yi],
                    color="grey", linewidth=1.2, zorder=1,
                )
                ax.plot(
                    base_val, yi,
                    marker="o", color="grey", markersize=5,
                    zorder=1, linestyle="none",
                )
        ax.errorbar(
            float(val),
            float(yi),
            xerr=float(err),
            color=color,
            marker="o",
            capthick=2.0,
            zorder=2,
        )

    # minimal TeX escaping for tick labels
    ax.set_yticks(y, [s.replace("&", r"\&").replace("_", r"\_") for s in labels])

    ax.set_xscale("log")
    ax.set_xlim(MIN_COUNT, 10.0)
    ax.set_xlabel(r"Background counts/(y/2t/FWHM)")
    ax.grid(which="major", axis="x", linewidth=1.5)
    ax.grid(which="minor", axis="x", linestyle="dashed", linewidth=1.0)

    ext_label = (
        "External components"
        if ext_scale is None
        else f"External components ({ext_scale:.3f}x)"
    )
    hfe_vol_label = (
        "HFE volume"
        if vol_scale is None
        else f"HFE volume ({vol_scale:.3f}x)"
    )
    legend_handles = [
        Line2D([], [], color="darkgreen", marker="o", linestyle="-",
               label="Internal components"),
        Line2D([], [], color="darkred", marker="o", linestyle="-",
               label=ext_label),
        Line2D([], [], color="darkorange", marker="o", linestyle="-",
               label=hfe_vol_label),
    ]
    if baseline_frame is not None:
        legend_handles.append(
            Line2D([], [], color="grey", marker="o", linestyle="-",
                   linewidth=1.2, markersize=5,
                   label="Baseline design")
        )
    ax.legend(handles=legend_handles)

    if r_mm is not None:
        if hfe_mass_t is not None:
            ax.set_title(
                f"Background Budget (HFE mass = {hfe_mass_t:.1f} t)", pad=20
            )
        else:
            ax.set_title(
                f"Background Budget (Inner Vessel r = {r_mm:.0f} mm)", pad=20
            )

    fig.tight_layout()
    return fig


base_df = df.copy()
output_dir = Path(__file__).resolve().parent / "figures"
output_dir.mkdir(parents=True, exist_ok=True)

_baseline_grouped, _, _ = build_grouped(base_df, BASELINE_RADIUS_MM)

for _r_mm in [1121.80, 1308.37]:
    _grouped, _ext_scale, _vol_scale = build_grouped(base_df, _r_mm)
    _hfe_mass_t = _hfe_mass_tonnes(_r_mm)
    print_budget(_grouped, _r_mm, _ext_scale, _vol_scale)
    _fig = plot_budget(
        _grouped, _r_mm, _ext_scale, _vol_scale, _hfe_mass_t,
        baseline_frame=_baseline_grouped,
    )
    _output_path = (
        output_dir / f"{Path(__file__).stem}_r{int(round(_r_mm))}mm.png"
    )
    _fig.savefig(_output_path)
    plt.close(_fig)
