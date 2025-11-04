#!/usr/bin/env python
"""
Create the BackgroundBudget_Intrinsic_ByComponent_ plot with HFE-based coloring.
Enhanced styling for better readability in LaTeX documents.
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

# ── LaTeX + Helvetica setup ──────────────────────────────────────────────
matplotlib.rc("text", usetex=True)
matplotlib.rcParams["text.latex.preamble"] = "\n".join(
    [
        r"\usepackage{tgheros}",
        r"\usepackage{sansmath}",
        r"\sansmath",
        r"\usepackage{siunitx}",
        r"\sisetup{detect-all}",
    ]
)

# ── Global style for figure legibility in LaTeX ──────────────────────────
plt.rcParams.update(
    {
        "font.size": 14,
        "axes.titlesize": 22,
        "axes.labelsize": 20,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "legend.fontsize": 16,
        "lines.linewidth": 3.0,
        "lines.markersize": 14,
        "errorbar.capsize": 6,
    }
)

# ── Components considered outside HFE (external) ─────────────────────────
# (Per request: removed "HV Cables" and "HV Feedthrough" from this list.)
OUTSIDE_HFE: List[str] = [
    "Cryopit concrete and shotcrete",
    "Outer Vessel Support",
    "Outer Cryostat",
    "Inner Vessel Support",
    "Inner Cryostat MLI",
    "Inner Cryostat",
    "CRE Transition Enclosures", #can be moved up if contribution is significant.
    "CRE Transition Boards", #can be moved up if contribution is significant.
    "PRE Transition Enclosures", #can be moved up if contribution is significant.
    "PRE Transition Boards", #can be moved up if contribution is significant.
    "OD: PMTs, PMT cable, and PMT mounts",
    "OD: Tank",
    #outer cryostat superstructure.
    # add HFE itself

]


def sqrt_sum_sq(series: pd.Series) -> float:
    """Return sqrt(sum(x^2)) for a numeric Series."""
    arr = series.to_numpy(dtype=float)
    return float(np.sqrt(np.sum(arr ** 2)))


def reformat_labels_latex(label: str) -> str:
    """Convert isotope labels into LaTeX-friendly strings."""
    mapping = {
        "U-238": r"$^{238}$U",
        "Th-232": r"$^{232}$Th",
        "K-40": r"$^{40}$K",
        "Co-60": r"$^{60}$Co",
        "Xe-137": r"$^{137}$Xe",
        "Rn-222": r"$^{222}$Rn",
        "Ar-42": r"$^{42}$Ar",
        "Al-26": r"$^{26}$Al",
        "bb2n": r"$2\nu\beta\beta$",
        "B8nu": r"Solar $\nu$",
    }
    return mapping.get(label, label)


class PlotConfig:
    """Container for plot configuration parameters."""

    def __init__(
        self,
        xlimits: Sequence[float] | None = None,
        inside_color: str = "darkgreen",
        outside_color: str = "darkblue",
        fontsize: int = 10,
        min_count: float | None = None,
        scale_factor: float | None = None,
        iv_radius: float | None = None,
    ) -> None:
        self.xlimits = [0.0001, 10] if xlimits is None else list(xlimits)
        self.inside_color = inside_color
        self.outside_color = outside_color
        self.fontsize = fontsize
        self.min_count = min_count
        self.scale_factor = scale_factor
        self.iv_radius = iv_radius


def _prepare_data(
    df: pd.DataFrame, groupby: Union[str, List[str]], min_count: float | None
) -> pd.DataFrame:
    """Group/filter the data for plotting."""
    grouped = df.groupby(groupby, dropna=False).agg(
        {"TG Mean": "sum", "TG Spread": sqrt_sum_sq}
    )
    grouped = grouped.sort_values("TG Mean", ascending=True)

    if min_count is not None:
        grouped = grouped[grouped["TG Mean"] >= float(min_count)]

    return grouped


def _setup_legend(config: PlotConfig) -> List[Line2D]:
    """Create legend handles."""
    marker = "o"
    inside_line = Line2D(
        [],
        [],
        color=config.inside_color,
        marker=marker,
        linestyle="-",
        label="Internal Components",
        markersize=14,
        linewidth=3.0,
    )
    if config.scale_factor is not None:
        increase_percent = (config.scale_factor - 1.0) * 100.0
        outside_label = f"External Components (+{increase_percent:.1f}\\%)"
    else:
        outside_label = "External Components"

    outside_line = Line2D(
        [],
        [],
        color=config.outside_color,
        marker=marker,
        linestyle="-",
        label=outside_label,
        markersize=14,
        linewidth=3.0,
    )

    return [inside_line, outside_line]


def make_plot(
    df: pd.DataFrame,
    groupby: Union[str, List[str]],
    filename: str,
    **kwargs: object,
) -> None:
    """
    Group the DataFrame by the specified key, compute error bars,
    and produce a horizontal errorbar plot saved to filename.
    Components are colored by whether they are inside or outside HFE.
    """
    config = PlotConfig(**kwargs)
    grouped = _prepare_data(df, groupby, config.min_count)

    fig, ax = plt.subplots(figsize=(16, 12))
    labels: List[str] = []
    marker = "o"
    legend_elements = _setup_legend(config)

    # Plot points
    for i, (idx, row) in enumerate(grouped.iterrows()):
        value = float(row["TG Mean"])
        err = float(row["TG Spread"])

        if isinstance(idx, tuple):
            label_str = " ".join(map(str, idx))
        else:
            label_str = str(idx)

        labels.append(label_str)

        color = (
            config.outside_color
            if label_str in OUTSIDE_HFE
            else config.inside_color
        )

        ax.errorbar(
            value,
            i * 2,
            xerr=err,
            lw=3.0,
            capsize=6,
            capthick=3.0,
            color=color,
            marker=marker,
            markersize=14,
        )

    # Axes formatting
    labels_ltx = [reformat_labels_latex(lbl) for lbl in labels]
    y_positions = np.arange(0, 2 * len(labels_ltx), 2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels_ltx, fontsize=16)

    ax.set_xscale("log")
    ax.set_xlim(config.xlimits[0], config.xlimits[1])

    def _format_x(x: float, _pos: int) -> str:
        return f"{x:f}".rstrip("0").rstrip(".") if x < 1 else f"{x:.0f}"

    ax.xaxis.set_major_formatter(mticker.FuncFormatter(_format_x))
    ax.set_xlabel(r"Background counts/(y/2t/FWHM)", fontsize=20)
    ax.grid(which="major", axis="x", linestyle="-", linewidth=1.5)
    ax.grid(which="minor", axis="x", linestyle="dashed", linewidth=1.0)

    ax.legend(handles=legend_elements, loc="upper left", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=18)
    ax.tick_params(axis="both", which="minor", labelsize=18)

    if (config.scale_factor is not None) and (config.iv_radius is not None):
        ax.set_title(
            f"Background Budget (Inner Vessel r = {config.iv_radius:.0f} mm)",
            fontsize=22,
            pad=20,
        )

    plt.tight_layout()
    plt.savefig(filename, dpi=400, transparent=True, bbox_inches="tight")
    plt.close(fig)


def calculate_hfe_scale(
    iv_radius: float, baseline_radius: float = 1691.0, mu: float = 0.00592496
) -> float:
    """
    Scaling factor for external components based on HFE thickness change.
    Returns exp(-mu * (iv_radius - baseline_radius)).
    """
    return float(np.exp(-mu * (iv_radius - baseline_radius)))


def main() -> None:
    """Parse CLI, process data, and generate background budget plots."""
    parser = argparse.ArgumentParser(
        description="Plot BackgroundBudget_Intrinsic_ByComponent_"
    )
    parser.add_argument(
        "--input_file",
        default=(
            "/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/"
            "nickel-cryostats-analysis/budget/"
            "Summary_D-047_v86_250113-233135_2025-09-11.xlsx"
        ),
        help="Path to input Excel file",
    )
    parser.add_argument(
        "--output_folder",
        default="./",
        help="Path to output folder",
    )
    parser.add_argument(
        "--min_count",
        type=float,
        default=0.0001,
        help="Minimum background count to show in plot",
    )
    parser.add_argument(
        "--iv_radius",
        type=float,
        help=(
            "Inner vessel radius in mm for HFE scaling. "
            "If provided, external components will be scaled."
        ),
    )
    args = parser.parse_args()

    in_file = Path(args.input_file)
    out_folder = Path(
        "/Users/antoine/My Drive/Documents/Thèse/Nickel Cryostats/"
        "nickel-cryostats-analysis/budget"
    )

    if not in_file.exists():
        sys.exit(f"ERROR: File not found: {in_file}")

    table_tag = in_file.stem  # tag output by file stem

    # Read the Excel file
    df = pd.read_excel(
        in_file,
        sheet_name="Summary",
        header=0,
        usecols="A:I",
        engine="openpyxl",
        skipfooter=1,
    ).copy()

    # Rename columns for clarity
    df = df.rename(
        columns={"Background [counts/y/2t/FWHM]": "TG Mean", "Error": "TG Spread"}
    )

    # Escape '&' for LaTeX in object columns
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].str.replace("&", r"\&", regex=False)

    # Normalize Component names to match OUTSIDE_HFE entries
    df.loc[df["Component"].str.startswith("Outer Cryostat Support"), "Component"] = (
        "Outer Vessel Support"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat Support"), "Component"] = (
        "Inner Vessel Support"
    )
    df.loc[df["Component"].str.startswith("Outer Cryostat ("), "Component"] = (
        "Outer Cryostat"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat ("), "Component"] = (
        "Inner Cryostat"
    )
    df.loc[df["Component"].str.startswith("Outer Cryostat Liner"), "Component"] = (
        "Outer Cryostat Liner"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat Liner"), "Component"] = (
        "Inner Cryostat Liner"
    )
    df.loc[df["Component"].str.startswith("Outer Cryostat Feedthrough"), "Component"] = (
        "Outer Cryostat Feedthrough"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat Feedthrough"), "Component"] = (
        "Inner Cryostat Feedthrough"
    )
    df.loc[df["Component"].str.startswith("Inactive LXe"), "Component"] = "Skin LXe"
    df.loc[df["Component"].str.startswith("Active LXe"), "Component"] = "TPC LXe"

    # Intrinsic radioactivity tagging
    df.loc[df["Isotope"].str.startswith("bb2n"), "Category"] = "Intrinsic Radioactivity"

    # Collapse LXe components under "LXe" for intrinsic backgrounds
    mask_intrinsic_lxe = df["Category"].str.startswith("Intrinsic") & df[
        "Component"
    ].str.contains("LXe")
    df.loc[mask_intrinsic_lxe, "Component"] = "LXe"

    # Remove rows not needed
    df = df[~df["Isotope"].isin(["bb0n", "Cs-137"])].copy()

    # Filter for intrinsic radioactivity only
    df_intrinsic = df[df["Category"].str.startswith("Intrinsic")].copy()

    # Apply HFE scaling to external components if radius provided
    scale_factor: float | None = None
    if args.iv_radius is not None:
        scale_factor = calculate_hfe_scale(args.iv_radius)
        external_mask = df_intrinsic["Component"].isin(OUTSIDE_HFE)
        df_intrinsic.loc[external_mask, "TG Mean"] *= scale_factor
        df_intrinsic.loc[external_mask, "TG Spread"] *= scale_factor

        output_file = (
            out_folder
            / f"BackgroundBudget_Intrinsic_ByComponent_{table_tag}_2tonne_r{args.iv_radius:.0f}mm.pdf"
        )
    else:
        output_file = (
            out_folder
            / f"BackgroundBudget_Intrinsic_ByComponent_{table_tag}_2tonne.pdf"
        )

    make_plot(
        df_intrinsic,
        ["Component"],
        str(output_file),
        inside_color="darkgreen",
        outside_color="darkblue",
        min_count=args.min_count,
        scale_factor=scale_factor,
        iv_radius=args.iv_radius,
    )


if __name__ == "__main__":
    main()