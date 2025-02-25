#!/usr/bin/env python
"""
Simplified code to create the BackgroundBudget_Intrinsic_ByComponent_ plot.
"""

import os
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import matplotlib

# Set up LaTeX rendering with Helvetica font
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


def SqrtSumSq(x):
    """Compute the square root of the sum of squares."""
    x = np.array(x.tolist())
    return np.sqrt(np.sum(x * x))


def reformat_labels_latex(label):
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


def make_plot(
    df,
    groupby,
    filename,
    xlimits=[1e-8, 1],
    color="darkgreen",
    fontsize=8,
    min_count=None,
):
    """
    Group the DataFrame by the specified key, compute error bars,
    and produce a horizontal errorbar plot saved to filename.

    The plot now shows the background counts directly (in counts/y/2t/FWHM).

    Parameters:
        min_count: float, optional
            Minimum background count to show in the plot. Components below this
            threshold will be excluded.
    """
    # Group and aggregate the data
    df_grouped = df.groupby(groupby).agg({"TG Mean": np.sum, "TG Spread": SqrtSumSq})
    df_grouped.sort_values("TG Mean", ascending=True, inplace=True)

    # Filter out components below the minimum count threshold
    if min_count is not None:
        df_grouped = df_grouped[df_grouped["TG Mean"] >= min_count]

    fig, ax = plt.subplots(figsize=(5, 4))
    labels = []
    marker = "."

    # Loop over groups to plot the background counts and error
    for i, (idx, row) in enumerate(df_grouped.iterrows()):
        value = row["TG Mean"]
        err = row["TG Spread"]
        label = idx if isinstance(idx, str) else " ".join(idx)
        labels.append(label)
        ax.errorbar(
            value,
            i * 2,
            xerr=err,
            lw=2,
            capsize=2,
            capthick=2,
            color=color,
            marker=marker,
            markersize=10,
        )

    # Format y-axis with LaTeX formatted labels
    labels = [reformat_labels_latex(l) for l in labels]
    y_positions = np.arange(0, 2 * len(labels), 2)
    ax.set_yticks(y_positions)
    ax.set_yticklabels(labels, fontsize=fontsize)

    ax.set_xscale("log")
    ax.set_xlim(xlimits)
    fmt = mticker.FuncFormatter(
        lambda x, pos: f"{x:f}".rstrip("0") if x < 1 else f"{x:.0f}"
    )
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(r"Background counts/(y/2t/FWHM)", fontsize=13)
    ax.grid(which="major", axis="x", linestyle="-")
    ax.grid(which="minor", axis="x", linestyle="dashed")

    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Plot BackgroundBudget_Intrinsic_ByComponent_"
    )
    parser.add_argument(
        "--input_file",
        default="Summary_D-047_v86_250113-233135_2025-02-21.xlsx",
        help="Path to input Excel file",
    )
    parser.add_argument(
        "--output_folder", default="./", help="Path to output folder"
    )
    parser.add_argument(
        "--min_count",
        type=float,
        default=1e-8,
        help="Minimum background count to show in plot",
    )
    args = parser.parse_args()

    inFile = Path(args.input_file)
    outFolder = Path(args.output_folder)
    outFolder.mkdir(parents=True, exist_ok=True)

    if not inFile.exists():
        sys.exit(f"ERROR: File {inFile} not found!")

    table = inFile.stem  # Use the file stem to tag the output file name

    # Read the Excel file (assumes sheet "Summary" and proper columns)
    df = pd.read_excel(
        inFile,
        sheet_name="Summary",
        header=0,
        usecols="A:I",
        engine="openpyxl",
        skipfooter=1,
    )

    # Rename columns for clarity
    df.rename(
        columns={"Background [counts/y/2t/FWHM]": "TG Mean", "Error": "TG Spread"},
        inplace=True,
    )
    df = df.applymap(lambda x: x.replace("&", r"\&") if isinstance(x, str) else x)

    # Clean up the Component names (only those relevant for grouping by Component)
    df.loc[df["Component"].str.startswith("Outer Cryostat Support"), "Component"] = (
        "Outer Vessel Support"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat Support"), "Component"] = (
        "Inner Vessel Support"
    )
    df.loc[df["Component"].str.startswith("Outer Cryostat ("), "Component"] = (
        "Outer Vessel"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat ("), "Component"] = (
        "Inner Vessel"
    )
    df.loc[df["Component"].str.startswith("Outer Cryostat Liner"), "Component"] = (
        "Outer Vessel Liner"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat Liner"), "Component"] = (
        "Inner Vessel Liner"
    )
    df.loc[
        df["Component"].str.startswith("Outer Cryostat Feedthrough"), "Component"
    ] = "Outer Vessel Feedthrough"
    df.loc[
        df["Component"].str.startswith("Inner Cryostat Feedthrough"), "Component"
    ] = "Inner Vessel Feedthrough"
    df.loc[df["Component"].str.startswith("Inactive LXe"), "Component"] = "Skin LXe"
    df.loc[df["Component"].str.startswith("Active LXe"), "Component"] = "TPC LXe"

    # Mark intrinsic radioactivity rows (e.g. bb2n becomes intrinsic)
    df.loc[df.Isotope.str.startswith("bb2n"), "Category"] = "Intrinsic Radioactivity"
    # For intrinsic backgrounds, do not distinguish between various LXe components
    df.loc[
        df["Category"].str.startswith("Intrinsic")
        & df["Component"].str.contains("LXe"),
        "Component",
    ] = "LXe"

    # Remove rows not needed
    df = df[~df["Isotope"].isin(["bb0n", "Cs-137"])]

    # Filter for intrinsic radioactivity
    df_intrinsic = df[df["Category"].str.startswith("Intrinsic")]

    # Produce the plot grouped by Component using direct background counts
    output_file = (
        outFolder / f"BackgroundBudget_Intrinsic_ByComponent_{table}_2tonne.pdf"
    )
    make_plot(
        df_intrinsic,
        ["Component"],
        str(output_file),
        color="darkgreen",
        min_count=args.min_count,
    )


if __name__ == "__main__":
    main()
