#!/usr/bin/env python
"""
Modified code to create the BackgroundBudget_Intrinsic_ByComponent_ plot with HFE-based coloring.
"""

import sys
import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

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

# Define components outside HFE
OUTSIDE_HFE = [
    "Cryopit concrete and shotcrete",
    "Outer Vessel Support",
    "Outer Cryostat",  # Changed from "Outer Vessel"
    "Inner Vessel Support",
    "Inner Cryostat MLI",
    "Inner Cryostat",  # Changed from "Inner Vessel"
    "CRE Transition Enclosures",
    "CRE Transition Boards",
    "PRE Transition Enclosures",
    "PRE Transition Boards",
    "OD: PMTs, PMT cable, and PMT mounts",
    "OD: Tank",
    "HV Tubes",
    "HV Cables",
    "HV Plunger",
    "HV Feedthrough",
    "HV Feedthrough Core (Teflon)",
    "HV Feedthrough Core (Cable)",
    "HV Feedthrough Core (Conductive PE)",
]


def sqrt_sum_sq(x):
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
    xlimits=None,
    inside_color="darkgreen",
    outside_color="blue",
    fontsize=8,
    min_count=None,
):
    """
    Group the DataFrame by the specified key, compute error bars,
    and produce a horizontal errorbar plot saved to filename.
    Components are colored differently based on whether they're inside or outside HFE.

    Parameters:
        min_count: float, optional
            Minimum background count to show in the plot. Components below this
            threshold will be excluded.
    """
    if xlimits is None:
        xlimits = [0.0001, 1]

    # Group and aggregate the data
    df_grouped = df.groupby(groupby).agg({"TG Mean": np.sum, "TG Spread": sqrt_sum_sq})
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

        # Choose color based on whether component is outside HFE
        color = outside_color if label in OUTSIDE_HFE else inside_color

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
    ax.set_xlim(xlimits[0], xlimits[1])
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
    """
    Parse command line arguments, read background data from Excel file,
    and generate a plot of intrinsic background components.
    """
    parser = argparse.ArgumentParser(
        description="Plot BackgroundBudget_Intrinsic_ByComponent_"
    )
    parser.add_argument(
        "--input_file",
        default="background/Summary_D-047_v86_250113-233135_2025-02-21.xlsx",
        help="Path to input Excel file",
    )
    parser.add_argument(
        "--output_folder", default="./", help="Path to output folder"
    )
    parser.add_argument(
        "--min_count",
        type=float,
        default=0.0001,
        help="Minimum background count to show in plot",
    )
    args = parser.parse_args()

    in_file = Path(args.input_file)
    out_folder = Path(args.output_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    if not in_file.exists():
        sys.exit(f"ERROR: File {in_file} not found!")

    table = in_file.stem  # Use the file stem to tag the output file name

    # Read the Excel file
    df = pd.read_excel(
        in_file,
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
    # Replace ampersands with escaped ampersands for LaTeX compatibility
    df = df.replace({"&": r"\&"}, regex=True)

    # Clean up the Component names
    # No need to clean up these names since they match OUTSIDE_HFE list
    df.loc[df["Component"].str.startswith("Outer Cryostat Support"), "Component"] = (
        "Outer Vessel Support"
    )
    df.loc[df["Component"].str.startswith("Inner Cryostat Support"), "Component"] = (
        "Inner Vessel Support"
    )
    # Keep original names for Outer/Inner Cryostat to match OUTSIDE_HFE list
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
    df.loc[
        df["Component"].str.startswith("Outer Cryostat Feedthrough"), "Component"
    ] = "Outer Cryostat Feedthrough"
    df.loc[
        df["Component"].str.startswith("Inner Cryostat Feedthrough"), "Component"
    ] = "Inner Cryostat Feedthrough"
    df.loc[df["Component"].str.startswith("Inactive LXe"), "Component"] = "Skin LXe"
    df.loc[df["Component"].str.startswith("Active LXe"), "Component"] = "TPC LXe"

    # Mark intrinsic radioactivity rows
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
        out_folder /"background"/ f"BackgroundBudget_Intrinsic_ByComponent_{table}_2tonne.pdf"
    )
    make_plot(
        df_intrinsic,
        ["Component"],
        str(output_file),
        inside_color="darkgreen",
        outside_color="blue",
        min_count=args.min_count,
    )


if __name__ == "__main__":
    main()
