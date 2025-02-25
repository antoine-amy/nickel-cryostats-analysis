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
from matplotlib.lines import Line2D
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
    outside_color="darkblue",
    fontsize=10,
    min_count=None,
    scale_factor=None,
):
    """
    Group the DataFrame by the specified key, compute error bars,
    and produce a horizontal errorbar plot saved to filename.
    Components are colored differently based on whether they're inside or outside HFE.

    Parameters:
        min_count: float, optional
            Minimum background count to show in the plot. Components below this
            threshold will be excluded.
        scale_factor: float, optional
            If provided, indicates the scaling factor applied to external components
    """
    # Set default xlimits if not provided
    if xlimits is None:
        xlimits = [0.0001, 10]

    # Group and aggregate the data
    df_grouped = df.groupby(groupby).agg({"TG Mean": np.sum, "TG Spread": sqrt_sum_sq})
    df_grouped.sort_values("TG Mean", ascending=True, inplace=True)

    # Filter out components below the minimum count threshold
    if min_count is not None:
        df_grouped = df_grouped[df_grouped["TG Mean"] >= min_count]

    fig, ax = plt.subplots(figsize=(12, 9))
    labels = []
    marker = "."

    # Create empty lines for legend
    inside_line = Line2D(
        [],
        [],
        color=inside_color,
        marker=marker,
        linestyle="-",
        label="Internal Components",
    )
    outside_line = Line2D(
        [],
        [],
        color=outside_color,
        marker=marker,
        linestyle="-",
        label="External Components",
    )
    legend_elements = [inside_line, outside_line]

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
    ax.set_xlabel(r"Background counts/(y/2t/FWHM)", fontsize=14)
    ax.grid(which="major", axis="x", linestyle="-")
    ax.grid(which="minor", axis="x", linestyle="dashed")

    # Add legend with scaling information if applicable
    if scale_factor is not None:
        increase_percent = (scale_factor - 1) * 100
        # Modify the external components label to include the scaling factor
        outside_line.set_label(f"External Components (+{increase_percent:.1f}\\%)")

    ax.legend(handles=legend_elements, loc="upper left", fontsize=fontsize + 1)

    # Increase tick label sizes
    ax.tick_params(axis="both", which="major", labelsize=fontsize + 1)
    ax.tick_params(axis="both", which="minor", labelsize=fontsize + 1)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True, bbox_inches="tight")
    plt.close(fig)


def calculate_hfe_scale(iv_radius, baseline_radius=1691.0, mu=0.0075):
    """
    Calculate scaling factor for external components based on HFE shielding thickness change.

    Parameters:
    -----------
    iv_radius : float
        New inner vessel radius in mm
    baseline_radius : float
        Baseline inner vessel radius in mm (default: 1691.0 mm)
    mu : float
        Attenuation coefficient in mm^-1 (default: 0.0075 mm^-1)

    Returns:
    --------
    float : Scaling factor for external component backgrounds
    """
    return np.exp(-mu * (iv_radius - baseline_radius))


def main():
    """Parse command-line arguments, process data, and generate background budget plots."""
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
    parser.add_argument(
        "--iv_radius",
        type=float,
        help="Inner vessel radius in mm for HFE scaling. "
        "If provided, external components will be scaled",
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
    # Replace '&' with '\&' in string columns
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.replace('&', r'\&', regex=False)

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
    df_intrinsic = df[df["Category"].str.startswith("Intrinsic")].copy()

    # Apply HFE scaling to external components if radius is provided
    if args.iv_radius is not None:
        scale_factor = calculate_hfe_scale(args.iv_radius)
        # Scale the background and error for external components
        mask = df_intrinsic["Component"].isin(OUTSIDE_HFE)
        df_intrinsic.loc[mask, "TG Mean"] *= scale_factor
        df_intrinsic.loc[mask, "TG Spread"] *= scale_factor

        # Add radius info to output filename
        output_file = (
            out_folder / "background"
            / f"BackgroundBudget_Intrinsic_ByComponent_{table}_2tonne_r{args.iv_radius:.0f}mm.pdf"
        )
    else:
        output_file = (
            out_folder /"background"/ f"BackgroundBudget_Intrinsic_ByComponent_{table}_2tonne.pdf"
        )

    make_plot(
        df_intrinsic,
        ["Component"],
        str(output_file),
        inside_color="darkgreen",
        outside_color="darkblue",
        min_count=args.min_count,
        scale_factor=scale_factor if args.iv_radius is not None else None,
    )


if __name__ == "__main__":
    main()
