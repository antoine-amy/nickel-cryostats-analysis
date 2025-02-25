#!/usr/bin/env python
#
# Samuele Sangiorgio

"""PlotBackgroundBudget.py: create figures of background budget

Input is from DB Excel file. Plots are generated grouping by Isotope, Material, or Component."""

import os.path
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


def get_iv_radius_from_thickness(thickness_cm):
    """Convert HFE thickness to inner vessel radius"""
    min_radius = 1026  # mm
    return min_radius + thickness_cm * 10  # convert cm to mm


def get_attenuation_factor(initial_radius, final_radius, attenuation_coeff=0.0075):
    """Calculate attenuation factor between two radii"""
    return np.exp(-attenuation_coeff * (final_radius - initial_radius))


def scale_external_background(row, reference_radius, new_radius):
    """Scale background for components outside HFE based on radius change"""
    # List of components affected by HFE attenuation
    external_components = [
        # External components
        "Water Shield",
        "Outer Vessel",
        "Outer Vessel Support",
        "Outer Vessel Feedthrough",
        "Outer Vessel Liner",
        # Vacuum space components
        "Multi-Layer Insulation",
        "Transition Box",
        "Support Rods and Spacers",  # Cryostat supports
        "Utility Lines",
        "Handling Hardware",
        # Inner vessel components (HFE between IV and TPC)
        "Inner Vessel",
        "Inner Vessel Support",
        "Inner Vessel Feedthrough",
        "Inner Vessel Liner",
    ]

    if row["Component"] in external_components:
        return row["TG Mean"] * get_attenuation_factor(reference_radius, new_radius)
    return row["TG Mean"]


def make_plot(
    df,
    groupby,
    filename,
    reference_radius=1691,
    new_radius=None,
    xlimits=[1e-2, 100],
    color="darkblue",
    fontsize=8,
    total=None,
):
    """
    Create background budget plot with optional HFE thickness scaling

    Parameters:
    df: DataFrame with background data
    groupby: Column to group results by
    filename: Output filename
    reference_radius: Reference IV radius in mm (default 1691mm)
    new_radius: New IV radius for scaling (if None, no scaling applied)
    xlimits: X-axis limits for plot
    color: Plot color
    fontsize: Font size for labels
    """
    print(f"\n{groupby=}")
    df["Original Mean"] = df["TG Mean"].copy()

    # Apply attenuation scaling if new_radius specified
    if new_radius is not None:
        df["TG Mean"] = df.apply(
            lambda row: scale_external_background(row, reference_radius, new_radius),
            axis=1,
        )
        # Scale spread proportionally
        scaling = df["TG Mean"] / df["Original Mean"]
        df["TG Spread"] = df["TG Spread"] * scaling

    # Group data
    df2 = df.groupby(groupby).agg({"TG Mean": np.sum, "TG Spread": SqrtSumSq})
    df2.sort_values("TG Mean", ascending=True, inplace=True)

    fig, ax0 = plt.subplots(figsize=(5, 4))

    if total is None:
        total = df["TG Mean"].sum() / 2000.0
    print("total = %s" % (total))

    # Plot data
    labels = []
    for index, row in df2.iterrows():
        if row["TG Mean"] / total * 100 < xlimits[0] * 2 * 2000:
            continue

        index = index if isinstance(index, str) else " ".join(index)
        labels.append(index)

        value = row["TG Mean"] / 2000.0 / total * 100
        err = row["TG Spread"] / 2000.0 / total * 100

        ax0.errorbar(
            value,
            len(labels) * 2 - 2,
            xerr=err,
            lw=2,
            capsize=2,
            capthick=2,
            color=color,
            marker=".",
            markersize=10,
        )

    # Format plot
    labels = [reformat_labels_latex(l) for l in labels]
    nn = len(labels)
    y = np.arange(0, 2 * nn, 2)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=fontsize)
    ax0.set_xscale("log")
    ax0.set_xlim(xlimits)

    fmt = mticker.FuncFormatter(
        lambda x, pos: f"{x:f}".rstrip("0") if x < 1 else f"{x:.0f}"
    )
    ax0.xaxis.set_major_formatter(fmt)
    ax0.grid(which="major", axis="x", linestyle="-")
    ax0.set_xlabel(r"\% of total SS counts/(FWHM$\cdot$2000kg)", fontsize=13)
    plt.xticks(fontsize=14)
    ax0.grid(which="minor", axis="x", linestyle="dashed")
    plt.subplots_adjust(left=0.32, bottom=0.15)
    plt.tick_params(axis="both", which="major", direction="in", length=8)
    plt.tick_params(axis="both", which="minor", direction="in", length=4)

    plt.tight_layout()
    plt.savefig(filename, dpi=300, transparent=True)

    # Convert data to format expected by summary_plot
    data = []
    for index, row in df2.iterrows():
        value = row["TG Mean"] / 2000.0 / total * 100
        err = row["TG Spread"] / 2000.0 / total * 100
        data.append((value, index, err, color, "."))

    return data


def reformat_labels_latex(l):
    if l == "U-238":
        return r"$^{238}$U"
    elif l == "Th-232":
        return r"$^{232}$Th"
    elif l == "K-40":
        return r"$^{40}$K"
    elif l == "Co-60":
        return r"$^{60}$Co"
    elif l == "Xe-137":
        return r"$^{137}$Xe"
    elif l == "Rn-222":
        return r"$^{222}$Rn"
    elif l == "Ar-42":
        return r"$^{42}$Ar"
    elif l == "Al-26":
        return r"$^{26}$Al"
    elif l == r"bb2n":
        return r"$2\nu\beta\beta$"
    elif l == r"B8nu":
        return r"Solar $\nu$"
    return l


def SqrtSumSq(x):
    """Calculate quadrature sum"""
    x = np.array(x.tolist())
    return np.sqrt(np.sum(np.multiply(x, x)))


def summary_plot(
    data_category, data_intrinsic, data_radon, data_invariant, data_exposure, filename
):
    fig, ax0 = plt.subplots(figsize=(5, 5))

    #    print(data_category)
    labels = []
    props = dict(boxstyle="round", facecolor="white", alpha=0.7, edgecolor="white")

    # TODO: refactor as a loop over the categories instead of repeating the same code over and over
    # exposure-based backgrounds sum
    for value, label, err, color, marker in data_exposure:
        labels.append(label)
        ax0.errorbar(
            value,
            len(labels) * 2 - 2,
            xerr=None,  # xuplims=xuplims,
            lw=2,
            capsize=2,
            capthick=2,
            color="darkred",
            marker=marker,
            markersize=12,
            markerfacecolor="none",
        )

    idx = [r[1] for r in data_category].index("Exposure-based")
    value, label, err, color, marker = data_category[idx]
    labels.append(r"\textbf{Total}")
    ax0.errorbar(
        value,
        len(labels) * 2 - 2,
        xerr=None,  # xuplims=xuplims,
        lw=2,
        capsize=2,
        capthick=2,
        color="darkred",
        marker=marker,
        markersize=12,
    )
    ax0.text(
        0.15, len(labels) * 2 - 2 - 0.2, r"Exposure-based", fontsize=12, bbox=props
    )
    labels.append(" ")
    ax0.axhline(len(labels) * 2 - 2, color="black")

    # Invariant backgrounds sum
    for value, label, err, color, marker in data_invariant:
        if label not in ["Xe-137", "B8nu"]:
            continue
        labels.append(label)
        ax0.errorbar(
            value,
            len(labels) * 2 - 2,
            xerr=None,  # xuplims=xuplims,
            lw=2,
            capsize=2,
            capthick=2,
            color="orange",
            marker=marker,
            markersize=10,
            markerfacecolor="none",
        )

    idx = [r[1] for r in data_category].index("Invariant")
    value, label, err, color, marker = data_category[idx]
    labels.append(r"\textbf{Total}")
    ax0.errorbar(
        value,
        len(labels) * 2 - 2,
        xerr=None,  # xuplims=xuplims,
        lw=2,
        capsize=2,
        capthick=2,
        color="orange",
        marker=marker,
        markersize=10,
    )

    ax0.text(0.15, len(labels) * 2 - 2 - 0.2, r"Invariant", fontsize=12, bbox=props)
    labels.append(" ")
    ax0.axhline(len(labels) * 2 - 2, color="black")

    # intrinsic backgrounds sum
    for value, label, err, color, marker in data_intrinsic:
        labels.append(label)
        ax0.errorbar(
            value,
            len(labels) * 2 - 2,
            xerr=err,  # xuplims=xuplims,
            lw=2,
            capsize=2,
            capthick=2,
            color="darkgreen",
            marker=marker,
            markersize=12,
            markerfacecolor="none",
        )

    idx = [r[1] for r in data_category].index("Intrinsic Radioactivity")
    value, label, err, color, marker = data_category[idx]
    labels.append(r"\textbf{Total}")
    ax0.errorbar(
        value,
        len(labels) * 2 - 2,
        xerr=err,  # xuplims=xuplims,
        lw=2,
        capsize=2,
        capthick=2,
        color="darkgreen",
        marker=marker,
        markersize=12,
    )

    ax0.text(
        0.15,
        len(labels) * 2 - 2 - 0.2,
        r"Intrinsic Radioactivity",
        fontsize=12,
        bbox=props,
    )
    labels.append(" ")
    ax0.axhline(len(labels) * 2 - 2, color="black")

    # radon backgrounds sum
    for value, label, err, color, marker in data_radon:
        labels.append(label)
        ax0.errorbar(
            value,
            len(labels) * 2 - 2,
            xerr=None,  # xuplims=xuplims,
            lw=2,
            capsize=2,
            capthick=2,
            color="darkblue",
            marker=marker,
            markersize=12,
            markerfacecolor="none",
        )

    idx = [r[1] for r in data_category].index("Radon Outgassing")
    value, label, err, color, marker = data_category[idx]
    labels.append(r"\textbf{Total}")
    ax0.errorbar(
        value,
        len(labels) * 2 - 2,
        xerr=None,  # xuplims=xuplims,
        lw=2,
        capsize=2,
        capthick=2,
        color="darkblue",
        marker=marker,
        markersize=12,
    )

    ax0.text(
        0.15,
        len(labels) * 2 - 2 - 0.2,
        r"$^{214}$Bi from $^{222}$Rn",
        fontsize=12,
        bbox=props,
    )

    # overall plot formatting (labels, etc...)
    labels = [reformat_labels_latex(l) for l in labels]
    nn = len(labels)
    y = np.arange(0, 2 * nn, 2)
    ax0.set_yticks(y)
    ax0.set_yticklabels(labels, fontsize=10)
    ax0.set_xlim([1e-1, 100])
    ax0.set_xscale("log")
    fmt = mticker.FuncFormatter(
        lambda x, pos: f"{x:f}".rstrip("0") if x < 1 else f"{x:.0f}"
    )
    ax0.xaxis.set_major_formatter(fmt)
    ax0.grid(which="major", axis="x", linestyle="-")
    # ax0.set_xlabel(r"cts/(FWHM$\cdot$kg$\cdot$year)")
    ax0.set_xlabel(r"\% of total SS counts/(FWHM$\cdot$2000kg)", fontsize=14)
    plt.xticks(fontsize=16)
    ax0.grid(which="minor", axis="x", linestyle="dashed")
    #    plt.subplots_adjust(left=0.32, bottom=0.15)
    plt.tick_params(axis="both", which="major", direction="in", length=8)
    plt.tick_params(axis="both", which="minor", direction="in", length=4)

    plt.tight_layout()
    # plt.show()

    plt.savefig(filename, dpi=300, transparent=True)


def resave_excel(path):
    import xlwings as xl

    app = xl.App(visible=False)
    book = app.books.open(path)
    book.save()
    app.kill()
    return


def main(argv):
    parser = argparse.ArgumentParser(description="Create figures of background budget")
    parser.add_argument(
        "--input_file",
        default="Summary_D-047_v86_250113-233135_2025-02-21.xlsx",
        type=str,
        help="Path to input Excel file",
    )
    parser.add_argument(
        "--output_folder", default="./", type=str, help="Path to output folder"
    )
    args = parser.parse_args()

    inTableName = Path(args.input_file)
    outFolder = Path(args.output_folder)
    os.makedirs(outFolder, exist_ok=True)

    if not inTableName.exists():
        sys.exit("ERROR: File %s was not found!" % inTableName)

    table = ".".join(os.path.basename(inTableName).split(".")[:-1])

    # Somehow this is required so that the values from the formulas are saved in the Excel file
    # and pandas can read them
    resave_excel(inTableName)

    # specific columns depend on the database version
    df = pd.read_excel(
        inTableName,
        sheet_name="Summary",
        header=0,
        usecols="A:I",
        engine="openpyxl",
        skipfooter=1,
    )

    df.rename(
        columns={"Background [counts/y/2t/FWHM]": "TG Mean", "Error": "TG Spread"},
        inplace=True,
    )
    df = df.applymap(lambda x: x.replace("&", r"\&") if isinstance(x, str) else x)

    # do some relabeling
    df.loc[df["Material"].str.endswith("Kapton"), "Material"] = "Polyimide"
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
    df.loc[df.Isotope.str.startswith("bb2n"), "Category"] = "Intrinsic Radioactivity"

    # no need to distinguish between various LXe "components" for intrinsic backgrounds (Ar-42 and bb2n)
    df.loc[
        df["Category"].str.startswith("Intrinsic")
        & df["Component"].str.contains("LXe"),
        "Component",
    ] = "LXe"

    # this is for printing without breaking into multiple lines
    pd.set_option("display.expand_frame_repr", False)

    # remove bb0n rows
    df = df[~df["Isotope"].isin(["bb0n", "Cs-137"])]

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    print(df)

    total = 0
    for index, row in df.iterrows():
        total += row["TG Mean"] / 2000.0

    # Plot by Material, Isotope, and Component separately
    make_plot(
        df[df["Category"].str.startswith("Intrinsic")],
        ["Material"],
        os.path.join(
            outFolder, "BackgroundBudget_Intrinsic_byMaterial_" + table + "_2tonne.pdf"
        ),
        total=total,
        color="darkgreen",
    )

    make_plot(
        df[df["Category"].str.startswith("Intrinsic")],
        ["Component"],
        os.path.join(
            outFolder, "BackgroundBudget_Intrinsic_ByComponent_" + table + "_2tonne.pdf"
        ),
        total=total,
        color="darkgreen",
    )

    data_intrinsic = make_plot(
        df[df["Category"].str.startswith("Intrinsic")],
        ["Isotope"],
        os.path.join(
            outFolder, "BackgroundBudget_Intrinsic_ByIsotope_" + table + "_2tonne.pdf"
        ),
        total=total,
        fontsize=16,
        color="darkgreen",  # Added to ensure consistent color in summary
    )

    data_category = make_plot(
        df, ["Category"], outFolder / f"BackgroundBudgetByCategory_{table}_2tonne.pdf"
    )

    data_radon = make_plot(
        df[df["Category"].str.startswith("Radon")],
        ["Component"],
        os.path.join(
            outFolder, "BackgroundBudgetByRadonComponent_" + table + "_2tonne.pdf"
        ),
        total=total,
        color="darkblue",  # Added to ensure consistent color in summary
    )

    # This is by Material & Isotope
    make_plot(
        df,
        ["Material", "Isotope"],
        os.path.join(
            outFolder, "BackgroundBudgetByMaterialIsotope_" + table + "_2tonne.pdf"
        ),
    )

    # This is by Component & Isotope
    make_plot(
        df,
        ["Component", "Isotope"],
        os.path.join(
            outFolder, "BackgroundBudgetByComponentIsotope_" + table + "_2tonne.pdf"
        ),
        fontsize=6,
    )

    # This is for exposure-based backgrounds
    data_exposure = make_plot(
        df[df["Category"].str.startswith("Exposure")],
        ["Material"],
        os.path.join(
            outFolder, "BackgroundBudgetByExposureMaterial_" + table + "_2tonne.pdf"
        ),
        total=total,
        color="darkred",  # Added to ensure consistent color in summary
    )

    data_invariant = make_plot(
        df[df["Category"].str.startswith("Invariant")],
        ["Isotope"],
        os.path.join(
            outFolder, "BackgroundBudgetInvariantByIsotope_" + table + "_2tonne.pdf"
        ),
        total=total,
        color="orange",  # Added to ensure consistent color in summary
    )

    summary_plot(
        data_category,
        data_intrinsic,
        data_radon,
        data_invariant,
        data_exposure,
        filename=os.path.join(
            outFolder, "BackgroundBudgetOverview_" + table + "_2tonne.pdf"
        ),
    )


if __name__ == "__main__":
    main(sys.argv)
