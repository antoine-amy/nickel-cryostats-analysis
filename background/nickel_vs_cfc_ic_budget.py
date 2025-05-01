#!/usr/bin/env python
"""
Script to create side-by-side plots comparing background budgets 
for CFC (Carbon Composite Fiber) and Nickel Cryostats.
The CFC data is partially generated from hard-coded values.
Modified to not share the x-axis for cryostat components.
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
    "HV Tubes",
    "HV Cables",
    "HV Plunger",
    "HV Feedthrough",
    "HV Feedthrough Core (Teflon)",
    "HV Feedthrough Core (Cable)",
    "HV Feedthrough Core (Conductive PE)",
    # Add CFC components to OUTSIDE_HFE list to ensure blue color
    "Outer Cryostat (CFC)",
    "Inner Cryostat (CFC)",
    "Outer Cryostat Liner (CFC)",
    "Inner Cryostat Liner (CFC)"
]

# Define components related to cryostats (to be replaced for CFC version)
NICKEL_CRYOSTAT_COMPONENTS = [
    "Outer Cryostat",
    "Inner Cryostat", 
    "Outer Cryostat Liner",
    "Inner Cryostat Liner"
]

# For CFC version, define a separate list
CFC_CRYOSTAT_COMPONENTS = [
    "Outer Cryostat (CFC)",
    "Inner Cryostat (CFC)",
    "Outer Cryostat Liner (CFC)",
    "Inner Cryostat Liner (CFC)"
]

# Combined list of all cryostat components across both types
ALL_CRYOSTAT_COMPONENTS = NICKEL_CRYOSTAT_COMPONENTS + CFC_CRYOSTAT_COMPONENTS

# Define color scheme for different component categories
COMPONENT_COLORS = {
    # Standard colors for error bars and markers
    "inside_hfe": "darkgreen",
    "outside_hfe": "darkblue",
    # Highlight colors for cryostat component bands
    "cryostat_nickel": "gold",        # Highlight color for Nickel cryostat components
    "cryostat_cfc": "firebrick",      # Highlight color for CFC cryostat components
}

# Define hard-coded CFC component data (based on the values provided)
CFC_COMPONENT_DATA = [
    # Inner Cryostat Liner
    ("Inner Cryostat Liner (CFC)", "U-238", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Inner Cryostat Liner (CFC)", "Th-232", "Intrinsic Radioactivity", 9.217e-3, 3.015e-3),
    ("Inner Cryostat Liner (CFC)", "K-40", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Inner Cryostat Liner (CFC)", "Co-60", "Intrinsic Radioactivity", 0.0, 0.0),
    
    # Outer Cryostat Liner
    ("Outer Cryostat Liner (CFC)", "U-238", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Outer Cryostat Liner (CFC)", "Th-232", "Intrinsic Radioactivity", 7.471e-3, 4.631e-3),
    ("Outer Cryostat Liner (CFC)", "K-40", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Outer Cryostat Liner (CFC)", "Co-60", "Intrinsic Radioactivity", 0.0, 0.0),
    
    # Outer Cryostat (Resin)
    ("Outer Cryostat (CFC)", "U-238 Resin", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Outer Cryostat (CFC)", "Th-232 Resin", "Intrinsic Radioactivity", 1.730e-3, 3.576e-3),
    ("Outer Cryostat (CFC)", "K-40 Resin", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Outer Cryostat (CFC)", "Co-60 Resin", "Intrinsic Radioactivity", 0.0, 0.0),
    
    # Outer Cryostat (Fiber)
    ("Outer Cryostat (CFC)", "U-238 Fiber", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Outer Cryostat (CFC)", "Th-232 Fiber", "Intrinsic Radioactivity", 1.994e-2, 1.305e-2),
    ("Outer Cryostat (CFC)", "K-40 Fiber", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Outer Cryostat (CFC)", "Co-60 Fiber", "Intrinsic Radioactivity", 0.0, 0.0),
    
    # Inner Cryostat (Resin)
    ("Inner Cryostat (CFC)", "U-238 Resin", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Inner Cryostat (CFC)", "Th-232 Resin", "Intrinsic Radioactivity", 1.014e-3, 2.193e-3),
    ("Inner Cryostat (CFC)", "K-40 Resin", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Inner Cryostat (CFC)", "Co-60 Resin", "Intrinsic Radioactivity", 0.0, 0.0),
    
    # Inner Cryostat (Fiber)
    ("Inner Cryostat (CFC)", "U-238 Fiber", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Inner Cryostat (CFC)", "Th-232 Fiber", "Intrinsic Radioactivity", 1.241e-2, 7.346e-3),
    ("Inner Cryostat (CFC)", "K-40 Fiber", "Intrinsic Radioactivity", 0.0, 0.0),
    ("Inner Cryostat (CFC)", "Co-60 Fiber", "Intrinsic Radioactivity", 0.0, 0.0),
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


class PlotConfig:
    """Container for plot configuration parameters."""
    def __init__(
        self,
        xlimits=None,
        colors=None,
        fontsize=20,
        min_count=None,
        title=None,
        cryostat_type="nickel",
    ):
        self.xlimits = [0.0001, 1] if xlimits is None else xlimits
        # Default colors if not provided
        self.colors = colors or {
            "inside_hfe": COMPONENT_COLORS["inside_hfe"],
            "outside_hfe": COMPONENT_COLORS["outside_hfe"],
            "cryostat": COMPONENT_COLORS[f"cryostat_{cryostat_type}"],
        }
        self.fontsize = fontsize
        self.min_count = min_count
        self.title = title
        self.cryostat_type = cryostat_type


def _prepare_data(df, groupby, min_count):
    """Prepare and filter the data for plotting."""
    df_grouped = df.groupby(groupby).agg({"TG Mean": "sum", "TG Spread": sqrt_sum_sq})
    
    # Properly sort by the actual TG Mean values
    df_grouped = df_grouped.sort_values("TG Mean", ascending=True)
    
    if min_count is not None:
        df_grouped = df_grouped[df_grouped["TG Mean"] >= min_count]
    
    return df_grouped


def _setup_legend(config=None):
    """Setup legend elements for the plot."""
    marker = "o"  # Changed from "." to "o" for better visibility
    
    # Create a patch for the highlighted areas
    from matplotlib.patches import Patch
    
    # Default colors if no config provided
    inside_color = COMPONENT_COLORS["inside_hfe"]
    outside_color = COMPONENT_COLORS["outside_hfe"]
    nickel_color = COMPONENT_COLORS["cryostat_nickel"]
    cfc_color = COMPONENT_COLORS["cryostat_cfc"]
    
    if config is not None:
        inside_color = config.colors["inside_hfe"]
        outside_color = config.colors["outside_hfe"]
    
    legend_elements = [
        # Component type markers (inside vs outside HFE)
        Line2D(
            [], [], color=inside_color, marker=marker, 
            linestyle="-", label="Internal Components", markersize=10, linewidth=2.5
        ),
        Line2D(
            [], [], color=outside_color, marker=marker,
            linestyle="-", label="External Components", markersize=10, linewidth=2.5
        ),
        # Highlighted cryostat components
        Patch(
            facecolor=nickel_color, alpha=0.2, edgecolor='gray',
            label="Nickel Cryostat Components"
        ),
        Patch(
            facecolor=cfc_color, alpha=0.2, edgecolor='gray',
            label="CFC Cryostat Components"
        )
    ]
    
    return legend_elements


def prepare_plot_data(df, groupby, min_count):
    """
    Prepare data for plotting and return positions and grouped data.
    
    Parameters:
        df: DataFrame containing the data
        groupby: Column(s) to group by
        min_count: Minimum count to include
        
    Returns:
        df_grouped: The grouped DataFrame
        labels: List of component labels
        y_positions: Array of y-positions for each component
    """
    # Group the data
    df_grouped = df.groupby(groupby).agg({"TG Mean": "sum", "TG Spread": sqrt_sum_sq})
    
    # Sort all components by mean value
    df_grouped = df_grouped.sort_values("TG Mean", ascending=True)
    
    # Apply minimum count filter
    if min_count is not None:
        df_grouped = df_grouped[df_grouped["TG Mean"] >= min_count]
    
    # Get the component labels
    labels = []
    for idx, _ in df_grouped.iterrows():
        label = idx if isinstance(idx, str) else " ".join(idx)
        labels.append(label)
    
    # Create y-positions for plotting
    y_positions = np.arange(0, 2 * len(labels), 2)
    
    return df_grouped, labels, y_positions


def make_single_plot(df, groupby, ax, config, cryostat_type="nickel"):
    """
    Create a background budget plot on the provided axis.
    Components are colored based on whether they're inside/outside HFE.
    Cryostat components get highlighted bands.
    
    Parameters:
        df: DataFrame containing the data
        groupby: Column(s) to group by
        ax: Matplotlib axis to plot on
        config: PlotConfig instance with plot settings
        cryostat_type: Type of cryostat ('nickel' or 'cfc')
    """
    # Select the appropriate cryostat components to highlight
    cryostat_components = NICKEL_CRYOSTAT_COMPONENTS if cryostat_type == "nickel" else CFC_CRYOSTAT_COMPONENTS
    
    # Prepare the data with all components sorted by background contribution
    df_grouped, labels, y_positions = prepare_plot_data(df, groupby, config.min_count)
    marker = "."
    
    # Set up axis properties
    ax.set_xscale("log")
    ax.set_xlim(config.xlimits[0], config.xlimits[1])
    
    # Add horizontal bands for cryostat components
    for i, label in enumerate(labels):
        if label in cryostat_components:
            # Add shaded background to highlight cryostat components
            ax.axhspan(y_positions[i] - 0.8, y_positions[i] + 0.8, 
                      color=config.colors["cryostat"], alpha=0.2, zorder=1)

    # Plot data points
    for i, (idx, row) in enumerate(df_grouped.iterrows()):
        value = row["TG Mean"]
        err = row["TG Spread"]
        label = labels[i]
        
        # Determine color based on component type - keep colors consistent for inside/outside
        if label in OUTSIDE_HFE:
            color = config.colors["outside_hfe"]
        else:
            color = config.colors["inside_hfe"]
        
        # Make all points the same size but adjust zorder to keep cryostat components on top
        zorder = 10 if label in cryostat_components else 5
        
        # Plot the error bars
        ax.errorbar(
            value, y_positions[i], xerr=err, lw=2, capsize=2, capthick=2,
            color=color, marker=marker, markersize=10, zorder=zorder
        )

    # Format x-axis
    fmt = mticker.FuncFormatter(
        lambda x, pos: f"{x:f}".rstrip("0") if x < 1 else f"{x:.0f}"
    )
    ax.xaxis.set_major_formatter(fmt)
    ax.set_xlabel(r"Background counts/(y/2t/FWHM)", fontsize=32)
    ax.grid(which="major", axis="x", linestyle="-")
    ax.grid(which="minor", axis="x", linestyle="dashed")
    
    # Remove y-ticks and labels (will be shown in a separate axis)
    ax.set_yticks([])
    ax.set_yticklabels([])
    
    # Add title
    if config.title:
        ax.set_title(config.title, fontsize=24)
        
    return y_positions, labels


def create_label_axis(ax, y_positions, labels, config):
    """Create a separate axis just for component names."""
    # Set up axis properties
    ax.set_ylim(min(y_positions) - 2, max(y_positions) + 2)
    ax.set_xlim(0, 1)
    
    # Hide all spines and ticks
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])  # Remove y-ticks (numbering on the left)
    
    # Add component names as text
    for i, (pos, label) in enumerate(zip(y_positions, labels)):
        # Format the label
        label_text = reformat_labels_latex(label)
        
        # Set color and formatting based on component type
        if label in ALL_CRYOSTAT_COMPONENTS:
            # Determine which type of cryostat component it is
            is_nickel = label in NICKEL_CRYOSTAT_COMPONENTS
            is_cfc = label in CFC_CRYOSTAT_COMPONENTS
            
            weight = 'bold'
            color = 'black'
            fontsize = 12
            
            # Add a background rectangle with appropriate color
            if is_nickel:
                bg_color = COMPONENT_COLORS["cryostat_nickel"]
                cryostat_label = "Nickel"
            else:  # CFC
                bg_color = COMPONENT_COLORS["cryostat_cfc"]
                cryostat_label = "CFC"
                
            # Add a background rectangle for cryostat components
            ax.axhspan(pos - 0.8, pos + 0.8, color=bg_color, alpha=0.2, zorder=1)
            
            # Add a left-side marker and type indicator
            ax.plot([0.02, 0.06], [pos, pos], color='black', linewidth=2, marker='>', markersize=8)
            
            # Add cryostat type marker
            ax.text(0.12, pos, f"({cryostat_label})", fontsize=20, ha='left', va='center', 
                   color='dimgray', style='italic')
        else:
            weight = 'normal'
            color = 'black'
            fontsize = 11
        
        # Add the component name - right-aligned
        ax.text(0.95, pos, label_text, ha='right', va='center', 
                fontsize=fontsize, weight=weight, color=color)
    
    # Add a title
    ax.set_title("Component", fontsize=24)
    
    # Create a comprehensive legend with both cryostat types
    legend_elements = _setup_legend()
    ax.legend(handles=legend_elements, loc="lower left", fontsize=20, 
              bbox_to_anchor=(0, -0.1))


def load_excel_data(file_path):
    """Load Excel file and process for plotting."""
    if not Path(file_path).exists():
        sys.exit(f"ERROR: File {file_path} not found!")

    # Read the Excel file
    df = pd.read_excel(
        file_path,
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
    df.loc[df["Component"].str.contains("Outer Cryostat Support", na=False), "Component"] = (
        "Outer Vessel Support"
    )
    df.loc[df["Component"].str.contains("Inner Cryostat Support", na=False), "Component"] = (
        "Inner Vessel Support"
    )
    # Keep original names for Outer/Inner Cryostat to match OUTSIDE_HFE list
    df.loc[df["Component"].str.contains(r"Outer Cryostat \(", na=False, regex=True), "Component"] = (
        "Outer Cryostat"
    )
    df.loc[df["Component"].str.contains(r"Inner Cryostat \(", na=False, regex=True), "Component"] = (
        "Inner Cryostat"
    )
    df.loc[df["Component"].str.contains("Outer Cryostat Liner", na=False), "Component"] = (
        "Outer Cryostat Liner"
    )
    df.loc[df["Component"].str.contains("Inner Cryostat Liner", na=False), "Component"] = (
        "Inner Cryostat Liner"
    )
    df.loc[
        df["Component"].str.contains("Outer Cryostat Feedthrough", na=False), "Component"
    ] = "Outer Cryostat Feedthrough"
    df.loc[
        df["Component"].str.contains("Inner Cryostat Feedthrough", na=False), "Component"
    ] = "Inner Cryostat Feedthrough"
    df.loc[df["Component"].str.contains("Inactive LXe", na=False), "Component"] = "Skin LXe"
    df.loc[df["Component"].str.contains("Active LXe", na=False), "Component"] = "TPC LXe"

    # Mark intrinsic radioactivity rows
    df.loc[df["Isotope"].str.contains("bb2n", na=False), "Category"] = "Intrinsic Radioactivity"
    # For intrinsic backgrounds, do not distinguish between various LXe components
    df.loc[
        df["Category"].str.contains("Intrinsic", na=False)
        & df["Component"].str.contains("LXe", na=False),
        "Component",
    ] = "LXe"

    # Remove rows not needed
    df = df[~df["Isotope"].isin(["bb0n", "Cs-137"])]

    # Filter for intrinsic radioactivity
    df_intrinsic = df[df["Category"].str.contains("Intrinsic", na=False)].copy()
    
    print(f"Loaded {len(df_intrinsic)} rows of intrinsic radioactivity data")
    
    return df_intrinsic


def create_cfc_data(nickel_data):
    """
    Create CFC plot data by using Nickel data as a base and replacing cryostat components.
    
    Parameters:
        nickel_data: DataFrame with Nickel cryostat component data
    
    Returns:
        DataFrame with CFC component data
    """
    # Clone the Nickel data
    cfc_data = nickel_data.copy()
    
    # Remove the Nickel cryostat components
    cfc_data = cfc_data[~cfc_data["Component"].isin(NICKEL_CRYOSTAT_COMPONENTS)]
    
    # Create a DataFrame with the hard-coded CFC component data
    cfc_components = pd.DataFrame(
        CFC_COMPONENT_DATA, 
        columns=["Component", "Isotope", "Category", "TG Mean", "TG Spread"]
    )
    
    # Pre-calculate sums for CFC components for proper ordering
    cfc_sums = cfc_components.groupby("Component").agg({"TG Mean": "sum"})
    print("CFC Component Sums:")
    for comp, row in cfc_sums.iterrows():
        print(f"{comp}: {row['TG Mean']:.6e}")
    
    # Add additional columns from the Nickel data to match the structure
    for col in nickel_data.columns:
        if col not in cfc_components.columns:
            cfc_components[col] = "Unknown"
    
    # Combine the data
    cfc_data = pd.concat([cfc_data, cfc_components], ignore_index=True)
    
    print(f"Created CFC data with {len(cfc_data)} rows")
    print(f"CFC cryostat components: {len(cfc_components)} rows")
    
    return cfc_data


def create_comparison_plot(nickel_file, output_path, min_count=0.0001):
    """
    Create a side-by-side comparison plot of Nickel and CFC cryostat background budgets.
    
    Parameters:
        nickel_file: Path to the Nickel cryostat background Excel file
        output_path: Path to save the output plot
        min_count: Minimum background count to show in the plot
    """
    # Load Nickel cryostat data
    nickel_data = load_excel_data(nickel_file)
    
    # Create CFC data by replacing cryostat components
    cfc_data = create_cfc_data(nickel_data)
    
    # Increase font sizes globally for better visibility in LaTeX
    plt.rcParams.update({
        'font.size': 18,
        'axes.titlesize': 26,
        'axes.labelsize': 24,
        'xtick.labelsize': 22,
        'ytick.labelsize': 22,
        'legend.fontsize': 20,
        'lines.linewidth': 3.0,
        'lines.markersize': 16,
        'errorbar.capsize': 8,
    })
    
    # Create a single figure with a wider size to accommodate component names
    fig = plt.figure(figsize=(24, 18))  # Increased size for better visibility
    
    # Create a main axis for component names (left 30% of the figure)
    ax_labels = fig.add_axes((0.01, 0.1, 0.29, 0.85))  # Using tuple instead of list
    
    # Create two axes for the plots (right 70% of the figure, split in half)
    ax1 = fig.add_axes((0.32, 0.1, 0.33, 0.85))  # Using tuple instead of list
    ax2 = fig.add_axes((0.66, 0.1, 0.33, 0.85))  # Using tuple instead of list
    
    # Same x-axis limits for both plots
    xlimits = [0.0001, 1]
    
    # Create plot configurations with the same x-axis limits
    nickel_config = PlotConfig(
        xlimits=xlimits,
        colors={
            "inside_hfe": COMPONENT_COLORS["inside_hfe"],
            "outside_hfe": COMPONENT_COLORS["outside_hfe"],
            "cryostat": COMPONENT_COLORS["cryostat_nickel"],
        },
        fontsize=21,
        min_count=min_count,
        title="Nickel Cryostat",
        cryostat_type="nickel"
    )
    
    cfc_config = PlotConfig(
        xlimits=xlimits,
        colors={
            "inside_hfe": COMPONENT_COLORS["inside_hfe"],
            "outside_hfe": COMPONENT_COLORS["outside_hfe"],
            "cryostat": COMPONENT_COLORS["cryostat_cfc"],
        },
        fontsize=21,
        min_count=min_count,
        title="Carbon Composite Fiber Cryostat",
        cryostat_type="cfc"
    )
    
    # Filter and group data
    nickel_grouped = nickel_data.groupby(["Component"]).agg({"TG Mean": "sum", "TG Spread": sqrt_sum_sq})
    cfc_grouped = cfc_data.groupby(["Component"]).agg({"TG Mean": "sum", "TG Spread": sqrt_sum_sq})
    
    # Apply minimum count filter
    if min_count is not None:
        nickel_grouped = nickel_grouped[nickel_grouped["TG Mean"] >= min_count]
        cfc_grouped = cfc_grouped[cfc_grouped["TG Mean"] >= min_count]
    
    # Sort by background contribution
    nickel_grouped = nickel_grouped.sort_values("TG Mean", ascending=True)
    cfc_grouped = cfc_grouped.sort_values("TG Mean", ascending=True)
    
    # Print out the CFC component values to verify their ordering
    print("CFC Components Sorted by Contribution:")
    for comp in cfc_grouped.index:
        if comp in CFC_CRYOSTAT_COMPONENTS:
            print(f"{comp}: {cfc_grouped.loc[comp]['TG Mean']:.6e}")
    
    # Get component lists
    nickel_components = [idx for idx in nickel_grouped.index]
    cfc_components = [idx for idx in cfc_grouped.index]
    
    # Extract cryostat components (already sorted by contribution)
    nickel_cryostat_items = [c for c in nickel_components if c in NICKEL_CRYOSTAT_COMPONENTS]
    cfc_cryostat_items = [c for c in cfc_components if c in CFC_CRYOSTAT_COMPONENTS]
    
    # Remove cryostat components from main lists
    nickel_non_cryostat = [c for c in nickel_components if c not in NICKEL_CRYOSTAT_COMPONENTS]
    cfc_non_cryostat = [c for c in cfc_components if c not in CFC_CRYOSTAT_COMPONENTS]
    
    # Create a combined list of non-cryostat components (common to both)
    common_components = sorted(
        list(set(nickel_non_cryostat).intersection(set(cfc_non_cryostat))),
        key=lambda x: (nickel_grouped.loc[x]["TG Mean"] + cfc_grouped.loc[x]["TG Mean"]) / 2
    )
    
    # Create dictionaries to map components to their actual TG Mean values
    nickel_values = {comp: nickel_grouped.loc[comp]["TG Mean"] for comp in nickel_components}
    cfc_values = {comp: cfc_grouped.loc[comp]["TG Mean"] for comp in cfc_components}
    
    # Combine all components into a unified list with their values
    all_components = []
    
    # Add common components
    for comp in common_components:
        # Average of both plots for common components
        value = (nickel_values[comp] + cfc_values[comp]) / 2
        all_components.append((comp, value, "common"))
    
    # Add Nickel cryostat components
    for comp in nickel_cryostat_items:
        all_components.append((comp, nickel_values[comp], "nickel"))
    
    # Add CFC cryostat components
    for comp in cfc_cryostat_items:
        all_components.append((comp, cfc_values[comp], "cfc"))
    
    # Sort all components by their TG Mean values
    all_components.sort(key=lambda x: x[1])
    
    # Now create the component lists for both plots based on the sorted unified list
    nickel_component_list = []
    cfc_component_list = []
    
    for comp, _, comp_type in all_components:
        if comp_type == "common":
            # Add common component to both lists
            nickel_component_list.append(comp)
            cfc_component_list.append(comp)
        elif comp_type == "nickel":
            # Add Nickel component with placeholder in CFC list
            nickel_component_list.append(comp)
            cfc_component_list.append(None)
        elif comp_type == "cfc":
            # Add CFC component with placeholder in Nickel list
            nickel_component_list.append(None)
            cfc_component_list.append(comp)
    
    # Create modified dataframes for each plot based on the custom component lists
    nickel_plot_data = []
    for comp in nickel_component_list:
        if comp is not None:
            # Add the actual component data
            component_data = nickel_grouped.loc[comp].to_dict()
            nickel_plot_data.append({"Component": comp, **component_data})
        else:
            # Add a placeholder (invisible)
            nickel_plot_data.append({"Component": "", "TG Mean": float('nan'), "TG Spread": float('nan')})
    
    cfc_plot_data = []
    for comp in cfc_component_list:
        if comp is not None:
            # Add the actual component data
            component_data = cfc_grouped.loc[comp].to_dict()
            cfc_plot_data.append({"Component": comp, **component_data})
        else:
            # Add a placeholder (invisible)
            cfc_plot_data.append({"Component": "", "TG Mean": float('nan'), "TG Spread": float('nan')})
    
    # Convert to DataFrames
    nickel_plot_df = pd.DataFrame(nickel_plot_data)
    cfc_plot_df = pd.DataFrame(cfc_plot_data)
    
    # Create the custom plot function that doesn't sort the components
    def make_custom_plot(df, ax, config, cryostat_type="nickel"):
        """Create plot with pre-ordered components (no sorting)"""
        cryostat_components = NICKEL_CRYOSTAT_COMPONENTS if cryostat_type == "nickel" else CFC_CRYOSTAT_COMPONENTS
        marker = "o"  # Changed from "." to "o" for better visibility
        
        # Set up axis properties
        ax.set_xscale("log")
        ax.set_xlim(config.xlimits[0], config.xlimits[1])
        
        # Get positions and labels
        y_positions = np.arange(0, 2 * len(df), 2)
        labels = df["Component"].tolist()
        
        # Add horizontal bands for cryostat components
        for i, label in enumerate(labels):
            if label in cryostat_components:
                ax.axhspan(y_positions[i] - 0.8, y_positions[i] + 0.8, 
                          color=config.colors["cryostat"], alpha=0.2, zorder=1)

        # Plot data points
        for i, (_, row) in enumerate(df.iterrows()):
            if pd.isna(row["TG Mean"]) or row["Component"] == "":
                continue  # Skip placeholder rows

            value = row["TG Mean"]
            err = row["TG Spread"]
            label = row["Component"]
            
            # Determine color based on component type
            if label in OUTSIDE_HFE:
                color = config.colors["outside_hfe"]
            else:
                color = config.colors["inside_hfe"]
            
            # Make cryostat components on top
            zorder = 10 if label in cryostat_components else 5
            
            # Plot the error bars
            ax.errorbar(
                value, y_positions[i], xerr=err, lw=3.0, capsize=8, capthick=3.0,
                color=color, marker=marker, markersize=16, zorder=zorder
            )

        # Format x-axis
        fmt = mticker.FuncFormatter(
            lambda x, pos: f"{x:f}".rstrip("0") if x < 1 else f"{x:.0f}"
        )
        ax.xaxis.set_major_formatter(fmt)
        ax.set_xlabel(r"Background counts/(y/2t/FWHM)", fontsize=32)
        ax.grid(which="major", axis="x", linestyle="-", linewidth=1.5)
        ax.grid(which="minor", axis="x", linestyle="dashed", linewidth=1.0)
        
        # Remove y-ticks and labels
        ax.set_yticks([])
        ax.set_yticklabels([])
        
        # Add title with larger font
        if config.title:
            ax.set_title(config.title, fontsize=32, pad=20)
            
        return y_positions, labels
    
    # Make the plots with our custom ordered datasets
    pos1, labels1 = make_custom_plot(nickel_plot_df, ax1, nickel_config, cryostat_type="nickel")
    pos2, labels2 = make_custom_plot(cfc_plot_df, ax2, cfc_config, cryostat_type="cfc")
    
    # Create a combined label axis using both sets of components
    def create_combined_label_axis(ax, positions, nickel_labels, cfc_labels):
        """Create a label axis with combined component names from both plots"""
        # Set up axis properties
        ax.set_ylim(min(positions) - 2, max(positions) + 2)
        ax.set_xlim(0, 1)
        
        # Hide all spines and ticks
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Add component names as text
        for i, pos in enumerate(positions):
            nickel_label = nickel_labels[i]
            cfc_label = cfc_labels[i]
            
            # Skip empty placeholders
            if nickel_label == "" and cfc_label == "":
                continue
                
            # Determine which label to use (prefer non-empty)
            display_label = nickel_label if nickel_label != "" else cfc_label
            label_text = reformat_labels_latex(display_label)
            
            # Format differently based on component type
            is_nickel_cryostat = nickel_label in NICKEL_CRYOSTAT_COMPONENTS
            is_cfc_cryostat = cfc_label in CFC_CRYOSTAT_COMPONENTS
            
            if is_nickel_cryostat or is_cfc_cryostat:
                weight = 'bold'
                color = 'black'
                fontsize = 20
                
                # Add a background rectangle with appropriate color
                if is_nickel_cryostat:
                    bg_color = COMPONENT_COLORS["cryostat_nickel"]
                    cryostat_label = "Nickel"
                    ax.axhspan(pos - 0.8, pos + 0.8, color=bg_color, alpha=0.2, zorder=1)
                else:  # CFC
                    bg_color = COMPONENT_COLORS["cryostat_cfc"]
                    cryostat_label = "CFC"
                    ax.axhspan(pos - 0.8, pos + 0.8, color=bg_color, alpha=0.2, zorder=1)
                
                # Add a left-side marker and type indicator
                ax.plot([0.02, 0.06], [pos, pos], color='black', linewidth=2.5, marker='>', markersize=10)
                
                # Add cryostat type marker
                ax.text(0.12, pos, f"({cryostat_label})", fontsize=24, ha='left', va='center', 
                       color='dimgray', style='italic', fontweight='bold')
            else:
                weight = 'normal'
                color = 'black'
                fontsize = 14
            
            # Add the component name
            ax.text(0.95, pos, label_text, ha='right', va='center', 
                    fontsize=26, weight=weight, color=color)
        
        # Add a title
        ax.set_title("Component", fontsize=32, pad=20)
        
        # Create a comprehensive legend
        legend_elements = _setup_legend()
        ax.legend(handles=legend_elements, loc="lower left", fontsize=26, 
                  bbox_to_anchor=(0, -0.12))
    
    # Create the combined component labels
    create_combined_label_axis(ax_labels, pos1, labels1, labels2)
    
    # Ensure all plots have the same y-axis limits for alignment
    ymin, ymax = ax1.get_ylim()
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)
    ax_labels.set_ylim(ymin, ymax)
    
    # Save the figure with high dpi for LaTeX inclusion
    plt.savefig(output_path, dpi=400, transparent=False, bbox_inches='tight')
    plt.close(fig)
    
    print(f"Comparison plot saved to {output_path}")


def main():
    """Parse command-line arguments and generate the comparison plot."""
    parser = argparse.ArgumentParser(
        description="Create side-by-side plots comparing Nickel and CFC cryostat background budgets"
    )
    parser.add_argument(
        "--nickel_file",
        default="background/Summary_D-047_v86_250113-233135_2025-02-21.xlsx",
        help="Path to Nickel cryostat Excel file",
    )
    parser.add_argument(
        "--output_path",
        default="background/nickel_vs_cfc_ic.pdf",
        help="Path to save the output plot",
    )
    parser.add_argument(
        "--min_count",
        type=float,
        default=0.0001,
        help="Minimum background count to show in plot",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    output_dir = Path(args.output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create the comparison plot
    create_comparison_plot(
        args.nickel_file,
        args.output_path,
        min_count=args.min_count
    )


if __name__ == "__main__":
    main()