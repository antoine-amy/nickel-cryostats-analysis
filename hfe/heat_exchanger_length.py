#!/usr/bin/env python3
"""
Helical-coil heat-exchanger geometry & tube-length calculator
Antoine Amy – July 2025
"""

import math
from typing import NamedTuple
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # type: ignore


# ───────────────────────────────
# USER PARAMETERS  (all in mm)
# ───────────────────────────────
TANK_HEIGHT = 216.0     # bottom → inlet/outlet plane
TANK_INNER_DIAMETER = 96.0     # 96 mm inner ID
COIL_HEIGHT = 200.0
COIL_OUTER_DIAMETER = 80.5
TUBE_OD = 6.35
LOOP_EDGE_SPACING = 19.86

BOTTOM_STUB = 0.0       # short up-stub after bottom bend
CLEARANCE_FROM_RISER = -20.0     # outlet ends 20 mm past riser axis
# ───────────────────────────────


class HeatExchangerGeometry(NamedTuple):
    """Container for heat exchanger geometry parameters."""
    coil_len: float
    riser_len: float
    bottom_radial: float
    bottom_vert: float
    top_radial: float
    top_vertical: float
    total_len: float
    n_loops: int


def calculate_heat_exchanger_geometry() -> HeatExchangerGeometry:
    """
    Calculate heat exchanger geometry and lengths.

    Returns:
        HeatExchangerGeometry containing all calculated parameters
    """
    # ─── derived coil specs ───
    centre_diam = COIL_OUTER_DIAMETER - TUBE_OD
    centre_radius = centre_diam / 2.0
    pitch = LOOP_EDGE_SPACING - TUBE_OD
    n_loops = math.floor((COIL_HEIGHT - TUBE_OD) / pitch) + 1
    coil_len = n_loops * math.pi * centre_diam
    coil_env_h = TUBE_OD + (n_loops - 1) * pitch

    # ─── straight riser ───
    riser_len = TANK_HEIGHT
    riser_radius = TUBE_OD / 2.0

    # ─── bottom entry ───
    bottom_radial = centre_radius
    bottom_vert = BOTTOM_STUB

    # ─── top outlet ───
    target_radius = riser_radius + CLEARANCE_FROM_RISER
    top_radial = centre_radius - target_radius
    top_vertical = TANK_HEIGHT - coil_env_h

    # ─── total length ───
    total_len = (coil_len + riser_len + bottom_radial + bottom_vert
                 + top_radial + top_vertical)

    return HeatExchangerGeometry(
        coil_len=coil_len,
        riser_len=riser_len,
        bottom_radial=bottom_radial,
        bottom_vert=bottom_vert,
        top_radial=top_radial,
        top_vertical=top_vertical,
        total_len=total_len,
        n_loops=n_loops
    )


def print_summary(geometry: HeatExchangerGeometry) -> None:
    """Print summary of heat exchanger tube lengths."""
    print("\nHEAT-EXCHANGER TUBE LENGTHS  (tank Ø = 96 mm)")
    print("─────────────────────────────────────────────────────────────────────")
    print(f"  Helical coil ({geometry.n_loops} turns) …… "
          f"{geometry.coil_len:7.1f} mm  ({geometry.coil_len/25.4:6.2f} in)")
    print(f"  Central riser …………………… {geometry.riser_len:7.1f} mm  ({geometry.riser_len/25.4:6.2f} in)")
    print(f"  Bottom radial ………………… {geometry.bottom_radial:7.1f} mm  ({geometry.bottom_radial/25.4:6.2f} in)")
    print(f"  Bottom vertical stub ……   {geometry.bottom_vert:7.1f} mm  ({geometry.bottom_vert/25.4:6.2f} in)")
    print(f"  Top radial ……………………   {geometry.top_radial:7.1f} mm  ({geometry.top_radial/25.4:6.2f} in)")
    print(f"  Top vertical …………………   {geometry.top_vertical:7.1f} mm  ({geometry.top_vertical/25.4:6.2f} in)")
    print("─────────────────────────────────────────────────────────────────────")
    print(f"  TOTAL tube length ………… {geometry.total_len:7.1f} mm  ({geometry.total_len/25.4:6.2f} in)\n")


def create_3d_plot(geometry: HeatExchangerGeometry) -> None:
    """Create 3D visualization of the heat exchanger."""
    # type: ignore
    # ─── derived coil specs ───
    centre_diam = COIL_OUTER_DIAMETER - TUBE_OD
    centre_radius = centre_diam / 2.0
    pitch = LOOP_EDGE_SPACING - TUBE_OD

    # ─── 3-D geometry ───
    pts = 200
    theta = np.linspace(0, 2*math.pi*geometry.n_loops, pts*geometry.n_loops)
    x_coil = centre_radius*np.cos(theta)
    y_coil = centre_radius*np.sin(theta)
    z_coil = pitch*theta/(2*math.pi)

    # bottom feed (axis → rim → stub)
    x_bot = np.linspace(0, x_coil[0], 30)
    y_bot = np.linspace(0, y_coil[0], 30)
    z_bot = np.zeros_like(x_bot)

    x_bot_stub = np.full(15, x_coil[0])
    y_bot_stub = np.full(15, y_coil[0])
    z_bot_stub = np.linspace(0, BOTTOM_STUB, 15)

    # top outlet (rim → target → up)
    ux, uy = -x_coil[-1]/centre_radius, -y_coil[-1]/centre_radius
    target_radius = (TUBE_OD / 2.0) + CLEARANCE_FROM_RISER
    x_out = np.linspace(x_coil[-1], ux*target_radius, 30)
    y_out = np.linspace(y_coil[-1], uy*target_radius, 30)
    z_out = np.full_like(x_out, z_coil[-1])

    x_out_up = np.full(15, ux*target_radius)
    y_out_up = np.full(15, uy*target_radius)
    z_out_up = np.linspace(z_coil[-1], TANK_HEIGHT, 15)

    # riser
    z_riser = np.linspace(0, TANK_HEIGHT, 50)
    x_riser = np.zeros_like(z_riser)
    y_riser = np.zeros_like(z_riser)

    # tank wire-frame
    h = np.linspace(0, TANK_HEIGHT, 30)
    a = np.linspace(0, 2*math.pi, 40)
    a_mesh, h_mesh = np.meshgrid(a, h)
    x_tank = (TANK_INNER_DIAMETER/2)*np.cos(a_mesh)
    y_tank = (TANK_INNER_DIAMETER/2)*np.sin(a_mesh)
    z_tank = h_mesh

    fig = plt.figure(figsize=(8, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")  # type: ignore

    # All exchanger parts in matplotlib blue - only one legend entry for the coil
    ax.plot(x_coil, y_coil, z_coil, lw=2, color="#1f77b4", label="LN2 coil")
    ax.plot(x_bot, y_bot, z_bot, lw=2, color="#1f77b4")
    ax.plot(x_bot_stub, y_bot_stub, z_bot_stub, lw=2, color="#1f77b4")
    ax.plot(x_out, y_out, z_out, lw=2, color="#1f77b4")
    ax.plot(x_out_up, y_out_up, z_out_up, lw=2, color="#1f77b4")
    ax.plot(x_riser, y_riser, z_riser, lw=2, color="#1f77b4")

    # Tank for context - made darker for better visibility
    ax.plot_wireframe(x_tank, y_tank, z_tank, alpha=0.25, lw=0.6,
                      color="black", label="Tank (96 mm ID)")  # type: ignore

    ax.set_xlabel("x [mm]")
    ax.set_ylabel("y [mm]")
    ax.set_zlabel("z [mm]")  # type: ignore
    lim = TANK_INNER_DIAMETER/2
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(0, TANK_HEIGHT)  # type: ignore
    #ax.set_title("Heat-Exchanger in 96 mm-ID Tank")
    ax.legend()
    plt.tight_layout()
    plt.show()


def main() -> None:
    """Main function to calculate and display heat exchanger geometry."""
    # Calculate geometry
    geometry = calculate_heat_exchanger_geometry()

    # Print summary
    print_summary(geometry)

    # Create 3D plot
    create_3d_plot(geometry)


if __name__ == "__main__":
    main()

