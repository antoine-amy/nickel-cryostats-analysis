#!/usr/bin/env python3
"""
Compute inner vessel radius and HFE properties from given mass.
"""
import math

# Constants
DENSITY_T_PER_M3 = 1.72       # tonne per m³
VOL_OFFSET_M3 = 2.2 - 0.58    # m³ correction term
PRICE_M_PER_TONNE = 2.2 / 32  # million USD per tonne


def summarize_hfe_properties(mass_tonnes):
    """
    Return a dict of HFE volume, IV radius, and price for a given mass.
    """
    volume_m3 = mass_tonnes / DENSITY_T_PER_M3
    radius_m = ((volume_m3 + VOL_OFFSET_M3) / ((4/3) * math.pi)) ** (1/3)
    return {
        "iv_radius_mm": radius_m * 1000,
        "hfe_mass_tonnes": mass_tonnes,
        "hfe_volume_m3": volume_m3,
        "hfe_price_millions": mass_tonnes * PRICE_M_PER_TONNE,
    }


def main():
    """Calculate and print HFE properties for a 12.8 tonne mass."""
    mass = 12.8  # tonnes
    props = summarize_hfe_properties(mass)
    print(f"Values for HFE mass: {props['hfe_mass_tonnes']:.2f} t")
    print(f"  IV radius: {props['iv_radius_mm']:.2f} mm")
    print(f"  HFE volume: {props['hfe_volume_m3']:.2f} m³")
    print(f"  HFE price: ${props['hfe_price_millions']:.2f} M")


if __name__ == "__main__":
    main()
