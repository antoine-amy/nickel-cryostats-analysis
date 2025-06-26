#!/usr/bin/env python3
"""
Compute HFE properties from a given inner‐vessel radius.
"""
import math

# Constants
DENSITY_T_PER_M3 = 1.72       # tonne per m³ of HFE
VOL_OFFSET_M3    = 2.2 - 0.58 # m³ correction term
PRICE_M_PER_TONNE = 2.2 / 32  # million USD per tonne of HFE

def summarize_from_radius(iv_radius_m):
    """
    Given an inner‐vessel radius (m), return a dict with:
      - hfe_volume_m3
      - hfe_mass_tonnes
      - hfe_price_millions
    """
    total_volume = (4.0/3.0) * math.pi * iv_radius_m**3
    hfe_volume_m3 = total_volume - VOL_OFFSET_M3
    hfe_mass_tonnes = hfe_volume_m3 * DENSITY_T_PER_M3
    hfe_price_millions = hfe_mass_tonnes * PRICE_M_PER_TONNE

    return {
        "iv_radius_m":          iv_radius_m,
        "hfe_volume_m3":        hfe_volume_m3,
        "hfe_mass_tonnes":      hfe_mass_tonnes-0.240, # 0.240 t of HFE is lost to the environment, value from background budget for 1691mm IV
        "hfe_price_millions":   hfe_price_millions,
    }

def main():
    """Calculate and print HFE properties for IV radius = 1.1 m."""
    radius = 1.1  # m
    props = summarize_from_radius(radius)
    print(f"Inner‐vessel radius: {props['iv_radius_m']:.2f} m")
    print(f"HFE volume:         {props['hfe_volume_m3']:.2f} m³")
    print(f"HFE mass:           {props['hfe_mass_tonnes']:.2f} t")
    print(f"HFE price:          ${props['hfe_price_millions']:.2f} M")

if __name__ == "__main__":
    main()
