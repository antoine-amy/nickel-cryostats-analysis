"""Calculate vessel volumes for inner and outer cryostat vessels.

This module computes the volumes of cylindrical side walls and spherical end caps
for both inner and outer vessels of a cryostat system.
"""

import math

# Inner vessel dimensions (m)
R_IV = 1.410
H_IV = 2.820
T_CYL_IV = 0.010
T_SPH_IV = 0.005

# Outer vessel dimensions (m)
R_OV = 1.54
H_OV = 3.072  # h_iv + 2 * vacuum gap (0.126 m)
T_CYL_OV = 0.020
T_SPH_OV = 0.010

# Nickel density (kg/m^3)
RHO_NI = 8350

# Compute volumes
V_SIDE_IV = math.pi * ((R_IV + T_CYL_IV)**2 - R_IV**2) * H_IV
V_CAPS_IV = 2 * math.pi * R_IV**2 * T_SPH_IV  # 2 circular disks
V_TOT_IV = V_SIDE_IV + V_CAPS_IV

V_SIDE_OV = math.pi * ((R_OV + T_CYL_OV)**2 - R_OV**2) * H_OV
V_CAPS_OV = 2 * math.pi * R_OV**2 * T_SPH_OV  # 2 circular disks
V_TOT_OV = V_SIDE_OV + V_CAPS_OV

# Compute masses
MASS_IV = V_TOT_IV * RHO_NI
MASS_OV = V_TOT_OV * RHO_NI

print(f"Inner Vessel Volumes (m^3): side = {V_SIDE_IV:.3f}, "
      f"caps = {V_CAPS_IV:.3f}, total = {V_TOT_IV:.3f}")
print(f"Outer Vessel Volumes (m^3): side = {V_SIDE_OV:.3f}, "
      f"caps = {V_CAPS_OV:.3f}, total = {V_TOT_OV:.3f}")
print(f"Inner Vessel Mass (kg): {MASS_IV:.1f}")
print(f"Outer Vessel Mass (kg): {MASS_OV:.1f}")

print("\n--- Comparison with Spherical Design ---")
print(f"Cylindrical IV: {V_TOT_IV:.3f} m³; {MASS_IV:.1f} kg ({MASS_IV/1000:.1f} tonnes)")
print(f"Spherical IV:   0.205 m³; 1600.0 kg (1.6 tonnes)")
print(f"Cylindrical OV: {V_TOT_OV:.3f} m³; {MASS_OV:.1f} kg ({MASS_OV/1000:.1f} tonnes)")
print(f"Spherical OV:   0.669 m³; 5660.0 kg (5.66 tonnes)")
