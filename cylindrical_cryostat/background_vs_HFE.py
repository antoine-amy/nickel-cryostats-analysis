import numpy as np
import matplotlib.pyplot as plt

# Constants
R_IV0 = 1.410      # m, original inner-vessel radius
H_IV0 = 2.820      # m, original inner-vessel height
T_CYL_IV = 0.010   # m, inner vessel cylinder thickness
T_SPH_IV = 0.005   # m, inner vessel cap thickness

R_OV0 = 1.54       # m, original outer-vessel radius
H_OV0 = 3.072      # m, original outer-vessel height
T_CYL_OV = 0.020   # m, outer vessel cylinder thickness
T_SPH_OV = 0.010   # m, outer vessel cap thickness

RHO_NI = 8350      # kg/m³, nickel density

# Activities (mBq/kg)
act_U238 = 0.00122   # mBq/kg
act_Th232 = 0.000265 # mBq/kg

# Spherical hit-efficiencies at 76 cm HFE
hit_sph = {
    'IV': {'Th232': 9.720e-9, 'U238': 9.0e-9},
    'OV': {'Th232': 7.2e-9,   'U238': 0.0}
}

# Transmission values
trans_sph = 0.0013   # 0.13%
trans_cyl = 0.0033   # 0.33%

# Compute scaling from spherical → cylindrical baseline
trans_scale = trans_cyl / trans_sph  # ≃ 2.538

# Build your cylindrical-baseline efficiencies
hit = {
    'IV': {
        'Th232': hit_sph['IV']['Th232'] * trans_scale,
        'U238' : hit_sph['IV']['U238']  * trans_scale
    },
    'OV': {
        'Th232': hit_sph['OV']['Th232'] * trans_scale,
        'U238' : hit_sph['OV']['U238']  * trans_scale  # still zero
    }
}

# Attenuation scaling parameter
MU_GAMMA = 0.007  # mm⁻¹

# Baseline HFE thickness (m)
original_hfe = 0.76

# Budget limits (counts/year)
budget_IV = 1.014e-3 + 1.241e-2
budget_OV = 1.730e-3 + 1.994e-2

# Seconds per year
sec_per_year = 365 * 24 * 3600

# Sweep HFE thickness from 10 cm to 76 cm
thickness = np.linspace(0.10, 0.76, 100)  # in meters

# Arrays to hold background counts/year
bg_IV = np.zeros_like(thickness)
bg_OV = np.zeros_like(thickness)

for i, t in enumerate(thickness):
    delta = original_hfe - t
    scale = np.exp(MU_GAMMA * delta * 1000)

    # Compute IV mass
    R_IV = R_IV0 - delta
    H_IV = H_IV0 - 2*delta
    V_side_IV = np.pi * ((R_IV + T_CYL_IV)**2 - R_IV**2) * H_IV
    V_caps_IV = 2 * np.pi * R_IV**2 * T_SPH_IV
    mass_IV = (V_side_IV + V_caps_IV) * RHO_NI

    # Compute OV mass
    R_OV = R_OV0 - delta
    H_OV = H_OV0 - 2*delta
    V_side_OV = np.pi * ((R_OV + T_CYL_OV)**2 - R_OV**2) * H_OV
    V_caps_OV = 2 * np.pi * R_OV**2 * T_SPH_OV
    mass_OV = (V_side_OV + V_caps_OV) * RHO_NI

    # Decays per year
    dec_IV_Th = mass_IV * act_Th232 * 1e-3 * sec_per_year
    dec_IV_U  = mass_IV * act_U238  * 1e-3 * sec_per_year
    dec_OV_Th = mass_OV * act_Th232 * 1e-3 * sec_per_year
    dec_OV_U  = mass_OV * act_U238  * 1e-3 * sec_per_year

    # Apply scaled hit efficiencies
    eff_IV_Th = hit['IV']['Th232'] * scale
    eff_IV_U  = hit['IV']['U238']  * scale
    eff_OV_Th = hit['OV']['Th232'] * scale
    eff_OV_U  = hit['OV']['U238']  * scale

    # Total background
    bg_IV[i] = dec_IV_Th * eff_IV_Th + dec_IV_U * eff_IV_U
    bg_OV[i] = dec_OV_Th * eff_OV_Th + dec_OV_U * eff_OV_U

# Plotting
plt.figure(figsize=(12,8))
plt.plot(thickness * 100, bg_IV, color='tab:blue', label='Inner Vessel')
plt.plot(thickness * 100, bg_OV, color='tab:red',  label='Outer Vessel')

plt.yscale('log')

# Budget lines in matching colors
plt.hlines(budget_IV, 10, 76, colors='tab:blue', linestyles='--', label='IV Budget')
plt.hlines(budget_OV, 10, 76, colors='tab:red',  linestyles='--', label='OV Budget')

plt.xlabel('HFE Thickness (cm)', fontsize=18)
plt.ylabel('Background (counts/year)', fontsize=18)
#plt.title('nEXO Cryostat Background vs. HFE Thickness')
plt.legend(fontsize=16)
plt.grid(True, which='both', ls='--')

# Increase tick label sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.tight_layout()
plt.show()