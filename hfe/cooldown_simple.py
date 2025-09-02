import math
import numpy as np
import matplotlib.pyplot as plt

# === Geometry & material for tube ===
L_tube = 4.5                            # m
OD_tube = 3/8 * 0.0254                  # m
thickness_tube = 0.035 * 0.0254         # m (0.035")
ID_tube = OD_tube - 2 * thickness_tube
h_i = 150.0                             # W/m²·K (inside)
h_o = 10.0                              # W/m²·K (outside)
k_steel = 16.0                          # W/m·K

# === Tank geometry ===
H_tank = 0.5                            # m
D_o_tank = 0.10                         # m
thickness_tank = 0.004                  # m (4 mm)
D_i_tank = D_o_tank - 2 * thickness_tank

# === Fluid & flow ===
V_DOT_HFE_LPM = 2.0                     # L/min
V_dot_hfe_m3_s = V_DOT_HFE_LPM / 60000.0 # m³/s
HOT_VOLUME_M3 = 0.0036                  # m³

# === Temperatures & simulation ===
T_AMB = 293.0                           # ambient (K)
T_LN2 = 77.0                            # LN2 inside (K)
T_INIT = 298.0                          # initial HFE (K)
SIM_DURATION_S = 3 * 3600               # 3 hours

# === Precompute conductances (W/K) ===
# Tube conduction to LN2
R_i_tube = 1/(h_i * math.pi * ID_tube * L_tube)
R_c_tube = math.log(OD_tube/ID_tube)/(2 * math.pi * k_steel * L_tube)
R_o_tube = 1/(h_o * math.pi * OD_tube * L_tube)
C_tube = 1 / (R_i_tube + R_c_tube + R_o_tube)

# Tank ambient conduction
R_i_tank = 1/(h_i * math.pi * D_i_tank * H_tank)
R_c_tank = math.log(D_o_tank/D_i_tank)/(2 * math.pi * k_steel * H_tank)
R_o_tank = 1/(h_o * math.pi * D_o_tank * H_tank)
C_tank = 1 / (R_i_tank + R_c_tank + R_o_tank)

# Cycle time (s)
dt_cycle = HOT_VOLUME_M3 / V_dot_hfe_m3_s

# Material properties
def get_rho_hfe(T):
    return 1430.0 - 2.88 * (T - 273.15)

def get_cp_hfe(T):
    return 1210.0 + 3.08 * (T - 273.15)

def simulate_conduction_curve(f_ambient):
    times, temps = [0.0], [T_INIT]
    t, T = 0.0, T_INIT
    while t < SIM_DURATION_S:
        # ambient heat gain
        Q_gain = C_tank * f_ambient * (T_AMB - T) * dt_cycle
        # heat removal to LN2
        Q_rem  = C_tube * (T - T_LN2) * dt_cycle

        rho = get_rho_hfe(T)
        cp = get_cp_hfe(T)
        mass = HOT_VOLUME_M3 * rho
        T += (Q_gain - Q_rem) / (mass * cp)

        t += dt_cycle
        times.append(t / 60.0)
        temps.append(T)
    return np.array(times), np.array(temps)

# Check baseline performance
times_base, temps_base = simulate_conduction_curve(1.0)
T_final_base = temps_base[-1]

if T_final_base <= 170.0:
    print("Baseline conduction model already reaches ≤170 K in 3 h → no reduction needed.")
    f_target = 1.0
    reduction_pct = 0.0
else:
    # binary search ambient reduction
    low, high = 0.0, 1.0
    for _ in range(30):
        mid = 0.5*(low + high)
        if simulate_conduction_curve(mid)[1][-1] <= 170.0:
            low = mid
        else:
            high = mid
    f_target = 0.5*(low + high)
    reduction_pct = (1 - f_target)*100.0
    print(f"Need to reduce ambient conduction by {reduction_pct:.1f}% to reach 170 K.")

# Simulate insulated case
times_ins, temps_ins = simulate_conduction_curve(f_target)

# Plot
plt.figure(figsize=(8,5))
plt.plot(times_base, temps_base, label='Baseline (no insulation)')
plt.plot(times_ins, temps_ins, '--', label=f'With {reduction_pct:.1f}% reduction')
plt.axhline(170, color='gray', ls=':', label='170 K target')
plt.xlabel('Elapsed Time (min)')
plt.ylabel('HFE Temperature (K)')
plt.title('First‑Order Conduction Model: Tube at 77 K')
plt.grid(True, ls='--', alpha=0.6)
plt.legend()
plt.tight_layout()
plt.show()