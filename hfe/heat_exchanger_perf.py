#!/usr/bin/env python3
"""
Closed-Loop Hot Fluid Cooling Simulation:
Combined plot of HFE inlet temperature (to 160 K) and cumulative LN₂ consumption vs. time.
"""
from typing import List, Tuple
import numpy as np
import matplotlib.pyplot as plt

# === CONSTANTS ===
LENGTH_M = 3.8217               # Heat exchanger length (m)
SEGMENTS = 100                  # Discretization segments
AREA_PER_M = 0.02               # Heat transfer area per m (m²/m)
U_COEFF = 11.47                 # Heat transfer coefficient (W/m²·K)

V_DOT_HOT_LPM = 2.0             # Hot fluid flow rate (L/min)
HOT_VOLUME_M3 = 0.0036         # Loop volume (m³)
T_HOT_INIT_K = 298.0            # Initial inlet temp (K)

V_DOT_LN2_LPM = 0.05            # Base LN₂ flow rate (L/min)
RHO_LN2 = 808.0                 # LN₂ density (kg/m³)
H_FG_LN2 = 199e3                # Latent heat of vapor (J/kg)
T_LN2_SAT_K = 77.0              # LN₂ boiling temp (K)

STOP_TEMP_K = 160.0             # Stop simulation when inlet reaches this
WORKING_TEMP_K = 170.0          # Horizontal line temperature

def get_rho_hot(temp_k: float) -> float:
    """Calculate hot fluid density as a function of temperature."""
    return 1430.0 - 2.88 * (temp_k - 273.15)

def get_cp_hot(temp_k: float) -> float:
    """Calculate hot fluid specific heat capacity as a function of temperature."""
    return 1210.0 + 3.08 * (temp_k - 273.15)

def get_vdot_ln2(temp_k: float) -> float:
    """
    Linearly taper LN2 flow from full at initial temp to zero at saturation.
    """
    fraction = max(0.0, (temp_k - T_LN2_SAT_K) /
                   (T_HOT_INIT_K - T_LN2_SAT_K))
    return V_DOT_LN2_LPM * fraction

def simulate_closed_loop() -> Tuple[List[float], List[float], float]:
    """
    Simulate until inlet drops to STOP_TEMP_K.
    Returns inlet_temps, ln2_per_cycle, time_per_cycle.
    """
    v_dot_hot = V_DOT_HOT_LPM / 60000.0
    time_per_cycle = HOT_VOLUME_M3 / v_dot_hot

    t_in = T_HOT_INIT_K
    inlet_temps: List[float] = []
    ln2_per_cycle: List[float] = []

    while t_in > STOP_TEMP_K:
        # compute LN2 mass usage this cycle
        vdot_ln2 = get_vdot_ln2(t_in)
        m_dot_ln2 = (vdot_ln2 / 60000.0) * RHO_LN2
        ln2_used = m_dot_ln2 * time_per_cycle

        inlet_temps.append(t_in)
        ln2_per_cycle.append(ln2_used)

        # simplistic cooling step: proportional step down
        # ideally replace with full profile integration
        t_drop = (T_HOT_INIT_K - STOP_TEMP_K) / 75.0
        t_in -= t_drop

    return inlet_temps, ln2_per_cycle, time_per_cycle

def main() -> None:
    """Run the heat exchanger performance simulation and plot results."""
    inlet_temps, ln2_usage, tpc = simulate_closed_loop()
    cycles = len(inlet_temps)
    # elapsed time array in minutes
    times_min = np.arange(1, cycles+1) * tpc / 60.0
    # cumulative LN2
    cumulative_ln2 = np.cumsum(ln2_usage)

    # Find time to reach working temperature
    working_temp_reached = False
    time_to_working = None
    for i, temp in enumerate(inlet_temps):
        if temp <= WORKING_TEMP_K and not working_temp_reached:
            time_to_working = times_min[i]
            working_temp_reached = True

    # Print results
    total_ln2 = cumulative_ln2[-1]
    print(f"Total LN₂ used: {total_ln2:.2f} kg")
    if time_to_working is not None:
        msg = (f"Time to reach working temperature ({WORKING_TEMP_K}K): "
               f"{time_to_working:.1f} minutes")
        print(msg)
    else:
        msg = f"Working temperature ({WORKING_TEMP_K}K) not reached in simulation"
        print(msg)

    # Direct energy balance for total cooling of hot fluid
    T_start = 298
    T_end = 173.15
    delta_T = T_start - T_end
    
    # Calculate average properties over temperature range
    rho_start = get_rho_hot(T_start)
    rho_end = get_rho_hot(T_end)
    rho_avg = (rho_start + rho_end) / 2
    
    cp_start = get_cp_hot(T_start)
    cp_end = get_cp_hot(T_end)
    cp_avg = (cp_start + cp_end) / 2
    
    mass_hot = HOT_VOLUME_M3 * rho_avg  # kg
    Q_needed = mass_hot * cp_avg * delta_T  # J
    mass_LN2_needed = Q_needed / H_FG_LN2  # kg
    
    print(f"\nDirect energy balance calculation:")
    print(f"Hot fluid mass: {mass_hot:.3f} kg")
    print(f"Temperature drop: {delta_T:.1f} K")
    print(f"Energy needed: {Q_needed/1000:.1f} kJ")
    print(f"LN₂ needed (theoretical): {mass_LN2_needed:.2f} kg")
    print(f"Simulation vs theoretical: {total_ln2:.2f} vs {mass_LN2_needed:.2f} kg")
    print(f"Difference: {abs(total_ln2 - mass_LN2_needed):.2f} kg ({abs(total_ln2 - mass_LN2_needed)/mass_LN2_needed*100:.1f}%)")

    # Calculate heat exchanger power
    total_area = LENGTH_M * AREA_PER_M  # m²
    # Use average temperature difference for power calculation
    # Assuming hot fluid at ~235K (average of 298K and 173K) and LN2 at 77K
    avg_temp_diff = (298 + 173) / 2 - T_LN2_SAT_K  # K
    heat_exchanger_power = U_COEFF * total_area * avg_temp_diff  # W
    
    print(f"\nHeat exchanger specifications:")
    print(f"Total heat transfer area: {total_area:.3f} m²")
    print(f"Average temperature difference: {avg_temp_diff:.1f} K")
    print(f"Heat exchanger power: {heat_exchanger_power:.1f} W ({heat_exchanger_power/1000:.2f} kW)")

    # Calculate mean temperature drop per cycle
    if len(inlet_temps) > 1:
        temp_drops = [inlet_temps[i] - inlet_temps[i+1] for i in range(len(inlet_temps)-1)]
        mean_temp_drop = np.mean(temp_drops)
        print(f"\nTemperature drop analysis:")
        print(f"Mean temperature drop per cycle: {mean_temp_drop:.3f} K")
        print(f"Total temperature drop: {inlet_temps[0] - inlet_temps[-1]:.1f} K")
        print(f"Number of cycles: {len(inlet_temps)}")
    else:
        print(f"\nTemperature drop analysis:")
        print(f"Only one cycle completed, no temperature drop to calculate")

    # combined plot with twin y-axis
    _, ax1 = plt.subplots(figsize=(10, 6))
    ax1.plot(times_min, inlet_temps, 'o-', label='Inlet HFE Temp (K)', markersize=3)
    ax1.axhline(WORKING_TEMP_K, color='gray', linestyle='--',
                label=f'Working Temp {WORKING_TEMP_K}K')
    ax1.set_xlabel('Elapsed Time (min)', fontsize=14)
    ax1.set_ylabel('Inlet HFE Temp (K)', fontsize=14)
    ax1.tick_params(axis='both', which='major', labelsize=12)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()
    ax2.plot(times_min, cumulative_ln2, 's-', color='tab:orange',
             label='Cumulative LN₂ (kg)', markersize=3)
    ax2.set_ylabel('Cumulative LN₂ Used (kg)', fontsize=14)
    ax2.tick_params(axis='y', which='major', labelsize=12)

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper center', fontsize=12)

    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    main()
