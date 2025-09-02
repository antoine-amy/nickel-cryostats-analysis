"""
This module simulates the cooling of HFE-7200 fluid by recirculating
liquid nitrogen (LN2) through a stainless steel tube heat exchanger.

The model calculates the temperature of the HFE-7200 as a function of time,
plots the cooling curve, and determines the time constant and the time
to reach a specified target temperature.

 Assumptions:
    1. The HFE-7200 in the tank is well-mixed, meaning its temperature is
       uniform throughout at any given time.
    2. The LN2 remains at its saturation temperature (77.15 K) throughout the
       tube length. The heat transfer from the HFE-7200 causes the LN2 to
       boil, but its temperature is assumed to be constant.
    3. Heat loss to the ambient environment from the tank and piping is
       neglected.
    4. The convective heat transfer coefficient for boiling LN2 inside the
       tube is estimated to be 1500 W/m^2·K, a typical value for pool boiling.
    5. The convective heat transfer coefficient for HFE-7200 flowing over
       the tube is estimated to be 500 W/m^2·K. This is an approximation
       for forced convection of a liquid.
    6. The initial temperature of the HFE-7200 is 25°C (298.15 K).
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# --- Constants and Initial Conditions ---

# HFE-7200 Properties
HFE_DENSITY = 1430  # kg/m^3
HFE_SPECIFIC_HEAT = 1220  # J/kg·K
HFE_VOLUME = 5.0 / 1000  # m^3 (5 L)
HFE_FLOW_RATE = 2.0 / (1000 * 60) # m^3/s (2 L/min)
HFE_INITIAL_TEMP_K = 298.15  # K (25 °C)
HFE_TARGET_TEMP_K = 165  # K

# Liquid Nitrogen (LN2) Properties
LN2_TEMP_K = 77.15  # K (Saturation temperature at 1 atm)

# Heat Exchanger Tubing Properties (SS 304L)
TUBE_LENGTH = 2.0  # m
TUBE_OUTER_DIA_M = 0.25 * 0.0254  # m (1/4 inch)
TUBE_WALL_THICKNESS_M = 0.049 * 0.0254  # m (0.049 inch)
TUBE_INNER_DIA_M = TUBE_OUTER_DIA_M - 2 * TUBE_WALL_THICKNESS_M
SS304L_THERMAL_CONDUCTIVITY = 16.2  # W/m·K (at room temp, approx.)

# --- Calculated Parameters ---

# Mass of HFE-7200
hfe_mass = HFE_DENSITY * HFE_VOLUME

# Heat transfer areas
tube_inner_area = np.pi * TUBE_INNER_DIA_M * TUBE_LENGTH
tube_outer_area = np.pi * TUBE_OUTER_DIA_M * TUBE_LENGTH

# Convective heat transfer coefficients (Assumed)
h_inner_ln2 = 1500  # W/m^2·K (boiling LN2)
h_outer_hfe = 500   # W/m^2·K (forced convection HFE-7200)

# Thermal resistances
r_conv_inner = 1 / (h_inner_ln2 * tube_inner_area)
r_conduction_wall = np.log(TUBE_OUTER_DIA_M / TUBE_INNER_DIA_M) / \
                    (2 * np.pi * SS304L_THERMAL_CONDUCTIVITY * TUBE_LENGTH)
r_conv_outer = 1 / (h_outer_hfe * tube_outer_area)

# Overall heat transfer coefficient (U) referenced to the outer area (A_o)
# 1 / (U * A_o) = R_total = R_conv_inner + R_cond + R_conv_outer
ua_product = 1 / (r_conv_inner + r_conduction_wall + r_conv_outer)


def model_cooling(temperature_k, time_s):
    """
    Defines the ordinary differential equation for the cooling process.
    This function calculates the rate of temperature change (dT/dt).

    Args:
        temperature_k (float): The temperature of the HFE-7200 in Kelvin.
        time_s (float): The current time in seconds (required by odeint).

    Returns:
        float: The rate of change of temperature (dT/dt) in K/s.
    """
    # Rate of heat transfer Q_dot = U * A * delta_T
    q_dot = ua_product * (temperature_k - LN2_TEMP_K)

    # Rate of temperature change dT/dt = -Q_dot / (m * c_p)
    d_temp_d_time = -q_dot / (hfe_mass * HFE_SPECIFIC_HEAT)
    return d_temp_d_time


def solve_and_analyze():
    """
    Solves the cooling model and analyzes the results.
    """
    # Time vector for the simulation
    # We run it for a long time to ensure we reach the target
    time_vector_s = np.linspace(0, 18000, 18001)  # 0 to 5 hours in seconds

    # Solve the ODE
    temperature_solution_k = odeint(
        model_cooling, HFE_INITIAL_TEMP_K, time_vector_s
    ).flatten()

    # --- Analysis ---
    # 1. Find the time to reach the target temperature
    try:
        # Find the first index where temperature is at or below the target
        target_time_index = np.where(
            temperature_solution_k <= HFE_TARGET_TEMP_K)[0][0]
        time_to_target_s = time_vector_s[target_time_index]
        time_to_target_min = time_to_target_s / 60
        print(
            f"Time to reach {HFE_TARGET_TEMP_K} K: "
            f"{time_to_target_min:.2f} minutes"
        )
    except IndexError:
        print(f"Target temperature of {HFE_TARGET_TEMP_K} K was not reached.")
        time_to_target_min = None

    # 2. Calculate the time constant (tau)
    # The time constant is the time it takes to reach 63.2% of the total
    # temperature change.
    total_temp_change = HFE_INITIAL_TEMP_K - LN2_TEMP_K
    temp_at_one_tau = HFE_INITIAL_TEMP_K - 0.632 * total_temp_change
    try:
        tau_index = np.where(temperature_solution_k <= temp_at_one_tau)[0][0]
        time_constant_s = time_vector_s[tau_index]
        time_constant_min = time_constant_s / 60
        print(f"Cooling time constant (tau): {time_constant_min:.2f} minutes")
    except IndexError:
        print("Could not determine the time constant.")
        time_constant_min = None

    return time_vector_s, temperature_solution_k, time_to_target_min


def plot_results(time_s, temp_k, time_to_target):
    """
    Plots the temperature of the HFE-7200 over time.

    Args:
        time_s (np.array): Array of time points in seconds.
        temp_k (np.array): Array of temperature points in Kelvin.
        time_to_target (float): Time in minutes to reach target temp.
    """
    # Convert to more readable units for plotting
    time_min = time_s / 60
    temp_c = temp_k - 273.15
    target_temp_c = HFE_TARGET_TEMP_K - 273.15

    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot the main cooling curve
    ax.plot(time_min, temp_c, label='HFE-7200 Temperature', color='b')

    # Add a horizontal line for the target temperature
    ax.axhline(
        y=target_temp_c, color='r', linestyle='--',
        label=f'Target Temp ({target_temp_c:.1f} °C)'
    )

    # Add a vertical line indicating the time to reach the target
    if time_to_target is not None:
        ax.axvline(
            x=time_to_target, color='g', linestyle='--',
            label=f'Time to Target ({time_to_target:.2f} min)'
        )

    # Formatting the plot
    ax.set_title('HFE-7200 Cooling Simulation', fontsize=16)
    ax.set_xlabel('Time (minutes)', fontsize=12)
    ax.set_ylabel('Temperature (°C)', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True)

    # Set reasonable axis limits
    plot_limit_time = (time_to_target * 1.5) if time_to_target else 60
    ax.set_xlim(0, plot_limit_time)
    ax.set_ylim(bottom=target_temp_c - 10, top=HFE_INITIAL_TEMP_K - 273.15 + 10)

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    # Run the simulation and analysis
    time_data_s, temp_data_k, target_time_min = solve_and_analyze()

    # Plot the results
    plot_results(time_data_s, temp_data_k, target_time_min)