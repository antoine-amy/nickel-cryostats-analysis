#!/usr/bin/env python3
"""
Simple nEXO MLI Scaling Analysis

This script calculates the required number of MLI layers for nEXO by:
1. Computing the radiative heat transfer in EXO-200
2. Maintaining the same radiative heat percentage in nEXO
3. Scaling for nEXO's 500W heat load and 3kW cooling capacity

Based on EXO-200's configuration (5 layers of single-sided aluminized PET)
and nEXO's thermal stability requirement of 0.1K.

An additional analysis is included to compute and plot the background contribution,
defined as: background = mass * activity * hit efficiency, converted into counts/year.
"""

import math
import matplotlib.pyplot as plt
import numpy as np

# Constants
STEFAN_BOLTZMANN = 5.67e-8  # W/m²K⁴

# EXO-200 parameters
EXO200_AREA = 0.9  # m² (approximate surface area of EXO-200 inner vessel)
EXO200_TOTAL_HEAT_LOAD = 90  # W (from the document)
EXO200_MLI_LAYERS = 5  # layers
EXO200_EMISSIVITY = 0.035  # emissivity of aluminized PET
EXO200_TEMP_OUTER = 300  # K (room temperature)
EXO200_TEMP_INNER = 170  # K (LXe temperature)

# nEXO parameters
NEXO_RADIUS = 1.691  # m (spherical vessel radius - 1691 mm)
NEXO_AREA = 4 * math.pi * NEXO_RADIUS**2  # m² (surface area)
NEXO_TOTAL_HEAT_LOAD = 500  # W (requirement)
NEXO_COOLING_CAPACITY = 3000  # W (3 kW cooling system)
NEXO_TEMP_STABILITY = 0.1  # K (requirement)
NEXO_TEMP_OUTER = 300  # K (room temperature)
NEXO_TEMP_INNER = 170  # K (LXe temperature)

# MLI option - both Type A and Type B have the same emissivity
MLI_EMISSIVITY = 0.035  # aluminized PET

# Isotope activity concentrations (in [g contaminant/g MLI])
# For Type A:
A_U_A = 5740e-12     # U-238 concentration for Type A
A_Th_A = 1800e-12    # Th-232 concentration for Type A (upper limit)
# For Type B:
A_U_B = 2500e-12     # U-238 concentration for Type B
A_Th_B = 1540e-12    # Th-232 concentration for Type B (upper limit)

# Hit efficiencies for the MLI (dimensionless)
hit_eff_U = 7.000e-9      # for U-238 (central value)
hit_eff_Th = 9.720e-9     # for Th-232 (central value)

def calculate_effective_emissivity(emissivity, n_layers):
    """Calculate effective emissivity of MLI stack."""
    return emissivity / (n_layers + 1)

def calculate_radiative_heat_load(area, emissivity_eff, temp_outer, temp_inner):
    """Calculate radiative heat transfer through MLI."""
    return emissivity_eff * STEFAN_BOLTZMANN * area * (temp_outer**4 - temp_inner**4)

def calculate_required_layers(area, emissivity, temp_outer, temp_inner, max_heat_load):
    """Calculate minimum number of layers needed to keep radiative heat load below target."""
    radiative_factor = STEFAN_BOLTZMANN * area * (temp_outer**4 - temp_inner**4)
    emissivity_eff_required = max_heat_load / radiative_factor
    n_layers = (emissivity / emissivity_eff_required) - 1
    return math.ceil(max(1, n_layers))

def calculate_background(mass, activity_U, activity_Th, hit_eff_U, hit_eff_Th):
    """
    Calculate the background contribution in counts/year.
    
    Parameters:
      mass         : mass of the MLI in kg.
      activity_U   : U-238 concentration in [g contaminant/g MLI].
      activity_Th  : Th-232 concentration in [g contaminant/g MLI].
      hit_eff_U    : hit efficiency for U-238.
      hit_eff_Th   : hit efficiency for Th-232.
    
    Returns:
      Total background contribution in counts/year.
    """
    mass_g = mass * 1000  # Convert kg to grams
    seconds_per_year = 3.15576e7  # seconds in one year
    
    # Specific activities in decays per second per gram (Bq/g)
    specific_activity_U = 1.24e4   # ~1.24e4 decays/s per gram for U-238
    specific_activity_Th = 4050    # ~4050 decays/s per gram for Th-232
    
    background_U = mass_g * activity_U * specific_activity_U * hit_eff_U * seconds_per_year
    background_Th = mass_g * activity_Th * specific_activity_Th * hit_eff_Th * seconds_per_year
    return background_U + background_Th

def main():
    print("\n===== SIMPLE NEXO MLI SCALING ANALYSIS =====\n")
    
    # Step 1: Calculate radiative heat load in EXO-200
    exo200_eff_emissivity = calculate_effective_emissivity(
        EXO200_EMISSIVITY, EXO200_MLI_LAYERS
    )
    
    exo200_radiative_load = calculate_radiative_heat_load(
        EXO200_AREA, exo200_eff_emissivity, 
        EXO200_TEMP_OUTER, EXO200_TEMP_INNER
    )
    
    # Calculate percentage of heat load that was radiative in EXO-200
    exo200_radiative_percentage = (exo200_radiative_load / EXO200_TOTAL_HEAT_LOAD) * 100
    
    print(f"EXO-200 Analysis:")
    print(f"- Surface area: {EXO200_AREA:.2f} m²")
    print(f"- Total heat load: {EXO200_TOTAL_HEAT_LOAD} W")
    print(f"- MLI layers: {EXO200_MLI_LAYERS}")
    print(f"- Effective emissivity: {exo200_eff_emissivity:.6f}")
    print(f"- Calculated radiative heat load: {exo200_radiative_load:.2f} W")
    print(f"- Radiative percentage of total heat: {exo200_radiative_percentage:.2f}%")
    
    # Step 2: Maintain same radiative percentage for nEXO
    nexo_radiative_budget = (exo200_radiative_percentage / 100) * NEXO_TOTAL_HEAT_LOAD
    nexo_conductive_budget = NEXO_TOTAL_HEAT_LOAD - nexo_radiative_budget
    
    # Step 3: Calculate required layers for both MLI types
    nexo_layers = calculate_required_layers(
        NEXO_AREA, MLI_EMISSIVITY, 
        NEXO_TEMP_OUTER, NEXO_TEMP_INNER,
        nexo_radiative_budget
    )
    
    # Step 4: Calculate actual radiative heat loads with these layers
    nexo_eff_emissivity = calculate_effective_emissivity(MLI_EMISSIVITY, nexo_layers)
    
    nexo_radiative_load = calculate_radiative_heat_load(
        NEXO_AREA, nexo_eff_emissivity,
        NEXO_TEMP_OUTER, NEXO_TEMP_INNER
    )
    
    # Calculate total heat loads and radiative percentages
    nexo_total_load = nexo_radiative_load + nexo_conductive_budget
    
    nexo_radiative_percentage = (nexo_radiative_load / nexo_total_load) * 100
    
    # Calculate impact on temperature stability
    # Simple model: ΔT ∝ (heat_load / cooling_capacity)
    # Normalizing to 0.1K at 500W
    temp_sensitivity = 0.1 * (NEXO_COOLING_CAPACITY / NEXO_TOTAL_HEAT_LOAD)
    
    nexo_temp_stability = temp_sensitivity * nexo_total_load / NEXO_COOLING_CAPACITY
    
    print("\nnEXO Analysis:")
    print(f"- Surface area: {NEXO_AREA:.2f} m²")
    print(f"- Target total heat load: {NEXO_TOTAL_HEAT_LOAD} W")
    print(f"- Target radiative heat budget: {nexo_radiative_budget:.2f} W")
    print(f"- Target conductive heat budget: {nexo_conductive_budget:.2f} W")
    
    print("\nAluminized PET (ε ≤ 0.035):")
    print(f"- Required layers: {nexo_layers}")
    print(f"- Effective emissivity: {nexo_eff_emissivity:.6f}")
    print(f"- Radiative heat load: {nexo_radiative_load:.2f} W ({nexo_radiative_percentage:.2f}%)")
    print(f"- Total heat load: {nexo_total_load:.2f} W")
    print(f"- Temperature stability: {nexo_temp_stability:.4f} K (requirement: {NEXO_TEMP_STABILITY} K)")
    
    # Recommendation
    print("\nRecommendation:")
    print(f"Aluminized PET with {nexo_layers} layers is recommended")
    
    # Generate a plot showing layers vs. radiative heat load
    try:
        max_layers = nexo_layers * 2
        layer_range = range(1, max_layers + 1)
        
        rad_loads = []
        
        for n in layer_range:
            emiss_eff = calculate_effective_emissivity(MLI_EMISSIVITY, n)
            
            rad_load = calculate_radiative_heat_load(
                NEXO_AREA, emiss_eff, NEXO_TEMP_OUTER, NEXO_TEMP_INNER
            )
            
            rad_loads.append(rad_load)
        
        plt.figure(figsize=(12, 9))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        
        plt.plot(layer_range, rad_loads, 'b-', linewidth=2, 
                 label='Aluminized PET (ε ≤ 0.035)')
        plt.axhline(y=nexo_radiative_budget, color='g', linestyle='--', linewidth=2,
                   label=f'Target Radiative Budget from EXO-200 scaling ({nexo_radiative_budget:.2f} W, {exo200_radiative_percentage:.2f}% of total heat)')
        plt.axvline(x=nexo_layers, color='r', linestyle=':', linewidth=2,
                   label=f'Min Required Layers ({nexo_layers})')
        
        plt.xlabel('Number of MLI Layers', fontsize=14)
        plt.ylabel('Radiative Heat Load (W)', fontsize=14)
        plt.title('nEXO: Radiative Heat Load vs. Number of MLI Layers\nInner Cryostat Vessel radius: 1691 mm', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.yscale('log')
        plt.xlim(0, 50)  # Set x-axis limit to 50 layers
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        plt.tight_layout()
        plt.savefig('thermal/nEXO_MLI_Simple_Analysis.png')
        print("\nAnalysis plot saved to 'nEXO_MLI_Simple_Analysis.png'")
    except Exception as e:
        print(f"\nWarning: Could not generate radiative heat load plot: {e}")
    
    # ----------------------------------------------------------------
    # New: Plot background contribution as a function of number of layers
    # ----------------------------------------------------------------
    # We assume that the total MLI mass scales linearly with the number of layers.
    # For the inner cryostat, 5 layers correspond to a total mass of 6.5 kg.
    try:
        nominal_layers = 5      # baseline number of layers
        nominal_mass = 6.5      # kg (for nominal 5-layer configuration)
        
        max_layers_bg = max(nexo_layers, 40)  # Use at least 40 for background plot
        layer_range_bg = range(1, max_layers_bg + 1)
        background_A_list = []
        background_B_list = []
        
        for n in layer_range_bg:
            # Scale mass linearly: mass_n = (n / nominal_layers) * nominal_mass
            mass_n = nominal_mass * (n / nominal_layers)
            bg_A = calculate_background(mass_n, A_U_A, A_Th_A, hit_eff_U, hit_eff_Th)
            bg_B = calculate_background(mass_n, A_U_B, A_Th_B, hit_eff_U, hit_eff_Th)
            background_A_list.append(bg_A)
            background_B_list.append(bg_B)
        
        plt.figure(figsize=(12, 9))
        plt.rcParams.update({'font.size': 12})  # Increase default font size
        
        plt.plot(layer_range_bg, background_A_list, 'b-', linewidth=2,
                 label='Type A (U-238: 5.74×10⁻⁹ g/g, Th-232: 1.80×10⁻⁹ g/g)')
        plt.plot(layer_range_bg, background_B_list, 'r-', linewidth=2,
                 label='Type B (U-238: 2.50×10⁻⁹ g/g, Th-232: 1.54×10⁻⁹ g/g)')
        plt.axvline(x=5, color='k', linestyle=':', linewidth=2,
                   label='5 Layers (EXO-200 baseline)')
        plt.axvline(x=nexo_layers, color='r', linestyle='--', linewidth=2,
                   label=f'Recommended Layers ({nexo_layers})')
        plt.xlabel('Number of MLI Layers', fontsize=14)
        plt.ylabel('Background Contribution (counts/year)', fontsize=14)
        plt.title('Background Contribution vs. Number of MLI Layers\nInner Cryostat Vessel radius: 1691 mm', fontsize=16)
        plt.grid(True)
        plt.legend(fontsize=12)
        plt.yscale('log')
        plt.xlim(0, 40)  # Set x-axis limit to 40 layers
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        plt.tight_layout()
        plt.savefig('thermal/nEXO_MLI_Background_Analysis.png')
        print("Background contribution plot saved to 'nEXO_MLI_Background_Analysis.png'\n")
    except Exception as e:
        print(f"\nWarning: Could not generate background plot: {e}")

if __name__ == "__main__":
    main()