import numpy as np

def calculate_radius_from_hfe_mass(mass_tonnes):
    """
    Calculate the inner cryostat vessel radius based on a given HFE mass.
    
    Args:
        mass_tonnes (float): The HFE mass in tonnes
        
    Returns:
        float: The inner cryostat vessel radius in millimeters
    """
    # First, calculate the volume from the mass
    # mass = volume * 1.72, so volume = mass / 1.72
    volume_m3 = mass_tonnes / 1.72
    
    # Now use the volume to calculate the radius
    # volume = (4/3) * np.pi * radius_m**3 - 2.2 + 0.58
    # Rearranging for radius_m:
    radius_m = ((volume_m3 - 0.58 + 2.2) / ((4/3) * np.pi)) ** (1/3)
    radius_mm = radius_m * 1000
    
    return radius_mm

def calculate_values_from_mass(mass_tonnes):
    """
    Calculate all related values from the HFE mass.
    
    Args:
        mass_tonnes (float): The HFE mass in tonnes
        
    Returns:
        dict: Dictionary containing radius, mass, volume, and price
    """
    radius_mm = calculate_radius_from_hfe_mass(mass_tonnes)
    volume_m3 = mass_tonnes / 1.72
    price = mass_tonnes * (2.2e6 / 32)  # price_per_tonne = 2.2e6 / 32
    
    return {
        "iv_radius_mm": radius_mm,
        "hfe_mass_tonnes": mass_tonnes,
        "hfe_volume_m3": volume_m3,
        "hfe_price_millions": price / 1e6
    }

# Print the results for 12.8 tonnes
def print_values_from_mass(mass_tonnes):
    values = calculate_values_from_mass(mass_tonnes)
    print(f"Values for HFE Mass of {mass_tonnes:.2f} tonnes:")
    print(f"  IV Radius: {values['iv_radius_mm']:.2f} mm")
    print(f"  HFE Mass: {values['hfe_mass_tonnes']:.2f} tonnes")
    print(f"  HFE Volume: {values['hfe_volume_m3']:.2f} m³")
    print(f"  HFE Price: ${values['hfe_price_millions']:.2f}M")

# Calculate for 12.8 tonnes
print_values_from_mass(12.8)