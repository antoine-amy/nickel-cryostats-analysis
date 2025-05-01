import numpy as np
import matplotlib.pyplot as plt

# Constants from the document
r0 = 1691  # baseline inner vessel radius in mm
B0 = 2.76e-8  # baseline background rate in counts/year
mu = 0.00685  # effective attenuation coefficient in 1/mm from Cryostat background section
mu_error = 0.0005  # uncertainty in attenuation coefficient (1σ)
background_requirement = 1e-5  # requirement in counts/year

# Function to calculate background rate for a given radius
def background_rate(r, mu_value):
    """
    Calculate background rate using the equation from the document:
    B(r) = B0 * exp(-μ(r-r0))
    
    Parameters:
    r (float or array): Inner vessel radius in mm
    mu_value (float): Attenuation coefficient in 1/mm
    
    Returns:
    float or array: Background rate in counts/year
    """
    return B0 * np.exp(-mu_value * (r - r0))

# Range of radii to consider (from 1000 to 1700 mm as in the document)
radii = np.linspace(1000, 1700, 100)

# Calculate background rates
background_rates = background_rate(radii, mu)
# Lower mu means higher background (less attenuation)
background_rates_upper = background_rate(radii, mu - mu_error)  
# Higher mu means lower background (more attenuation)
background_rates_lower = background_rate(radii, mu + mu_error)  

# Create the plot
plt.figure(figsize=(12, 8))

# Set bigger font sizes
plt.rcParams.update({'font.size': 14, 'axes.labelsize': 16, 'axes.titlesize': 18, 
                     'xtick.labelsize': 14, 'ytick.labelsize': 14, 'legend.fontsize': 14})

# Plot the background rates with uncertainty band
plt.semilogy(radii, background_rates, 'b-', linewidth=3, label='Expected background rate')
plt.fill_between(radii, background_rates_lower, background_rates_upper, color='b', alpha=0.3, 
                 label='1σ statistical uncertainty')

# Plot the requirement line
plt.axhline(y=background_requirement, color='r', linestyle='--', linewidth=2.5,
            label=f'Background requirement ({background_requirement:.0e} counts/year)')

# Add annotation for baseline
plt.axvline(x=r0, color='g', linestyle=':', alpha=0.7, linewidth=2)
# Move the green text inside the plot area
plt.text(1590, 7e-6, f'Baseline: {r0} mm\n{B0:.2e} counts/year', 
         bbox=dict(facecolor='white', alpha=0.7, edgecolor='green', boxstyle='round,pad=0.5'),
         va='center', ha='center', color='green', fontsize=14, fontweight='bold')

# Add labels and title
plt.xlabel('Inner Vessel Radius (mm)', fontweight='bold')
plt.ylabel('Background Rate (counts/year)', fontweight='bold')
#plt.title('Evolution of Water Shield Background Rate with Inner Vessel Radius', fontweight='bold', pad=15)
plt.grid(True, which="both", ls="-", alpha=0.3)
plt.legend(loc='lower left', framealpha=0.9, fontsize=16, frameon=True, fancybox=True, shadow=True)

# Set y-axis limits to match the figure in the document
#plt.ylim(1e-9, 1e-4)

plt.tight_layout()
plt.savefig("water_analysis/water_bkgd_vs_HFE.png")
plt.show()

# Print some key values for reference
print(f"Baseline radius: {r0} mm")
print(f"Baseline background rate: {B0:.2e} counts/year")
print(f"Attenuation coefficient: {mu:.5f} per mm (mean value from Cryostat Background section)")

# Calculate and print the minimum radius that keeps the background below the requirement
r_min = r0 - (1/mu) * np.log(background_requirement / B0)
print(f"Minimum radius to stay below requirement: {r_min:.1f} mm")