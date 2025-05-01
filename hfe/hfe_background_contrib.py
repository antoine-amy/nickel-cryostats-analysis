# pylint: disable=invalid-name, too-many-locals, too-many-arguments, line-too-long
import numpy as np
from scipy.integrate import quad
import matplotlib.pyplot as plt
import warnings
import datetime

# --- Script Information ---
# Purpose: Calculate and plot HFE/Inner Vessel backgrounds and HFE mass
#          as a function of Inner Vessel Radius, comparing with simulation/external data.
# Date: 2025-04-11
# Key Features:
#   - Calculates HFE mass based on spherical IV minus cylindrical TPC volume.
#   - Calculates HFE background using a spherical integral approximation.
#   - Calculates Inner Vessel background based on an exponential fit y=a*exp(-mu*(R_mm-r0)). # CORRECTED MODEL
#   - Allows setting a minimum effective radius for plotting/calculation.
#   - Includes Image (blue) data points for comparison.

# --- Functions ---
def calculate_hfe_background(
    inner_vessel_radius_cm: float,
    base_background_contribution_counts_per_year: float,
    hfe_attenuation_coefficient_per_cm: float,
    tpc_radius_cm: float,
    tpc_half_height_cm: float, # Argument kept for consistency
    initial_inner_vessel_radius_cm: float
) -> tuple[float, float]:
    """
    Computes the background contribution of HFE shielding liquid (using a
    spherical integral approximation) as a function of the inner vessel radius.
    Normalizes using the base_background at initial_inner_vessel_radius_cm.
    Uses memoization for the normalization constant C.
    """
    memo_attr = '_c_normalization_memo'
    float_tolerance = 1e-6
    # Physical minimum radius based on TPC geometry (corner)
    r_min_physical = np.sqrt(tpc_radius_cm**2 + tpc_half_height_cm**2)

    # If requested radius is below physical minimum, HFE background is zero
    if inner_vessel_radius_cm < r_min_physical - float_tolerance:
        # Try to get memoized C even if returning early, needed for subsequent calls
        c_norm_val_local = getattr(calculate_hfe_background, memo_attr, np.nan)
        if not hasattr(calculate_hfe_background, memo_attr):
             # If C not memoized yet, calculate it (will be stored if calc succeeds)
            try:
                _, c_norm_val_local = calculate_hfe_background(
                    initial_inner_vessel_radius_cm, base_background_contribution_counts_per_year,
                    hfe_attenuation_coefficient_per_cm, tpc_radius_cm, tpc_half_height_cm,
                    initial_inner_vessel_radius_cm)
            except RuntimeError:
                 c_norm_val_local = np.nan # Handle case where norm calc fails
        return 0.0, c_norm_val_local # Return 0 BG below physical min

    mu = hfe_attenuation_coefficient_per_cm
    # Effective inner radius for HFE calculation (approx using TPC radius)
    r_tpc_effective = tpc_radius_cm
    r_iv_initial = initial_inner_vessel_radius_cm

    # Calculate Normalization Constant C only once (Memoization)
    if not hasattr(calculate_hfe_background, memo_attr):
        # Integrate from TPC surface up to the initial IV radius for normalization
        integral_base_unnormalized, abserr_base = quad(
            differential_background_integrand, r_tpc_effective, r_iv_initial,
            args=(1.0, mu, r_tpc_effective), epsabs=1e-9, epsrel=1e-7)

        if integral_base_unnormalized <= 1e-12:
            if hasattr(calculate_hfe_background, memo_attr):
                delattr(calculate_hfe_background, memo_attr)
            raise RuntimeError(
                f"Base HFE integration for norm yielded negligible value: "
                f"{integral_base_unnormalized:.3e} (R_tpc={r_tpc_effective:.1f}, R_iv_init={r_iv_initial:.1f}, mu={mu:.3f}). Check params.")
        c_normalization = base_background_contribution_counts_per_year / integral_base_unnormalized
        setattr(calculate_hfe_background, memo_attr, c_normalization)
    else:
        c_normalization = getattr(calculate_hfe_background, memo_attr)

    # Calculate HFE background integral for the *current* inner_vessel_radius_cm
    effective_lower_bound_for_calc = r_tpc_effective
    effective_upper_bound_for_calc = max(r_tpc_effective, inner_vessel_radius_cm)

    new_background_contribution = 0.0
    if effective_upper_bound_for_calc > effective_lower_bound_for_calc + float_tolerance:
        new_background_contribution, abserr_new = quad(
            differential_background_integrand, effective_lower_bound_for_calc,
            effective_upper_bound_for_calc, args=(c_normalization, mu, r_tpc_effective),
            epsabs=1e-9, epsrel=1e-7)

    # Ensure non-negative result
    new_background_contribution = max(0.0, new_background_contribution)

    # Redundant check, but safe: ensure zero if below physical min
    if inner_vessel_radius_cm < r_min_physical - float_tolerance:
         new_background_contribution = 0.0

    return new_background_contribution, c_normalization

def differential_background_integrand(r, c_norm: float, mu_val: float, r_tpc_eff: float) -> float:
    """
    Integrand for the differential HFE background contribution assuming spherical
    shells starting from r_tpc_eff. Attenuation depends on distance from TPC.
    """
    distance_from_tpc = np.maximum(0.0, r - r_tpc_eff)
    attenuation_factor = np.exp(np.clip(-mu_val * distance_from_tpc, -700, 700))
    volume_element_factor = 4 * np.pi * r**2
    return c_norm * attenuation_factor * volume_element_factor

# --- Parameters ---
# HFE Background Calculation Parameters
BASE_BG_HFE = 5e-3 # Assumed background level at INITIAL_IV_RAD (units: counts/y/ROI/2t or counts/year)
MASS_ATTENUATION_COEFF_HFE = 0.04 # cm^2/g
HFE_DENSITY = 1.72 # g/cm^3
ATTENUATION_COEFF_HFE = MASS_ATTENUATION_COEFF_HFE * HFE_DENSITY # Mu for HFE BG (green line) in cm^-1

# Geometry Parameters
TPC_RAD = 56.665  # cm
TPC_HALF_H = 59.15 # cm
TPC_HEIGHT = 2 * TPC_HALF_H # cm
INITIAL_IV_RAD = 168.5 # Base design radius for HFE normalization (cm)

# Inner Vessel Background Fit Parameters (Model: y = a * exp(-mu * (R_mm - r0)))
IV_BG_AMPLITUDE_AT_R0 = 4.601024e-03 # 'a' parameter from the fit (value at r0)
IV_BG_MU = 0.006223              # 'mu' parameter from the fit (slope in mm^-1)
# Determine r0 (reference radius in mm) from the first data point assumed used in the fit
# These data points are also plotted as blue triangles
img_iv_radii_mm_for_fit = [1025, 1225, 1510, 1690]
IV_BG_R0 = min(img_iv_radii_mm_for_fit) if img_iv_radii_mm_for_fit else 0 # Get the minimum radius (mm)

# Plotting and Calculation Range Parameters
R_MIN_PHYS_CM = np.sqrt(TPC_RAD**2 + TPC_HALF_H**2) # Min radius based on TPC geometry
R_MIN_EFFECTIVE_CM = 95.0 # User-defined minimum radius for plotting/calculation start
CALCULATION_START_RADIUS_CM = R_MIN_EFFECTIVE_CM
V_TPC_CYLINDER_CM3 = np.pi * (TPC_RAD**2) * TPC_HEIGHT # Volume of TPC
PLOT_MAX_RADIUS_CM = 170.0 # Max radius for plot x-axis

# --- External Data Points (Blue Markers) ---
# Assumed to be the data points the fit was performed on (units match fit: counts/y/ROI/2t or counts/year)
img_iv_radii_mm = img_iv_radii_mm_for_fit # Use the same data defined above for consistency
img_iv_bg = [4.5e-3, 1.4e-3, 2.5e-4, 7e-5]
# Convert image data mm to cm for plotting on the primary x-axis
img_iv_radii_cm = [r / 10.0 for r in img_iv_radii_mm]

# --- Print Parameters ---
print("--- Parameters Used ---")
print(f"Timestamp: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"HFE Base BG (for norm @ {INITIAL_IV_RAD:.1f} cm): {BASE_BG_HFE:.2e}")
print(f"HFE Attenuation Coeff (for HFE calc): {ATTENUATION_COEFF_HFE:.4f} cm^-1")
print(f"IV Fit Model: a * exp(-mu * (R_mm - r0))") # Explicitly state model
print(f"IV Fit Amplitude (a @ r0): {IV_BG_AMPLITUDE_AT_R0:.6e}")
print(f"IV Fit Slope (mu): {IV_BG_MU:.6f} mm^-1")
print(f"IV Fit Ref Radius (r0): {IV_BG_R0:.1f} mm")
print(f"TPC Radius: {TPC_RAD:.3f} cm")
print(f"TPC Half Height: {TPC_HALF_H:.2f} cm")
print(f"Initial IV Radius (HFE Norm): {INITIAL_IV_RAD:.1f} cm")
print(f"Physical Min Radius: {R_MIN_PHYS_CM:.1f} cm")
print(f"Effective Min Plot/Calc Radius: {R_MIN_EFFECTIVE_CM:.1f} cm")
print(f"Plot Max Radius: {PLOT_MAX_RADIUS_CM:.1f} cm")
print("-----------------------")
print("--- Image Data Points (Blue Markers) ---")
print(f"IV BG Radii (cm): {[f'{r:.1f}' for r in img_iv_radii_cm]} -> BG: {[f'{b:.1e}' for b in img_iv_bg]}")
print("--------------------------")

# --- Clear Memoization Before Calculation ---
MEMO_ATTR_NAME = '_c_normalization_memo'
if hasattr(calculate_hfe_background, MEMO_ATTR_NAME):
    delattr(calculate_hfe_background, MEMO_ATTR_NAME)
    # print("Cleared HFE background memoization.") # Optional confirmation

try:
    # --- Calculations ---
    # Define calculation range, ensuring it covers plot range and necessary points
    _min_calc_rad_cm = CALCULATION_START_RADIUS_CM
    _max_calc_rad_cm = max(INITIAL_IV_RAD, PLOT_MAX_RADIUS_CM, max(img_iv_radii_cm, default=0))
    if INITIAL_IV_RAD > _max_calc_rad_cm: _max_calc_rad_cm = INITIAL_IV_RAD + 1
    if INITIAL_IV_RAD < _min_calc_rad_cm: _min_calc_rad_cm = INITIAL_IV_RAD - 1 # Ensure norm radius is included

    radii_to_check_cm = np.linspace(_min_calc_rad_cm, _max_calc_rad_cm, 300) # Array of radii in cm

    # --- HFE Background Calculation ---
    print(f"\nCalculating HFE background contribution...")
    hfe_backgrounds = []
    c_norm_val = np.nan
    try:
        # Ensure Normalization Constant C is calculated first (will be memoized)
        _, c_norm_val = calculate_hfe_background(
                INITIAL_IV_RAD, BASE_BG_HFE, ATTENUATION_COEFF_HFE,
                TPC_RAD, TPC_HALF_H, INITIAL_IV_RAD
            )
        print(f"HFE BG Normalization Constant (C): {c_norm_val:.3e}")
    except RuntimeError as e:
        print(f"ERROR calculating HFE normalization constant: {e}")
        raise # Stop execution if normalization fails

    # Calculate HFE background over the full range using the memoized C
    for r_cm in radii_to_check_cm:
        bg_hfe, _ = calculate_hfe_background(
            r_cm, BASE_BG_HFE, ATTENUATION_COEFF_HFE,
            TPC_RAD, TPC_HALF_H, INITIAL_IV_RAD
        )
        hfe_backgrounds.append(bg_hfe)
    hfe_backgrounds = np.array(hfe_backgrounds)
    print("HFE background calculation complete.")

    # --- Inner Vessel Background Calculation (Using Corrected Fit Model) ---
    print(f"\nCalculating IV background using fit: a*exp(-mu*(R_mm-r0))...")
    # Convert the radius array from cm to mm for the calculation
    radii_to_check_mm = radii_to_check_cm * 10.0
    # Apply the correct fit formula using radius in mm, mu in mm^-1, and r0 in mm
    iv_backgrounds_fit = IV_BG_AMPLITUDE_AT_R0 * np.exp(np.clip(-IV_BG_MU * (radii_to_check_mm - IV_BG_R0), -700, 700))
    iv_backgrounds_fit = np.maximum(iv_backgrounds_fit, 0) # Ensure non-negative
    print("IV background calculation complete.")

    # --- Print Check Values ---
    print("\nBackground values near key radii:")
    idx_min_eff = np.abs(radii_to_check_cm - R_MIN_EFFECTIVE_CM).argmin()
    idx_initial = np.abs(radii_to_check_cm - INITIAL_IV_RAD).argmin()
    idx_max_img = np.abs(radii_to_check_cm - max(img_iv_radii_cm)).argmin()
    print(f" - IV Radius {radii_to_check_cm[idx_min_eff]:>6.1f} cm: HFE BG = {hfe_backgrounds[idx_min_eff]:>9.3e} | IV BG (Fit) = {iv_backgrounds_fit[idx_min_eff]:>9.3e}")
    print(f" - IV Radius {radii_to_check_cm[idx_initial]:>6.1f} cm: HFE BG = {hfe_backgrounds[idx_initial]:>9.3e} | IV BG (Fit) = {iv_backgrounds_fit[idx_initial]:>9.3e}")
    print(f" - IV Radius {radii_to_check_cm[idx_max_img]:>6.1f} cm: HFE BG = {hfe_backgrounds[idx_max_img]:>9.3e} | IV BG (Fit) = {iv_backgrounds_fit[idx_max_img]:>9.3e}")

    # --- HFE Mass Calculation ---
    print("\nCalculating HFE mass...")
    volume_iv_sphere_cm3 = (4/3) * np.pi * (radii_to_check_cm**3)
    volume_tpc_cylinder_cm3 = V_TPC_CYLINDER_CM3 # Use pre-calculated constant
    volume_hfe_cm3 = np.maximum(0.0, volume_iv_sphere_cm3 - volume_tpc_cylinder_cm3)
    mass_tonnes = (volume_hfe_cm3 * HFE_DENSITY) / 1e6 # 1 tonne = 1e6 g
    print("HFE mass calculation complete.")

    # --- Plotting ---
    print("\nGenerating plot...")
    plt.style.use('seaborn-v0_8-darkgrid')
    fig1, ax1 = plt.subplots(figsize=(12, 8))

    # Plot Calculated HFE Background (Green Solid Line)
    color1 = 'forestgreen'
    ax1.set_xlabel(f"Inner Vessel Radius (cm) [Effective Min: {R_MIN_EFFECTIVE_CM:.1f} cm]")
    # Set Y-axis label (assuming units are consistent or clearly distinct)
    ax1.set_ylabel("Background [counts/y/ROI/2t or counts/year]", color=color1)
    lines_ax1 = []
    line1, = ax1.plot(radii_to_check_cm, hfe_backgrounds, color=color1, linewidth=2, label='HFE BG (Calc. Approx.)')
    lines_ax1.append(line1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_xlim(left=R_MIN_EFFECTIVE_CM, right=PLOT_MAX_RADIUS_CM) # Use defined plot limits
    ax1.set_yscale('log')

    # Plot Calculated Inner Vessel Background using the CORRECTED FIT model (Blue Dotted Line)
    color3 = 'dodgerblue'
    # Update Label to accurately reflect the model y = a*exp(-mu*(R_mm-r0))
    line3, = ax1.plot(radii_to_check_cm, iv_backgrounds_fit, color=color3, linestyle=':', linewidth=2.5, label=f'IV BG (Fit: a*exp[-{IV_BG_MU:.4f}*(R_mm-{IV_BG_R0:.0f})])')
    lines_ax1.append(line3)

    # Plot Image Data Points (Blue Markers) using radii in cm
    img_marker_size = 70
    scatter_iv_img = ax1.scatter(img_iv_radii_cm, img_iv_bg, color='blue', marker='^', s=img_marker_size, label='IV BG (Image Data)', zorder=5)
    lines_ax1.append(scatter_iv_img)

    # Adjust y-limits based on all plotted background data
    valid_hfe_bg = hfe_backgrounds[np.isfinite(hfe_backgrounds) & (hfe_backgrounds > 0)]
    valid_iv_bg_fit = iv_backgrounds_fit[np.isfinite(iv_backgrounds_fit) & (iv_backgrounds_fit > 0)]
    valid_img_bg = np.array(img_iv_bg)[np.array(img_iv_bg) > 0]
    all_bg_data_for_ylim = np.concatenate([valid_hfe_bg, valid_iv_bg_fit, valid_img_bg])
    if len(all_bg_data_for_ylim) > 0:
        min_bg_val = all_bg_data_for_ylim.min()
        max_bg_val = all_bg_data_for_ylim.max()
        ax1.set_ylim(bottom=min_bg_val * 0.2, top=max_bg_val * 5.0) # Log scale padding
    else:
        ax1.set_ylim(bottom=1e-7, top=1e-2) # Fallback limits

    # Plot Crossover Point (between HFE line and corrected IV Fit line)
    try:
        sign_diff = np.sign(hfe_backgrounds - iv_backgrounds_fit)
        valid_diff_indices = np.where(np.isfinite(sign_diff) & np.isfinite(hfe_backgrounds) & np.isfinite(iv_backgrounds_fit))[0]
        if len(valid_diff_indices) > 1:
            diff_changes = np.diff(sign_diff[valid_diff_indices])
            idx_change = np.argwhere(diff_changes != 0).flatten()
            original_indices = valid_diff_indices[idx_change]
            valid_cross_idx = [i for i in original_indices if i + 1 < len(radii_to_check_cm) and radii_to_check_cm[i] >= R_MIN_EFFECTIVE_CM and radii_to_check_cm[i+1] <= PLOT_MAX_RADIUS_CM]

            if len(valid_cross_idx) > 0:
                crossover_idx = valid_cross_idx[0]
                r1, r2 = radii_to_check_cm[crossover_idx], radii_to_check_cm[crossover_idx+1]
                bg1_hfe, bg2_hfe = hfe_backgrounds[crossover_idx], hfe_backgrounds[crossover_idx+1]
                bg1_iv, bg2_iv = iv_backgrounds_fit[crossover_idx], iv_backgrounds_fit[crossover_idx+1]
                diff1 = bg1_hfe - bg1_iv
                diff2 = bg2_hfe - bg2_iv
                if np.abs(diff1 - diff2) > 1e-12:
                     crossover_radius_cm = r1 - diff1 * (r2 - r1) / (diff2 - diff1)
                     # Interpolate on log scale might be better
                     log_bg1_hfe, log_bg2_hfe = np.log(bg1_hfe), np.log(bg2_hfe)
                     crossover_log_bg = log_bg1_hfe + (log_bg2_hfe - log_bg1_hfe) * (crossover_radius_cm - r1) / (r2 - r1)
                     crossover_bg = np.exp(crossover_log_bg)

                     if (min(r1, r2) <= crossover_radius_cm <= max(r1, r2)):
                         line_cross, = ax1.plot(crossover_radius_cm, crossover_bg, 'ko', markersize=8, markerfacecolor='none', markeredgewidth=1.5, label=f'Calc. Crossover ({crossover_radius_cm:.1f} cm)', zorder=4)
                         lines_ax1.append(line_cross)
                         print(f"\nApproximate Calculated Crossover Radius: {crossover_radius_cm:.1f} cm (BG ~ {crossover_bg:.3e})")
                     # else: print("\nInterpolated crossover radius is outside the segment.") # Optional debug
                # else: print("\nLines are parallel or identical in the crossover segment.") # Optional debug
            else:
                print("\nNo calculated crossover point found within the plotted radius range.")
        # else: print("\nNot enough valid data points to find crossover.") # Optional debug
    except Exception as e:
        import traceback
        print(f"\nCould not determine crossover point: {type(e).__name__}: {e}")
        # print(traceback.format_exc()) # Uncomment for detailed error traceback

    # Create secondary y-axis for HFE Mass (Purple Dash-Dot Line)
    ax2 = ax1.twinx()
    color2 = 'purple'
    ax2.set_ylabel("Approx. HFE Mass [tonnes] (Linear Scale)", color=color2)
    lines_ax2 = []
    line2, = ax2.plot(radii_to_check_cm, mass_tonnes, color=color2, linestyle='-.', linewidth=2, label='HFE Mass (Sphere - Cylinder)')
    lines_ax2.append(line2)
    ax2.tick_params(axis='y', labelcolor=color2)
    min_mass = 0
    valid_mass = mass_tonnes[np.isfinite(mass_tonnes)]
    max_mass = valid_mass.max() if len(valid_mass) > 0 else 1.0
    ax2.set_ylim(bottom=min_mass, top=max_mass * 1.1 + 1) # Add padding + ensure non-zero top limit
    ax2.spines['right'].set_position(('outward', 10))

    # Combine handles and labels for the legend
    lines = lines_ax1 + lines_ax2
    labels = [l.get_label() for l in lines]
    # Place legend below the axes
    ax1.legend(lines, labels, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=3, fontsize='medium')

    # Add title and grid
    ax1.set_title("HFE & Inner Vessel Backgrounds & HFE Mass vs. Inner Vessel Radius", fontsize=14, pad=20)
    ax1.grid(True, which='major', linestyle='-', linewidth=0.6, alpha=0.7)
    ax1.grid(True, which='minor', linestyle=':', linewidth=0.4, alpha=0.5)

    # Adjust layout to prevent overlap
    fig1.tight_layout(rect=[0, 0.12, 1, 0.95]) # Adjust bottom margin for legend

    # Display the final plot
    print("\nPlot generation complete.")
    plt.show()

except Exception as e:
    print(f"\nERROR during script execution: {type(e).__name__}: {e}")
    import traceback
    print(traceback.format_exc())
finally:
    # Ensure memoized C is cleared regardless of success or failure
    if hasattr(calculate_hfe_background, MEMO_ATTR_NAME):
        try:
            delattr(calculate_hfe_background, MEMO_ATTR_NAME)
            # print("Final cleanup: Memoization cleared.") # Optional confirmation
        except AttributeError:
            pass # Already deleted or never existed