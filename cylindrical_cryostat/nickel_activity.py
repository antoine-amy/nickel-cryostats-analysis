# Values used in the collaboration
# From data assays of R-207-1-1-1

# --- Th-232 detections from the report (μBq/kg) ---
th_vals_ubq = [0.09, 0.37, 0.36, 0.20, 0.30]  # ignore the "<0.08" censored point

# Simple mean
th_mean_ubq = sum(th_vals_ubq) / len(th_vals_ubq)

# Sample standard deviation (Bessel-corrected)
n = len(th_vals_ubq)
th_stdev_ubq = (sum((x - th_mean_ubq)**2 for x in th_vals_ubq) / (n - 1)) ** 0.5

# Convert μBq/kg -> Bq/kg
th_mean_bqkg  = th_mean_ubq  * 1e-6
th_sigma_bqkg = th_stdev_ubq * 1e-6

# --- U-238 convention used by the collaboration ---
u_mean_bqkg  = 0.0
u_sigma_bqkg = 7.430e-7  # Bq/kg (best 1σ UL adopted)

# Print in scientific notation like the sheet
print(f"Th-232: {th_mean_bqkg:.3E} {th_sigma_bqkg:.3E}")
print(f"U-238:  {u_mean_bqkg:.0f} {u_sigma_bqkg:.3E}")