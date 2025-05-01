"""
Plot mass attenuation coefficients for HFE-7000 across different photon energies.
"""
from io import StringIO
import numpy as np
import matplotlib.pyplot as plt

# Data as string (same as before)
DATA_STR = """Photon    Coherent Incoher. Photoel. Nuclear  Electron Tot. w/  Tot. wo/ 
Energy    Scatter. Scatter. Absorb.  Pr. Prd. Pr. Prd. Coherent Coherent 
1.000E-01 5.923E-03 1.438E-01 3.444E-03 0.000E+00 0.000E+00 1.532E-01 1.473E-01 
1.500E-01 2.707E-03 1.305E-01 9.146E-04 0.000E+00 0.000E+00 1.341E-01 1.314E-01 
2.000E-01 1.539E-03 1.199E-01 3.621E-04 0.000E+00 0.000E+00 1.218E-01 1.203E-01 
3.000E-01 6.896E-04 1.045E-01 1.026E-04 0.000E+00 0.000E+00 1.053E-01 1.046E-01 
4.000E-01 3.891E-04 9.377E-02 4.398E-05 0.000E+00 0.000E+00 9.420E-02 9.382E-02 
5.000E-01 2.493E-04 8.567E-02 2.372E-05 0.000E+00 0.000E+00 8.594E-02 8.569E-02 
6.000E-01 1.733E-04 7.927E-02 1.478E-05 0.000E+00 0.000E+00 7.945E-02 7.928E-02 
8.000E-01 9.754E-05 6.967E-02 7.460E-06 0.000E+00 0.000E+00 6.977E-02 6.967E-02 
1.000E+00 6.245E-05 6.265E-02 4.637E-06 0.000E+00 0.000E+00 6.272E-02 6.265E-02 
1.022E+00 5.978E-05 6.200E-02 4.333E-06 0.000E+00 0.000E+00 6.206E-02 6.200E-02 
1.250E+00 3.997E-05 5.603E-02 2.940E-06 1.854E-05 0.000E+00 5.609E-02 5.605E-02 
1.500E+00 2.776E-05 5.093E-02 2.133E-06 1.027E-04 0.000E+00 5.106E-02 5.103E-02 
2.000E+00 1.562E-05 4.345E-02 1.340E-06 4.091E-04 0.000E+00 4.388E-02 4.386E-02 
2.044E+00 1.495E-05 4.292E-02 1.296E-06 4.411E-04 0.000E+00 4.338E-02 4.336E-02 
2.500E+00 9.998E-06 3.818E-02 9.633E-07 7.867E-04 2.068E-06 3.898E-02 3.897E-02 
3.000E+00 6.943E-06 3.419E-02 7.476E-07 1.170E-03 1.196E-05 3.538E-02 3.537E-02 
4.000E+00 3.906E-06 2.852E-02 5.126E-07 1.898E-03 4.884E-05 3.047E-02 3.047E-02 
5.000E+00 2.499E-06 2.463E-02 3.885E-07 2.547E-03 9.731E-05 2.728E-02 2.727E-02 
6.000E+00 1.736E-06 2.177E-02 3.123E-07 3.129E-03 1.495E-04 2.505E-02 2.505E-02 
7.000E+00 1.275E-06 1.957E-02 2.607E-07 3.649E-03 2.015E-04 2.342E-02 2.342E-02 
8.000E+00 9.764E-07 1.781E-02 2.237E-07 4.116E-03 2.521E-04 2.218E-02 2.217E-02 
9.000E+00 7.714E-07 1.637E-02 1.958E-07 4.539E-03 3.005E-04 2.121E-02 2.121E-02 
1.000E+01 6.249E-07 1.516E-02 1.741E-07 4.925E-03 3.468E-04 2.044E-02 2.044E-02"""

# Convert string data to numpy array
data = np.genfromtxt(StringIO(DATA_STR), skip_header=2)

# Extract columns
energy = data[:, 0]  # MeV
coherent = data[:, 1]
incoherent = data[:, 2]
photoelectric = data[:, 3]
nuclear = data[:, 4]
electron = data[:, 5]
total = data[:, 6]

# Set figure parameters
plt.figure(figsize=(9, 7))
plt.grid(True, which="both", ls="-", alpha=0.2)

# Plot individual components with different colors and styles
plt.loglog(
    energy, coherent, "--", color="#1f77b4", label="Coherent Scattering", alpha=0.7
)
plt.loglog(
    energy, incoherent, "--", color="#ff7f0e", label="Incoherent Scattering", alpha=0.7
)
plt.loglog(
    energy,
    photoelectric,
    "--",
    color="#2ca02c",
    label="Photoelectric Absorption",
    alpha=0.7,
)
plt.loglog(
    energy, nuclear, "--", color="#d62728", label="Nuclear Pair Production", alpha=0.7
)
plt.loglog(
    energy, electron, "--", color="#9467bd", label="Electron Pair Production", alpha=0.7
)
plt.loglog(energy, total, "k-", label="Total", linewidth=2)

# Customize the plot
plt.xlabel("Photon Energy (MeV)", fontsize=20)
plt.ylabel("Mass Attenuation Coefficient (cm²/g)", fontsize=20)

# Add legend
plt.legend(fontsize=16)

# Highlight 2.5 MeV point
idx_2_5MeV = np.where(abs(energy - 2.5) < 1e-10)[0][0]
plt.plot(2.5, total[idx_2_5MeV], "ro", markersize=8)
plt.annotate(
    "2.5 MeV: 0.04 cm²/g",
    xy=(2.5, float(total[idx_2_5MeV])),
    xytext=(1.5, 0.006),  # Adjusted position
    arrowprops=dict(facecolor="red", shrink=0.05),
    fontsize=20,
)

# Add minor grid
plt.grid(True, which="minor", linestyle=":", alpha=0.2)

# Adjust layout
plt.tight_layout()
plt.show()
