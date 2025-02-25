import numpy as np
import matplotlib.pyplot as plt

# Data for selected events
radii = ["1026mm", "1226mm", "1510mm", "1691mm"]
isotopes = ["Th232", "U238"]

# Selected events data
recon_ic = {
    "Th232": {
        "values": [251.58, 60.0198, 6.1098, 2.1564],
        "errors": [np.sqrt(251.58), np.sqrt(60.0198), np.sqrt(6.1098), np.sqrt(2.1564)],
    },
    "U238": {
        "values": [170.0, 33.0, 1.0, 3.0],
        "errors": [np.sqrt(170.0), np.sqrt(33.0), np.sqrt(1.0), np.sqrt(3.0)],
    },
}

oldrecon_ic = {
    "Th232": {
        "values": [256.6116, 63.9732, 6.4692, 2.8752],
        "errors": [
            np.sqrt(256.6116),
            np.sqrt(63.9732),
            np.sqrt(6.4692),
            np.sqrt(2.8752),
        ],
    },
    "U238": {
        "values": [161.0, 36.0, 3.0, 2.0],
        "errors": [np.sqrt(161.0), np.sqrt(36.0), np.sqrt(3.0), np.sqrt(2.0)],
    },
}

oldrecon_oc = {
    "Th232": {
        "values": [264.159, 60.7386, 7.5474, 2.5158],
        "errors": [
            np.sqrt(264.159),
            np.sqrt(60.7386),
            np.sqrt(7.5474),
            np.sqrt(2.5158),
        ],
    },
    "U238": {
        "values": [165.0, 32.0, 6.0, 0.0],
        "errors": [np.sqrt(165.0), np.sqrt(32.0), np.sqrt(6.0), np.sqrt(0.0)],
    },
}


def calculate_asymmetric_errors(N):
    """
    Calculate asymmetric Poisson errors for small N
    Using Gehrels 1986 approximation
    """
    if N == 0:
        upper = 1.841  # 84% upper limit for 0 observed events
        lower = 0
    else:
        upper = np.sqrt(N + 0.75) + 1
        lower = np.sqrt(N - 0.25) if N > 0.25 else 0
    return lower, upper


# Set up the plot
fig, axes = plt.subplots(1, 2, figsize=(12, 6))

# Bar width and positions
width = 0.25
r1 = np.arange(len(radii))
r2 = [x + width for x in r1]
r3 = [x + width for x in r2]

for i, isotope in enumerate(isotopes):
    ax = axes[i]

    for dataset, positions, color, label in [
        (recon_ic[isotope]["values"], r1, "royalblue", "My new recon code IC"),
        (oldrecon_ic[isotope]["values"], r2, "seagreen", "Old recon code IC"),
        (oldrecon_oc[isotope]["values"], r3, "orange", "Old recon code IC 2"),
    ]:
        yerr_low = []
        yerr_high = []
        for val in dataset:
            low, high = calculate_asymmetric_errors(val)
            yerr_low.append(low)
            yerr_high.append(high)

        ax.bar(positions, dataset, width, label=label, color=color)
        # Plot asymmetric error bars
        ax.errorbar(
            positions,
            dataset,
            yerr=[yerr_low, yerr_high],
            fmt="none",
            color="black",
            capsize=3,
        )

    # Customize the plot
    ax.set_yscale("log")
    ax.set_xticks([r + width for r in range(len(radii))])
    ax.set_xticklabels(radii, rotation=45)
    ax.set_title(f"{isotope} Selected Events")
    ax.set_ylabel("Number of Selected Events")
    ax.grid(True, which="both", linestyle="--", alpha=0.7)
    ax.legend()

# Add title with adjusted layout
fig.suptitle("Number of Selected Background events from IC (1e9 events)")
plt.tight_layout()
plt.subplots_adjust(top=0.9)

plt.show()

# Print numerical comparisons with asymmetric uncertainties
print("\nNumerical Comparisons (Selected Events with asymmetric errors):")
for isotope in isotopes:
    print(f"\n{isotope}:")
    for i, radius in enumerate(radii):
        print(f"\n{radius}:")
        for dataset, label in [
            (recon_ic[isotope]["values"][i], "My new recon code IC"),
            (oldrecon_ic[isotope]["values"][i], "Old recon code IC"),
            (oldrecon_oc[isotope]["values"][i], "Old recon code IC 2"),
        ]:
            low, high = calculate_asymmetric_errors(dataset)
            print(f"{label}: {dataset:.2f} (+{high:.2f}/-{low:.2f})")
