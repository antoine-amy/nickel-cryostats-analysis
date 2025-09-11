## Nickel Cryostats Analysis

Analysis and plotting scripts for background estimates, hit efficiencies, thermal scaling, and related studies for the Nickel cryostat configurations.

### Repository layout

- `budget/`: Background budget comparisons and summaries
- `cylindrical_cryostat/`: Geometry-driven IV/OV/HFE scaling studies
- `hit_efficiencies/`: IV hit-efficiency test scripts and fits
- `thermal/`: Thermal mass and MLI scaling utilities
- `TG_Spread/`: Truncated-Gaussian utilities for spread/limits
- Top-level helpers: `iv_mass.py`, `hfe_*.py`, `water_bkgd_*.py`, etc.

### Requirements

Python 3.10+ recommended.

- See `requirements.txt` for runtime dependencies.
- For development (linting): `pylint` (optional).

### Quick start

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Run a script, for example, background vs IV radius:

```bash
python background/bkg_vs_iv.py
```

Or plot/fit IV hit efficiencies vs IV radius:

```bash
python hit_efficiencies/iv_hit_efficiencies_tests.py
python hit_efficiencies/iv_hit_efficiencies_tests_vs_hfe.py
```

Water background analyses:

```bash
python water_bkgd_analysis.py
python water_bkgd_contrib.py
python water_bkgd_vs_iv.py
```

Many scripts produce figures and print best‑fit parameters to stdout.


