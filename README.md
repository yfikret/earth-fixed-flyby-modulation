# Earth Flyby Land–Sea Coupling Analysis

This project computes Earth-fixed land–sea coupling metrics from precomputed spacecraft trajectories. It is designed to explore potential correlations between reported spacecraft Earth flyby anomalies and the distribution of land versus ocean visible from the spacecraft in an Earth-fixed frame.

## Code Overview

- `fetch_kernels.py`  
  Downloads and verifies the required planetary ephemeris and spacecraft SPICE kernels from NASA NAIF.

- `parse_trajectory_data.py`  
  Extracts and caches spacecraft position and velocity features from SPICE kernels into compressed `.npz` files.

- `build_landmask.py`  
  Converts Natural Earth vector geography data into a binary land–ocean mask on a regular latitude–longitude grid.

- `get_land_fraction.py`  
  Contains the core mathematical routines for computing visible land fraction. The visible Earth disk is sampled deterministically using a **Fibonacci (golden-angle) lattice** projected onto the Earth’s surface from the spacecraft’s viewpoint.

- `run_experiment.py`  
  **Main control interface.** Orchestrates analyses, visualizations, and statistical tests (e.g., saturation, drift, window dependence) based on a user-specified YAML configuration file.

## Installation & Requirements

### Environment Preparation

This project has been written and tested for **Python 3.12.2**.

Install required packages:
```bash
pip install -r requirements.txt
````

Core dependencies include:
`pyyaml`, `numpy`, `matplotlib`, `spiceypy`, `shapely`, `fiona`

### Preparing Trajectory Data

Download and verify all required SPICE kernels:

```bash
python scripts/fetch_kernels.py
```

Parse and cache spacecraft trajectory features:

```bash
python scripts/parse_trajectory_data.py
```

### Geographic Data

Prepare the land–ocean mask cache:

```bash
python scripts/build_landmask.py
```

### Running Analyses

All analyses are controlled via explicit YAML configuration files.

Run an experiment with:

```bash
python scripts/run_experiment.py --config <name>
```

Each configuration file specifies parameters such as:

* analysis mode (e.g., window test, saturation test)
* distance weighting
* flyby selection
* distance range

Experimental configurations are provided under `config/`.

Examples:
```bash
python scripts/run_experiment.py --config generate_cached_map

python scripts/run_experiment.py --config window_test
```

> **Note:** Certain visualization modes require generating cached maps first, as specified in the corresponding configuration files.

## Data and Reproducibility

### Surface Geography Data

Geographic data is provided by the Natural Earth dataset.

* **Source:** [https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-land/](https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-land/)
* **License:** Public domain

This dataset is small (~100 KB) and is vendored in full with the repository.

### Spacecraft Trajectory and Ephemeris Data

Spacecraft trajectories and planetary ephemerides are provided via SPICE kernels obtained from NASA NAIF. Due to their size (~117 MB total), these kernels are **not included** in the repository.

The exact kernel URLs and SHA256 checksums are recorded in:

```
data/manifest.yml
```

The script `fetch_kernels.py` retrieves the kernels from NAIF, verifies their checksums, and places them in the expected directory structure under:

```
data/kernels/
```

All reported results are deterministic given identical kernels, surface data, and configuration files.