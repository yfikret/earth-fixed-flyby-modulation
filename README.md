## Data and Reproducibility


### Surface Geography Data

Ocean/land geometry is based on the Natural Earth dataset:

* **Dataset**: Natural Earth — 110m Land
* **File**: `ne_110m_ocean.*`
* **Source**: [https://www.naturalearthdata.com/downloads/110m-physical-vectors/110m-land/](https://www.naturalearthdata.com)
* **License**: Public domain

The extracted shapefile is included directly in the repository under:

```
data_surface/ne_110m_land/
```

This dataset is small (~100 KB) and is vendored to allow the code to run out-of-the-box.

### Spacecraft Trajectory and Ephemeris Data

Spacecraft trajectories and planetary ephemerides are provided via SPICE kernels obtained from NASA NAIF.
Due to their size (~117 MB total), these kernels are **not included** in the repository.

Instead, exact kernel URLs and SHA256 checksums are recorded in:

```
data/manifest.yml
```

A helper script is provided to download and verify all required kernels:

```bash
python scripts/fetch_kernels.py
```

This script retrieves the kernels from NAIF, verifies their checksums, and places them in the expected directory structure under:

```
data/kernels/
```

### SPICE Configuration

SPICE kernels are loaded via a metakernel (`.tm`) file using relative paths.
The code has been tested with Python 3.12.2 and standard scientific Python packages.

### Trajectory Feature Generation

Spacecraft trajectory features are generated using SPICE kernels via parse_trajectory_data.py.
Inputs: NAIF SPICE kernels (see data/manifest.yml).
Outputs: compressed .npz files containing spherical position/velocity features in the Earth-fixed frame.
Parameters: 

DT = 10 s, 