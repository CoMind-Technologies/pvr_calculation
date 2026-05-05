# pvr_calculation

A Python package for segmenting pulsatile BFI measurements and calculating Pulse Variance Ratio (PVR).

This is designed to accompany the paper: *CoMind R1: A time-resolved interferometric optical neuromonitoring system for pulsatile cerebral blood flow measurement at late times-of-flight*, https://doi.org/10.1117/1.NPh.13.2.025002.


## Installation

Clone or download the repository, open a terminal in the **repository root** (the folder that contains this `README.md` and `pyproject.toml` — often named `pvr_calculation`).

```bash
conda create -n pvr_calculation python=3.10 poetry=1.6.1 -c conda-forge
conda activate pvr_calculation
poetry install
```

Use `poetry run …` or `poetry shell` so commands use this environment. Dependencies are declared in `pyproject.toml` (core numerics: `numpy`, `scipy`, `matplotlib`, plus other packages listed there). Segmentation and FIR helpers used by `Segmenter` / `AlignSignals` live in `pvr_calculation/utils/align_signals.py` and are **self-contained** (no `comind_utils` import).


## Directory layout

Repository root:

```
pvr_calculation/                 # repository root (example name)
├── README.md
├── pyproject.toml
└── pvr_calculation/             # Python package
    ├── data/
    │   ├── time_series_data/       # example bfi.npy, abp.npy
    │   ├── segmentation_and_pvr_results/   # default script output (pickles, pvr.npy)
    │   └── figures/                # segmented-pulse overlay PNG from the example script
    ├── examples/
    │   └── run_segmentation_and_pvr_calculation.py
    └── utils/
        ├── absolute_paths.py
        ├── segmenter_utils.py
        └── pvr_utils.py
```

Paths like `pvr_calculation/data/...` below are **relative to the repository root** (your shell’s current working directory).


## Data

- **`pvr_calculation/data/time_series_data/`** — example `bfi.npy` and `abp.npy` (synchronised); ABP is used as the reference for segmentation.
- **`pvr_calculation/data/segmentation_and_pvr_results/`** — default location for script outputs when you omit `--outdir` (see below).
- **`pvr_calculation/data/figures/`** — default location for the segmented-pulses figure (gray overlays, red mean, mean PVR in the legend).


## Examples

### Segment BFI and compute PVR

The script loads whatever paths you pass in; **`--bfi`**, **`--abp`**, and **`--fs`** are **required**. Each array file is read **once**, and segmentation plus alignment run **once**; PVR and the optional figure reuse that result. Outputs: edge pickles and ``*_pvr.npy`` under ``--outdir``. It also saves a PNG under **`pvr_calculation/data/figures/`** (all z-scored pulses in gray at alpha 0.3, mean pulse in red, legend with mean PVR) unless you pass **`--no-figure`**. For **1-D BFI**, PVR is computed in **non-overlapping blocks** of **`--n-pulses`** consecutive pulses (same `batch_pvr` logic as multi-TOF data); any trailing pulses that do not complete a full block are omitted.

From the repository root, using the bundled example files and a sampling rate that matches those traces:

```bash
poetry run python -m pvr_calculation.examples.run_segmentation_and_pvr_calculation \
    --bfi pvr_calculation/data/time_series_data/bfi.npy \
    --abp pvr_calculation/data/time_series_data/abp.npy \
    --fs 47.68
```

If you omit **`--outdir`**, outputs go to **`./pvr_calculation/data/segmentation_and_pvr_results/`** (under the same directory you run from — keep your cwd at the repository root so that path matches the layout above).

| Argument | Required | Description |
|----------|----------|-------------|
| `--bfi` | yes | Path to the BFI `.npy` file |
| `--abp` | yes | Path to the ABP `.npy` file |
| `--fs` | yes | Sampling rate of both signals (Hz) |
| `--outdir` | no | Output directory (default: `./pvr_calculation/data/segmentation_and_pvr_results` relative to cwd) |
| `--tof_ix` | no | TOF index for multi-TOF BFI alignment (default: `6`) |
| `--n-pulses` | no | Pulses per PVR estimate, 1-D and 2-D BFI (default: `10`) |
| `--figure-name` | no | PNG filename under `data/figures/` (default: `<bfi_stem>_segmented_pulses_mean_pvr.png`) |
| `--no-figure` | no | Skip writing the figure |

## Citation

If you use this package in your research, please cite the associated publication.

V. Parfentyeva et al., *CoMind R1: A time-resolved interferometric optical neuromonitoring system for pulsatile cerebral blood flow measurement at late times-of-flight* (2026).


## License

See LICENSE for details.
