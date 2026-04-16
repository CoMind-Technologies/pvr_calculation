# pvr_analysis


A Python package for segmenting pulsaile BFI measurements and calculating Pulse Varience Ration (PVR).
This is designed to accompany the paper: "CoMind R1: A time-resolved interferometric optical neuromonitoring system for pulsatile cerebral blood flow measurement at late times-of-flight".


## Installation

Create a new environment and install dependencies using Poetry:

```commandline
conda create -n pvr_analysis python=3.10 poetry=1.6.1 -c conda-forge
conda activate pvr_analysis
poetry install
```

# Directory Structure
 - pvr_analysis/
 - data/: Contains example pulsatile BFI time series, segmentation and PVR calculation results.
 - examples/: Example scripts for segmentation, PVR calculation, and plotting.
 - utils/: Utility functions for plotting, paths, and calculations.

## Data
The `data/` folder contains:
- `bfi/`: example bfi time series to be run segmentation and PVR calculation on.
- `segmentation_and_pvr_results/`: Empty. Segmentation and PVR calculation results are saved here by default.


# Examples
## Segment BFI time series
To segment pulsatile BFI time series. Before PVR calculation, BFI time series is supposed to be segmented to isolate single waveforms and align them on the common time axis.

`pvr_analysis/examples/segment_pulsatile_bfi_series.py` can then be run to segment a pulsatile BFI time series. By default, the time seriesis loaded from:
`pvr_analysis/data/bfi/bfi_time_series.hdf5`.

```commandline
    python pvr_analysis/examples/segment_pulsatile_bfi_series.py --folder data/bfi/bfi_time_series.hdf5 

Arguments:
    --folder: Path to folder containing `.hdf5` file 
    --save_folder: Output folder for results (default: data/segmentation_and_pvr_results/)
```
The code will output a `segmented_bfi.hdf5` file in the save_folder containing the results of the segmentation (array of segmented waveforms).

### Run PVR calculation
PVR calculation may be run on an array of segmented pulses using the following command:
```commandline
    python pvr_analysis/examples/run_pvr_calculation.py --folder <input_folder_or_file> --outdir <output_folder>

Arguments:
    --folder: Path to input `.hdf5` file or folder containing `.hdf5` files (required)
    --ending: File ending to match (default: `.hdf5`)
    --outdir: Output directory for fit results (default: data/segmentation_and_pvr_results/)
```
Results are saved as `pvr.hdf5` files in the specified output directory.


### Plotting the results - stacked segmented pulses with calculated PVR values in the legend.
To plot the results run the following script:
```commandline
python pvr_analysis/examples/plot_results.py --file </path/to/segmentation_and_pvr_results_results>
```
where
- `--file`: Path to the folder with `pvr.hdf5` and `segmented_bfi.hdf5` files.


## Citation
If you use this package in your research, please cite the associated publication.
V. Parfentyeva et al. "CoMind R1: A time-resolved interferometric optical neuromonitoring system for pulsatile cerebral blood flow measurement at late times-of-flight",  (2026)

## License
See LICENSE for details.
