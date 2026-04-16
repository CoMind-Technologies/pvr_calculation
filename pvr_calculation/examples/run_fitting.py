"""
Batch fitting script for simulated or experimental g1 time series.

This script fits all `.pkl` recordings in a folder (or a single file) using a specified fitting method:
- Time domain (td)
- Continuous wave (cw)
- Speckle contrast optical spectroscopy (scos)


Usage:
    python mc_analysis.examples/run_fitting.py --folder <input_folder_or_file> --sds <SDS> \
    --fit_type <fit_type> --outdir <output_folder> --n_proc <n_proc>

Arguments:
    --folder: Path to input `.pkl` file or folder containing `.pkl` files (required)
    --ending: File ending to match (default: `.pkl`)
    --fit_type: Fitting method: 'td', 'cw', or 'scos' (default: 'td')
    --outdir: Output directory for fit results (default: data/fits/)
    --sds: Source-detector separation in mm (default: 10, used for cw fitting)
    --n_proc: Number of parallel processes (default: 1). If more than one file is to be fitted,
    multiprocessing may be used.

Results are saved as `.hdf5` files in the specified output directory.

"""
import glob
import time
from multiprocessing import Pool
from pathlib import Path
from typing import Literal

import numpy as np

from mc_analysis.fitting.bfi_fitting import run_bfi_fit
from mc_analysis.fitting.loaders import _load_g1_cw, load_g1_pkl, save_fit, save_fit_scos
from mc_analysis.fitting.scos_utils import compute_speckle_contrast
from mc_analysis.utils.absolute_paths import DATA_PATH

MIN_POINTS = 6
MIN_G1 = 0.5


def fit_one_recording(dir_name, out_folder):
    print(f"Fitting {dir_name} with single exp\n output to: {out_folder}")
    return _fit_one_recording_fit(dir_name, out_folder, fit_name="exp")


def fit_one_recording_cw(dir_name, out_folder, sds):
    print(f"Fitting {dir_name} with semi-infinite homogeneous dcs model\n output to: {out_folder}")
    return _fit_one_recording_fit_cw(dir_name, out_folder, fit_name="dcs", sds=int(sds))


def fit_one_recording_scos(dir_name, out_folder, exposure_times=(250e-6, 1000e-6)):
    results = []
    for exposure_time in exposure_times:
        print(f"Fitting {dir_name} with scos\n output to: {out_folder}")
        _result = _fit_one_recording_fit_scos(dir_name, out_folder, "scos", exposure_time=exposure_time)
        results.append(_result)
    return results


def _fit_one_recording_fit_cw(G1_file, out_folder, fit_name, prefix=None, sds=10):
    """Fit one recording with continuous wave DCS solution."""
    starttime = time.time()
    if prefix is None:
        prefix = Path(G1_file).stem
    save_path = Path(out_folder, fit_name)
    min_g1 = MIN_G1
    min_lag_points = MIN_POINTS
    out_hdf5_path = Path(save_path, f"{prefix}_{fit_name}_min{MIN_POINTS}_min_g1{min_g1}.hdf5")
    Path(save_path).mkdir(exist_ok=True, parents=True)

    sds_cm = sds / 10  # convert to cm
    g1, lags, meta, times, tofs = _load_g1_cw(G1_file)
    assert g1.shape[1] == 1
    g1, lags = remove_nans(g1, lags)
    bfi = run_bfi_fit(lags, g1, fit_name, min_g1=min_g1, min_lag_points=min_lag_points, sds=sds_cm)
    endtime = time.time()
    runtime = endtime - starttime
    print(f"writing to {out_hdf5_path},runtime {runtime}")
    save_fit(
        G1_file,
        bfi,
        fit_name,
        lags,
        meta,
        min_g1,
        min_lag_points,
        out_hdf5_path,
        times,
        tofs,
    )
    return G1_file, bfi


def _fit_one_recording_fit_scos(G1_file, out_folder, fit_name, exposure_time=100e-6, prefix=None):
    """Convert g1 to squared SCOS speckle contrast and convert to relative 'blood flow index (rBFI)."""
    starttime = time.time()
    if prefix is None:
        prefix = Path(G1_file).stem
    save_path = Path(out_folder, fit_name)
    min_lag_points = 6  # not used - required for save function consistency.
    min_g1 = 0.5  # not used for scos - required for save function consistency.
    fit_str = "scos"
    out_hdf5_path = Path(
        save_path,
        f"{prefix}_{fit_str}_{exposure_time * 1e6:.0f}us_min{MIN_POINTS}_min_g1{MIN_G1}_06_90_adpt.hdf5",
    )

    Path(save_path).mkdir(exist_ok=True, parents=True)

    g1_cw, lags, meta, times, tofs = _load_g1_cw(G1_file)
    assert g1_cw.shape[1] == 1
    g1_cw = g1_cw[:, 0, :]
    g1_cw, lags = remove_nans(g1_cw, lags)

    meta["exposure_time"] = exposure_time

    k2 = compute_speckle_contrast(np.moveaxis(g1_cw, 0, 1), lags, exposure_time)
    bfi = 1 - (k2 - k2[0]) / k2[0]

    endtime = time.time()
    runtime = endtime - starttime
    print(f"writing to {out_hdf5_path},runtime {runtime}")
    save_fit_scos(
        G1_file,
        bfi,
        exposure_time,
        fit_str,
        lags,
        meta,
        min_g1,
        min_lag_points,
        out_hdf5_path,
        times,
        tofs,
    )
    return G1_file, bfi


def remove_nans(g1_cw, lags):
    lag_mask = np.isnan(lags)
    g1_cw = g1_cw[..., ~lag_mask]
    lags = lags[~lag_mask]
    return g1_cw, lags


def _fit_one_recording_fit(G1_file, out_folder, fit_name, prefix=None, sds=None):
    if prefix is None:
        prefix = Path(G1_file).stem
    save_path = Path(out_folder, fit_name)
    Path(save_path).mkdir(exist_ok=True, parents=True)
    starttime = time.time()
    try:
        g1, times, tofs, lags, meta = load_g1_pkl(G1_file)
    except Exception as e:
        print(f"failed to load {G1_file} with {e}")
        return

    if sds is None:
        sds = meta.get("sds", None)
    sds = sds / 10  # convert to cm

    g1 = g1 / g1[..., 0, None]
    g1, lags = remove_nans(g1, lags)

    if g1.shape[-1] < len(lags):
        lags = lags[: g1.shape[-1]]
    bfi = run_bfi_fit(lags, g1, fit_name, min_g1=MIN_G1, min_lag_points=MIN_POINTS, sds=sds)
    out_hdf5_path = Path(save_path, f"{prefix}_{fit_name}_min{MIN_POINTS}_min_g1{MIN_G1}.hdf5")
    endtime = time.time()
    runtime = endtime - starttime
    print(f"writing to {out_hdf5_path},runtime {runtime}")
    save_fit(
        G1_file,
        bfi,
        fit_name,
        lags,
        meta,
        MIN_G1,
        MIN_POINTS,
        out_hdf5_path,
        times,
        tofs,
    )
    return G1_file, bfi


def fit_all_recordings(
    folder,
    n_proc=1,
    outdir=str(DATA_PATH / "fits"),
    ending=".pkl",
    fit_type="td",
):
    """Fit all recordings in a folder with specified fitting method."""
    files = glob.glob(folder + f"/*{ending}" if not folder.endswith(ending) else folder)
    if outdir is None:
        outdir = "./"
    print(f"Fitting {len(files)} files in {folder} with {n_proc} processes\n output to: {outdir}")
    Path(outdir).mkdir(exist_ok=True, parents=True)

    args = [(file, outdir) for file in files]
    if fit_type == "td":
        with Pool(processes=n_proc) as p:
            results = p.starmap(fit_one_recording, args)
    elif fit_type == "cw":
        with Pool(processes=n_proc) as p:
            results = p.starmap(fit_one_recording_cw, args)
    elif fit_type == "scos":
        with Pool(processes=n_proc) as p:
            results = p.starmap(fit_one_recording_scos, args)
    else:
        raise ValueError(f"Unknown fit_type {fit_type}. Use 'td', 'cw' or 'scos'.")
    return results


def get_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Fit all recordings")
    parser.add_argument(
        "--folder",
        type=str,
        default=str(
            DATA_PATH
            / "simulations"
            / "pulsatile_g1_td"
            / "20250904212745g1_2024_06_11_4layer_cw_dcs_1000_log_10_200000.pkl"
        ),
        help="Folder or regex containing .pkl files to fit.",
    )
    parser.add_argument(
        "--ending",
        type=str,
        default=".pkl",
        help="Folder or regex containing .pkl files to fit.",
    )
    parser.add_argument(
        "--fit_type",
        type=Literal["td", "cw", "scos"],
        default="td",
        help="Type of fit - must be one of 'td', 'cw' or 'scos' for time-domain, continuous wave or speckle contrast"
        " fitting.",
    )
    parser.add_argument("--outdir", type=str, default=str(DATA_PATH / "fits"), help="Output directory to save fits.")
    parser.add_argument(
        "--sds", type=int, default=10, help="Source-detector separation in mm - only used for cw fitting."
    )
    parser.add_argument(
        "--n_proc", type=int, default=1, help="Number of processors - only used if there is more than one file to fit."
    )

    return parser.parse_args()


if __name__ == "__main__":
    cli_args = get_cli()
    print(f"Running with args: {cli_args}")
    fit_all_recordings(
        folder=cli_args.folder,
        ending=cli_args.ending,
        n_proc=cli_args.n_proc,
        outdir=cli_args.outdir,
        fit_type=cli_args.fit_type,
    )
