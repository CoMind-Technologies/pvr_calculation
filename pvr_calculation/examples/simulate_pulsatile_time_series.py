"""
Simulate pulsatile g1 time series from ground truth waveforms.

This script generates g1 time series using MCX `.mch` files and a ground truth waveform.
It supports batch processing with multiple source-detector separations and acquisition frequencies,
parallel execution, and saving results as `.pkl` files.

Usage:
    python mc_analysis.examples/simulate_pulsatile_time_series.py \
           --folder /path/to/mch_files --n_proc 4 --sds 10 20 --fs_sweep 200000 400000

Arguments:
    --folder: Path to folder containing `.mch` files (required unless using --basename)
    --n_proc: Number of parallel processes (default: 1)
    --save_folder: Output folder for results (default: DATA_PATH/simulations/pulsatile_g1_td/)
    --is_td: If True, run time domain simulation; otherwise, continuous wave (default: False)
    --sds: List of source-detector separations in mm (default: 10)
    --fs_sweep: List of acquisition frequencies in Hz (default: 200000)

Results are saved as `.pkl` files in the specified folder.
"""
import argparse
import glob
import itertools
import os
import pathlib
import time
from dataclasses import dataclass
from datetime import datetime
from multiprocessing import Pool
from typing import Literal, Union

import numpy as np
from tqdm import tqdm

from mc_analysis.fitting.loaders import _load_groundtruth_csv, save_g1_pkl
from mc_analysis.simulator.g1_simulator import G1Simulator, get_G1_sim, resample_g1
from mc_analysis.utils.absolute_paths import DATA_PATH, WAVEFORM_PATH
from mc_analysis.utils.pulsatile_time_series_utils import Waveform, get_troughs

MAX_LAG = 1.0e-2  # MAX G1 lag in seconds


@dataclass
class PulsatileSimConfig:
    basename: str  # Path to the .mch file
    sds: int  # Source-detector separation in mm
    path_template_gt: str  # Path to ground truth waveform file
    path_save: str  # Path to save generated g1 files
    fs_sweep: int = 200_000  # acuiqsition frequency in Hz
    lag_type: Literal["log", "linear", "resampled"] = "log"
    nt: int = 1000  # number of time points to simulate
    is_td: bool = False  # True for time domain, False for CW
    irf_std_tof: Union[str, float, None] = None
    suffix: str = ""  # In case you want to add an identifying suffix to the output filename.pkl

    def get_suffix(self):
        return (
            f"g1_{pathlib.Path(self.basename).stem}_{self.nt}_{self.lag_type}"
            f"_{self.sds}_{self.fs_sweep}{self.suffix}.pkl"
        )

    def get_out_file_path(self, date_time_str):
        filepath = os.path.join(
            self.path_save,
            f"{date_time_str}{self.get_suffix()}",
        )
        return filepath

    def get_output_filepath_glob(self):
        filepath_glob = os.path.join(
            self.path_save,
            f"*{self.get_suffix()}",
        )
        return filepath_glob


def data_time_str():
    return datetime.now().strftime("%Y%m%d%H%M%S")


def get_lag_axis(fs_sweep, lag_type):
    if lag_type == "linear":
        resample = False
        l_all = np.arange(0, 0.01, 1 / fs_sweep)
        lags = l_all[l_all < MAX_LAG]
    elif lag_type == "resampled":
        n_lags_linear = int(MAX_LAG * fs_sweep)
        lags_full = np.arange(0, n_lags_linear) / fs_sweep
        # always maintain a block size with the same absolute lag time.
        n_lag_bins = int((fs_sweep / 200_000) * 20)
        lags = resample_g1(lags_full, fs_sweep, n_lag_bins)
        resample = False
    elif lag_type == "log":
        resample = True
        lags = None
    else:
        raise ValueError(f"lag_type {lag_type} not recognised")
    return lags, resample


def main(config: PulsatileSimConfig = None):
    """Generate g1 time series from a ground truth waveform and save to file."""
    is_td = config.is_td
    date_time_str = data_time_str()
    filepath = config.get_out_file_path(date_time_str)

    wf = load_groundtruth_waveform(config)
    # Get segmentation indices
    index_pulses = get_troughs(wf)

    lag_type = config.lag_type
    mua_skull = 0.013
    mua_csf = 0.012
    db_csf = 5e-8
    db_skull = 2e-8
    refractive_indices = np.array([1.4, 1.4, 1.33, 1.4])

    # Select the lag axis
    lags_in, resample = get_lag_axis(config.fs_sweep, lag_type)

    g1sim = get_g1_simulator(config, is_td)

    start_time = time.perf_counter()
    g1s = []
    for _t, _db_b, _db_s, _mua_b, _mua_s in tqdm(
        zip(
            wf.times,
            wf.db_brain,
            wf.db_scalp,
            wf.mua_brain,
            wf.mua_scalp,
        )
    ):
        mua_list = np.array([_mua_s, mua_skull, mua_csf, _mua_b])
        aDb = np.array([_db_s, db_skull, db_csf, _db_b])
        if lag_type == "log":
            n_lags_linear = int(MAX_LAG * config.fs_sweep)
            # Allow fitter class to generate the lag axis from sweep freq and block size.
            G1, tofs, lags = get_G1_sim(
                g1sim,
                aDb,
                refractive_indices,
                mua_list=mua_list,
                irf_std_tof=config.irf_std_tof,
                resample=resample,
                n_lags_linear=n_lags_linear,
                fs_sweep=config.fs_sweep,
                tof_offset=0,
                lags=None,
                n_lag_bins=int((config.fs_sweep / 200_000) * 20) if resample else None,
                pad_tof_lhs=None,
            )
        else:
            lags = lags_in
            G1, tofs, _ = get_G1_sim(
                g1sim,
                aDb,
                refractive_indices,
                config.irf_std_tof,
                mua_list=mua_list,
                resample=resample,
                n_lags_linear=len(lags_in),
                fs_sweep=config.fs_sweep,
                tof_offset=0,
                lags=lags_in,
                pad_tof_lhs=None,
            )

        g1s.append(G1)

    end_time = time.perf_counter()
    print(f"Generating {len(wf.times[:config.nt])} g1s took {end_time - start_time:.1f} s")
    g1s = np.array(g1s)
    save_g1_pkl(filepath, config.basename, g1s, lags, tofs, wf, index_pulses, sds=config.sds)


def get_g1_simulator(config, is_td):
    g1fit = G1Simulator(config.basename, config.sds)
    if is_td:
        g1fit.set_tof_bins(np.arange(40) * 6e-11)
    else:
        g1fit.set_tof_bins(np.array([0.0, 10e-9]))
    g1fit.groupby_tofs()
    return g1fit


def load_groundtruth_waveform(config):
    if config.path_template_gt.endswith(".hdf5"):
        wf = Waveform.from_file(config.path_template_gt)
    else:
        db_brain, db_scalp, mua_brain, mua_scalp = _load_groundtruth_csv(
            config.path_template_gt, include_absorption=True
        )
        waveform_gt = Waveform(
            db_brain,
            db_scalp,
            mua_brain,
            mua_scalp,
            times=np.arange(len(db_brain)),
        )

        wf = waveform_gt
    # Cut to nt 'times'
    wf.trim(0, config.nt)
    return wf


def run_g1_generation(
    basename,
    sds,
    fs_sweep,
    save_path=None,
    is_td=False,
    path_template_gt=str(WAVEFORM_PATH / "ground_truth_wf.hdf5"),
    lag_type="resampled",
    overwrite=False,
):
    print(f"{basename=}, {save_path=}, {is_td=}, {lag_type=}")

    if save_path is None:
        if is_td:
            is_td_str = "td_"
        else:
            is_td_str = "cw_"
        save_path = os.path.join(pathlib.Path(basename).parent, f"pulsatile_g1_{is_td_str}/")

    pathlib.Path(save_path).mkdir(exist_ok=True, parents=True)
    config = PulsatileSimConfig(
        basename=basename,
        sds=sds,
        path_template_gt=path_template_gt,
        path_save=save_path,
        fs_sweep=fs_sweep,
        lag_type="log" if is_td else lag_type,
        nt=1000,
        is_td=is_td,
        irf_std_tof="irf_phantom" if is_td else None,
    )
    filepaths = glob.glob(config.get_output_filepath_glob())
    if len(filepaths) == 1 and not overwrite:
        print(f"{filepaths=} already exists skipping.")
        return
    main(config=config)


def simulate_pulsatile_timeseries():
    parser = argparse.ArgumentParser(description="Simulate pulsatile g1 time series from ground truth waveform.")
    parser.add_argument("--folder", type=str, default=None, help="Folder or regex containing .mch files to simulate.")
    parser.add_argument("--n_proc", type=int, default=1)
    parser.add_argument(
        "--ground_truth",
        type=str,
        default=str(WAVEFORM_PATH / "ground_truth_wf.hdf5"),
        help="Path to ground truth waveform file.",
    )
    parser.add_argument(
        "--save_folder", type=str, default=str(DATA_PATH / pathlib.Path("simulations/pulsatile_g1_td/"))
    )
    parser.add_argument(
        "--is_td",
        type=bool,
        default=False,
        help="If True run time domain simulation if False simulate continuous wave.",
    )
    parser.add_argument(
        "--sds",
        type=int,
        default=10,
        nargs="+",
        help="Source-detector separations to simulate in mm - must be in target simulated .mch file.",
    )
    parser.add_argument(
        "--fs_sweep", type=int, default=200_000, nargs="+", help="Acquisition frequencies to simulate in Hz"
    )
    cli_args = parser.parse_args()
    basenames = (
        glob.glob(cli_args.folder) if str(cli_args.folder).endswith(".mch") else glob.glob(cli_args.folder + "/*.mch")
    )

    args = list(
        itertools.product(
            basenames,
            cli_args.sds,
            cli_args.fs_sweep,
            [
                cli_args.save_folder,
            ],
            [
                cli_args.is_td,
            ],
            [
                cli_args.ground_truth,
            ],
        )
    )
    n_proc = min([cli_args.n_proc, len(args)])

    with Pool(processes=n_proc) as pool:
        pool.starmap(run_g1_generation, args)


if __name__ == "__main__":
    simulate_pulsatile_timeseries()
