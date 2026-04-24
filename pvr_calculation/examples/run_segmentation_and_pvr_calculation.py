"""
Segmentation and PVR calculation script for pulsatile BFI time series.

This script segments a pulsatile BFI time series based on the reference ABP
signal and calculates the PVR.

Usage:
    python -m pvr_calculation.examples.run_segmentation_and_pvr_calculation \
        --bfi <bfi.npy> --abp <abp.npy> --fs <sampling_rate> [--outdir <dir>]
"""
import argparse
import time
from pathlib import Path

import numpy as np
from scipy import stats

from pvr_calculation.utils.absolute_paths import DATA_PATH, RESULTS_PATH
from pvr_calculation.utils.pvr_utils import batch_pvr, nan_pvr
from pvr_calculation.utils.segmenter_utils import AlignSignals, Segmenter, split_continuous


def _extract_bfi_for_alignment(bfi, tof_ix):
    """Return a 1-D BFI trace suitable for AlignSignals."""
    if bfi.ndim == 1:
        return bfi
    if bfi.ndim == 2:
        return bfi[:, tof_ix]
    return bfi[:, tof_ix, 0]


def _segment_one_recording(bfi_file, abp_file, fs, out_folder, prefix=None, tof_ix=6):
    """Segment one recording: find cardiac edges from ABP, then align BFI.

    Args:
        bfi_file: Path to .npy file containing BFI data (1-D, 2-D, or 3-D).
        abp_file: Path to .npy file containing ABP reference signal (1-D).
        fs: Sampling rate of the data in Hz.
        out_folder: Directory to save segmentation results.
        prefix: Filename prefix for saved outputs; defaults to bfi_file stem.
        tof_ix: Time-of-flight index for BFI alignment (default 6).

    Returns:
        Tuple of (Segmenter for ABP, AlignSignals for BFI).
    """
    starttime = time.time()
    if prefix is None:
        prefix = Path(bfi_file).stem
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    bfi = np.load(bfi_file)
    abp = np.load(abp_file)

    seg_ref = Segmenter(
        data=abp,
        fs=fs,
        method="foGD",
        method_preproc="ssf",
        peak_sign=1,
    )

    bfi_1d = _extract_bfi_for_alignment(bfi, tof_ix)
    align_bfi = AlignSignals(
        data=bfi_1d,
        fs=fs,
        ref_peaks=seg_ref.edges,
        method="troughs",
    )
    align_bfi.align(method="troughs")

    seg_ref.save(save_folder=out_folder, edges_name=f"{prefix}_abp_edges.pkl")
    align_bfi.save(save_folder=out_folder, edges_name=f"{prefix}_bfi_edges.pkl")

    runtime = time.time() - starttime
    print(f"Segmented {bfi_file} -> {out_folder}  ({runtime:.1f}s)")
    return seg_ref, align_bfi


def _calculate_pvr_one_recording(
    bfi_file, abp_file, fs, out_folder, prefix=None, tof_ix=6,
    upsamp_len=151, n_pulses=10,
):
    """Calculate PVR for one recording.

    Segments ABP, aligns BFI, splits into pulses, z-scores, then computes
    batch PVR.  Handles 1-D BFI (single channel) or 3-D
    (n_times, n_tofs, n_params).

    Args:
        bfi_file: Path to .npy file with BFI data.
        abp_file: Path to .npy file with ABP reference signal (1-D).
        fs: Sampling rate in Hz.
        out_folder: Directory to save PVR results.
        prefix: Filename prefix; defaults to bfi_file stem.
        tof_ix: TOF gate used for segmentation alignment (default 6).
        upsamp_len: Resample length per pulse (default 151).
        n_pulses: Number of pulses per PVR batch (default 10).

    Returns:
        np.ndarray: PVR values.
    """
    starttime = time.time()
    if prefix is None:
        prefix = Path(bfi_file).stem
    out_folder = Path(out_folder)
    out_folder.mkdir(exist_ok=True, parents=True)

    bfi = np.load(bfi_file)
    abp = np.load(abp_file)

    seg_ref = Segmenter(
        data=abp, fs=fs, method="foGD", method_preproc="ssf", peak_sign=1,
    )

    bfi_1d = _extract_bfi_for_alignment(bfi, tof_ix)
    align_bfi = AlignSignals(
        data=bfi_1d, fs=fs, ref_peaks=seg_ref.edges, method="troughs",
    )
    align_bfi.align(method="troughs")
    aligned_edges = seg_ref.edges - align_bfi.aligned[1]

    if bfi.ndim == 1:
        pulses = split_continuous(bfi, aligned_edges, upsamp_len, axis=0)
        for i in range(pulses.shape[0]):
            pulses[i, :] = stats.zscore(pulses[i, :])
        pvr_vals = nan_pvr(pulses, within_axis=1, across_axis=0)
    else:
        bfi_t = bfi.transpose(1, 0, 2)
        pulses = split_continuous(bfi_t, aligned_edges, upsamp_len, axis=1)
        for i in range(pulses.shape[1]):
            pulses[:, i, :, :] = stats.zscore(pulses[:, i, :, :], axis=1)
        pvr_vals = batch_pvr(pulses, batch_size=n_pulses, within_axis=2, across_axis=1)

    np.save(out_folder / f"{prefix}_pvr.npy", pvr_vals)

    runtime = time.time() - starttime
    print(f"PVR {bfi_file} -> {out_folder}  ({runtime:.1f}s)")
    return pvr_vals


def get_cli():
    parser = argparse.ArgumentParser(description="Segmentation and PVR calculation")
    parser.add_argument("--bfi", type=str, required=True, help="Path to bfi.npy")
    parser.add_argument("--abp", type=str, required=True, help="Path to abp.npy")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz")
    parser.add_argument("--outdir", type=str, default=str(RESULTS_PATH),
                        help="Output directory")
    parser.add_argument("--tof_ix", type=int, default=6,
                        help="TOF index for alignment (default 6)")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = get_cli()
    print(f"Running with: bfi={cli_args.bfi}, abp={cli_args.abp}, fs={cli_args.fs}")

    seg_ref, align_bfi = _segment_one_recording(
        bfi_file=cli_args.bfi,
        abp_file=cli_args.abp,
        fs=cli_args.fs,
        out_folder=cli_args.outdir,
        tof_ix=cli_args.tof_ix,
    )
    print(f"  edges: {len(seg_ref.edges)}, offset: {align_bfi.aligned[1]}")

    pvr_vals = _calculate_pvr_one_recording(
        bfi_file=cli_args.bfi,
        abp_file=cli_args.abp,
        fs=cli_args.fs,
        out_folder=cli_args.outdir,
        tof_ix=cli_args.tof_ix,
    )
    print(f"  PVR: {pvr_vals}")
