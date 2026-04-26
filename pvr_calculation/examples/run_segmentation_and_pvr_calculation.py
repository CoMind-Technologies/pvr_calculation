"""
Segmentation and PVR calculation script for pulsatile BFI time series.

Loads BFI and ABP once, segments and aligns once, then saves edge pickles,
computes batch PVR and saves ``*_pvr.npy``, and optionally writes a figure of
overlaid z-scored pulses under ``pvr_calculation/data/figures/``.

Usage:
    python -m pvr_calculation.examples.run_segmentation_and_pvr_calculation \
        --bfi <bfi.npy> --abp <abp.npy> --fs <sampling_rate> [--outdir <dir>]
"""
import argparse
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from pvr_calculation.utils.absolute_paths import FIGURE_PATH, ensure_dir
from pvr_calculation.utils.pvr_utils import batch_pvr
from pvr_calculation.utils.segmenter_utils import AlignSignals, Segmenter, split_continuous


def _extract_bfi_for_alignment(bfi, tof_ix):
    """Return a 1-D BFI trace suitable for AlignSignals."""
    if bfi.ndim == 1:
        return bfi
    if bfi.ndim == 2:
        return bfi[:, tof_ix]
    return bfi[:, tof_ix, 0]


def _segment_and_align(bfi, abp, fs, tof_ix):
    """Run ``Segmenter`` on ABP and ``AlignSignals`` on the alignment BFI trace."""
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
    return seg_ref, align_bfi


def _pulses_and_pvr(bfi, seg_ref, align_bfi, upsamp_len, n_pulses):
    """Z-scored pulses and batch PVR from an already-aligned segmentation."""
    aligned_edges = seg_ref.edges - align_bfi.aligned[1]

    if bfi.ndim == 1:
        pulses = split_continuous(bfi, aligned_edges, upsamp_len, axis=0)
        for i in range(pulses.shape[0]):
            pulses[i, :] = stats.zscore(pulses[i, :])
        pvr_vals = batch_pvr(
            pulses, batch_size=n_pulses, within_axis=1, across_axis=0
        )
    else:
        bfi_t = bfi.transpose(1, 0, 2)
        pulses = split_continuous(bfi_t, aligned_edges, upsamp_len, axis=1)
        for i in range(pulses.shape[1]):
            pulses[:, i, :, :] = stats.zscore(pulses[:, i, :, :], axis=1)
        pvr_vals = batch_pvr(
            pulses, batch_size=n_pulses, within_axis=2, across_axis=1
        )
    return pvr_vals, pulses


def _pulses_to_overlay_matrix(pulses):
    """(n_pulses, L) for plotting; average over ToF / param axes when needed."""
    if pulses.ndim == 2:
        return pulses
    if pulses.ndim == 4:
        return np.mean(pulses, axis=(0, 2))
    raise ValueError(f"Unexpected pulses ndim={pulses.ndim}; expected 2 or 4.")


def _save_segmented_pulses_figure(
    pulses: np.ndarray,
    pvr_vals: np.ndarray,
    figure_path: Path,
) -> Path:
    """Gray pulse overlays (alpha 0.3), red mean pulse, legend with mean PVR."""
    pulses_plot = _pulses_to_overlay_matrix(pulses)
    x = np.arange(pulses_plot.shape[1], dtype=float)
    fig, ax = plt.subplots(figsize=(8, 4))
    for i in range(pulses_plot.shape[0]):
        ax.plot(x, pulses_plot[i], color="0.5", alpha=0.3, linewidth=0.9)
    mean_pulse = np.mean(pulses_plot, axis=0)
    mean_pvr = float(np.mean(pvr_vals))
    ax.plot(
        x,
        mean_pulse,
        color="red",
        linewidth=2.0,
        label=f"Mean pulse\n(mean PVR = {mean_pvr:.4f})",
    )
    ax.set_xlabel("Resampled pulse sample")
    ax.set_ylabel("z-scored BFI")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)
    return figure_path


def process_segmentation_and_pvr(
    bfi_file,
    abp_file,
    fs,
    out_folder,
    *,
    prefix=None,
    tof_ix=6,
    upsamp_len=151,
    n_pulses=10,
):
    """Load data once, segment and align once, save edges and PVR, return results.

    PVR is one value per non-overlapping block of ``n_pulses`` pulses (trailing
    incomplete block dropped), same ``batch_pvr`` logic for 1-D and 3-D BFI.

    Returns:
        ``seg_ref``, ``align_bfi``, ``pvr_vals``, ``pulses``
    """
    if prefix is None:
        prefix = Path(bfi_file).stem
    out_folder = Path(out_folder)
    out_folder.mkdir(parents=True, exist_ok=True)

    bfi = np.load(bfi_file)
    abp = np.load(abp_file)

    t0 = time.time()
    seg_ref, align_bfi = _segment_and_align(bfi, abp, fs, tof_ix)
    seg_ref.save(save_folder=out_folder, edges_name=f"{prefix}_abp_edges.pkl")
    align_bfi.save(save_folder=out_folder, edges_name=f"{prefix}_bfi_edges.pkl")
    print(
        f"Segmented {bfi_file} -> {out_folder}  ({time.time() - t0:.1f}s)"
    )

    t1 = time.time()
    pvr_vals, pulses = _pulses_and_pvr(
        bfi, seg_ref, align_bfi, upsamp_len, n_pulses
    )
    np.save(out_folder / f"{prefix}_pvr.npy", pvr_vals)
    print(f"PVR {bfi_file} -> {out_folder}  ({time.time() - t1:.1f}s)")

    return seg_ref, align_bfi, pvr_vals, pulses


def _default_outdir() -> str:
    """Results under the repository root when cwd is that root (see README)."""
    return str(Path.cwd() / "pvr_calculation" / "data" / "segmentation_and_pvr_results")


def get_cli():
    parser = argparse.ArgumentParser(description="Segmentation and PVR calculation")
    parser.add_argument("--bfi", type=str, required=True, help="Path to bfi.npy")
    parser.add_argument("--abp", type=str, required=True, help="Path to abp.npy")
    parser.add_argument("--fs", type=float, required=True, help="Sampling rate in Hz")
    parser.add_argument(
        "--outdir",
        type=str,
        default=_default_outdir(),
        help=(
            "Output directory (default: ./pvr_calculation/data/segmentation_and_pvr_results "
            "relative to the current working directory)"
        ),
    )
    parser.add_argument("--tof_ix", type=int, default=6,
                        help="TOF index for alignment (default 6)")
    parser.add_argument(
        "--n-pulses",
        type=int,
        default=10,
        metavar="N",
        help="Number of consecutive pulses per PVR estimate (default 10)",
    )
    parser.add_argument(
        "--figure-name",
        type=str,
        default=None,
        help=(
            "PNG filename under pvr_calculation/data/figures/ "
            "(default: <bfi_stem>_segmented_pulses_mean_pvr.png)"
        ),
    )
    parser.add_argument(
        "--no-figure",
        action="store_true",
        help="Skip writing the segmented-pulses figure",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = get_cli()
    print(f"Running with: bfi={cli_args.bfi}, abp={cli_args.abp}, fs={cli_args.fs}")

    seg_ref, align_bfi, pvr_vals, pulses = process_segmentation_and_pvr(
        cli_args.bfi,
        cli_args.abp,
        cli_args.fs,
        cli_args.outdir,
        tof_ix=cli_args.tof_ix,
        n_pulses=cli_args.n_pulses,
    )
    print(f"  edges: {len(seg_ref.edges)}, offset: {align_bfi.aligned[1]}")
    print(f"  PVR: {pvr_vals}")

    if not cli_args.no_figure:
        stem = Path(cli_args.bfi).stem
        fname = cli_args.figure_name or f"{stem}_segmented_pulses_mean_pvr.png"
        fig_path = _save_segmented_pulses_figure(
            pulses,
            pvr_vals,
            ensure_dir(FIGURE_PATH) / fname,
        )
        print(f"  Figure: {fig_path}")
