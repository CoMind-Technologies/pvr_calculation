""""Plot BFI and correlation vs ToF for different ToF values."""
import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt

from mc_analysis.fitting.correlation_metrics import get_td_metrics
from mc_analysis.fitting.loaders import load_fit
from mc_analysis.utils.absolute_paths import DATA_PATH
from mc_analysis.utils.plotting import save_fig


def normalise(_bfi):
    bfi_norm = _bfi - _bfi.min()
    bfi_norm = bfi_norm / bfi_norm.max()
    return bfi_norm


def _result_to_metric(fit_result):
    db_c = fit_result.meta["db_c"]
    db_s = fit_result.meta["db_s"]
    bfi = fit_result.fits
    bfi = np.moveaxis(bfi, 0, 1)
    # Expect bfi to be (tof, time, lags)
    _metrics_td = get_td_metrics(bfi, db_c, db_s, return_metric_obj=True)
    _metrics_td.tof = fit_result.tofs
    _metrics_td.sds = 10
    return _metrics_td, db_c, db_s


def main(file_fit, save_name="bfi_correlation_vs_tof.png"):
    fit_result = load_fit(file_fit)
    metrics_td, db_c, db_s = _result_to_metric(fit_result=fit_result)
    tof_to_plot = (4, 9, 14, 19, 23)
    tofs = metrics_td.tof
    tofs = tofs[tofs < 1.6e-9]
    times = np.arange(len(metrics_td.bfi[0])) / 25
    dt = np.diff(tofs)[0] * 0.5
    tofs -= dt
    tof_labels = [f"{(tofs[itof]) * 1e12:.0f} ps" for itof in tof_to_plot]
    db_c_norm = normalise(db_c)
    db_s_norm = normalise(db_s)

    fig = plt.figure(figsize=(12 * 0.7, 3.5))
    gs0 = gridspec.GridSpec(2, 2, figure=fig, width_ratios=[0.5, 1])
    plt.subplots_adjust(hspace=0.1, bottom=0.15, right=0.95, left=0.07, top=0.95)
    ax_gt = fig.add_subplot(gs0[0, 0])
    ax_gt.set_xticks([])
    axfit = fig.add_subplot(gs0[1, 0])
    axcr = fig.add_subplot(gs0[:, 1])

    colors = plt.cm.get_cmap("gnuplot2")(np.linspace(0.1, 0.8, len(tof_to_plot)))
    p_list = []
    for itof, _c, _tof_label in zip(tof_to_plot, colors, tof_labels):
        bfi_norm = normalise(metrics_td.bfi[itof])
        (p1,) = axfit.plot(times, bfi_norm, label=_tof_label, c=_c)
        p_list.append(p1)
    labels = [p.get_label() for p in p_list]
    ax_gt.plot(times, db_c_norm, linestyle="-", c="k", label="rCBF g.t.", alpha=0.5)
    ax_gt.plot(times, db_s_norm, linestyle="--", c="k", label="rSBF g.t.", alpha=0.5)
    axfit.legend(loc="upper right", fontsize=10)

    ax_gt.legend(loc="upper right")
    for ax in [ax_gt, axfit]:
        ax.set_xlim(times[10], times[62])
        ax.set_ylabel("BFI (norm.)")
        ax.set_yticks([0.0, 0.5, 1.0])
        ax.set_ylim(0.0, 1.5)
    axfit.set_xlabel("Time (s)")

    ax = axcr
    ax.plot(
        metrics_td.tof * 1e12,
        metrics_td.corr_per_pulse_brain,
        label="rCBF",
        c="k",
    )
    ax.plot(
        metrics_td.tof * 1e12,
        metrics_td.corr_per_pulse_scalp,
        label="rSBF",
        c="k",
        linestyle="--",
    )
    for itof, _c in zip(tof_to_plot, colors):
        ax.axvline(tofs[itof] * 1e12, linestyle="--", c=_c, alpha=0.8)

    ax.set_xlim(200, 2100)
    ax.set_xlabel("ToF (ps)")
    ax.set_ylabel("Correlation")
    _ax = ax.twinx()
    ax.set_ylim(-0.1, 1.1)
    _ax.legend(p_list, labels, title="ToF", loc="upper right")
    _ax.set_yticks([])
    ax.legend(loc="lower right")
    save_fig(fig, save_name, dpi=800)


def get_cli_args():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--file",
        type=str,
        default=str(
            DATA_PATH
            / "fits/"
            / "exp/20250904212745g1_2024_06_11_4layer_cw_dcs_1000_log_10_200000_exp_min6_min_g10.5.hdf5"
        ),
        help="Path to the fit file",
    )
    parser.add_argument(
        "--save_name",
        type=str,
        default="figure10_bfi_correlation_vs_tof.png",
        help="Name of the output figure",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_cli_args()
    main(args.file, args.save_name)
