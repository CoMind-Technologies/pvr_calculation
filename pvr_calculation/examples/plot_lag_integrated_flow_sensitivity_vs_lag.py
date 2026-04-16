import os
from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy
from matplotlib import pyplot as plt
from numpy.typing import NDArray

from mc_analysis.simulator.g1_simulator_utils import get_cw_jac_list, get_td_jac
from mc_analysis.utils.absolute_paths import DATA_PATH
from mc_analysis.utils.plotting import save_fig


CMAP = plt.cm.gnuplot2
YLABEL = r"Brain Sensitivity"
XLABEL = r"$\tau_{min} \, (s)$"
LEG_ARGS = {"framealpha": 1.0, "ncol": 1, "fontsize": 10, "title_fontsize": 12}
XLIMITS = (1e-7, 1e-3)


@dataclass
class SimulationOpticalProperties:
    mua: NDArray = np.array([0.012, 0.013, 0.012, 0.017])
    aDb: NDArray = np.array([1e-6, 1e-8, 2e-8, 5e-6])
    refractive_indices: NDArray = np.array([1.4, 1.4, 1.33, 1.4])
    basename: Optional[str] = os.path.join(
        DATA_PATH, "simulations/2025_06_11_4layer_cw_dcs/2024_06_11_4layer_cw_dcs.mch"
    )


def integrate_from_min_lag(target, x_axis, integrate_func=scipy.integrate.simpson):
    assert target.shape[-1] == x_axis.shape[-1]
    y = np.array([integrate_func(target[..., i:], x=x_axis[i:], axis=-1) for i in range(target.shape[-1])])
    return np.moveaxis(y, 0, -1)


def get_cmap_colors(tofs_to_plot, modality_type=None):
    """Get linearly spaced colors from a colormap based on the modality type."""
    if modality_type is None:
        _cmap = CMAP(np.linspace(0, 0.8, len(tofs_to_plot)))
    elif modality_type == "cw":
        _cmap = CMAP(np.linspace(0, 0.4, len(tofs_to_plot)))
    elif modality_type == "td":
        _cmap = CMAP(np.linspace(0.5, 0.8, len(tofs_to_plot)))
    else:
        raise ValueError("Invalid modality_type")
    return _cmap


def get_single_fig():
    fig, ax = plt.subplots(1, 1, figsize=(4, 4))
    plt.subplots_adjust(bottom=0.18, wspace=0.4, left=0.15, top=0.82, hspace=0.35, right=0.95)
    return ax


def set_axes_labels(_ax):
    _ax.set_ylabel(YLABEL)
    _ax.set_xlabel(XLABEL)
    _ax.set_xlim(XLIMITS)
    _ax.set_xscale("log")
    _ax.set_yticks([0.0, 0.2, 0.4, 0.6])
    _ax.set_ylim(0, 0.5)


def run_brain_sensitivity_analysis_cw(lags, simulation_config, sds, ax=None, plot_leg_on=True, **kwargs):
    jac_list = get_cw_jac_list(
        simulation_config.aDb,
        simulation_config.basename,
        lags,
        simulation_config.mua,
        simulation_config.refractive_indices,
        sds,
    )

    plot_cw_brain_sensitivity_vs_lag(
        ax, simulation_config.aDb, jac_list, lags, sds, plot_leg_on=plot_leg_on, **kwargs
    )


def plot_cw_brain_sensitivity_vs_lag(ax, aDb, jac_list, lags, sds_to_plot, save_name=None, plot_leg_on=True):
    lag_cols = [f"g1_{i}" for i in range(len(lags))]
    if ax is None:
        ax = get_single_fig()
    else:
        pass
    colors = get_cmap_colors(sds_to_plot, modality_type="cw")
    for sds, jac, _c in zip(sds_to_plot, jac_list, colors):
        # simulation layer 3 = brain, layer 0 = scalp
        g1_jac_cbf = jac.query("layer == 3")
        g1_jac_sbf = jac.query("layer == 0")
        g1_cbf = g1_jac_cbf[lag_cols].to_numpy().flatten()
        g1_sbf = g1_jac_sbf[lag_cols].to_numpy().flatten()
        mask = lags < 1.0
        s_cbf = integrate_from_min_lag(g1_cbf[mask] * aDb[3], lags[mask])
        s_total = integrate_from_min_lag(g1_sbf[mask] * aDb[0] + g1_cbf[mask] * aDb[3], lags[mask])
        ax.plot(
            lags[mask],
            s_cbf / s_total,
            linestyle="-",
            c=_c,
            label=f"{sds} mm",
        )
        set_axes_labels(ax)
    ax.set_ylabel(YLABEL)

    if plot_leg_on:
        ax.legend(title="SDS (CW)", bbox_to_anchor=(-0.68, 1), loc="upper left", **LEG_ARGS)

    if save_name is not None:
        print(f"Saving {save_name}")
        plt.savefig(save_name, dpi=800)


def run_brain_sensitivity_analysis_td(
    lags, simulation_config, sds, tofs_to_plot=None, ax=None, plot_leg_on=True, save_name=None
):
    """Run the TD Jacobian analysis and plot the lag-integrated sensitivity to flow in the brain vs lag time.

    Parameters
    ----------
    lags : array-like
        Array of lag times.
    sds : int
        Source-detector separation in mm.
    simulation_config : SimulationOpticalProperties
        Configuration object containing simulation parameters. If None, default values are used.
    tofs_to_plot : list of int, optional
        Indices of ToFs to plot. If None, default ToFs are used.
    ax : matplotlib.axes.Axes, optional
        The axes to plot on. If None, a new figure and axes are created.
    plot_leg_on : bool, optional
        Whether to display the legend. Default is True.
    save_name : str or None, optional
        If provided, the name to save the figure as.
    Returns
    -------
    ax : matplotlib.axes.Axes
        The axes containing the plot.
    """
    jac, tof = get_td_jac(
        simulation_config.aDb,
        simulation_config.basename,
        np.copy(lags),
        simulation_config.mua,
        simulation_config.refractive_indices,
        sds,
        tof_bins=np.arange(30) * 65e-12,
    )

    ax = plot_td_brain_sensitivity_vs_lag(
        simulation_config.aDb, ax, jac, lags, plot_leg_on, save_name, tof, tofs_to_plot
    )
    return ax


def plot_td_brain_sensitivity_vs_lag(aDb, ax, jac, lags, plot_leg_on, save_name, tof, tofs_to_plot):
    """Plot the lag-integrated sensitivity to flow in the brain vs lag time for selected ToFs.

    Parameters
    ----------
    aDb : array-like
        List of Brownian diffusion coefficients for each layer.
    ax : matplotlib.axes.Axes
        The axes to plot on. If None, a new figure and axes are created.
    jac : pandas.DataFrame
        The Jacobian DataFrame containing sensitivity data.
    lags : array-like
        Array of lag times.
    plot_leg_on : bool
        Whether to display the legend.
    save_name : str or None
        If provided, the name to save the figure as.
    tof : array-like
        Array of time-of-flight values.
    tofs_to_plot : list of int or None
        Indices of ToFs to plot. If None, default ToFs are used.

    """
    lag_cols = [f"g1_{i}" for i in range(len(lags))]
    if tofs_to_plot is None:
        equiv_tofs = np.array([300e-12, 570e-12, 900e-12, 1200e-12])
        # convert to indices
        tofs_to_plot = [np.argmin(np.abs(tof - _tof)) for _tof in equiv_tofs]
    elif np.max(tofs_to_plot) > len(tof):
        raise IndexError("Jacobian binning error - tof index out of range. Insufficient tof bins in Jacobian.")
    tof = np.array(tof)
    if ax is None:
        ax = get_single_fig()
    if jac.index.name != "ind":
        raise ValueError(f"Jacobian index name is {jac.index.name}, expected 'ind'")
    colors = get_cmap_colors(tofs_to_plot, modality_type="td")
    for itof, _c in zip(tofs_to_plot, colors):
        # simulation layer 3 = brain, layer 0 = scalp
        g1_jac_cbf = jac.query(f"layer == 3 and index == {itof}")
        g1_jac_sbf = jac.query(f"layer == 0 and index == {itof}")
        if len(g1_jac_sbf) == 0:
            print(f"{jac.shape} itof = {itof} tof = {tof[itof]}")
            raise ValueError(f"Invalid tof index {itof}\n Potential values={jac.index.unique()}")
        mask = lags < 1.0
        g1_cbf = np.abs(g1_jac_cbf[lag_cols].to_numpy().flatten())[mask]
        g1_sbf = np.abs(g1_jac_sbf[lag_cols].to_numpy().flatten())[mask]
        num = integrate_from_min_lag(g1_cbf * aDb[3], lags[mask])
        denom = integrate_from_min_lag(g1_sbf * aDb[0] + g1_cbf * aDb[3], lags[mask])
        sensitivity = num / denom
        ax.plot(
            lags[mask],
            sensitivity,
            linestyle="-",
            c=_c,
            label=f"{tof[itof] * 1e12:.0f} ps",
        )
    set_axes_labels(ax)
    if plot_leg_on:
        ax.legend(title="ToF", bbox_to_anchor=(1.02, 1), loc="upper left", **LEG_ARGS)
    if save_name is not None:
        print(f"Saving {save_name}")
        plt.savefig(save_name, dpi=800)
    return ax


def main():
    lags = np.logspace(-8, 1e-3, 500)
    simulation_config = SimulationOpticalProperties()
    fontsize = 14

    fig, ax = plt.subplots(1, 2, figsize=(9, 3))
    plt.subplots_adjust(left=0.2, right=0.86, wspace=0.32, bottom=0.15)
    run_brain_sensitivity_analysis_td(lags, simulation_config=simulation_config, sds=10, tofs_to_plot=None, ax=ax[1])
    ax[1].set_title("TD", fontsize=fontsize)
    run_brain_sensitivity_analysis_cw(lags, simulation_config=simulation_config, sds=[10, 20, 30, 40], ax=ax[0])
    ax[0].set_title("CW", fontsize=fontsize)
    save_fig(fig, "figure6_lag_integrated_flow_sensitivity_vs_lag.png")
    plt.show(block=True)


if __name__ == "__main__":
    main()
