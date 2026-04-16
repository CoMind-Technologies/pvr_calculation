import os
import re

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from mc_analysis.examples.get_brain_sensitivity_and_equivalent_tofs import PARQUET_FILEPATH
from mc_analysis.fitting.loaders import load_pickle
from mc_analysis.utils.plotting import save_fig, set_default_fontsizes

set_default_fontsizes(14)

SDS_TO_PLOT = [10, 20, 30, 40]


def sweep_to_label(fs_sweep):
    if fs_sweep == "hemophotonics":
        sweep_key = "5000kHz"
    else:
        sweep_key = fs_sweep
    # Write this more efficiently with regex in the future!
    if isinstance(fs_sweep, str):
        sweep_lookup = {
            "5000kHz": "5000 kHz",
            "200kHz": "200 kHz",
            "100kHz": "100 kHz",
            "50kHz": "50 kHz",
            "20kHz": "20 kHz",
            "10kHz": "10 kHz",
        }
        sweep_label = sweep_lookup[sweep_key]
    else:
        sweep_label = f"{fs_sweep} kHz"
    return sweep_label


def _pkl_to_df(indices, fs_sweep="200kHz"):
    _df_out = []
    for _index_dict in indices:
        tof = _index_dict["tof"]
        if not isinstance(tof, np.ndarray):
            tof = np.array(tof)
        index_equiv = _index_dict["index_equiv_tofs"]
        cbf_sensitivity = _index_dict.get("cbf_sensitivity_cw", 4 * [None])
        sbf_sensitivity = _index_dict.get("sbf_sensitivity_cw", 4 * [None])
        sds_list = np.sort(SDS_TO_PLOT)
        equiv_tofs = tof[index_equiv]
        for index_sds, (_sds, _tof, _index_equiv) in enumerate(zip(sds_list, equiv_tofs, index_equiv)):
            _df_out.append(
                {
                    "sub": _index_dict.get("sub", None),
                    "mus": _index_dict["mus"],
                    "skull_thickness": _index_dict["skull_thickness"],
                    "sds": _sds,
                    "equivalent_tof": _tof,
                    "cbf_sensitivity": cbf_sensitivity[index_sds],
                    "sbf_sensitivity": sbf_sensitivity[index_sds],
                    "CBF": _index_dict["dbc"],
                    "SBF": _index_dict["dbs"],
                    "index_equiv_tofs": _index_equiv,
                }
            )
    df_200 = pd.json_normalize(_df_out)
    df_200["fs_sweep"] = fs_sweep
    df_200 = df_200.query("mus == 9.0 and skull_thickness == 6")
    df_200["fs_sweep_kHz"] = int(fs_sweep.replace("kHz", ""))
    return df_200


def load_cw_equiv_tofs():
    folder = PARQUET_FILEPATH
    prefix = "depth_sensitivity_"
    file1 = os.path.join(folder, f"{prefix}200_5000.pkl")
    file2 = os.path.join(folder, f"{prefix}200_200.pkl")
    file3 = os.path.join(folder, f"{prefix}200_100.pkl")
    file4 = os.path.join(folder, f"{prefix}200_20.pkl")
    file5 = os.path.join(folder, f"{prefix}200_5.pkl")
    file6 = os.path.join(folder, f"{prefix}200_1.pkl")
    file_list = [file1, file2, file3, file4, file5, file6]
    df_all = pkls_to_equiv_tof_df(file_list)
    return df_all


def pkls_to_equiv_tof_df(file_list):
    dfs = []
    for file in file_list:
        indices = load_pickle(file)
        acq_freq = re.search(r"_(\d+).pkl", file).group(1)
        df_200 = _pkl_to_df(indices, fs_sweep=f"{acq_freq}kHz")
        dfs.append(df_200)
    df_all = pd.concat(dfs)
    return df_all


def _equivalent_tof_fill_between(ax, df, colors=None, **kwargs):
    """Plot equivalent tof with fill between min and max values representing the simulation errorbars.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        The axes to plot on.
    df : pandas.DataFrame
        DataFrame containing equivalent ToF data.
    colors : list of str, optional
        List of colors for each fs_sweep group. If None, defaults to ['k',
        'r'].
    **kwargs : dict
        Additional keyword arguments passed to ax.legend().
    Returns
    -------
    None
    """

    df["Equivalent ToF (ps)"] = df["equivalent_tof"] * 1e12

    if colors is None:
        colors = ["k", "r"]
    df = df.sort_values("fs_sweep_kHz")
    for index, (fs_sweep, df_sub) in enumerate(df.groupby("fs_sweep_kHz")):
        df_sub = df_sub[df_sub.columns[df_sub.columns != "fs_sweep"]]
        # sort values
        df_sub = df_sub.sort_values("sds")
        _max_vals = df_sub.groupby("sds").agg("max").reset_index()
        _min_vals = df_sub.groupby("sds").agg("min").reset_index()
        _median = df_sub.groupby("sds").agg("mean").reset_index()
        print(f"fs_sweep = {fs_sweep} {_median}")
        p1 = ax.plot(
            _median["sds"], _median["Equivalent ToF (ps)"], label=f"{sweep_to_label(fs_sweep)}", c=colors[index]
        )
        ax.fill_between(
            _median["sds"],
            _min_vals["Equivalent ToF (ps)"],
            _max_vals["Equivalent ToF (ps)"],
            color=p1[0].get_color(),
            alpha=0.2,
        )
    if kwargs.pop("plot_legend_on", True):
        ax.legend(title="CW Acq. freq.", **kwargs)
    ax.set_xlabel("SDS (mm)")
    ax.set_ylabel("Equivalent ToF (ps) - (200kHz, 10mm)")
    ax.set_xlim(10, 40)
    ax.set_ylim(180, 1440)


def _main(df_all, suffix=""):
    # plot equivalent tof
    print(f"Number of rows before dropping duplicates: {len(df_all)}")
    df_all = df_all.drop_duplicates(
        subset=["sds", "fs_sweep", "equivalent_tof", "CBF", "SBF", "mus", "skull_thickness"]
    )
    print(f"Number of rows after dropping duplicates: {len(df_all)}")
    # Plot for presentation
    colors = plt.cm.gnuplot2(np.linspace(0, 0.8, len(df_all["fs_sweep"].unique())))
    fig, ax = plt.subplots(1, 1, figsize=(5, 4.6))
    _equivalent_tof_fill_between(ax, df_all, colors=colors, framealpha=1.0, plot_legend_on=True)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    save_fig(fig, f"figure7_equiv_tof_{suffix}.png", dpi=700)


def main():
    df_all = load_cw_equiv_tofs()
    _main(df_all)
    plt.show()


if __name__ == "__main__":
    main()
