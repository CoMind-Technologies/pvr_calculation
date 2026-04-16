"""This is a script to compute equivalent time-of-flight (ToF) values where
 TD brain sensitivities match CW brain sensitivities for different acquisition frequencies.


 Output sare saved to PARQUET_FILEPATH.
"""
import os
import pathlib
import pickle

import numpy as np
import pandas as pd

from mc_analysis.fitting.loaders import load_pickle
from mc_analysis.simulator.g1_simulator_utils import get_cw_jac_list, get_g1_cols, get_td_jac
from mc_analysis.utils.absolute_paths import DATA_PATH
from mc_analysis.utils.brain_sensitivity_calculation import integrate_from_min_lag_index

BASE_PATH = str(DATA_PATH / "simulations" / "2025_08_05_dcs_comparison")
CT_FILE = str(DATA_PATH / "simulations" / "values_to_simulate.pkl")
PARQUET_FILEPATH = DATA_PATH / "equivalent_tof"

if not os.path.exists(PARQUET_FILEPATH):
    os.makedirs(PARQUET_FILEPATH)

N_PROC = 8
FILTER_ON = True

REFRACTIVE_INDICES = np.array([1.4, 1.4, 1.33, 1.4])
MUA_LIST = np.array([0.012, 0.013, 0.012, 0.017])  # mm-1


def iter_simulations(simulations_unwrapped, yield_full=False):
    """Iterate over a list of simulations"""
    for index, row in enumerate(simulations_unwrapped):
        file_stem = row["file_stem"]
        basename = f"{BASE_PATH}/{file_stem}.mch"
        dbs = row["dbs"]
        dbc = row["dbc"]
        if yield_full:
            yield dbs, dbc, basename, row
        else:
            yield dbs, dbc, basename


def load_ct_errorbar_simulations(file=CT_FILE):
    """ Load a range of aDb and aDs values representative of in vivo person-to-person variability."""
    return load_pickle(file=file)


def get_td_brain_sensitivity_integrated(aDb, jac, lags, mask, tof):
    """Get time-domain brain sensitivity

    Parameters
    ----------
    aDb : array-like
        List of Brownian diffusion coefficients [dbs, db_skull, db_csf, dbc]
    jac : pd.DataFrame
        DataFrame containing the Jacobian.
    lags : array-like
        List of lag times to compute the Jacobian for.
    mask : array-like
        Boolean array to mask lags.
    tof : array-like
        List of time-of-flight values.
    Returns
    -------
    cbf_sensitivity_td : np.ndarray
        Array of cerebral blood flow sensitivity values for each ToF bin.
    brain_sensitivity_norm_td : np.ndarray
        Array of normalized brain sensitivity values for each ToF bin.
    sbf_sensitivity_td : np.ndarray
        Array of scalp blood flow sensitivity values for each ToF bin.

    """
    n_tof = int(max(jac.index))
    tof_full = np.zeros(n_tof)
    tof_full[jac.index.unique().astype(int) - 1] = tof
    brain_sensitivity_norm_td = np.zeros(n_tof)
    cbf_sensitivity_td = np.zeros(n_tof)
    sbf_sensitivity_td = np.zeros(n_tof)
    # loop over tof bins and compute brain sensitivity
    for itof in jac.index.unique().astype(int):
        itof = int(itof)
        g1_jac_cbf = get_g1_cols(jac.query(f"layer == 3 and index == {itof}"), index_min=0)
        g1_jac_sbf = get_g1_cols(jac.query(f"layer == 0 and index == {itof}"), index_min=0)
        if len(g1_jac_sbf) == 0:
            print(f"{jac.shape} itof = {itof} tof = {tof[itof]}")
            raise ValueError(f"Invalid tof index {itof}\n Potential values={jac.index.unique()}")

        (_cbf_sensitivity, _brain_sensitivity_norm, _sbf_sensitivity,) = get_cw_brain_sensitivity_integrated(
            aDb, g1_jac_cbf, g1_jac_sbf, min_lag=1 / 200_000, lags=lags, mask=mask
        )

        cbf_sensitivity_td[itof - 1] = _cbf_sensitivity
        sbf_sensitivity_td[itof - 1] = _sbf_sensitivity
        brain_sensitivity_norm_td[itof - 1] = _brain_sensitivity_norm
        # print(f"Loaded jacobian for {file_stem} - {len(jac_list_cw)} sds")
    return cbf_sensitivity_td, brain_sensitivity_norm_td, sbf_sensitivity_td, tof_full


def get_cw_brain_sensitivity_integrated(aDb, g1_jac_cbf, g1_jac_sbf, min_lag, lags, mask=None):
    """Get continuous-wave brain sensitivity for a given minimum lag time.

    Parameters
    ----------
    aDb : array-like
        List of Brownian diffusion coefficients [dbs, db_skull, db_csf,
        dbc]
    g1_jac_cbf : np.ndarray
        Array of cerebral blood flow Jacobian values.
    g1_jac_sbf : np.ndarray
        Array of scalp blood flow Jacobian values.
    min_lag : float
        Minimum lag time to integrate from.
    lags : array-like
        List of lag times to compute the Jacobian for.
    mask : array-like, optional
        Boolean array to mask lags. Default is None.
    Returns
    -------
    _cbf_sensitivity : float
        Cerebral blood flow sensitivity value.
    _depth_sensitivity : float
        Normalized brain sensitivity value (brain senistivity / (brain + scalp sensitivity)).
    _sbf_sensitivity : float
        Scalp blood flow sensitivity value.

    """
    # check that ndim of g1_jac is 1
    if (g1_jac_cbf.ndim > 1 and g1_jac_cbf.shape[0] != 1) or (g1_jac_sbf.ndim > 1 and g1_jac_sbf.shape[0] != 1):
        raise ValueError(
            f"g1_jac_cbf and g1_jac_sbf should be 1D arrays, got shapes {g1_jac_cbf.shape} and {g1_jac_sbf.shape}"
        )

    g1_cbf = g1_jac_cbf.flatten()
    g1_sbf = g1_jac_sbf.flatten()

    if mask is not None:
        lags_in = np.copy(lags[mask])
        g1_cbf = g1_cbf[mask]
        g1_sbf = g1_sbf[mask]
    else:
        lags_in = np.copy(lags)

    jac_cbf_relative = g1_cbf * aDb[3]
    jac_sbf_relative = g1_sbf * aDb[0]
    index_min_lag = np.argmin(np.abs(lags_in - min_lag))
    print(f"Integrating from min lag index {index_min_lag} for lags {lags_in[index_min_lag]} <- target {min_lag} ")
    _cbf_sensitivity = integrate_from_min_lag_index(jac_cbf_relative, lags_in, index=index_min_lag)
    _sbf_sensitivity = integrate_from_min_lag_index(jac_sbf_relative, lags_in, index=index_min_lag)
    denom = integrate_from_min_lag_index(jac_sbf_relative + jac_cbf_relative, lags_in, index=index_min_lag)
    _depth_sensitivity = np.abs(_cbf_sensitivity) / np.abs(denom)
    return _cbf_sensitivity, _depth_sensitivity, _sbf_sensitivity


def get_td_and_cw_jacobians(aDb, basename, lags_200kHz, lags_cw, sweep_rate_cw, sds_to_plot):
    if lags_cw is None:
        lags_cw = lags_200kHz
    # sweep_rate_cw = (lags_cw[1] - lags_cw[0]) ** -1
    jac, tof, lag = get_td_jac_from_parquet(aDb, basename=basename, lags=lags_200kHz)
    jac_list_cw = get_cw_jac_from_parquet(aDb, basename, lags_cw, tof, sweep_rate_cw, sds_to_plot)

    return jac, jac_list_cw, tof.to_numpy()


def get_cw_jac_from_parquet(
    aDb, basename, lags_cw, tof, sweep_rate_cw, sds_to_plot, refractive_indices=REFRACTIVE_INDICES, mua=MUA_LIST
):
    jac_list_cw = []
    for _sds in sds_to_plot:
        filename_jac, filename_info = get_parquet_filename_cw(
            dbs=aDb[0],
            dbc=aDb[3],
            sds=_sds,
            sweep_rate=sweep_rate_cw,
            basename=basename,
        )
        if os.path.exists(filename_jac):
            print(f"Loading from parquet - {sweep_rate_cw} {_sds}")
            info, _jac_cw = _load_jac(filename_jac, filename_info)

            # jac_cw, info = load_from_parquet(aDb[-1], aDb[0], sds, sweep_rate_cw, modality='cw')
        else:
            print(f"File not found {filename_jac}")
            _jac_cw = get_cw_jac_list(aDb, basename, lags_cw, mua, refractive_indices, (_sds,))[0]
            # Save to parquet?
            save_jac(filename_info, filename_jac, _jac_cw, lags_cw, tof)

        jac_list_cw.append(_jac_cw.copy())
    return jac_list_cw


def save_jac(filename_info, filename_jac, _jac_cw, lags_cw, tof):
    _jac_cw.to_parquet(filename_jac)
    with open(filename_info, "wb") as fid:
        pickle.dump({"tof": tof, "lags": lags_cw}, fid)


def get_td_jac_from_parquet(aDb, basename, lags):
    """Get time domain Jacobian from parquet or compute it if not available.

    Parameters
    ----------
    aDb : array-like
        List of Brownian diffusion coefficients [dbs, db_skull, db_csf, dbc]
    basename : str
    lags :  array-like
        List of lag times to compute the Jacobian for.

    Returns
    -------
    jac : pd.DataFrame
        DataFrame containing the Jacobian.
    tof : np.ndarray
        Array of time-of-flight values.
    lags : np.ndarray

    """
    dbs, dbc = aDb[0], aDb[3]
    filename_td = get_td_filename(dbc=dbc, dbs=dbs, basename=basename)
    filename_td_info = get_td_filename_info(dbc=dbc, dbs=dbs, basename=basename)
    if os.path.exists(filename_td) and os.path.exists(filename_td_info):
        jac = pd.read_parquet(filename_td)
        info = load_parquet_info(dbc, dbs, basename)
        tof = info["tof"]
        lags = info["lags"]
    else:
        jac, tof = get_td_jac(aDb, basename, lags=lags, mua=MUA_LIST, refractive_indices=REFRACTIVE_INDICES, sds=10)
        # jac, tof, ds_int = get_brain_sensitivity(aDb, lags=[lags], basename=basename)
        jac.to_parquet(filename_td)
        # save info
        with open(filename_td_info, "wb") as fid:
            pickle.dump({"tof": tof, "lags": lags, "basename": basename}, fid)

    return jac, tof, lags


def load_parquet_info(dbc, dbs, basename):
    filename = get_td_filename_info(dbc, dbs, basename)
    with open(filename, "rb") as fid:
        info = pickle.load(fid)
    return info


def get_parquet_filename_td(dbs, dbc, basename):
    filename_td = get_td_filename(dbc, dbs, basename=basename)
    filename_td_info = get_td_filename_info(dbc, dbs, basename)
    return filename_td, filename_td_info


def get_td_filename(dbc, dbs, basename):
    suffix = get_td_stem(basename)
    filename_td = get_parquet_filename(dbc, dbs, suffix=suffix)
    return filename_td


def get_td_stem(basename):
    base_stem = pathlib.Path(basename).stem
    suffix = f"{base_stem}_td_200kHz"
    return suffix


def get_td_filename_info(dbc, dbs, basename):
    suffix = get_td_stem(basename)
    return os.path.join(PARQUET_FILEPATH, f"jac_out_info_{suffix}_{dbs:.3e}_{dbc:.3e}.pkl")


def jac_to_ds_integrated(aDb, jac_):
    g1_jac_superficial = get_g1_cols(jac_.query("layer == 0") * aDb[0])
    g1_jac_deep = get_g1_cols(jac_.query("layer == 3") * aDb[3])
    ds_integrated = np.sum(g1_jac_deep, axis=-1) / np.sum(g1_jac_superficial, axis=-1)
    return ds_integrated


def get_parquet_filename_cw(dbs, dbc, sds, sweep_rate, basename=None):
    base_stem = pathlib.Path(basename).stem

    if sweep_rate == 1000:
        suffix = f"{base_stem}_{int(np.round(sweep_rate / 1000))}kHz_cw_{sds}"
    else:
        suffix = f"{base_stem}_1000to5000kHz_cw_{sds}"

    filename = get_parquet_filename(dbc, dbs, suffix)
    return filename, os.path.join(PARQUET_FILEPATH, f"jac_out_{suffix}_{dbs:.3e}_{dbc:.3e}.pkl")


def get_parquet_filename(dbc, dbs, suffix="200kHz"):
    return os.path.join(PARQUET_FILEPATH, f"jac_out_{suffix}_{dbs:.2e}_{dbc:.2e}.parquet")


def _load_jac(filename, filename_info):
    print(filename)
    if os.path.exists(filename):
        jac = pd.read_parquet(filename)
        with open(filename_info, "rb") as fid:
            info = pickle.load(fid)
    else:
        raise ValueError(f"File {filename} does not exist")
    return info, jac


def get_brain_sensitivity_all(save_folder, sweep_rate_cw):
    """Compute brain sensitivity for a given CW acquisition frequency. and the equivalent ToF
    where a TD device has the same brain sensitivity.


    """
    lags_log = np.logspace(-7, 1, 500)
    lags_log = np.geomspace(2e-7, 1, 2000)
    lags = np.sort(
        np.concatenate(
            (
                lags_log,
                np.array(
                    [
                        1 / 5000_000,
                        1 / 200_000,
                        1 / 100_000,
                        1 / 20_000,
                        1 / 5_000,
                        1 / 1_000,
                    ]
                ),
            )
        )
    )

    if sweep_rate_cw != 1000:
        lags_cw = np.copy(lags)
    else:
        lags_cw = np.logspace(-3, 1, 200)

    mask = None

    save_name = os.path.join(save_folder, f"depth_sensitivity_200_{int(sweep_rate_cw / 1000)}.pkl")
    if os.path.exists(save_name):
        print(f"Loading from {save_name}")
        with open(save_name, "rb") as f:
            indices = pickle.load(f)
        return indices

    sds_to_plot = [10, 20, 30, 40]
    simulations_unwrapped = load_ct_errorbar_simulations()

    print(f"Running on {len(simulations_unwrapped)} simulations")
    indices = []
    for index, row in enumerate(simulations_unwrapped):
        file_stem = row["file_stem"]
        basename = f"{BASE_PATH}/{file_stem}.mch"
        dbs = row["dbs"]
        dbc = row["dbc"]
        aDb = np.array([dbs, 2e-8, 5e-8, dbc])
        print(f"Processing {index + 1}/{len(simulations_unwrapped)} - {file_stem} {dbc=} {dbs=}")

        jac, jac_list_cw, tof = get_td_and_cw_jacobians(
            aDb, basename, lags, lags_cw, sweep_rate_cw=sweep_rate_cw, sds_to_plot=sds_to_plot
        )
        (
            cbf_sensitivity_td,
            brain_sensitivity_norm_td,
            sbf_sensitivity_td,
            tof_full,
        ) = get_td_brain_sensitivity_integrated(aDb, jac, lags, mask, tof)

        # Normalized brain sensitivity = Sensitivity_CBF / (Sensitivity_CBF + Sensitivity_SBF)
        brain_sensitivity_norm_cw = np.zeros((len(sds_to_plot)))
        sbf_sensitivity_cw = np.zeros((len(sds_to_plot)))
        cbf_sensitivity_cw = np.zeros((len(sds_to_plot)))
        index_equiv_tofs = np.zeros((len(sds_to_plot)), dtype=int)

        for index_sds, (sds, jac_cw) in enumerate(zip(sds_to_plot, jac_list_cw)):
            g1_jac_sbf = get_g1_cols(jac_cw.query("layer == 0"), index_min=0)
            g1_jac_cbf = get_g1_cols(jac_cw.query("layer == 3"), index_min=0)
            (_cbf_sensitivity, _brain_sensitivity_norm, _sbf_sensitivity,) = get_cw_brain_sensitivity_integrated(
                aDb,
                g1_jac_cbf,
                g1_jac_sbf,
                min_lag=1 / sweep_rate_cw,
                lags=lags_cw,
                mask=mask,
            )
            sbf_sensitivity_cw[index_sds] = _sbf_sensitivity
            cbf_sensitivity_cw[index_sds] = _cbf_sensitivity
            brain_sensitivity_norm_cw[index_sds] = _brain_sensitivity_norm
            index_equiv_tofs[index_sds] = np.argmin(np.abs(_brain_sensitivity_norm - brain_sensitivity_norm_td))
            print(
                f"Loaded jacobian for {file_stem} - {sds} mm - {index_sds + 1}/{len(sds_to_plot)} sds "
                f"{cbf_sensitivity_cw[index_sds]=:.3f} {sbf_sensitivity_cw[index_sds]} "
                f"index_equiv_tof = {index_equiv_tofs[index_sds]}"
            )

        indices.append(
            {
                "brain_sensitivity_norm_cw": brain_sensitivity_norm_cw,
                "brain_sensitivity_norm_td": brain_sensitivity_norm_td,
                "tof": tof_full,
                "cbf_sensitivity_cw": cbf_sensitivity_cw,
                "sbf_sensitivity_cw": sbf_sensitivity_cw,
                "cbf_sensitivity_td": cbf_sensitivity_td,
                "sbf_sensitivity_td": sbf_sensitivity_td,
                "dbs": dbs,
                "dbc": dbc,
                "sds": sds_to_plot,
                "index_equiv_tofs": index_equiv_tofs,
                "mus": row["musp"],
                "skull_thickness": row["skull_thickness"],
                "file_stem": file_stem,
            }
        )
    print(save_name)
    with open(save_name, "wb") as f:
        pickle.dump(indices, f)
    return indices


if __name__ == "__main__":
    # Loop over the following acquisition frequencies and compute brain sensitivity
    for sweep_rate_cw in (1000, 5000, 5000_000, 20_000, 100_000, 200_000):
        get_brain_sensitivity_all(save_folder=PARQUET_FILEPATH, sweep_rate_cw=sweep_rate_cw)
