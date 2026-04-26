"""
Segmentation / FIR helpers and :class:`AlignSignals`.

DSP routines are vendored from CoMind internal ``comind_utils`` (``dsp.fir`` and
``dsp.segmentation.segmenter_utils``) so this package does **not** depend on
``comind_utils`` at runtime.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Union

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage as sp_nd
import scipy.signal as sps
from numpy.typing import NDArray
from scipy.interpolate import interp1d
from scipy.signal import (
    convolve,
    correlate,
    firwin,
    kaiser_atten,
    kaiser_beta,
    kaiserord,
)

# ---------------------------------------------------------------------------
# FIR (vendored from comind_utils.dsp.fir)
# ---------------------------------------------------------------------------


def kaiser_fir(f_type, fc, tbw, fs, att=None, filter_length=None):
    nyq = fs / 2
    fc = fc / nyq
    tbw = tbw / nyq

    if not ((att is None) ^ (filter_length is None)):
        raise ValueError(
            "Exactly one out of `att` and `filter_length` must be provided"
        )

    if att is None:
        if filter_length % 2 == 0:
            filter_length = filter_length + 1

        att = kaiser_atten(filter_length, tbw)
        beta = kaiser_beta(att)

    else:
        filter_length, beta = kaiserord(att, tbw)
        if filter_length % 2 == 0:
            filter_length = filter_length + 1

    filter_coeffs = firwin(filter_length, fc, window=("kaiser", beta), pass_zero=f_type)

    return filter_coeffs, att


def apply_fir(x, fir_coeffs, padtype="edge", causal=False, axis=-1):
    axis = axis % x.ndim
    filter_length = fir_coeffs.shape[0]

    if padtype is not None:
        pad_t = [(0, 0)] * x.ndim

        if causal:
            n_pad = filter_length - 1
            pad_t[axis] = (n_pad, 0)
        else:
            n_pad = int((filter_length - 1) / 2)
            pad_t[axis] = (n_pad, n_pad)

        x = np.pad(x, pad_t, mode=padtype)

    dims = axis * (1,) + fir_coeffs.shape + (x.ndim - axis - 1) * (1,)

    if causal:
        x = convolve(x, np.reshape(fir_coeffs, dims), mode="full")
        n = x.shape[axis]
        x = np.take(x, range(0, n - filter_length + 1), axis=axis)
    else:
        x = convolve(x, np.reshape(fir_coeffs, dims), mode="same")

    if padtype is not None:
        n = x.shape[axis]
        if causal:
            _s = slice(n_pad, n)
        else:
            _s = slice(n_pad, n - n_pad)

        x = x[(slice(None),) * axis + (_s,)]

    return x


# ---------------------------------------------------------------------------
# Segmentation DSP (vendored from comind_utils.dsp.segmentation.segmenter_utils)
# ---------------------------------------------------------------------------


def highpass_filter(data, cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    padlen = int(len(data) * 0.5)
    b, a = sps.butter(order, normal_cutoff, btype="high", analog=False)
    y = sps.filtfilt(b, a, data, padlen=padlen, method="pad", padtype="constant")
    return y


def lowpass_filter(data, cutoff: float, fs: float, order: int = 4):
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    padlen = int(len(data) * 0.5)
    b, a = sps.butter(order, normal_cutoff, btype="low", analog=False)
    y = sps.filtfilt(b, a, data, padlen=padlen, method="pad", padtype="constant")
    return y


def min_peak_distance(arr, fs):
    f, p = sps.welch(arr, fs=fs, nfft=512)
    peak_loc = np.argmax(p)
    return 0.8 * fs / f[peak_loc]


def ssf_(data, fs, w: float = 100e-3):
    y_ = lowpass_filter(data, 16, fs)
    w = int(w * fs)
    y_diff = np.gradient(y_, 1)
    z = []
    for i in range(w, len(y_)):
        this_diff = y_diff[i - w : i]
        z.append(np.sum(this_diff[this_diff > 0]))
    return z


def aggregate(seg_data, warp_len):
    resamp_seg = []

    for seg in seg_data:
        try:
            resamp_seg.append(sps.resample(seg, warp_len))
        except Exception:
            resamp_seg.append(np.full(warp_len, np.nan))
    aggregated_seg = np.stack(resamp_seg)
    return aggregated_seg


def detect_beats_fogd(data, fs, ransac_window_size=5.0, lowfreq=5.0, highfreq=15.0):
    ransac_window_size = int(ransac_window_size * fs)

    data_low = lowpass_filter(data, highfreq, fs)
    data_band = highpass_filter(data_low, lowfreq, fs)

    ddata = np.gradient(data_band)
    ddata_power = ddata**2

    thresholds = []
    max_powers = []
    for i in range(int(len(ddata_power) / ransac_window_size)):
        sample = slice(i * ransac_window_size, (i + 1) * ransac_window_size)
        d = ddata_power[sample]
        thresholds.append(0.5 * np.std(d))
        max_powers.append(np.max(d))

    threshold = np.median(thresholds)
    max_power = np.median(max_powers)
    ddata_power[ddata_power < threshold] = 0

    ddata_power /= max_power
    ddata_power[ddata_power > 1.0] = 1.0
    square_decg_power = ddata_power**2
    square_decg_power[square_decg_power < 1e-12] = 1e-12

    shannon_energy = -square_decg_power * np.log(square_decg_power)
    shannon_energy[~np.isfinite(shannon_energy)] = 0.0

    mean_window_len = int(fs * 0.125 + 1)
    lp_energy = np.convolve(
        shannon_energy, [1.0 / mean_window_len] * mean_window_len, mode="same"
    )

    lp_energy = sp_nd.gaussian_filter1d(lp_energy, fs / 8.0)
    lp_energy_diff = np.diff(lp_energy)

    zero_crossings = (lp_energy_diff[:-1] > 0) & (lp_energy_diff[1:] < 0)
    zero_crossings = np.flatnonzero(zero_crossings)
    return zero_crossings


def rsp_interp(
    data: NDArray[np.number], resamp_len: int, axis: int
) -> NDArray[np.number]:
    interp = interp1d(
        np.linspace(0, 1, data.shape[axis]),
        data,
        kind="linear",
        axis=axis,
        assume_sorted=True,
    )

    x_out = np.arange(resamp_len) / resamp_len

    return interp(x_out)


def split_continuous(
    data: NDArray[np.number],
    segmentation_idx: NDArray[np.int_],
    resampl_len: int,
    axis: int,
) -> NDArray[np.number]:
    axis = axis % data.ndim
    segmentation_idx = segmentation_idx[0 <= segmentation_idx]
    segmentation_idx = segmentation_idx[segmentation_idx < data.shape[axis]]
    n_pulses = len(segmentation_idx) - 1

    in_shape = data.shape

    out_shape = in_shape[:axis] + (n_pulses, resampl_len) + in_shape[(axis + 1) :]

    pulses_stacked = np.full(out_shape, fill_value=np.nan, dtype=data.dtype)

    n_prev = axis
    n_trailing = pulses_stacked.ndim - axis - 1

    for i_pulse in range(n_pulses):
        seg_start_idx = segmentation_idx[i_pulse]
        seg_end_idx = segmentation_idx[i_pulse + 1] + 1
        segment = np.take(data, range(seg_start_idx, seg_end_idx), axis=axis)

        dest = n_prev * (slice(None),) + (i_pulse,) + n_trailing * (slice(None),)
        pulses_stacked[dest] = rsp_interp(segment, resampl_len, axis)

    return pulses_stacked


# ---------------------------------------------------------------------------
# AlignSignals (vendored from comind_utils.dsp.segmentation.segmenter)
# ---------------------------------------------------------------------------


class AlignSignals:
    """
    Align segmented BFI (i.e. one data chunk) - Signals are at same sampling frequency
    """

    def __init__(
        self,
        ref_peaks: Union[np.ndarray, list],
        data: Union[np.ndarray, list],
        fs: float,
        method: str = "troughs",
        method_preproc: str | None = None,
        fixed_offset: Union[int, None] = 5,
        ref_data: Union[np.ndarray, None] = None,
        use_xcorr: bool = False,
        warp_len: int = 151,
        save_path: Union[str, Path, None] = None,
    ):
        self.data = data

        self.fs = fs
        self.ref_data = ref_data
        self.ref_peaks = ref_peaks
        self.method = method
        self.method_preproc = method_preproc
        if self.ref_data is not None:
            self.use_xcorr = use_xcorr
        else:
            self.use_xcorr = False
        self.warp_len = warp_len

        self.fixed_offset = (
            fixed_offset  # user-provided offset for 'constant' alignment method
        )
        self.save_path = save_path
        self.params = {
            "method": method,
            "use_xcorr": self.use_xcorr,
            "warp_len": self.warp_len,
            "fixed_offset": self.fixed_offset,
        }
        self.data_filt = highpass_filter(
            self.data, cutoff=0.5, fs=self.fs, order=5
        )
        if self.method_preproc == "ssf":
            self.w = 100e-3
            self.data_filt = ssf_(self.data_filt, self.fs, w=self.w)
        else:
            self.data_filt = self.data_filt**5

    def align(self, method: str):
        if self.use_xcorr:
            offset_0 = self._estimate_offset_xcorr()
        else:
            offset_0 = 0

        if method == "constant":
            self.aligned = self._align_from_constant()
        elif method == "troughs":
            self.aligned = self._align_from_min_troughs(init_offset=offset_0)
        elif method == "slope":
            self.aligned = self._align_from_slope(init_offset=offset_0)

    def _align_from_constant(self):
        segs = np.split(self.data, self.ref_peaks - self.fixed_offset)

        whole_segs = segs[1:-1]
        segmented_pulses = aggregate(whole_segs, self.warp_len)

        return segmented_pulses, self.fixed_offset

    def _align_from_min_troughs(self, init_offset: int):
        search_win_samps = int(np.median(np.gradient(self.ref_peaks)))
        best_offset = init_offset
        lowest_value = 1e3
        hpf, _ = kaiser_fir("highpass", 0.25, 0.5, self.fs, att=54)
        data_filt_hpf = apply_fir(x=self.data_filt, fir_coeffs=hpf)
        for offset in range(
            0, search_win_samps * 2
        ):  # TODO - update to allow any of the two signals leading
            segs = np.split(data_filt_hpf, self.ref_peaks - offset)
            whole_segs = segs[1:-1]
            aggregated_pulses = aggregate(whole_segs, self.warp_len)
            val_troughs = (
                np.nanmedian(aggregated_pulses, 0)[0]
                + np.nanmedian(aggregated_pulses, 0)[-1]
            )

            if val_troughs <= lowest_value:
                lowest_value = val_troughs
                best_offset = offset
        if self.method_preproc == "ssf":
            extra_offset = int(self.w * self.fs)
        else:
            extra_offset = 0
        final_segs = np.split(self.data, self.ref_peaks - best_offset + extra_offset)
        best_whole_segs = final_segs[1:-1]
        segmented_pulses = aggregate(best_whole_segs, self.warp_len)
        return segmented_pulses, best_offset

    def _align_from_slope(self, init_offset: int):
        search_win_samps = int(min_peak_distance(self.data_filt, self.fs) / 2)

        best_offset = init_offset
        highest_value = 0
        for offset in range(
            0, search_win_samps // 2
        ):  # TODO - update to allow any of the two signals leading
            ssf_data = np.gradient(ssf_(self.data_filt, self.fs))
            segs = np.split(ssf_data, self.ref_peaks - offset)
            whole_segs = segs[1:-1]
            aggregated_pulses = aggregate(whole_segs, self.warp_len)
            val_slope = np.nanmedian(aggregated_pulses, axis=0)[0]
            if val_slope > highest_value:
                highest_value = val_slope
                best_offset = offset + 1
                np.split(self.data_filt, self.ref_peaks - best_offset)
        if self.method_preproc == "ssf":
            extra_offset = int(self.w * self.fs)
        else:
            extra_offset = 0
        final_segs = np.split(self.data, self.ref_peaks - best_offset + extra_offset)
        best_whole_segs = final_segs[1:-1]

        segmented_pulses = aggregate(best_whole_segs, self.warp_len)

        return segmented_pulses, best_offset

    def _estimate_offset_xcorr(self):
        if self.ref_data is not None:
            ref_data_filt = highpass_filter(
                self.ref_data, cutoff=0.5, fs=self.fs, order=4
            )
            xcorr = correlate(self.data_filt, ref_data_filt)
            if np.max(xcorr) > -np.min(xcorr):
                return np.argmax(xcorr)
            else:
                return np.argmin(xcorr)
        else:
            return 0

    def plot_segments(
        self,
        fig_show: bool = True,
        fig_save: bool = True,
        fig_close: bool = False,
        fig_folder: Union[str, Path, None] = None,
        save_name: str = None,
    ):
        fig, ax = plt.subplots()
        if hasattr(self, "aligned"):
            for i_seg in range(self.aligned[0].shape[0]):
                ax.plot(self.aligned[0][i_seg, :], c="k", alpha=0.2)
            ax.plot(np.nanmean(self.aligned[0], axis=0), c="r")
            ax.margins(x=0)

            if fig_show:
                plt.show()

            if fig_close:
                plt.close(fig)

            if fig_save:
                if fig_folder is None:
                    fig_folder = Path.cwd()
                if save_name is None:
                    save_name = "plot_alignsignals_segments.png"
                plt.savefig(Path(fig_folder, save_name), dpi=200, bbox_inches="tight")

    def plot_edges(
        self,
        fig_show: bool = True,
        fig_save: bool = True,
        fig_close: bool = False,
        fig_folder: Union[str, Path, None] = None,
        save_name: str = None,
    ):
        fig, ax = plt.subplots(figsize=(25, 5))

        if hasattr(self, "aligned"):
            ax.plot(self.data)
            for peak in self.ref_peaks:
                ax.axvline(peak - self.aligned[1], c="k", ls=":")
            ax.margins(x=0)

        if fig_show:
            plt.show()

        if fig_close:
            plt.close(fig)

        if fig_save:
            if fig_folder is None:
                fig_folder = Path.cwd()
            if save_name is None:
                save_name = "plot_alignsignals_segments.png"
            plt.savefig(Path(fig_folder, save_name), dpi=200, bbox_inches="tight")

    def save(
        self,
        save_folder: Union[str, Path, None] = None,
        edges_name: str = None,
        save_segments: bool = False,
        segments_name: str = None,
        params_name: str = None,
    ):
        """Save offset to reference data, and AlignSignals parameters."""

        if save_folder is None:
            save_folder = Path.cwd()
        if params_name is None:
            params_name = "align_params.json"
        if edges_name is None:
            edges_name = "align_edges.pkl"
        if segments_name is None:
            segments_name = "align_segments.pkl"

        with open(Path(save_folder, params_name), "w", encoding="utf-8") as f:
            json.dump(self.params, f, ensure_ascii=False, indent=4)

        if hasattr(self, "aligned"):
            with open(Path(save_folder, edges_name), "wb") as fp_edges:
                pickle.dump(
                    {
                        "params": self.params,
                        "edges": self.ref_peaks - self.aligned[1],
                        "offset": self.aligned[1],
                    },
                    fp_edges,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )

            if save_segments:
                with open(Path(save_folder, segments_name), "wb") as fp_segments:
                    pickle.dump(
                        {
                            "params": self.params,
                            "segments": self.aligned[0],
                        },
                        fp_segments,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
