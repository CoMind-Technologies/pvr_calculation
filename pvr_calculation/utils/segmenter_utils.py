import numpy as np
from numpy.typing import NDArray

import itertools
import json
import pickle
import traceback
from pathlib import Path
from typing import Optional, Union

from scipy.signal import find_peaks

from comind_utils.dsp.segmentation import segmenter_utils as segutils


def batch(
    data: NDArray[np.number],
    batch_size: int,
    axis: int,
) -> NDArray[np.number]:
    """
    Split `data` in batches of batch_size `batch_size` elements along
    axis axis.

    Args:
        data (NDArray[np.number]): input data
        batch_size (int): number of items in one batch
        axis (int): axis along which to split the data

    Returns:
        NDArray[np.number]: batched data, axis `axis` dropped, and
            axes (n_batches, batch_size) pre-pended
    """

    # ensure 0 <= axis < n_dim
    axis = axis % data.ndim

    n_batches = data.shape[axis] // batch_size
    # move axis to start for convenience
    data = np.moveaxis(data, axis, 0)
    data = np.reshape(
        data[: (n_batches * batch_size), ...], (n_batches, batch_size) + data.shape[1:]
    )

    return data

def overlaps(edges, intervals):
    overlapping_seg = []
    for interval in intervals:
        try:
            if any(i < interval.min() for i in edges):
                ix_closest_edge_left = (
                    np.where(edges == np.max(edges[edges < (interval.min())]))[0][0] + 1
                )
            else:
                ix_closest_edge_left = 0
            if any(i > (interval.max() - 1) for i in edges):
                ix_closest_edge_right = np.where(
                    edges == np.min(edges[edges > (interval.max())])
                )[0][0]
            else:
                ix_closest_edge_right = len(edges) - 1
            range_overlap = list(range(ix_closest_edge_left, ix_closest_edge_right))
        except Exception:
            range_overlap = []
        overlapping_seg.append(range_overlap)
    return overlapping_seg


class Segmenter:
    """
    Segmentation class for single signal (i.e. one data chunk)
    Assumes a pulsatile signal such as CBF or ABP


    Args:
        data
        fs
        method
        method_preproc
        peak_sign
        warp
        warp_len
        prominence_ratio
        ref_indices [Optional]
    """

    def __init__(
        self,
        data,
        fs,
        method,
        method_preproc,
        peak_sign,
        warp: bool = True,
        warp_len: int = 151,
        prominence_ratio: Union[float, None] = 0.7,
        ref_indices: Union[np.ndarray, list, None] = None,
        save_path: Union[str, Path, None] = None,
    ):
        if method not in ["find_peaks", "foGD", "indices"]:
            raise NotImplementedError("invalid method")
        if peak_sign not in [-1, 1]:
            raise ValueError("peak_sign needs to be -1 or 1")
        self.data = data
        self.fs = fs
        self.method = method
        self.method_preproc = method_preproc
        self.warp = warp
        if self.warp:
            self.warp_len = warp_len
        else:
            self.warp_len = None
        self.peak_sign = peak_sign
        if self.method == "find_peaks":
            self.prominence_ratio = prominence_ratio
        else:
            self.prominence_ratio = None
        self.ref_indices = ref_indices
        self.save_path = save_path
        self.params = {
            "method": self.method,
            "method_preproc": self.method_preproc,
            "warp": self.warp,
            "warp_len": self.warp_len,
            "peak_sign_detect": self.peak_sign,
            "prominence_ratio": self.prominence_ratio,
            "ref_indices": ref_indices,
        }

        self.edges = self.find_edges(
            method=self.method,
            method_preproc=self.method_preproc,
            peak_sign=self.peak_sign,
        )

        # probably separate that in its own call
        self.segments = self.segment(self.edges)

    def find_edges(
        self,
        method: str = "find_peaks",
        method_preproc: Union[str, None] = None,
        peak_sign=1,
    ):
        """
        segment signal
        Args:
            method: str, one of 'find_peaks', 'foGD'
            method_preproc: str, one of 'ssf', None
            peak_sign: [-1, 1] use data or -data (i.e. find positive or negative peaks)
        """
        self.seg_source = self._transform_data(
            method_preproc=method_preproc, peak_sign=peak_sign
        )
        if method == "find_peaks":
            if self.prominence_ratio is not None:
                prominence_threshold = self.prominence_ratio * (
                    np.nanmax(self.seg_source) - np.nanmean(self.seg_source)
                )
                edges = find_peaks(
                    self.seg_source,
                    distance=segutils.min_peak_distance(self.seg_source, self.fs),
                    prominence=prominence_threshold,
                )[0]
            else:
                edges = find_peaks(
                    self.seg_source,
                    distance=segutils.min_peak_distance(self.seg_source, self.fs),
                )[0]
        elif method == "foGD":
            edges = segutils.detect_beats_fogd(self.seg_source, self.fs)
        elif method == "indices":
            edges = self.ref_indices
        else:
            raise NotImplementedError("")
        return edges

    def segment(self, edges):
        segments = np.split(self.data, edges)
        segments = segments[1:-1]
        if self.warp:
            segments = segutils.aggregate(segments, self.warp_len)
        else:
            segments = np.array(
                list(itertools.zip_longest(*segments, fillvalue=np.nan))
            ).T
        return segments

    def _transform_data(
        self,
        method_preproc: Union[str, None] = None,
        peak_sign: int = 1,
        cutoff: float = 0.5,
        order: int = 4,
    ):
        """
        convert data appropriately for segmentation (e.g. apply filter, invert, use SSF)
        """
        transf_data = np.array(self.data.copy())
        transf_data *= peak_sign
        transf_data = segutils.highpass_filter(transf_data, cutoff, self.fs, order)
        if method_preproc == "ssf":
            transf_data = segutils.ssf_(transf_data, self.fs)
        elif method_preproc is None:
            transf_data = transf_data
        else:
            raise NotImplementedError("can only preprocess using ssf or None")

        return transf_data

    def plot_segments(
        self,
        fig_show: bool = True,
        fig_save: bool = True,
        fig_close: bool = False,
        fig_folder: Union[str, Path, None] = None,
        save_name: str = None,
    ):
        fig, ax = plt.subplots()
        if hasattr(self, "segments"):
            for i_seg in range(len(self.segments)):
                ax.plot(self.segments[i_seg, :], c="k", alpha=0.2)
            ax.plot(np.nanmean(self.segments, axis=0), c="r")
            ax.margins(x=0)

            if fig_show:
                plt.show()

            if fig_close:
                plt.close(fig)

            if fig_save:
                if fig_folder is None:
                    fig_folder = Path.cwd()
                if save_name is None:
                    save_name = "plot_segmenter_segments.png"
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
        if hasattr(self, "edges"):
            ax.plot(self.data)
            for edge in self.edges:
                ax.axvline(edge, c="k", ls=":")
            ax.margins(x=0)

            if fig_show:
                plt.show()

            if fig_close:
                plt.close(fig)

            if fig_save:
                if fig_folder is None:
                    fig_folder = Path.cwd()
                if save_name is None:
                    save_name = "plot_segmenter_edges.png"
                plt.savefig(Path(fig_folder, save_name), dpi=200, bbox_inches="tight")

    def save(
        self,
        save_folder: Union[str, Path, None] = None,
        edges_name: str = None,
        segments_name: str = None,
        params_name: str = None,
        save_segments: bool = False,
    ):
        """
        save segments, segmentation indices and Segmenter parameters
        """
        if save_folder is None:
            save_folder = Path.cwd()
        if params_name is None:
            params_name = "segmenter_params.json"
        if edges_name is None:
            edges_name = "segmenter_edges.pkl"
        if segments_name is None:
            segments_name = "segmenter_segments.pkl"

        with open(Path(save_folder, params_name), "w", encoding="utf-8") as f:
            json.dump(self.params, f, ensure_ascii=False, indent=4)

        with open(Path(save_folder, edges_name), "wb") as fp_edges:
            pickle.dump(
                {
                    "params": self.params,
                    "edges": self.edges,
                },
                fp_edges,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        if save_segments:
            with open(Path(save_folder, segments_name), "wb") as fp_segments:
                pickle.dump(
                    {
                        "params": self.params,
                        "segments": self.segments,
                    },
                    fp_segments,
                    protocol=pickle.HIGHEST_PROTOCOL,
                )
