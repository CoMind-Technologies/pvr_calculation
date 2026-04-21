from typing import Optional, Union

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

# from comind_utils.dsp.segmentation.segmenter import SegmentFromOmnibus
from pvr_calculation.utils.segmenter_utils import batch, overlaps


def pvr(data: NDArray[np.number], within_axis: int, across_axis: int):
    """
    pulse variance ratio
    """

    # keepdims to ensure axes stay in the same position after applying
    # var & mean;
    # squeeze to drop the reduced within_axis / across_axis dimensions &
    # thus avoid broadcasting
    # specifying arguments in squeeze to preserve singleton dimensions in the
    # input
    return 1 - np.squeeze(
        np.mean(
            np.var(data, axis=across_axis, keepdims=True),
            axis=within_axis,
            keepdims=True,
        ),
        axis=(across_axis, within_axis),
    ) / np.squeeze(
        np.mean(
            np.var(data, axis=within_axis, keepdims=True),
            axis=across_axis,
            keepdims=True,
        ),
        axis=(across_axis, within_axis),
    )

def batch_pvr(
    data: NDArray[np.number], batch_size: int, within_axis: int, across_axis: int
):
    """
    Compute PVR with a fixed batch size (fixed number of pulses per
    PVR estimate).
    """

    # ensure 0 <= axis < n_dim
    within_axis = within_axis % data.ndim
    across_axis = across_axis % data.ndim

    if within_axis == across_axis:
        raise ValueError("within and across axes should be different")

    data = batch(data, batch_size, across_axis)

    # compute pvr
    # new position of within_axis after batching
    if within_axis < across_axis:
        within_axis += 2
    else:
        within_axis += 1

    return pvr(data, within_axis, 1)
