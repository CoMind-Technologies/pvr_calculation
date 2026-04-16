import scipy


def integrate_from_min_lag_index(target, x_axis, index=0):
    """
    Integrate the jacobian from the minimum lag index to the end of the lags.
    """
    return scipy.integrate.simpson(target[..., index:], x=x_axis[index:], axis=-1)


def map_col(x, index, block_size=1000):
    if x.startswith("g1"):
        return f'g1_{int(x.replace("g1_", "")) + index * block_size}'
    else:
        return x
