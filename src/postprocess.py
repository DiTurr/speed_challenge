"""

"""
import numpy as np


def low_pass_filter(x, y_init=None, fps=30, time_constant=1):
    """

    """
    # parameters of the low pass filter
    lowpass_cutoff = 1 / time_constant
    dt = 1 / fps
    rc = 1 / (2 * np.pi * lowpass_cutoff)
    alpha = dt / (rc + dt)
    # create output array
    y = np.zeros_like(x)
    if y_init is None:
        y[0] = x[0]
    else:
        y[0] = y_init
    for idx in range(1, x.shape[0]):
        y[idx] = x[idx] * alpha + (1 - alpha) * y[idx - 1]
    # return output
    return y
