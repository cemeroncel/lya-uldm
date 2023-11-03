"""Various routines used throught the package."""
from scipy.interpolate import CubicSpline
import numpy as np


def log_cubic_spline(x, y):
    logx = np.log(x)
    logy = np.log(y)
    lin_interp = CubicSpline(logx, logy)

    def log_interp(z):
        return np.exp(lin_interp(np.log(z)))

    return log_interp
