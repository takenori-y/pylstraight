# ------------------------------------------------------------------------ #
# Copyright 2025 Takenori Yoshimura                                        #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ------------------------------------------------------------------------ #

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Sequence

import numpy as np
from scipy import signal
from scipy.interpolate import interp1d, splev, splrep

TAU = 2 * np.pi


def abs2(x: np.ndarray) -> np.ndarray:
    """Return the squared magnitude of the input array.

    Parameters
    ----------
    x : np.ndarray
        The input array.

    Returns
    -------
    out : np.ndarray
        The squared magnitude of the input array.

    """
    return np.square(np.abs(x))


def butter(n: int, cutoff: float, ftype: str = "high") -> tuple[np.ndarray, np.ndarray]:
    """Design an n-th order Butterworth filter.

    Parameters
    ----------
    n : int
        The filter order.

    cutoff : float
        The cutoff frequency of the filter.

    ftype : ['low', 'high']
        The type of the filter.

    Returns
    -------
    b : np.ndarray [shape=(n + 1,)]
        The numerator coefficients of the filter.

    a : np.ndarray [shape=(n + 1,)]
        The denominator coefficients of the filter.

    """
    ftype_to_btype = {"low": "lowpass", "high": "highpass"}
    return signal.butter(n, cutoff, btype=ftype_to_btype[ftype])


def conv(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Convolve the signal x with the impulse response h.

    Parameters
    ----------
    x : np.ndarray [shape=(t,)]
        The input signal.

    h : np.ndarray [shape=(m,)]
        The impulse response.

    Returns
    -------
    out : np.ndarray [shape=(t + m - 1,)]
        The convolved signal.

    """
    return signal.convolve(x, h, mode="full")


def cut(x: np.ndarray, begin: int, length: int) -> np.ndarray:
    """Cut the signal x.

    Parameters
    ----------
    x : np.ndarray
        The input array.

    begin : int
        The beginning of the cut.

    length : int
        The length of the cut.

    Returns
    -------
    out : np.ndarray
        The cut array.

    """
    return x[..., begin : begin + length]


def decimate(x: np.ndarray, r: int) -> np.ndarray:
    """Decimate the signal by a factor of r.

    Parameters
    ----------
    x : np.ndarray [shape=(t,)]
        The input signal.

    r : int
        The decimation factor.

    Returns
    -------
    out : np.ndarray [shape=(t / r,)]
        The decimated signal.

    """

    def _decimate(x: np.ndarray, r: int, n: int) -> np.ndarray:
        b, a = signal.cheby1(n, 0.05, 0.8 / r)
        y = signal.filtfilt(b, a, x, padlen=3 * (len(b) - 1))
        n = len(x)
        s = r - (r * math.ceil(n / r) - n) - 1
        return y[s::r]

    tol = 2 * np.amax(np.abs(x))
    for n in range(8, 0, -1):
        y = _decimate(x, r, n)
        if np.amax(np.abs(y)) <= tol:
            return y

    msg = "Decimation failed."
    raise RuntimeError(msg)


def fftfilt(b: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter the signal x using the FIR filter b in the frequency domain.

    Parameters
    ----------
    b : np.ndarray [shape=(..., m)]
        The FIR filter coefficients.

    x : np.ndarray [shape=(..., t)]
        The input signal.

    Returns
    -------
    out : np.ndarray [shape=(..., t)]
        The filtered signal.

    """
    nb = b.shape[-1]
    nx = x.shape[-1]
    nfft = 2 ** nextpow2(nb + nx - 1)
    B = np.fft.fft(b, n=nfft, axis=-1)
    X = np.fft.fft(x, n=nfft, axis=-1)
    y = np.fft.ifft(X * B)[..., :nx]
    if np.iscomplexobj(b) or np.iscomplexobj(x):
        return y
    return y.real


def fir1(n: int, cutoff: float) -> np.ndarray:
    """Design a low-pass FIR filter using the window method.

    Parameters
    ----------
    n : int
        The filter order.

    cutoff : float
        The cutoff frequency of the filter.

    Returns
    -------
    out : np.ndarray [shape=(n,)]
        The filter coefficients.

    """
    return signal.firwin(n + 1, cutoff, window="hamming")


def hanning(n: int, *, square: bool = False, norm: bool = False) -> np.ndarray:
    """Return the Hanning window.

    Parameters
    ----------
    n : int
        The window length.

    square : bool
        Whether to use the squared Hanning window.

    norm : bool
        Whether to normalize the window.

    Returns
    -------
    out : np.ndarray [shape=(n,)]
        The Hanning window.

    """
    w = np.hanning(n + 2)[1:-1]
    if square:
        w **= 2
    if norm:
        w /= np.sum(w)
    return w


def interp1(
    x: np.ndarray, y: np.ndarray, xq: np.ndarray | int, method: str = "linear"
) -> np.ndarray:
    """Interpolate the signal y at the points xq.

    Parameters
    ----------
    x : np.ndarray [shape=(m,)]
        The sample points.

    y : np.ndarray [shape=(m,)]
        The sample values.

    xq : np.ndarray [shape=(n,)]
        The query points.

    method : ['linear', '*linear']
        The method of interpolation.

    Returns
    -------
    out : np.ndarray [shape=(n,)]
        The interpolated values.

    """
    if method == "linear":
        return interp1d(
            x, y, kind="linear", bounds_error=False, fill_value=(y[..., 0], y[..., -1])
        )(xq)
    if method == "*linear":
        return interp1d(x, y, kind="linear", fill_value="extrapolate")(xq)
    msg = f"Unknown interpolation method: {method}"
    raise ValueError(msg)


def mfilter(b: np.ndarray, a: np.ndarray, x: np.ndarray) -> np.ndarray:
    """Filter the signal x using the IIR filter defined by b and a.

    Parameters
    ----------
    b : np.ndarray [shape=(m,)]
        The numerator coefficients of the filter.

    a : np.ndarray [shape=(n,)]
        The denominator coefficients of the filter.

    x : np.ndarray [shape=(t,)]
        The input signal.

    Returns
    -------
    out : np.ndarray [shape=(t,)]
        The filtered signal.

    """
    return signal.lfilter(b, a, x)


def mrange(start: float, step: float, stop: float) -> np.ndarray:
    """Generate a range of numbers.

    Parameters
    ----------
    start : float
        The start of the range.

    step : float
        The step size.

    stop : float
        The stop of the range.

    Returns
    -------
    out : np.ndarray
        The range of numbers.

    """
    return np.arange(start, stop + 1e-8, step)


def mround(x: np.ndarray | float) -> np.ndarray | int:
    """Round the number to the nearest integer.

    Parameters
    ----------
    x : np.ndarray or float
        The number to be rounded.

    Returns
    -------
    out : np.ndarray or int
        The rounded number.

    """
    out = np.where(x - np.floor(x) < 0.5, np.floor(x), np.ceil(x))
    if isinstance(x, float):
        return int(out)
    return out.astype(np.int64)


def mstd(x: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Compute the standard deviation of the input.

    Parameters
    ----------
    x : np.ndarray
        The input.

    ddof : int
        The delta degrees of freedom.

    Returns
    -------
    out : np.ndarray
        The standard deviation.

    """
    return np.std(x, ddof=ddof)


def nextpow2(x: float) -> int:
    """Return the smallest power of 2 that is greater than or equal to x.

    Parameters
    ----------
    x : float
        The number.

    Returns
    -------
    out : int
        The smallest power of 2.

    """
    return np.ceil(np.log2(x)).astype(np.int64)


def spline(x: np.ndarray, y: np.ndarray, xq: np.ndarray) -> np.ndarray:
    """Interpolate the signal y at the points xq using a spline.

    Parameters
    ----------
    x : np.ndarray [shape=(m,)]
        The sample points.

    y : np.ndarray [shape=(m,)]
        The sample values.

    xq : np.ndarray [shape=(n,)]
        The query points.

    Returns
    -------
    out : np.ndarray [shape=(n,)]
        The interpolated values.

    """
    spl = splrep(x, y)
    return splev(xq, spl)


def randn(shape: int | Sequence[int], scale: float = 1) -> np.ndarray:
    """Generate random numbers from a standard normal distribution.

    Parameters
    ----------
    shape : int | Sequence[int]
        The shape of the output samples.

    scale : float
        The standard deviation of the distribution.

    Returns
    -------
    out : np.ndarray
        The generated samples.

    """
    rng = np.random.default_rng()
    return rng.standard_normal(shape) * scale


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Compute the sigmoid function.

    Parameters
    ----------
    x : np.ndarray
        The input.

    Returns
    -------
    out : np.ndarray
        The output.

    """
    return 1 / (1 + np.exp(-x))
