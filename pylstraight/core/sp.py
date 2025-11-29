# ------------------------------------------------------------------------ #
# Copyright 2025 Takenori Yoshimura                                        #
# Copyright 2018 Hideki Kawahara (Original Author)                         #
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

from dataclasses import dataclass

import numpy as np

from .utils.mat import (
    abs2,
    butter,
    conv,
    cut,
    fftfilt,
    hanning,
    interp1,
    mfilter,
    mrange,
    mround,
    mstd,
    randn,
    sigmoid,
    spline,
)
from .utils.misc import get_fft_length, is_debug_mode


@dataclass
class SpParam:
    """Control parameters for the spectrum extraction algorithm."""

    default_frame_length: float = 80.0
    spectral_update_interval: float = 1.0
    spectral_time_window_stretch: float = 1.0
    spectral_exponent_for_nonlinearity: float = 0.6
    spectral_time_domain_compensation: float = 0.2
    f0_frame_update_interval: float = 1.0


def exstraightspec(
    x: np.ndarray,
    f0raw: np.ndarray,
    fs: int,
    prm: SpParam | None = None,
) -> np.ndarray:
    """Perform spectral information extraction.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    f0raw : np.ndarray [shape=(nframe,)]
        Fundamental frequency in Hz.

    fs : int
        Sampling frequency in Hz.

    prm : SpParam or None
        Control parameters.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        Spectrogram.

    """
    if prm is None:
        prm = SpParam()  # pragma: no cover

    # Set parameters.
    framem = prm.default_frame_length
    shiftm = prm.spectral_update_interval
    eta = prm.spectral_time_window_stretch
    pc = prm.spectral_exponent_for_nonlinearity
    mag = prm.spectral_time_domain_compensation

    fftl = get_fft_length(fs, framem, "full")

    # Perform spectral estimation.
    xamp = mstd(x)
    scaleconst = 2200
    xold = x if xamp == 0 else x / xamp * scaleconst
    n2sgrambk, _ = straightBodyC03ma(xold, fs, shiftm, framem, fftl, f0raw, eta, pc)
    n3sgram = specreshape(fs, n2sgrambk, eta, pc, mag, f0raw) if 0 < mag else n2sgrambk
    if xamp != 0:
        n3sgram *= xamp / scaleconst
    return n3sgram


def straightBodyC03ma(
    x: np.ndarray,
    fs: int,
    shiftm: float,
    framem: float,
    fftl: int,
    f0raw: np.ndarray,
    eta: float,
    pc: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Perform spectral information extraction.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    shiftm : float
        Frame shift in msec.

    framem : float
        Window length in msec.

    fftl : int
        Length of FFT.

    f0raw : np.ndarray [shape=(nframe,)]
        Fundamental frequency in Hz.

    eta : float
        Time window stretching factor.

    pc : float
        Exponent for nonlinearity.

    Returns
    -------
    n2sgram : np.ndarray [shape=(nframe, nfreq)]
        Smoothed spectrogram.

    nsgram : np.ndarray [shape=(nframe, nfreq)]
        Isometric spectrogram.

    """
    f0l = f0raw
    framel = mround(framem * fs / 1000)
    shiftl = mround(shiftm * fs / 1000)

    b, a = butter(6, 70 / fs * 2)
    xh = mfilter(b, a, x)
    rmsp = mstd(xh)

    b, a = butter(6, 300 / fs * 2)
    xh2 = mfilter(b, a, x)

    b, a = butter(6, 3000 / fs * 2)
    xhh = mfilter(b, a, x)

    rmsp = 0 if is_debug_mode() else mstd(xh)
    tx = np.concatenate(
        [
            randn(mround(framel / 2), rmsp / 4000),
            xh,
            randn(framel, rmsp / 4000),
        ]
    )
    nframe = min(len(f0l), mround(len(x) / shiftl))

    tt = (np.arange(framel) + 1 - framel / 2) / fs
    d = fftl // 2 + 1

    cfv = np.asarray([0.36, 0.30, 0.26, 0.21, 0.17, 0.14, 0.10])
    muv = np.asarray([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6])
    bcf = spline(muv, cfv, eta)

    ovc = optimumsmoothing(eta, pc)

    fNominal = 40
    wGaussian = np.exp(-np.pi * (tt * fNominal / eta) ** 2)
    wSynchronousBartlett = np.maximum(0, 1 - np.abs(tt * fNominal))
    wPSGSeed = fftfilt(
        wSynchronousBartlett[0 < wSynchronousBartlett], np.pad(wGaussian, (0, len(tt)))
    )
    wPSGSeed /= np.max(wPSGSeed)
    maxLocation = np.argmax(wPSGSeed)
    tNominal = (np.arange(len(wPSGSeed)) - maxLocation) / fs

    ttm = np.concatenate([[1e-5], np.arange(1, d), np.arange(-d + 2, 0)]) / fs
    lft = sigmoid((np.abs(np.arange(fftl) - fftl // 2) - fftl / 30) / 2)

    f0 = f0l[:nframe].copy()
    f0[f0 == 0] = 160
    ttf = tt * f0[:, None]

    wxe = interp1(tNominal, wPSGSeed, ttf / fNominal, method="*linear")
    wxe /= np.sqrt(np.sum(wxe**2, axis=1, keepdims=True))
    wxd = bcf * wxe * np.sin(np.pi * ttf)

    txiix = [tx[i : i + framel] for i in range(0, len(tx) - framel // 2, shiftl)]
    txiix = np.stack(txiix[:nframe])
    txiix -= np.mean(txiix, axis=1, keepdims=True)
    pw = (
        abs2(np.fft.fft(txiix * wxe, n=fftl)) + abs2(np.fft.fft(txiix * wxd, n=fftl))
    ) ** (pc / 2)
    nsgram = pw[:, :d].copy()

    f0pr = f0 * (fftl / fs) + 1
    f0p = np.ceil(f0pr).astype(int)
    f0p2 = np.floor((f0pr + 1) / 2).astype(int)

    f0pm = np.max(f0p)
    pwx = np.arange(1, f0pm + 1)
    f0p2m = np.max(f0p2)
    pwxq = f0pr[:, None] - np.arange(f0p2m)
    for ii in range(nframe):
        tmppw = interp1(pwx, pw[ii, :f0pm], pwxq[ii])
        pw[ii, : f0p2[ii]] = tmppw[: f0p2[ii]]
        pw[ii, -(f0p2[ii] - 1) :] = np.flip(pw[ii, 1 : f0p2[ii]])

    ttmf = np.pi * ttm * f0[:, None]
    ww2t = (np.sin(3 * ttmf) / (3 * ttmf)) ** 2
    spw2 = np.real(np.fft.ifft(ww2t * np.fft.fft(pw) * lft))
    spw2[spw2 == 0] = 1e-10

    wwt = (np.sin(ttmf) / ttmf) ** 2
    wwt *= ovc[0] + ovc[1] * 2 * np.cos(2 * ttmf) + ovc[2] * 2 * np.cos(4 * ttmf)
    spw = np.real(np.fft.ifft(wwt * np.fft.fft(pw / spw2))) / wwt[:, :1]
    n2sgram = spw2[:, :d] * (
        0.175 * np.log(2 * np.cosh(4 / 1.4 * spw[:, :d])) + 0.5 * spw[:, :d]
    )

    nsgram **= 1 / pc
    n2sgram **= 2 / pc

    ttlv = np.sum(n2sgram)
    ncw = mround(2 * fs / 1000)
    lbb = mround(300 / fs * fftl) - 1
    h3 = conv(hanning(ncw // 2), np.exp(-1400 / fs * np.arange(2 * ncw + 1)))
    pwc = fftfilt(h3, abs2(np.pad(xh2, (0, 10 * ncw))))
    pwc = pwc[mround(mrange(0, shiftm * fs / 1000, len(pwc) - 1))]
    pwc = pwc[:nframe]
    pwc *= np.sum(n2sgram[:, lbb:]) / (np.sum(pwc) + 1e-10)

    pwch = fftfilt(h3, abs2(np.pad(xhh, (0, 10 * ncw))))
    pwch = pwch[mround(mrange(0, shiftm * fs / 1000, len(pwch) - 1))]
    pwch = pwch[:nframe]
    pwch *= ttlv / (np.sum(pwch) + 1e-10)

    ipwm = 7
    ipl = mround(ipwm / shiftm)
    ww = hanning(ipl * 2 + 1, norm=True)
    apwt = fftfilt(ww, np.pad(pwch, (0, len(ww) * 2)))
    apwt = cut(apwt, ipl, len(pwch))
    dpwt = fftfilt(ww, np.pad(np.diff(pwch) ** 2, (0, len(ww) * 2)))
    dpwt = np.maximum(0, cut(dpwt, ipl, len(pwch)))
    mmaa = np.max(apwt)
    apwt[apwt <= 0] = mmaa
    rr = np.sqrt(dpwt) / (apwt + 1e-10)
    lmbd = sigmoid((np.sqrt(rr) - 0.75) * 20)

    pwc = lmbd * pwc + (1 - lmbd) * np.sum(n2sgram, axis=1)

    mask = f0raw[:nframe] == 0
    n2sgram[mask] *= pwc[mask][:, None] / np.sum(n2sgram[mask], axis=1, keepdims=True)
    n2sgram = np.sqrt(np.abs(n2sgram + 1e-10))

    return n2sgram, nsgram


def optimumsmoothing(eta: float, pc: float) -> np.ndarray:
    """Calculate the optimum smoothing factor.

    Parameters
    ----------
    eta : float
        Time window stretching factor.

    pc : float
        Exponent for nonlinearity.

    Returns
    -------
    out : np.ndarray [shape=(4,)]
        Coefficients for the 2nd-order cardinal B-spline.

    """
    fx = mrange(-8, 0.05, 8)
    cb = np.maximum(0, 1 - np.abs(fx))
    gw = np.exp(-np.pi * (fx * eta * 1.4) ** 2) ** pc
    cmw = conv(cb, gw)
    cmw = cut(cmw, (len(cb) - 1) // 2, len(cb)) / np.max(cmw)
    ss = (np.abs(fx - mround(fx)) < 0.025) * np.arange(1, len(fx) + 1)
    ss = ss[0 < ss]
    cmws = cmw[ss - 1]

    nn = len(cmws)
    idv = np.arange(nn)

    hh = np.zeros((2 * nn, nn))
    rows = idv[:, None] + idv
    cols = idv
    hh[rows, cols] = cmws[:, None]
    h = hh.T @ hh
    ov = np.linalg.solve(h, hh[nn])
    return cut(ov, (nn + 1) // 2, 4)


def specreshape(
    fs: int, n2sgram: np.ndarray, eta: float, pc: float, mag: float, f0: np.ndarray
) -> np.ndarray:
    """Perform spectral compensation using time domain compensation.

    Parameters
    ----------
    fs : int
        Sampling frequency in Hz.

    n2sgram : np.ndarray [shape=(nframe, nfreq)]
        Straight smoothed spectrogram.

    eta : float
        Temporal stretching factor.

    pc : float
        Power exponent for nonlinearity.

    mag : float
        Magnification factor.

    f0 : np.ndarray [shape=(nframe,)]
        F0 contour in Hz.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        Compensated spectrogram.

    """
    mm, nn = n2sgram.shape
    fftl = (nn - 1) * 2

    ovc = optimumsmoothing(eta, pc)
    hh = np.array(
        [
            [1, 1, 1, 1],
            [0, 1 / 2, 2 / 3, 3 / 4],
            [0, 0, 1 / 3, 2 / 4],
            [0, 0, 0, 1 / 4],
        ]
    )
    bb = np.linalg.solve(hh, ovc)
    cc = np.array([1, 4, 9, 16])
    tt = np.arange(fftl) / fs
    pb2 = (np.pi / eta**2 + np.pi**2 / 3 * np.sum(bb * cc)) * tt**2

    ffs = np.pad(n2sgram, ((0, 0), (0, nn - 2)), mode="reflect")
    ccs2 = np.real(np.fft.fft(ffs)) * np.minimum(
        20, (1 + mag * pb2 * f0[:mm, None] ** 2)
    )
    ccs2[:, nn:] = np.flip(ccs2[:, 1 : nn - 1], axis=1)
    n2sgram3 = np.real(np.fft.ihfft(ccs2))
    return (np.abs(n2sgram3) + n2sgram3) / 2 + 0.1
