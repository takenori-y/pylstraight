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
from dataclasses import dataclass

import numpy as np

from .f0 import zrefineF06m
from .utils.mat import (
    TAU,
    cut,
    decimate,
    fftfilt,
    hanning,
    interp1,
    mrange,
    mround,
    randn,
    sigmoid,
)
from .utils.misc import get_fft_length, is_debug_mode


@dataclass
class ApParam:
    """Control parameters for the aperiodicity extraction algorithm."""

    default_window_length: float = 80.0
    periodicity_frame_update_interval: float = 1.0
    f0_search_upper_bound: float = 800.0
    f0_frame_update_interval: float = 1.0
    refine_fft_length: int = 1024
    refine_time_stretching_factor: float = 1.1
    refine_number_of_harmonic_component: int = 3


def exstraightAPind(
    x: np.ndarray,
    fs: int,
    f0: np.ndarray,
    ecr: np.ndarray | None = None,
    prm: ApParam | None = None,
) -> np.ndarray:
    """Perform aperiodicity index extraction.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    f0 : np.ndarray [shape=(nframe,)]
        Fundamental frequency in Hz.

    ecr : np.ndarray [shape=(nframe,)] or None
        C/N.

    prm : ApParam or None
        Control parameters.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        Amount of aperiodic component in dB.

    """
    if prm is None:
        prm = ApParam()  # pragma: no cover

    # Set parameters.
    framem = prm.default_window_length
    f0shiftm = prm.f0_frame_update_interval
    iPeriodicityInterval = prm.periodicity_frame_update_interval

    if ecr is None:
        f0ceil = prm.f0_search_upper_bound
        fftlf0r = prm.refine_fft_length
        tstretch = prm.refine_time_stretching_factor
        nhmx = prm.refine_number_of_harmonic_component
        if is_debug_mode():
            dn = max(1, math.floor(fs / (3 * 2 * 400)))
            y = np.fromfile("tests/reference/data.dec.3", dtype=np.float64)
        else:
            dn = max(1, math.floor(fs / (3 * 2 * f0ceil)))
            y = decimate(x, dn)
        _, ecr, _ = zrefineF06m(y, fs / dn, f0, fftlf0r, tstretch, nhmx, f0shiftm)

    fftl = get_fft_length(fs, framem, "full")
    apvq, dpvq, _, _ = aperiodicpartERB2(
        x, fs, f0, f0shiftm, iPeriodicityInterval, fftl // 2 + 1
    )
    apv = 10 * np.log10(apvq)
    dpv = 10 * np.log10(dpvq)

    dpv = correctdpv(apv, dpv, iPeriodicityInterval, f0, ecr, f0shiftm, fs)

    return aperiodiccomp(apv, dpv, iPeriodicityInterval, f0, f0shiftm)


def aperiodicpartERB2(
    x: np.ndarray,
    fs: int,
    f0raw: np.ndarray,
    shiftm: float,
    intshiftm: float,
    mm: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Estimate relative aperiodic energy with ERB smoothing.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    f0raw : np.ndarray [shape=(nframe,)]
        Fundamental frequency in Hz.

    shiftm : float
        Frame shift in ms for input F0 data.

    intshiftm : float
        Frame shift in ms for internal processing.

    mm : int
        Length of frequency axis.

    Returns
    -------
    apv : np.ndarray [shape=(nframe, nfreq)]
        Smoothed upper power envelope on linear scale.

    dpv : np.ndarray [shape=(nframe, nfreq)]
        Smoothed lower power envelope on linear scale.

    apve : np.ndarray [shape=(nframe, nerb)]
        Smoothed upper power envelope on ERB scale.

    dpve : np.ndarray [shape=(nframe, nerb)]
        Smoothed lower power envelope on ERB scale.

    """
    f0 = f0raw.copy()
    avf0 = 180 if len(f0[0 < f0]) == 0 else np.mean(f0[0 < f0])

    lowerF0limit = 40
    fftl = 2 ** math.ceil(np.log2(6.7 * fs / lowerF0limit) + 1)

    f0[np.isnan(f0)] = avf0
    f0[f0 == 0] = avf0
    f0[f0 < lowerF0limit] = lowerF0limit

    t0 = np.arange(len(x)) / fs
    f0i = interp1(
        np.arange(len(f0) + 3) * shiftm / 1000, np.pad(f0, (0, 3), mode="edge"), t0
    )
    phr = np.cumsum(TAU * f0 * shiftm / 1000)
    phri = interp1(
        np.arange(len(phr)), phr, np.arange(len(x)) / (len(x) - 1) * (len(phr) - 1)
    )
    phc = mrange(phr[0], TAU * 40 / fs, phr[-1])
    xi = interp1(phri, x, phc)
    f0ii = interp1(phri, f0i, phc)
    ti = interp1(phri, t0, phc)
    tidx = interp1(ti, np.arange(len(ti)), mrange(0, intshiftm / 1000, ti[-1])) + 1

    fftl2 = fftl // 2
    fxa = np.arange(mm) / (mm - 1) * fs / 2
    fxfi = np.arange(fftl2 + 1) / fftl * fs

    xii = np.pad(xi, (fftl, fftl))
    if is_debug_mode():
        xii += np.fromfile("tests/reference/data.rand", dtype=np.float64)
    else:
        xii += randn(len(xii), max(1e-5, np.max(np.abs(xii))) * 1e-5)

    bb = np.arange(fftl) - fftl2 + 1
    tt = bb / fs
    w = np.exp(-np.pi * (tt * 40) ** 2)
    wb = np.maximum(0, 1 - np.abs(tt * 40 / 2))
    wb = wb[0 < wb]
    wcc = fftfilt(wb, np.pad(w, (fftl, fftl)))
    wcc /= np.max(wcc)
    mxp = np.argmax(wcc)
    wcc -= wcc[0]
    wcc /= np.sum(wcc)
    ww = wcc[bb + mxp]

    qx = np.arange(fftl) / fs
    lft = sigmoid(-(qx - 1.4 / 40) * 1000)
    lft[fftl2 + 1 :] = np.flip(lft[1:fftl2])

    evv = np.arange(mm) / (mm - 1) * HzToErbRate(fs / 2)
    eew = 1
    lh = mround(2 * eew / evv[1])
    we = hanning(lh, norm=True)
    hvv: np.ndarray = ErbRateToHz(evv)
    hvv[0] = 0
    hvv[-1] = fs / 2
    evx = mrange(0, 0.5, evv[-1])

    bss = (slice(None), slice(0, fftl2 - 1))
    bss2 = (slice(None), slice(0, fftl2))
    bss21 = (slice(None), slice(1, fftl2 + 1))

    bias = fftl - 1
    idp = mround(tidx) + bias
    sw = np.abs(np.fft.fft(xii[idp[:, None] + bb] * ww))
    sws = (sw * 2 + np.roll(sw, -1, axis=-1) + np.roll(sw, 1, axis=-1)) / 4
    sms = (
        np.real(np.fft.ihfft(np.real(np.fft.fft(np.log(sws))) * lft)) / np.log(10) * 20
    )
    plits1 = (np.diff(sms[bss2], axis=-1) * np.diff(sms[bss21], axis=-1)) < 0
    plits2 = sms[bss] * (0 < np.diff(sms[bss2], axis=-1))
    plits = np.pad(plits1 * plits2, ((0, 0), (1, 1)))
    dlits1 = plits1
    dlits2 = sms[bss] * (np.diff(sms[bss2], axis=-1) < 0)
    dlits = np.pad(dlits1 * dlits2, ((0, 0), (1, 1)))

    ape = np.empty((len(tidx), mm))
    dpe = np.empty((len(tidx), mm))

    for ii in range(len(tidx)):
        gg = fxfi[0 < np.abs(plits[ii])]
        gfg = sms[ii, 0 < np.abs(plits[ii])]
        dd = fxfi[0 < np.abs(dlits[ii])]
        dfd = sms[ii, 0 < np.abs(dlits[ii])]
        jj = min(len(f0ii) - 1, mround(tidx[ii]))
        gg = np.pad(gg, (1, 1))
        gg[-1] = fs / 2
        gga = gg * (f0ii[jj] / 40)
        dd = np.pad(dd, (1, 1))
        dd[-1] = fs / 2
        dda = dd * (f0ii[jj] / 40)
        gfga = np.pad(gfg, (1, 1), mode="edge")
        dfda = np.pad(dfd, (1, 1), mode="edge")
        gfgap = 10 ** (gfga / 10)
        dfgap = 10 ** (dfda / 10)
        ape[ii] = interp1(HzToErbRate(gga), gfgap, evv)
        dpe[ii] = interp1(HzToErbRate(dda), dfgap, evv)

    apef = np.pad(ape, ((0, 0), (lh - 1, lh - 1)), mode="reflect")
    dpef = np.pad(dpe, ((0, 0), (lh - 1, lh - 1)), mode="reflect")
    apefs = fftfilt(we, apef)
    dpefs = fftfilt(we, dpef)
    apefs = cut(apefs, lh - 1 + mround(lh / 2), len(evv))
    dpefs = cut(dpefs, lh - 1 + mround(lh / 2), len(evv))

    apv = interp1(hvv, apefs, fxa)
    dpv = interp1(hvv, dpefs, fxa)
    apve = interp1(evv, apefs, evx)
    dpve = interp1(evv, dpefs, evx)

    return apv, dpv, apve, dpve


def HzToErbRate(x: np.ndarray | float) -> np.ndarray | float:
    """Convert frequency in Hz to ERB rate.

    Parameters
    ----------
    x : np.ndarray or float
        Frequency in Hz.

    Returns
    -------
    out : np.ndarray or float
        ERB rate.

    """
    return 21.4 * np.log10(4.37e-3 * x + 1)


def ErbRateToHz(x: np.ndarray | float) -> np.ndarray | float:
    """Convert ERB rate to frequency in Hz.

    Parameters
    ----------
    x : np.ndarray or float
        ERB rate.

    Returns
    -------
    out : np.ndarray or float
        Frequency in Hz.

    """
    if is_debug_mode():
        return (10 ** (0.0467 * x) - 1) * 228.8
    return (10 ** (x / 21.4) - 1) / 4.37e-3


def correctdpv(
    apv: np.ndarray,
    dpv: np.ndarray,
    shiftap: float,
    f0raw: np.ndarray,
    ecrt: np.ndarray,
    shiftm: float,
    fs: int,
) -> np.ndarray:
    """Perform apperiodicity correction based on C/N estimation.

    Parameters
    ----------
    apv : np.ndarray [shape=(nframe, nfreq)]
        Lower spectral envelope.

    dpv : np.ndarray [shape=(nframe, nfreq)]
        Upper spectral envelope.

    shiftap : float
        Frame shift for the envelopes.

    f0raw : np.ndarray [shape=(nframe',)]
        F0 contour.

    ecrt : np.ndarray [shape=(nframe',)]
        C/N ratio.

    shiftm : float
        Frame shift for the F0.

    fs : int
        Sampling frequency.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        Corrected spectral envelope.

    """
    mm, nn = apv.shape
    nf0 = len(f0raw)
    fx = np.arange(nn) / (nn - 1) * fs / 2
    f0 = f0raw.copy()
    mask = f0 == 0
    f0[mask] = 40

    # codespell:ignore-begin
    iif = np.minimum(nf0 - 1, mround(np.arange(mm) * shiftap / shiftm))
    f0i = f0[iif][:, None]
    ecri = 1 / ecrt[iif][:, None]
    # codespell:ignore-end

    bdr = sigmoid((fx - 2.5 * f0i) / f0i * 4)
    bdr = (bdr + ecri) / (1 + ecri)
    out = np.empty_like(dpv)
    if nf0 < mm:
        mask = np.pad(mask, (0, mm - nf0), constant_values=True)
    elif mm < nf0:
        mask = mask[:mm]
    out[~mask] = np.minimum(dpv[~mask], apv[~mask] + 20 * np.log10(bdr[~mask]))
    out[mask] = dpv[mask]
    return out


def aperiodiccomp(
    apv: np.ndarray, dpv: np.ndarray, ashift: float, f0: np.ndarray, nshift: float
) -> np.ndarray:
    """Match the length of aperiodicity with the length of f0.

    Parameters
    ----------
    apv : np.ndarray [shape=(nframe, nfreq)]
        Upper spectral envelope.

    dpv : np.ndarray [shape=(nframe, nfreq)]
        Lower spectral envelope.

    ashift : float
        Frame shift for the aperiodicity.

    f0 : np.ndarray [shape=(nframe',)]
        F0 contour.

    nshift : float
        Frame shift for the F0.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        Aperiodicity index.

    """
    mm = len(f0)
    m2 = len(apv)
    x = np.arange(m2) * ashift
    xi = np.arange(mm) * nshift
    xi = np.minimum(x[-1], xi)
    return interp1(x, (dpv - apv).T, xi, method="*linear").T
