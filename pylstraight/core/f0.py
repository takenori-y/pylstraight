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
from types import SimpleNamespace

import numpy as np

from .utils.mat import (
    TAU,
    conv,
    cut,
    decimate,
    fftfilt,
    fir1,
    hanning,
    interp1,
    mrange,
    mround,
    mstd,
    randn,
    sigmoid,
)
from .utils.misc import is_debug_mode


@dataclass
class F0Param:
    """Control parameters for the F0 extraction algorithm."""

    f0_search_lower_bound: float = 40.0
    f0_search_upper_bound: float = 800.0
    f0_frame_update_interval: float = 1.0
    number_of_channels_in_octave: int = 24
    if_window_stretch: float = 1.2
    if_smoothing_length_relative_to_fc: float = 1.0
    if_minimum_smoothing_length: float = 5.0
    if_exponent_for_nonlinear_sum: float = 0.5
    if_number_of_harmonic_for_initial_estimate: int = 1
    ac_time_window_length: int = 60
    ac_number_of_frequency_segments: int = 8
    ac_frequency_domain_window_width: int = 2200
    ac_power_exponent_for_nonlinearity: float = 0.5
    ac_amplitude_compensation_in_short_lag: float = 1.6
    ac_exponent_for_ac_distance: float = 4.0
    ac_lag_smoothing_length: float = 0.0001
    ac_temporal_smoothing_length: float = 20.0
    weight_for_autocorrelation_map: float = 1.0
    weight_for_instantaneous_frequency_map: float = 1.0
    time_constant_for_power_calculation: float = 10.0
    sd_for_normalize_mixing_distance: float = 0.3
    sd_for_tracking_normalization: float = 0.2
    maximum_permissible_octave_jump: float = 0.4
    refine_fft_length: int = 1024
    refine_time_stretching_factor: float = 1.1
    refine_number_of_harmonic_component: int = 3


def _log_ratio(a: np.ndarray | float, b: np.ndarray | float) -> np.ndarray | float:
    """Compute the logarithm of the ratio of two numbers.

    Parameters
    ----------
    a : np.ndarray or float
        a.

    b : np.ndarray or float
        b.

    Returns
    -------
    out : np.ndarray or float
        log2(a / b).

    """
    return np.log2(a) - np.log2(b)


def SourceInfobyMultiCues050111(
    x: np.ndarray,
    fs: int,
    prm: F0Param | None = None,
) -> tuple[np.ndarray, np.ndarray, SimpleNamespace]:
    """Perform source information extraction using multiple cues.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    prm : F0Param or None
        Control parameters.

    Returns
    -------
    f0raw : np.ndarray [shape=(nframe,)]
        Fundamental frequency in Hz.

    vuv : np.ndarray [shape=(nframe,)]
        Voiced/unvoiced flag.

    auxouts : SimpleNamespace
        Auxiliary outputs.

    """
    if prm is None:
        prm = F0Param()  # pragma: no cover

    # Check the length of the input signal.
    minlen = 30
    if len(x) < minlen:
        x = np.pad(x, (0, minlen - len(x)))

    # Set parameters.
    f0floor = prm.f0_search_lower_bound
    f0ceil = prm.f0_search_upper_bound
    shiftm = prm.f0_frame_update_interval
    nvo = prm.number_of_channels_in_octave
    mu = prm.if_window_stretch
    smp = prm.if_smoothing_length_relative_to_fc
    minm = prm.if_minimum_smoothing_length
    pcIF = prm.if_exponent_for_nonlinear_sum
    ncIF = prm.if_number_of_harmonic_for_initial_estimate
    nvc = math.ceil(_log_ratio(f0ceil, f0floor) * nvo)

    # Select multiple F0 candidates based on instantaneous frequency.
    f0v, vrv, _, _ = zfixpF0VexMltpBG4(
        x, fs, f0floor, nvc, nvo, mu, shiftm, smp, minm, pcIF, ncIF
    )
    if f0v.shape[1] == 0:
        return np.zeros(f0v.shape[0]), np.zeros(f0v.shape[0]), SimpleNamespace()
    val, pos = zmultiCandIF(f0v, vrv)
    y, ind, _ = zremoveACinduction(x, fs, pos)
    if ind == 1:
        x = y
        f0v, vrv, _, _ = zfixpF0VexMltpBG4(
            x, fs, f0floor, nvc, nvo, mu, shiftm, smp, minm, pcIF, ncIF
        )
        val, pos = zmultiCandIF(f0v, vrv)

    # Set parameters.
    wtlm = prm.ac_time_window_length
    ndiv = prm.ac_number_of_frequency_segments
    wflf = prm.ac_frequency_domain_window_width
    pcAC = prm.ac_power_exponent_for_nonlinearity
    ampAC = prm.ac_amplitude_compensation_in_short_lag
    betaAC = prm.ac_exponent_for_ac_distance
    lagslAC = prm.ac_lag_smoothing_length
    timeslAC = prm.ac_temporal_smoothing_length
    dn = max(1, math.floor(fs / max(8000, 3 * 2 * f0ceil)))

    # Select multiple F0 candidates based on modified autocorrelation.
    if is_debug_mode():
        y = np.fromfile("tests/reference/data.dec.1", dtype=np.float64)
    else:
        y = decimate(x, dn)
    lagspec, lx = zlagspectestnormal(
        y, fs / dn, shiftm, len(x) / fs * 1000, shiftm, wtlm, ndiv, wflf, pcAC, ampAC
    )
    if len(lagspec) == 0:
        msg = "Failed to extract F0."
        raise RuntimeError(msg)
    f02, pl2 = zmultiCandAC(lx, lagspec, betaAC, lagslAC, timeslAC)

    # Set parameters.
    wAC = prm.weight_for_autocorrelation_map
    wIF = prm.weight_for_instantaneous_frequency_map
    mixsd = prm.sd_for_normalize_mixing_distance
    tcpower = prm.time_constant_for_power_calculation
    f0jumpt = prm.maximum_permissible_octave_jump
    nsdt = prm.sd_for_tracking_normalization

    # Combine multiple source information with dynamic range normalization.
    f0cand, relv = zcombineRanking4(
        pos, val, f02, pl2, nvo, mixsd, wAC, wIF, f0floor, f0ceil
    )
    pws = zVpowercalc(x, fs, tcpower, shiftm, 2000)
    pwsdb = 10 * np.log10(np.abs(pws) + 1e-11)
    noiselevel = calc_noise_level(pwsdb)

    # Perform F0 tracking.
    f0s, rels, csegs = zcontiguousSegment10(pwsdb, f0cand, relv, shiftm, f0jumpt, nsdt)
    f0raw0 = zfillf0gaps6(f0s, csegs, f0cand, relv, pwsdb, f0jumpt, nsdt, noiselevel)

    # F0 refinement using harmonic components.
    fftlf0r = prm.refine_fft_length
    tstretch = prm.refine_time_stretching_factor
    nhmx = prm.refine_number_of_harmonic_component
    f0raw0 = np.nan_to_num(f0raw0, nan=0)
    f0raw0[f0ceil < f0raw0] = f0ceil
    f0raw0[(0 < f0raw0) & (f0raw0 < f0floor)] = f0floor
    f0raw2, ecr, _ = zrefineF06m(y, fs / dn, f0raw0, fftlf0r, tstretch, nhmx, shiftm)
    vuv = zvuvdecision4(f0raw2, rels, pwsdb, shiftm, noiselevel)
    nnll = min(len(f0raw2), len(vuv))

    auxouts = SimpleNamespace(refined_cn=ecr)

    return f0raw2[:nnll], vuv[:nnll], auxouts


def zfixpF0VexMltpBG4(
    x: np.ndarray,
    fs: int,
    f0floor: float,
    nvc: int,
    nvo: int,
    mu: float,
    shiftm: float,
    smp: float,
    minm: float,
    pc: float,
    nc: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Perform fixed point analysis to extract F0.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    f0floor : float
        Lower bound of F0 search in Hz.

    nvc : int
        Total number of filter channels.

    nvo : int
        Number of channels per octave.

    mu : float
        Temporal stretching factor.

    shiftm : float
        Frame shift in msec.

    smp : float
        Relative smoothing length.

    minm : float
        Minimum smoothing length in msec.

    pc : float
        Exponent to represent nonlinear summation.

    nc : int
        Number of harmonic component.

    Returns
    -------
    f0v : np.ndarray [shape=(nframe, nq)]
        Fixed point frequencies.

    vrv : np.ndarray [shape=(nframe, nq)]
        Fixed point reliability.

    dfv : np.ndarray [shape=(nframe, nq)]
        Third intermediates.

    aav : np.ndarray [shape=(nframe, nq)]
        Fourth intermediates.

    """
    x = clean_low_noise(x, fs, f0floor)
    fxx = f0floor * np.power(2, np.arange(nvc) / nvo)
    fxh = fxx[-1]
    dn = max(1, math.floor(fs / (fxh * 6.3)))
    dn3 = dn * 3
    if is_debug_mode():
        y = np.fromfile("tests/reference/data.dec.2", dtype=np.float64)
    else:
        y = decimate(x, dn3)
    pm1 = zmultanalytFineCSPB(y, fs / dn3, f0floor, nvc, nvo, mu, 1)
    mxpm1 = np.max(np.abs(pm1))
    eeps = mxpm1 * 1e-7
    pm1[pm1 == 0] = eeps
    pif1 = zwvlt2ifq(pm1, fs / dn3)
    mm1 = pif1.shape[0]

    if 3 <= nc:
        pm3 = zmultanalytFineCSPB(decimate(x, dn), fs / dn, f0floor, nvc, nvo, mu, 3)
        pif3 = zwvlt2ifq(pm3, fs / dn)
        pm3 = pm3[::3]
        pif3 = pif3[::3]
        mm3 = pif3.shape[0]
        if mm1 < mm3:
            pif3 = pif3[:mm1]
            pm3 = pm3[:mm1]
        elif mm3 < mm1:
            pif1 = pif1[:mm3]
            pm1 = pm1[:mm3]
            mm1 = mm3

    if 2 <= nc:
        pm2 = zmultanalytFineCSPB(decimate(x, dn), fs / dn, f0floor, nvc, nvo, mu, 2)
        pif2 = zwvlt2ifq(pm2, fs / dn)
        pm2 = pm2[::3]
        pif2 = pif2[::3]
        mm2 = pif2.shape[0]
        if mm1 < mm2:
            pif2 = pif2[:mm1]
            pm2 = pm2[:mm1]
        elif mm2 < mm1:
            pif1 = pif1[:mm2]
            pm1 = pm1[:mm2]

    if 3 <= nc:
        s1 = (
            pif1 * np.abs(pm1) ** pc
            + pif2 / 2 * np.abs(pm2) ** pc
            + pif3 / 3 * np.abs(pm3) ** pc
        )
        s2 = np.abs(pm1) ** pc + np.abs(pm2) ** pc + np.abs(pm3) ** pc
        pif2 = s1 / s2
    elif nc == 2:
        s1 = pif1 * np.abs(pm1) ** pc + pif2 / 2 * np.abs(pm2) ** pc
        s2 = np.abs(pm1) ** pc + np.abs(pm2) ** pc
        pif2 = s1 / s2
    elif nc == 1:
        pif2 = pif1
    else:
        msg = f"Unsupported number of harmonic components: {nc}"
        raise ValueError(msg)

    pif2 *= TAU
    slp, _ = zifq2gpm2(pif2, f0floor, nvo)
    dpif = np.diff(pif2, axis=0) * fs / dn3
    dpif = np.pad(dpif, ((0, 1), (0, 0)), mode="edge")
    dslp, _ = zifq2gpm2(dpif, f0floor, nvo)

    # The following code is not used in the original implementation,
    # as the result is eventually multiplies by zero.
    # damp = np.abs(np.diff(pm1, axis=1)) * fs / dn3
    # damp = np.pad(damp, ((0, 0), (0, 1)), mode="edge")
    # damp /= np.abs(pm1)

    fxx = f0floor * TAU * 2 ** (np.arange(nvc) / nvo)
    c1, c2b = znrmlcf2(1)

    c2 = c2b * (fxx / TAU) ** 2
    cff = 0
    cff1 = 1 + cff**2
    mmp = (dslp / cff1 / np.sqrt(c2)) ** 2 + (slp / np.sqrt(cff1) / np.sqrt(c1)) ** 2
    smap = zsmoothmapB(mmp, fs / dn3, f0floor, nvo, smp, minm, 0.4)

    r = mround(np.arange(0, len(pif2) - 1, shiftm / dn3 * fs / 1000))
    f0v, vrv, dfv, aav = zfixpfreq3(fxx, pif2[r], smap[r], dpif[r] / TAU, pm1[r])
    return f0v / TAU, vrv, dfv, aav


def clean_low_noise(x: np.ndarray, fs: int, f0floor: float) -> np.ndarray:
    """Filter out low frequency noise.

    Parameters
    ----------
    x : np.ndarray [shape=(nx,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    f0floor : float
        Lower bound of F0 search in Hz.

    Returns
    -------
    out : np.ndarray [shape=(nx,)]
        Low-pass filtered signal.

    """
    flm = 50
    flp = int(mround(fs * flm / 1000))
    nn = len(x)
    wlp = fir1(flp * 2, f0floor / (fs / 2))
    wlp[flp] -= 1
    wlp = -wlp

    tx = np.pad(x, (0, 2 * len(wlp)))
    ttx = fftfilt(wlp, tx)
    return ttx[flp : flp + nn]


def zmultanalytFineCSPB(
    x: np.ndarray,
    fs: float,
    f0floor: float,
    nvc: int,
    nvo: int,
    mu: float,
    mlt: int,
) -> np.ndarray:
    """Perform dual wavelet analysis using cardinal spline manipulation.

    Parameters
    ----------
    x : np.ndarray [shape=(nx,)]
        Input signal.

    fs : float
        Sampling frequency in Hz.

    f0floor : float
        Lower bound of F0 search in Hz.

    nvc : int
        Number of total voices for wavelet analysis.

    nvo : int
        Number of voices in an octave.

    mu : float
        Temporal stretching factor.

    mlt : int
        Harmonic ID.

    Returns
    -------
    out : np.ndarray [shape=(nx, nvc)]
        Wavelet transform using iso-metric Gabor function.

    """
    t0 = 1 / f0floor
    lmx = mround(6 * t0 * fs * mu)
    wl = 2 ** math.ceil(np.log2(lmx))
    nx = len(x)
    tx = np.pad(x, (0, wl))
    gent = (np.arange(wl) + 1 - wl / 2) / fs

    pm = np.empty((nx, nvc), dtype=np.complex128)
    mpv = 1.0
    for i in range(nvc):
        tb = gent * mpv
        t = tb[abs(tb) < 3.5 * mu * t0]
        ttm = t / t0 / mu
        wd1 = np.exp(-np.pi * ttm**2)
        wd2 = 1 - abs(ttm)
        wd2 = wd2[0 < wd2]
        wwd = conv(wd2, wd1)
        wwd = wwd[1e-5 < abs(wwd)]
        wbias = mround((len(wwd) - 1) / 2)
        idx = mround(np.arange(len(wwd)) - wbias + len(t) / 2)
        wwdc = wwd * np.exp(1j * TAU * mlt * t[idx] / t0)
        pmtmp1 = fftfilt(wwdc, tx)
        pm[:, i] = pmtmp1[wbias : wbias + nx] * np.sqrt(mpv)
        mpv *= 2 ** (1 / nvo)
    return pm


def zwvlt2ifq(pm: np.ndarray, fs: float) -> np.ndarray:
    """Convert wavelet transform to instantaneous frequency.

    Parameters
    ----------
    pm : np.ndarray [shape=(nx, nvc)]
        Wavelet transform.

    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    out : np.ndarray [shape=(nx, nvc)]
        Instantaneous frequency.

    """
    npm = pm / (np.abs(pm) + 1e-10)
    pif = np.abs(np.diff(npm, axis=0))
    pif = np.pad(pif, ((1, 0), (0, 0)), mode="edge")
    pif = np.clip(pif, 0, 2)
    return fs / np.pi * np.arcsin(pif / 2)


def zifq2gpm2(
    pif: np.ndarray, f0floor: float, nvo: int
) -> tuple[np.ndarray, np.ndarray]:
    """Convert instantaneous frequency to geometric parameters.

    Parameters
    ----------
    pif : np.ndarray [shape=(nx, nvc)]
        Instantaneous frequency.

    f0floor : float
        Lower bound of F0 search in Hz.

    nvo : int
        Number of voices in an octave.

    Returns
    -------
    slp : np.ndarray [shape=(nx, nvc)]
        First order coefficients.

    pbl : np.ndarray [shape=(nx, nvc)]
        Second order coefficients.

    """
    nn = pif.shape[1]
    fx = f0floor * TAU * 2 ** (np.arange(nn) / nvo)

    c = 2 ** (1 / nvo)
    g = np.asarray(
        [
            [1 / (c * c), 1 / c, 1],
            [1, 1, 1],
            [c * c, c, 1],
        ]
    )
    h = np.linalg.inv(g)

    slp = (
        np.diff(pif[:, :-1], axis=1) / (1 - g[0, 1])
        + np.diff(pif[:, 1:], axis=1) / (g[2, 1] - 1)
    ) / 2
    slp = np.pad(slp, ((0, 0), (1, 1)), mode="edge")

    pbl = pif[:, :-2] * h[1, 0] + pif[:, 1:-1] * h[1, 1] + pif[:, 2:] * h[1, 2]
    pbl = np.pad(pbl, ((0, 0), (1, 1)), mode="edge")

    slp /= fx
    pbl /= fx
    return slp, pbl


def znrmlcf2(f: float) -> tuple[float, float]:
    """Calculate normalization coefficients.

    Parameters
    ----------
    f : float
        A constant factor.

    Returns
    -------
    c1 : float
        First normalization coefficient.

    c2 : float
        Second normalization coefficient.

    """

    def zGcBs(x: np.ndarray, k: float) -> np.ndarray:
        """Compute values.

        Parameters
        ----------
        x : np.ndarray
            Input values.

        k : float
            A constant factor.

        Returns
        -------
        out : np.ndarray
            Output values.

        """
        tt = x + 1e-7
        ptt = np.pi * tt + 1e-4
        return tt**k * np.exp(-np.pi * tt**2) * (np.sin(ptt) / ptt) ** 2

    n = 100
    x = np.linspace(0, 3, 3 * n + 1)
    g = zGcBs(x, 0)
    dg = np.pad(n * np.diff(g), (0, 1))
    dgs = dg / (TAU * f)
    xx = TAU * f * x
    c1 = np.sum((xx * dgs) ** 2) / n * 2
    c2 = np.sum((xx**2 * dgs) ** 2) / n * 2
    return c1, c2


def zsmoothmapB(
    mmp: np.ndarray,
    fs: float,
    f0floor: float,
    nvo: int,
    mu: float,
    mlim: float,
    pex: float,
) -> np.ndarray:
    """Perform dual wavelet analysis using cardinal spline manipulation.

    Parameters
    ----------
    mmp : np.ndarray [shape=(nx, nvc)]
        Input signal.

    fs : float
        Sampling frequency in Hz.

    f0floor : float
        Lower bound of F0 search in Hz.

    nvo : int
        Number of voices in an octave.

    mu : float
        Temporal stretching factor.

    mlim : float
        Minimum smoothing length.

    pex : float
        A constant factor.

    Returns
    -------
    out : np.ndarray [shape=(nx, nvc)]
        Wavelet transform using iso-metric Gabor function.

    """
    mm, nvc = mmp.shape
    t0 = 1 / f0floor
    lmx = mround(6 * t0 * fs * mu)
    wl = 2 ** math.ceil(np.log2(lmx))
    gent = (np.arange(wl) + 1 - wl / 2) / fs

    smap = np.empty_like(mmp)
    mpv = 1.0
    iiv = np.arange(mm)
    for i in range(nvc):
        t = gent * mpv
        t = t[abs(t) < 3.5 * mu * t0]
        wbias = mround((len(t) - 1) / 2)
        wd1 = np.exp(-np.pi * (t / (mu * t0 * (1 - pex))) ** 2)
        wd2 = np.exp(-np.pi * (t / (mu * t0 * (1 + pex))) ** 2)
        wd1 /= wd1.sum()
        wd2 /= wd2.sum()
        tm = fftfilt(wd1, np.pad(mmp[:, i], (0, wl))) + 1e-10
        tm = fftfilt(wd2, np.pad(1 / tm[iiv + wbias], (0, wl)))
        smap[:, i] = 1 / tm[iiv + wbias]
        if mlim < mu * t0 / mpv * 1000:
            mpv *= 2 ** (1 / nvo)
    return smap


def zfixpfreq3(
    fxx: np.ndarray, pif2: np.ndarray, mmp: np.ndarray, dfv: np.ndarray, pm: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Fix peak frequencies.

    Parameters
    ----------
    fxx : np.ndarray [shape=(nvc,)]
        Set of frequencies.

    pif2 : [shape=(nframe, nvc)]
        Instantaneous frequency.

    mmp : [shape=(nframe, nvc)]
        Frequency map.

    dfv : [shape=(nframe, nvc)]
        Difference of instantaneous frequency.

    pm : [shape=(nframe, nvc)]
        Wavelet transform.

    Returns
    -------
    ff : np.ndarray [shape=(nframe, nq)]
        Fixed instantaneous frequency.

    vv : np.ndarray [shape=(nframe, nq)]
        Fixed frequency map.

    df : np.ndarray [shape=(nframe, nq)]
        Fixed difference of instantaneous frequency.

    aa : np.ndarray [shape=(nframe, nq)]
        Fixed wavelet transform.

    """
    fxx = np.tile(fxx, (mmp.shape[0], 1))
    aav = np.abs(pm)
    mm = pif2.shape[0]
    cd1 = pif2 - fxx
    cd2 = np.diff(cd1, axis=1)
    cd2 = np.pad(cd2, ((0, 0), (0, 1)), mode="edge")
    cdd1 = np.pad(cd1[:, 1:], ((0, 0), (0, 1)), mode="edge")
    ixx = (cd1 * cdd1 < 0) & (cd2 < 0)
    ixx1 = np.roll(ixx, 1, axis=1)
    nf = np.sum(ixx, axis=1)
    nq = np.max(nf)
    mask = np.arange(nq) < nf[:, None]

    ff = np.full((mm, nq), 1e10)
    vv = np.full((mm, nq), 1e10)
    df = np.full((mm, nq), 1e10)
    aa = np.full((mm, nq), 1e10)
    ff[mask] = pif2[ixx] + (pif2[ixx1] - pif2[ixx]) * cd1[ixx] / (cd1[ixx] - cdd1[ixx])
    z = (ff[mask] - fxx[ixx]) / (fxx[ixx1] - fxx[ixx])
    vv[mask] = mmp[ixx] + (mmp[ixx1] - mmp[ixx]) * z
    df[mask] = dfv[ixx] + (dfv[ixx1] - dfv[ixx]) * z
    aa[mask] = aav[ixx] + (aav[ixx1] - aav[ixx]) * z
    return ff, vv, df, aa


def zmultiCandIF(f0v: np.ndarray, vrv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Find F0 candidates based on instantaneous frequency fixed points.

    Parameters
    ----------
    f0v : np.ndarray [shape=(nframe, nq)]
        Fixed point frequencies in Hz.

    vrv : np.ndarray [shape=(nframe, nq)]
        Fixed point ratio.

    Returns
    -------
    val : np.ndarray [shape=(nframe, 3)]
        Values.

    pos : np.ndarray [shape=(nframe, 3)]
        Positions.

    """
    nc = f0v.shape[0]
    vrvdb = -10 * np.log10(vrv)

    f = f0v
    v = vrvdb
    mxfq = 1e6
    mnval = -50
    v[mxfq <= f] = mnval

    mxp = np.argsort(-v, axis=1)[..., :3]
    pos = np.take_along_axis(f, mxp, axis=1)
    val = np.take_along_axis(v, mxp, axis=1)

    ramp = np.arange(nc)
    mask = val == mnval
    first_true = np.argmax(mask, axis=1)
    prev_first_true = np.maximum(0, first_true - 1)
    pos[ramp, first_true] = pos[ramp, prev_first_true]
    val[ramp, first_true] = val[ramp, prev_first_true]
    mask = val == mnval
    pos[mask] = 1
    val[mask] = 0
    return val, pos


def zremoveACinduction(
    x: np.ndarray, fs: int, pos: np.ndarray
) -> tuple[np.ndarray, int, float]:
    """Remove AC induction.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    pos : np.ndarray [shape=(nframe, 3)]
        Locations of top-three F0 candidates.

    Returns
    -------
    y : np.ndarray [shape=(nsample,)]
        Output signal without the AC induction.

    ind : int
        1 indicates that the signal has AC induction.

    fq : float
        Frequency of the AC induction.

    """
    ind = 0
    f = pos
    h50 = np.sum(np.abs(f - 50) < 5) / np.sum(0 < f)
    h60 = np.sum(np.abs(f - 60) < 5) / np.sum(0 < f)
    if h50 < 0.2 and h60 < 0.2:
        return x, ind, 0
    ind = 1
    fq = 50 if h60 < h50 else 60
    tx = (np.arange(len(x)) + 1) / fs
    fqv = mrange(-0.3, 0.025, 0.3) + fq
    txv = tx[:, None] * fqv
    fk = x @ np.exp(-1j * TAU * txv) / len(x)
    ix = np.argmax(np.abs(fk))
    fq = fqv[ix]
    y = x - 2 * np.real(fk[ix] * np.exp(1j * TAU * fq * tx))
    return y, ind, fq


def zlagspectestnormal(
    x: np.ndarray,
    fs: float,
    stp: float,
    edp: float,
    shiftm: float,
    wtlm: int,
    ndiv: int,
    wflf: int,
    pc: float,
    amp: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute lag spectrum for F0 extraction.

    Parameters
    ----------
    x : np.ndarray
        Waveform.

    fs : float
        Sampling frequency in Hz.

    stp : float
        Start position in msec.

    edp : float
        End position in msec.

    shiftm : float
        Frame shift in msec.

    wtlm : int
        Time window length in msec.

    ndiv : int
        Number of segment in the frequency domain.

    wflf : int
        Window length in the frequency domain.

    pc : float
        Power exponent.for nonlinearity.

    amp : float
        Amount of lag window compensation.

    Returns
    -------
    lagspec : np.ndarray [shape=(nfft, nftm)]
        Lag spectrum.

    lx : np.ndarray [shape=(nfft,)]
        Lag axis.

    """
    nftm = math.floor((edp - stp) / shiftm)
    if nftm <= 0:
        return np.empty((0, 0)), np.empty(0)
    lagspecs = []
    for ii in range(nftm):
        pmmul = stp + ii * shiftm
        acc, _, _, lx = ztestspecspecnormal(x, fs, pmmul, wtlm, ndiv, wflf, pc, amp)
        lagspecs.append(np.mean(acc, axis=1) / np.mean(acc[0]))
    lagspec = np.stack(lagspecs, axis=1)
    return lagspec, lx


def ztestspecspecnormal(
    x: np.ndarray,
    fs: float,
    pm: float,
    wtlm: int,
    ndiv: int,
    wflf: int,
    pc: float,
    amp: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute lag spectrum for F0 extraction.

    Parameters
    ----------
    x : np.ndarray
        Waveform.

    fs : float
        Sampling frequency in Hz.

    pm : float
        Position to be tested in msec.

    wtlm : int
        Time window length in msec.

    ndiv : int
        Number of segment in the frequency domain.

    wflf : int
        Window length in the frequency domain.

    pc : float
        Power exponent.for nonlinearity.

    amp : float
        Amount of lag window compensation.

    Returns
    -------
    acc : np.ndarray [shape=(nfft, ndiv+1)]
        Spectrogram on frequency axis.

    abase : np.ndarray [shape=(nfft,)]
        An intermediate array.

    fx : np.ndarray [shape=(nfft,)]
        Frequency axis.

    lx : np.ndarray [shape=(nfft,)]
        Lag axis.

    """
    wtlms = mround(wtlm / 1000 * fs)
    wtlmso = math.floor(wtlms / 2) * 2 + 1
    bb = np.arange(wtlmso) - (wtlmso - 1) // 2
    fftl = 2 ** math.ceil(np.log2(wtlmso))
    fftl2 = fftl // 2
    x = np.pad(x, (fftl, fftl))

    p = mround(pm / 1000 * fs)
    fx = np.arange(fftl) / fftl * fs
    tx = np.arange(fftl)
    tx[fftl2 < tx] -= fftl
    txf = tx / fs
    lagw = np.exp(-((txf / 0.0035) ** 2))
    lagw2 = np.exp(-((txf / 0.0016) ** 2))

    xt = x[fftl + bb + p]
    if np.sum(np.abs(xt)) < 1e-10:
        xt += randn(len(xt))
    abase = np.abs(np.fft.fft(xt * np.blackman(wtlmso), n=fftl))
    ac = np.fft.ifft(abase**2)
    npw = np.real(np.fft.fft(ac * lagw))
    npw = np.maximum(1e-10, npw)
    pw = abase**2 / npw

    fsp = fs / fftl
    wflfs = mround(wflf / fsp)
    wflfso = math.floor(wflfs / 2) * 2 + 1
    bbf = np.arange(wflfso) - (wflfso - 1) // 2
    fftlf = 2 ** math.ceil(np.log2(wflfso) + 2)
    fftlf2 = fftlf // 2
    lx = np.arange(fftlf2) / (fsp * fftlf)

    nsht = fftl2 / ndiv
    ampw = 1 - lagw * (1 - 1 / amp)
    ampw = (1 - lagw2[:fftlf2] * (1 - 1 / amp)) / ampw[:fftlf2]
    ii = np.arange(ndiv + 1)
    p = np.remainder(mround(fftl2 + 1 + bbf[:, None] + ii * nsht), fftl)
    ac1 = np.abs(np.fft.fft(pw[p] * hanning(wflfso)[:, None], n=fftlf, axis=0))
    ac2 = npw[p[(wflfso - 1) // 2] - 1] ** pc
    acc = (ac1[:fftlf2] * ac2) * ampw[:, None]
    return acc, abase, fx, lx


def zmultiCandAC(
    lx: np.ndarray, lagspec: np.ndarray, beta: float, lagsp: float, timesp: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract F0 candidates from time-lag representation.

    Parameters
    ----------
    lx : np.ndarray [shape=(nfft,)]
        Lag axis.

    lagspec : np.ndarray [shape=(nfft, nftm)]
        Time-lag representation.

    beta : float
        Nonlinear distance measure.

    lagsp : float
        Lag smoothing parameter.

    timesp : float
        Temporal smoothing parameter.

    Returns
    -------
    f0 : np.ndarray [shape=(nftm, 3)]
        Fundamental frequency.

    pl : np.ndarray [shape=(nftm, 3)]
        Peak level.

    """
    nr, nc = lagspec.shape
    dlag = np.pad(np.diff(lagspec, axis=0), ((1, 0), (0, 0)))
    imm = dlag * np.roll(dlag, -1, axis=0)
    lagspecz = lagspec.copy()
    lagspecz[lx < 0.002] -= np.exp(-((lx[lx < 0.002] / 0.00055) ** 2))[:, None]

    def zgendeconvmatrix(n: int, a: float) -> np.ndarray:
        """Generate deconvolution matrix.

        Parameters
        ----------
        n : int
            Size of the matrix.

        a : float
            Reduction factor.

        Returns
        -------
        out : np.ndarray [shape=(n, n)]
            The deconvolution matrix.

        """
        mapm = np.zeros((n, n))
        for ii in range(2, n):
            for k in [2, 3, 5]:
                bet = ii / k
                lbet = math.floor(bet)
                ubet = lbet + 1
                mapm[ii, lbet] += (1 - (bet - lbet)) * a ** (k - 1)
                mapm[ii, ubet] += (1 - (ubet - bet)) * a ** (k - 1)
        return mapm

    mapm = zgendeconvmatrix(nr, 0.6)
    lagspecz = np.log(np.exp((lagspecz - mapm @ lagspecz) * 20) + 1) / 20
    lagspec = lagspecz
    tls = np.concatenate([lagspec, lagspec[-1:], lagspec[-1:0:-1]]) ** beta
    llx = np.concatenate([lx, lx[-1:], lx[-1:0:-1]])
    lagw = np.exp(-((llx / lagsp * 1000) ** 2))
    lagw /= np.sum(lagw)
    flagw = np.real(np.fft.fft(lagw))
    tls = np.real(np.fft.ifft(np.fft.fft(tls, axis=0) * flagw[:, None], axis=0))
    tmsm = int(mround((timesp - 1) / 2)) * 2 + 1
    wt = hanning(tmsm, norm=True)
    lagsms = fftfilt(wt, np.pad(tls[:nr], ((0, 0), (tmsm, tmsm))))
    lagsms = cut(lagsms, (tmsm - 1) // 2 * 3, nc)
    lagsms = np.abs(lagsms) ** (1 / beta)

    mnval = 0
    ix = (imm < 0) & (0 < dlag)
    lagsms[~ix] = mnval

    ramp = np.arange(nc)[:, None]
    ramp3 = np.arange(-1, 2)[:, None, None]
    mxp = np.argsort(-lagsms, axis=0)[:3].T
    pl, pos = zzParabolicInterp(lagspec[ramp3 + mxp, ramp], mxp)
    f0 = 1 / (pos * lx[1])
    return f0, pl


def zzParabolicInterp(
    yv: np.ndarray, xo: np.ndarray, *, clip: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """Perform parabolic interpolation.

    Parameters
    ----------
    yv : np.ndarray [shape=(3, ...)]
       Y values.

    xo : np.ndarray [shape=(...,)]
       X values.

    clip : bool
       Whether to clip the interpolated positions.

    Returns
    -------
    val : np.ndarray [shape=(...,)]
       Interpolated values.

    pos : np.ndarray [shape=(...,)]
       Interpolated positions.

    """
    lp = np.diff(yv, axis=0)
    a = lp[0] - lp[1]
    b = (lp[0] + lp[1]) / 2
    xp = b / a + xo
    val = yv[1] + 1.5 * b * b / a
    if clip:
        mask = xo + 1 < xp
        xp[mask] = xo[mask] + 1
        val[mask] = yv[2, mask]
        mask = xp < xo - 1
        xp[mask] = xo[mask] - 1
        val[mask] = yv[0, mask]
    pos = xp
    return val, pos


def zcombineRanking4(
    f0if: np.ndarray,
    cnif: np.ndarray,
    f0ac: np.ndarray,
    acac: np.ndarray,
    nvo: int,
    beta: float,
    wAC: float,
    wIF: float,
    f0floor: float,
    f0ceil: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Combine F0 estimation results from IF and AC methods.

    Parameters
    ----------
    f0if : np.ndarray [shape=(nframe, 3)]
        Fundamental frequency from IF method.

    cnif : np.ndarray [shape=(nframe, 3)]
        Confidence from IF method.

    f0ac : np.ndarray [shape=(nframe, 3)]
        Fundamental frequency from AC method.

    acac : np.ndarray [shape=(nframe, 3)]
        Autocorrelation from AC method.

    nvo : int
        Number of channels in one octave.

    beta : float
        Normalization factor for mixing F0 distance

    wAC : float
        Weight for AC method.

    wIF : float
        Weight for IF method.

    f0floor : float
        Lower limit of F0.

    f0ceil : float
        Upper limit of F0.

    Returns
    -------
    f0 : np.ndarray [shape=(nframe, 6)]
        Candidates of fundamental frequency.

    pl : np.ndarray [shape=(nframe, 6)]
        Score of candidates.

    """
    n = min(len(f0if), len(f0ac))

    nvc = math.ceil(_log_ratio(f0ceil, f0floor)) * nvo
    fx = f0floor * (2 ** (np.arange(nvc) / nvo))
    lfx = np.log2(fx)
    logf0if = np.log2(f0if)
    logf0ac = np.log2(f0ac)
    relif = np.maximum(
        1e-9,
        (cnif - np.min(cnif[:, 0])) / (np.max(cnif[:, 0]) - np.min(cnif[:, 0]) + 1e-9),
    )
    relac = np.maximum(
        1e-9,
        (acac - np.min(acac[:, 0])) / (np.max(acac[:, 0]) - np.min(acac[:, 0]) + 1e-9),
    )
    IFmap = relif[..., None] ** 2 * np.exp(-(((logf0if[..., None] - lfx) / beta) ** 2))
    IFmap = np.sum(IFmap, axis=1)
    ACmap = relac[..., None] ** 2 * np.exp(-(((logf0ac[..., None] - lfx) / beta) ** 2))
    ACmap = np.sum(ACmap, axis=1)
    f0map = np.sqrt((wIF * IFmap[:n] + wAC * ACmap[:n]) / 2)
    f0mapbak = f0map.copy()

    ix1 = np.pad(np.diff(f0map, axis=1), ((0, 0), (1, 0)))
    ix2 = np.pad(np.diff(f0map, axis=1), ((0, 0), (0, 1)))
    ix = ix1 * ix2 < 0
    f0map[~ix] = 0

    ramp = np.arange(n)[:, None]
    ramp3 = np.arange(-1, 2)[:, None, None]
    mxp = np.argsort(-f0map, axis=1)[:, :6]
    pl1, f01 = zzParabolicInterp(f0mapbak[ramp, ramp3 + mxp], mxp, clip=False)
    mask = f0map[ramp, mxp] == 0
    shifted_mask = np.roll(mask, -1, axis=1)
    extended_mask = mask | shifted_mask

    pl1[mask] = 0
    f01[mask] = 0
    pl2 = np.cumsum(pl1 * extended_mask, axis=1)
    f02 = np.cumsum(f01 * extended_mask, axis=1)
    pl = pl1 + pl2 * mask
    f0 = f01 + f02 * mask
    f0 = f0floor * 2 ** (f0 / nvo)
    return f0, pl


def zVpowercalc(
    x: np.ndarray, fs: int, wtc: float, shiftm: float, fc: float
) -> np.ndarray:
    """Compute power of the input signal.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,)]
        Input signal.

    fs : int
        Sampling frequency in Hz.

    wtc : float
        Window time constant in msec.

    shiftm : float
        Frame update interval in msec.

    fc : float
        LPC cutoff frequency in Hz.

    Returns
    -------
    out : np.ndarray [shape=(nframe,)]
        Power of the input signal.

    """
    t = mrange(0, 1 / fs, wtc * 5 / 1000)
    w = np.exp(-t / wtc * 1000)
    w -= w[-1]
    w /= np.sum(w)

    lw = int(mround(fs / fc * 2))
    b = fir1(lw - 1, 2 * fc / fs)
    nn = len(x)
    x = fftfilt(b, np.pad(x, (0, lw)))
    x = cut(x, mround(lw / 2) - 1, nn)

    x2 = x**2
    yf = fftfilt(w, x2)
    yb = fftfilt(w, x2[::-1])[::-1]
    y = np.minimum(yf, yb)

    z = np.arange(nn) * 1000 / fs
    return interp1(z, y, mrange(0, shiftm, z[-1]))


def calc_noise_level(pwsdb: np.ndarray) -> float:
    """Calculate noise level from power.

    Parameters
    ----------
    pwsdb : np.ndarray [shape=(nframe,)]
        Power in dB.

    Returns
    -------
    out : float
        Noise level in dB.

    """
    mxpwsdb = np.max(pwsdb)
    hstgrm, binlvl = np.histogram(
        np.clip(pwsdb, mxpwsdb - 60 - 0.5, mxpwsdb + 2 + 0.5),
        bins=63,
        range=(mxpwsdb - 60 - 0.5, mxpwsdb + 2 + 0.5),
    )
    binlvl = binlvl[:-1] + 0.5
    q10 = interp1(np.cumsum(hstgrm + 1e-9) / np.sum(hstgrm) * 100, binlvl, 10)
    minid = np.argmin(np.abs(q10 - binlvl))
    bb = np.clip(np.arange(minid - 5, minid + 6), 0, len(binlvl) - 1)
    return np.sum(hstgrm[bb] * binlvl[bb]) / np.sum(hstgrm[bb])


def zcontiguousSegment10(
    pwsdb: np.ndarray,
    f0cand: np.ndarray,
    relv: np.ndarray,
    shiftm: float,
    f0jumpt: float,
    nsdt: float,
) -> tuple[np.ndarray, np.ndarray, list[list[int]]]:
    """Search for contiguous segments that consists of best candidates.

    Parameters
    ----------
    pwsdb : np.ndarray [shape=(nn,)]
        Instantaneous power in dB.

    f0cand : np.ndarray [shape=(nn, 6)]
        F0 candidates.

    relv : np.ndarray [shape=(nn, 6)]
        Reliability values.

    shiftm : float
        Frame shift in msec.

    f0jumpt : float
        F0 jump threshold.

    nsdt : float
        SD for tracking normalization.

    Returns
    -------
    f0 : np.ndarray [shape=(nn,)]
        Fundamental frequency.

    rel : np.ndarray [shape=(nn,)]
        Reliability.

    cseg : list[list[int]]
        Segment indices.

    """
    nn = min(len(pwsdb), len(f0cand), len(relv))
    pwsdb = pwsdb[:nn]
    f0cand = f0cand[:nn]
    relv = relv[:nn]
    relv[relv == 0] = 1e-5

    noiselevel = calc_noise_level(pwsdb[:nn])
    mxpwsdb = max(pwsdb)
    wellovernoise = (4 * noiselevel + mxpwsdb) / 5
    if mxpwsdb - 10 < wellovernoise:
        wellovernoise = mxpwsdb - 10
        noiselevel = (5 * wellovernoise - mxpwsdb) / 4

    maskr = np.ones_like(f0cand)
    idx = np.argsort(-relv[:, 0])
    idx = idx[0.16 < relv[idx, 0]]
    segv = []
    f0segs = []
    relsegs = []
    sratev = []
    minlen = 13 / shiftm
    for i in idx:
        if 0 < maskr[i, 0] and wellovernoise < pwsdb[i]:
            f0seg, relseg, lb, ub, srate = zsearchforContiguousSegment(
                f0cand, relv, maskr, i, pwsdb, noiselevel
            )
            if minlen < len(f0seg) and 0.12 < srate:
                segv.append([lb, ub])
                f0segs.append(f0seg)
                relsegs.append(relseg)
                sratev.append(
                    srate * (1 - 1 / max(1.4, np.sqrt(len(f0seg) * shiftm / 40)))
                )

    if len(segv) == 0:
        return np.zeros(nn), np.zeros(nn), []

    f0 = np.zeros_like(f0cand[:, 0])
    rel = np.zeros_like(relv[:, 0])
    idrel = np.argsort(np.stack(sratev))
    nseg = len(idrel)
    for ii in range(nseg):
        icp = idrel[ii]
        lb, ub = segv[icp]
        validind = np.all(f0[lb:ub] == 0)
        if validind:
            f0[lb:ub] = f0segs[icp]
            rel[lb:ub] = relsegs[icp]

    cseg = []
    InInd = 0
    ubs = np.asarray([segv[ii][1] for ii in range(nseg)])
    for ii in range(nn):
        if InInd == 0 and 0 < f0[ii]:
            cseg.append([ii])
            InInd = 1
        elif InInd == 1 and (
            f0[ii] == 0
            or (np.any(ubs == ii) and (pwsdb[ii] < (noiselevel + 4 * mxpwsdb) / 5))
        ):
            cseg[-1].append(ii)
            InInd = 0
    if len(cseg[-1]) == 1:
        cseg[-1].append(nn)
    crseg = len(cseg)

    for ii in range(len(cseg)):
        lb, ub = cseg[ii]
        maxjmp = np.max(np.abs(np.diff(np.log2(f0[lb:ub]))))
        if maxjmp <= 0.4:
            continue
        ixmx = np.unravel_index(np.argmax(relv[lb:ub]), relv[lb:ub].shape)[0]
        cpos = lb + ixmx
        bp = lb
        ep = ub - 1
        args1 = [f0cand, relv, pwsdb]
        args2 = [lb, ub, f0jumpt, nsdt, noiselevel]
        f0raw0 = f0.copy()
        f0raw1 = f0raw0.copy()
        f0raw2 = f0raw0.copy()
        f0raw3 = f0raw0.copy()
        lastf0 = f0raw0[cpos] = f0cand[cpos, 0]
        sprob0 = ztraceInAsegment2(f0raw0, *args1, cpos, lastf0, *args2)
        lastf0 = f0raw1[cpos] = f0cand[cpos, 1]
        sprob1 = ztraceInAsegment2(f0raw1, *args1, cpos, lastf0, *args2)
        lastf0 = f0raw2[bp] = f0[bp]
        sprob2 = ztraceInAsegment2(f0raw2, *args1, bp, lastf0, *args2)
        lastf0 = f0raw3[ep] = f0cand[ep, 0]
        sprob3 = ztraceInAsegment2(f0raw3, *args1, ep, lastf0, *args2)
        imx = np.argmax([sprob0, sprob1, sprob2, sprob3])
        f0raws = [f0raw0, f0raw1, f0raw2, f0raw3]
        f0[lb:ub] = f0raws[imx][lb:ub]

    hgf0 = np.sort(np.log2(f0[0 < f0]))
    id10 = mround(0.1 * len(hgf0)) - 1
    id90 = mround(0.9 * len(hgf0))
    rsd = mstd(hgf0[id10:id90], ddof=(0 if len(hgf0[id10:id90]) == 1 else 1))
    mf0 = np.mean(hgf0[id10:id90])
    csego = cseg
    cseg = []
    f0o = f0
    f0 = np.zeros_like(f0o)
    for ii in range(crseg):
        lb, ub = csego[ii]
        cond1 = np.abs(np.mean(np.log2(f0o[lb:ub])) - mf0) < min(1.2, max(0.9, 5 * rsd))
        cond2 = (2 * noiselevel + mxpwsdb) / 3 < np.mean(pwsdb[lb:ub])
        if cond1 and cond2:
            cseg.append([lb, ub])
            f0[lb:ub] = f0o[lb:ub]

    f0bk = f0
    f0 = np.zeros_like(f0bk)
    lastend = 0
    f0bk[lastend] = 1
    nrseg = len(cseg)
    minlen = 50 / shiftm
    for ii in range(nrseg):
        lb, ub = cseg[ii]
        nexttop = ub if ii == nrseg - 1 else cseg[ii + 1][0] + 1
        ipause = lb - lastend + 1
        fpause = nexttop - ub + 1
        cond1 = ipause < minlen or fpause < minlen
        cond2 = 0.6 < np.abs(_log_ratio(f0bk[lastend], f0bk[lb]))
        cond3 = 0.6 < np.abs(_log_ratio(f0bk[ub - 1], f0bk[nexttop - 1]))
        cond4 = ub - lb < minlen and np.mean(rel[lb:ub]) < 0.5
        if not (cond1 and cond2 and cond3 and cond4):
            f0[lb:ub] = f0bk[lb:ub]
        lastend = ub - 1

    if np.all(f0 == 0):
        return f0, rel, []

    pv, dv = zpeakdipdetect(pwsdb, mround(81 / shiftm))
    avf0 = np.mean(f0[0 < f0])
    logavf0 = np.log2(avf0)
    relv2 = relv * np.exp(-((np.log2(f0cand) - logavf0) ** 2))
    reliablelevel = (noiselevel + 2 * mxpwsdb) / 3
    for ii in range(len(pv)):
        if reliablelevel < pwsdb[pv[ii]] and f0[pv[ii]] == 0:
            ary = dv[dv < pv[ii]]
            lb = 0 if len(ary) == 0 else np.max(ary)
            ary = dv[pv[ii] < dv]
            ub = nn if len(ary) == 0 else np.min(ary)
            peaklvl = pwsdb[pv[ii]]
            for bp in reversed(range(lb, pv[ii])):
                if pwsdb[bp] < peaklvl - 9:
                    break
            for ep in range(pv[ii], ub):
                if pwsdb[ep] < peaklvl - 9:
                    break
            lb = bp
            ub = ep + 1
            imx2 = np.unravel_index(np.argmax(relv2[lb:ub]), relv2[lb:ub].shape)[0]
            cpos = lb + imx2

            args1 = [f0cand, relv2, pwsdb + 10]
            args2 = [lb, ub, f0jumpt, nsdt, noiselevel]
            f0raw0 = f0cand[:, 0].copy()
            f0raw1 = f0raw0.copy()
            lastf0 = f0raw0[cpos] = f0cand[cpos, 0]
            sprob0 = ztraceInAsegment2(f0raw0, *args1, cpos, lastf0, *args2)
            lastf0 = f0raw1[cpos] = f0cand[cpos, 1]
            sprob1 = ztraceInAsegment2(f0raw1, *args1, cpos, lastf0, *args2)
            imx = np.argmax([sprob0, sprob1])
            f0raws = [f0raw0, f0raw1]
            f0[lb:ub] = f0raws[imx][lb:ub]

    cseg = []
    InInd = 0
    for ii in range(nn):
        if InInd == 0 and 0 < f0[ii]:
            cseg.append([ii])
            InInd = 1
        elif InInd == 1 and f0[ii] == 0:
            cseg[-1].append(ii)
            InInd = 0
    if InInd == 1:
        cseg[-1].append(nn)

    return f0, rel, cseg


def zsearchforContiguousSegment(
    f0cand: np.ndarray,
    relv: np.ndarray,
    maskr: np.ndarray,
    acp: int,
    pwsdb: np.ndarray,
    noiselevel: float,
) -> tuple[np.ndarray, np.ndarray, int, int, float]:
    """Search for contiguous segment.

    Parameters
    ----------
    f0cand : np.ndarray [shape=(nframe, 6)]
        F0 candidates.

    relv : np.ndarray [shape=(nframe, 6)]
        Reliability values.

    maskr : np.ndarray [shape=(nframe, 6)]
        Masker for preventing multiple assignment.

    acp : int
        Current point.

    pwsdb : np.ndarray [shape=(nframe,)]
        Power in dB.

    noiselevel : float
        Noise level in dB.

    Returns
    -------
    f0seg : np.ndarray [shape=(ub-lb,)]
        F0 values in the segment.

    relseg : np.ndarray [shape=(ub-lb,)]
        Reliability values in the segment.

    lb : int
        Lower index of the segment.

    ub : int
        Upper index of the segment.

    srate : float
        Sum of reliabilities in the segment.

    """
    f0segs: list[np.ndarray] = []
    relsegs: list[np.ndarray] = []
    srate = 0.0
    nn = len(f0cand)

    lb = acp
    lastf0 = f0cand[acp, 0]
    for ii in reversed(range(acp)):
        ff = np.abs(_log_ratio(lastf0, f0cand[ii]))
        idx = np.argmin(ff)
        if (
            0.1 < ff[idx]
            or pwsdb[ii] < noiselevel + 6
            or maskr[ii, idx] == 0
            or relv[ii, idx] < 0.17
        ):
            break
        lb = ii
        lastf0 = f0cand[ii, idx]
        f0segs.insert(0, lastf0)
        relsegs.insert(0, relv[ii, idx])
        srate += relv[ii, idx]

    f0segs.append(f0cand[acp, 0])
    relsegs.append(relv[acp, 0])

    ub = acp
    lastf0 = f0cand[acp, 0]
    for ii in range(acp + 1, nn):
        ff = np.abs(_log_ratio(lastf0, f0cand[ii]))
        idx = np.argmin(ff)
        if (
            0.1 < ff[idx]
            or pwsdb[ii] < noiselevel + 6
            or maskr[ii, idx] == 0
            or relv[ii, idx] < 0.05
        ):
            break
        ub = ii
        lastf0 = f0cand[ii, idx]
        f0segs.append(lastf0)
        relsegs.append(relv[ii, idx])
        srate += relv[ii, idx]
        maskr[ii] = 0

    ub += 1
    f0seg = np.stack(f0segs)
    relseg = np.stack(relsegs)
    maskr[acp] = 0
    maskr[lb:ub] = 0
    srate /= ub - lb
    return f0seg, relseg, lb, ub, srate


def ztraceInAsegment2(
    f0raw0: np.ndarray,
    f0cand: np.ndarray,
    relv: np.ndarray,
    pwsdb: np.ndarray,
    acp: int,
    lastf0in: float,
    lb: int,
    ub: int,
    f0jump: float,
    nsd: float,
    noiselevel: float,
) -> float:
    """Trace in a segment.

    Parameters
    ----------
    f0raw0 : np.ndarray [shape=(nn,)]
        Input and output fundamental frequency.

    f0cand : np.ndarray [shape=(nn, 6)]
        F0 candidates.

    relv : np.ndarray [shape=(nn, 6)]
        Reliability values.

    pwsdb : np.ndarray [shape=(nn,)]
        Power in dB.

    acp : int
        Current point.

    lastf0in : float
        Last fundamental frequency.

    lb : int
        Lower index of the segment.

    ub : int
        Upper index of the segment.

    f0jump : float
        Threshold for F0 jump.

    nsd : float
        Standard deviation for normalization.

    noiselevel : float
        Noise level in dB.

    Returns
    -------
    out : float
        Score.

    """
    nn = len(f0raw0)
    maxpower = np.max(pwsdb)
    reliablepowerth = max(noiselevel + 10, (3 * maxpower + noiselevel) / 4)

    sprob = 0.0
    if acp == lb:
        f0raw0[lb] = lastf0in
    if acp == ub - 1:
        f0raw0[ub - 1] = lastf0in

    lastf0 = lastf0in
    for jj in reversed(range(lb, acp)):
        bsb = np.maximum(0, np.arange(jj, jj - 6, -1))
        val = np.exp(-((_log_ratio(f0cand[bsb], lastf0) / nsd) ** 2)) * relv[bsb]
        idxx = np.unravel_index(np.argmax(val), val.shape)
        idx = idxx[1]
        jjmx = bsb[idxx[0]]
        if np.abs(_log_ratio(f0cand[jjmx, idx], lastf0)) < f0jump:
            dd = abs(jjmx - jj)
            if dd == 0:
                f0raw0[jj] = f0cand[jjmx, idx]
            else:
                f0raw0[jj] = lastf0 + (f0cand[jjmx, idx] - lastf0) / (dd + 1)
            if reliablepowerth < pwsdb[jj]:
                sprob += np.log2(val[idxx])
        else:
            f0raw0[jj] = lastf0
            if reliablepowerth < pwsdb[jj]:
                sprob += np.log2(val[idxx]) - 10
        lastf0 = f0raw0[jj]

    lastf0 = lastf0in
    for jj in range(acp + 1, ub):
        bsb = np.minimum(nn - 1, np.arange(jj, jj + 6))
        val = np.exp(-((_log_ratio(f0cand[bsb], lastf0) / nsd) ** 2)) * relv[bsb]
        idxx = np.unravel_index(np.argmax(val), val.shape)
        idx = idxx[1]
        jjmx = bsb[idxx[0]]
        if np.abs(_log_ratio(f0cand[jjmx, idx], lastf0)) < f0jump:
            dd = abs(jjmx - jj)
            if dd == 0:
                f0raw0[jj] = f0cand[jjmx, idx]
            else:
                f0raw0[jj] = lastf0 + (f0cand[jjmx, idx] - lastf0) / (dd + 1)
            if reliablepowerth < pwsdb[jj]:
                sprob += np.log2(val[idxx])
        else:
            f0raw0[jj] = lastf0
            if reliablepowerth < pwsdb[jj]:
                sprob += np.log2(val[idxx]) - 10
        lastf0 = f0raw0[jj]

    return 2 ** (sprob / (ub - lb))


def zfillf0gaps6(
    f0: np.ndarray,
    cseg: list[list[int]],
    f0cand: np.ndarray,
    relv: np.ndarray,
    pwsdb: np.ndarray,
    f0jumpt: float,
    nsdt: float,
    noiselevel: float,
) -> np.ndarray:
    """Fill F0 gaps.

    Parameters
    ----------
    f0 : np.ndarray [shape=(nn,)]
        Fundamental frequency.

    cseg : list[list[int]]
        Segment indices.

    f0cand : np.ndarray [shape=(nn, 6)]
        F0 candidates.

    relv : np.ndarray [shape=(nn, 6)]
        Reliability values.

    pwsdb : np.ndarray [shape=(nn,)]
        Power in dB.

    f0jumpt : float
        Threshold for F0 jump.

    nsdt : float
        Standard deviation for normalization.

    noiselevel : float
        Noise level in dB.

    Returns
    -------
    out : np.ndarray [shape=(nn,)]
        Filled fundamental frequency.

    """
    f0c = f0.copy()
    nr = len(cseg)
    nf0 = len(f0)
    f0raw0 = f0.copy()
    for ii in range(nr):
        lb = 0 if ii == 0 else mround((cseg[ii][0] + cseg[ii - 1][1] - 1) / 2)
        ub = min(nf0, cseg[ii][0] + 1)
        bp = lb
        ep = ub - 1
        args1 = [f0cand, relv, pwsdb]
        args2 = [lb, ub, f0jumpt, nsdt, noiselevel]
        lastf0 = f0raw0[ep] = f0[ep]
        ztraceInAsegment2(f0raw0, *args1, ep, lastf0, *args2)
        f0c[lb:ub] = f0raw0[lb:ub]

        lb = cseg[ii][1] - 1
        ub = nf0 if ii == nr - 1 else mround((cseg[ii][1] + cseg[ii + 1][0] + 1) / 2)
        bp = lb
        ep = ub - 1
        args1 = [f0cand, relv, pwsdb]
        args2 = [lb, ub, f0jumpt, nsdt, noiselevel]
        lastf0 = f0raw0[bp] = f0[bp]
        ztraceInAsegment2(f0raw0, *args1, bp, lastf0, *args2)
        f0c[lb:ub] = f0raw0[lb:ub]

    return f0c


def zrefineF06m(
    x: np.ndarray,
    fs: float,
    f0raw: np.ndarray,
    fftl: int,
    eta: float,
    nhmx: int,
    shiftm: float,
    nl: int = 0,
    nu: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Refine estimated F0.

    Parameters
    ----------
    x : np.ndarray [shape=(nx,)]
        Input signal.

    fs : float
        Sampling frequency.

    f0raw : np.ndarray [shape=(nn,)]
        Estimated F0.

    fftl : int
        FFT length.

    eta : float
        Temporal stretching factor.

    nhmx : int
        Highest harmonic number.

    shiftm : float
        Frame shift in msec.

    nl : int
        Lower frame number.

    nu : int
        Upper frame number.

    Returns
    -------
    f0r : np.ndarray [shape=(nn,)]
        Refined F0.

    ecr : np.ndarray [shape=(nn,)]
        Auxiliary output.

    ac1 : np.ndarray [shape=(nn,)]
        Auxiliary output.

    """
    f0i = f0raw.copy()
    f0i[f0i == 0] = 160
    f0i[f0i < 40] = 40
    fax = np.arange(fftl) / fftl * fs
    nfr = len(f0i)

    shiftl = shiftm / 1000 * fs
    x = np.pad(x, (fftl, fftl))

    tf = TAU / fftl
    ec1 = np.cos(tf * np.arange(fftl))
    ac1 = np.zeros_like(f0raw)

    tt = (np.arange(fftl) + 1 - fftl // 2) / fs
    th = np.arange(fftl) * tf
    rr = np.exp(-1j * th)

    f0t = 100
    w1 = np.maximum(0, 1 - np.abs(tt * f0t / eta))
    w1 = w1[0 < w1]
    wg = np.exp(-np.pi * (tt * f0t / eta) ** 2)
    wgg = wg[0.0002 < np.abs(wg)]
    wo = fftfilt(wgg, np.pad(w1, (0, len(wgg))))

    nlo = len(wo) - 1
    xo = np.arange(len(wo)) / nlo
    if nu <= nl:
        nu = nfr

    nh = fftl // 2 + 1
    pif = np.zeros((nh, nfr))
    dpif = np.zeros((nh, nfr))
    pwm = np.zeros((nh, nfr))
    rmsValue = mstd(x)
    for kk in range(nl, nu):
        f0t = f0i[kk]
        xi = mrange(0, 1 / nlo * f0t / 100, 1)
        wa = interp1(xo, wo, xi, method="*linear")
        wal = len(wa)
        bias = mround(fftl - wal / 2 + kk * shiftl)
        txm1 = x[bias - 1 : bias - 1 + wal]
        tx0 = x[bias : bias + wal]
        txp1 = x[bias + 1 : bias + 1 + wal]
        if mstd(txm1) * mstd(tx0) * mstd(txp1) == 0:
            xtmp = randn(wal + 2, rmsValue * 1e-5)
            txm1 = xtmp[:-2]
            tx0 = xtmp[1:-1]
            txp1 = xtmp[2:]
        dcl = np.mean(tx0)
        ff0 = np.fft.fft((txm1 - dcl) * wa, n=fftl)
        ff1 = np.fft.fft((tx0 - dcl) * wa, n=fftl)
        ff2 = np.fft.fft((txp1 - dcl) * wa, n=fftl)
        ff0[ff0 == 0] = 1e-9
        ff1[ff1 == 0] = 1e-9
        ff2[ff2 == 0] = 1e-9
        fd = ff2 * rr - ff1
        fd0 = ff1 * rr - ff0
        fa1 = np.abs(ff1) ** 2
        fa0 = np.abs(ff0) ** 2
        crf = (
            fax * TAU
            + (np.real(ff1) * np.imag(fd) - np.imag(ff1) * np.real(fd)) / fa1 * fs
        )
        crf0 = (
            fax * TAU
            + (np.real(ff0) * np.imag(fd0) - np.imag(ff0) * np.real(fd0)) / fa0 * fs
        )
        pif[:, kk] = crf[:nh]
        dpif[:, kk] = crf[:nh] - crf0[:nh]
        pwm[:, kk] = np.abs(ff1[:nh])
        ac1[kk] = np.sum(fa1 * ec1) / np.sum(fa1)

    slp = np.diff(pif, axis=0) / (fs * tf)
    slp = np.pad(slp, ((0, 1), (0, 0)))
    dslp = np.diff(dpif, axis=0) / tf
    dslp = np.pad(dslp, ((0, 1), (0, 0)))

    c1, c2 = znrmlcf2(shiftm)
    c1 /= 2
    c2 /= 2
    fxx = (np.arange(nh) + 0.5) * (fs * tf)
    c2f = c2 * (fxx[:, None] / TAU) ** 2
    mmp = (dslp / np.sqrt(c2f)) ** 2 + (slp / np.sqrt(c1)) ** 2

    sml = mround(1.5 * fs / 1000 / 2 / shiftm) * 2 + 1
    smb = (sml - 1) // 2

    w2 = hanning(sml, square=True, norm=True)
    w3 = hanning(sml, norm=True)
    smmp = fftfilt(w2, np.pad(mmp, ((0, 0), (0, 2 * sml))) + np.max(mmp) * 1e-7)
    smmp = 1 / fftfilt(w3, 1 / smmp)
    smmp = cut(smmp, max(0, sml - 2), nfr)

    spwm = fftfilt(w3, np.pad(pwm, ((0, 0), (0, 2 * sml))) + 1e-5)
    spfm = fftfilt(w3, np.pad(pwm * pif, ((0, 0), (0, 2 * sml))) + 1e-5)
    spif = spfm / spwm
    spif = spif[:, smb : smb + nfr]

    idx = f0i * (fftl / fs)
    ramp = np.arange(nfr)
    vvv = np.empty((nhmx, nfr))
    fqv = np.empty((nhmx, nfr))
    for ii in range(1, nhmx + 1):
        iidx = ii * idx
        fiidx = np.floor(iidx).astype(np.int64)
        fiidx = np.minimum(fiidx, nh - 2)
        fiidx1 = fiidx + 1
        diff = iidx - fiidx
        vvv[ii - 1] = (ii * ii) / (
            smmp[fiidx, ramp] + diff * (smmp[fiidx1, ramp] - smmp[fiidx, ramp])
        )
        fqv[ii - 1] = (
            spif[fiidx, ramp] + diff * (spif[fiidx1, ramp] - spif[fiidx, ramp])
        ) / (TAU * ii)

    vvvf = np.sum(vvv, axis=0)
    f0r = np.sum(fqv * np.sqrt(vvv), axis=0) / np.sum(np.sqrt(vvv), axis=0)
    f0r *= 0 < f0raw
    ecr = np.sqrt(vvvf)
    ecr = ecr * (0 < f0raw) + (f0raw <= 0)
    return f0r, ecr, ac1


def zvuvdecision4(
    f0: np.ndarray, rel: np.ndarray, pwsdb: np.ndarray, shiftm: float, noiselevel: float
) -> np.ndarray:
    """Perform V/UV decision.

    Parameters
    ----------
    f0 : np.ndarray [shape=(nn,)]
        Fundamental frequency.

    rel : np.ndarray [shape=(nn,)]
        Reliability values.

    pwsdb : np.ndarray [shape=(nn,)]
        Power in dB.

    shiftm : float
        Frame shift in msec.

    noiselevel : float
        Noise level in dB.

    Returns
    -------
    vuv : np.ndarray [shape=(nn,)]
        V/UV indicator.

    """
    maxpwsdb = np.max(pwsdb)

    nw = math.ceil(40 / shiftm)
    nrw = 3 / shiftm
    tt = np.arange(-nw, nw + 1)
    pws = 10 ** (pwsdb / 20)
    wwh = np.exp(-((tt / (nw / 2.5)) ** 2)) * (0.5 - sigmoid(tt / nrw))
    dpw = fftfilt(wwh, np.pad(pws, (0, 2 * nw)))
    dpw = dpw[nw : nw + len(pws)]
    biast = math.ceil(nrw * 3 / shiftm)

    ddpw = np.pad(np.diff(dpw), (1, 0))
    ddpwm = np.roll(ddpw, -1)
    onv = np.where((ddpw * ddpwm < 0) & (ddpwm <= 0))[0]

    pv, _ = zpeakdipdetect(pwsdb, mround(81 / shiftm))
    nn = min(len(pwsdb), len(f0))
    vuv = np.zeros(nn)
    if len(onv) == 0 or np.all(f0 == 0):
        return vuv

    lastp = 1
    for ii in range(len(pv)):
        if (1.2 * maxpwsdb + noiselevel) / 2.2 < pwsdb[pv[ii]] and lastp < pv[ii]:
            lb = lastp
            ub = nn - 1
            bp = ep = cp = pv[ii]
            for bp in reversed(range(lb, cp)):
                cond1 = pwsdb[bp] < (maxpwsdb + 2.3 * noiselevel) / 3.3
                cond2 = pwsdb[bp] < (1.5 * maxpwsdb + noiselevel) / 2.5
                cond3 = rel[bp] < 0.3
                cond4 = cond2
                cond5 = 0.1 < np.abs(_log_ratio(f0[bp], f0[bp - 1]))
                if cond1 or (cond2 and cond3) or (cond4 and cond5):
                    break
            val = np.abs(onv - bp)
            ix = np.argmin(val)
            if val[ix] < 20 / shiftm:
                bp = max(0, onv[ix] - biast)
            for ep in range(cp + 1, ub):
                cond1 = pwsdb[ep] < (maxpwsdb + 5 * noiselevel) / 6
                cond2 = pwsdb[ep] < (maxpwsdb + 1.3 * noiselevel) / 2.3
                cond3 = rel[ep] < 0.25
                cond4 = pwsdb[ep] < (maxpwsdb + 0.7 * noiselevel) / 1.7
                cond5 = 0.1 < np.abs(_log_ratio(f0[ep], f0[ep + 1]))
                if cond1 or (cond2 and cond3) or (cond4 and cond5):
                    break
            vuv[bp : ep + 1] = 1
            lastp = ep

    return vuv


def zpeakdipdetect(pwsdb: np.ndarray, wsml: int) -> tuple[np.ndarray, np.ndarray]:
    """Detect peaks and dips.

    Parameters
    ----------
    pwsdb : np.ndarray
        Power in dB.

    wsml : int
        Window length.

    Returns
    -------
    pv : np.ndarray
        Index of peaks.

    dv : np.ndarray
        Index of dips.

    """
    pwsdbl = np.pad(pwsdb, (wsml, 2 * wsml), mode="edge")
    pwsdbs = fftfilt(hanning(wsml, norm=True), pwsdbl)
    pwsdbs = cut(pwsdbs, mround(3 * wsml / 2), len(pwsdb))
    dpwsdbs = np.pad(np.diff(pwsdbs), (1, 0))
    dpwsdbsm = np.roll(dpwsdbs, -1)
    pv = np.where((dpwsdbs * dpwsdbsm < 0) & (dpwsdbsm <= 0))[0]
    dv = np.where((dpwsdbs * dpwsdbsm < 0) & (0 < dpwsdbsm))[0]
    return pv, dv


MulticueF0v14 = SourceInfobyMultiCues050111
