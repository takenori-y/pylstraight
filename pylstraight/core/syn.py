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

import math
from dataclasses import dataclass

import numpy as np

from .utils.mat import TAU, cut, fftfilt, hanning, interp1, mround, randn, sigmoid


@dataclass
class SynParam:
    """Control parameters for the synthesis."""

    spectral_update_interval: float = 1.0
    group_delay_standard_deviation: float = 0.5
    group_delay_spatial_bandwidth: float = 70.0
    group_delay_randomize_corner_frequency: float = 4000.0
    ratio_to_fundamental_period: float = 0.2
    ratio_mode_indicator: int = 0
    time_axis_mapping_table: np.ndarray | None = None  # imap
    time_axis_stretching_factor: float = 1.0  # sconv
    frequency_axis_mapping_table: np.ndarray | None = None  # fconv
    frequency_axis_stretching_factor: float = 1.0  # fconv
    pitch_stretching_factor: float = 1.0  # pconv
    lowest_f0: float = 40.0


def exstraightsynth(
    f0raw: np.ndarray,
    n3sgram: np.ndarray,
    ap: np.ndarray,
    fs: int,
    prm: SynParam | None,
) -> np.ndarray:
    """Synthesize waveform using STRAIGHT parameters with linear modifications.

    Parameters
    ----------
    f0raw : np.ndarray [shape=(nframe,)]
        Fundamental frequency in Hz.

    n3sgram : np.ndarray [shape=(nframe, nfreq)]
        Spectrogram in absolute scale.

    ap : np.ndarray [shape=(nframe, nfreq)]
        Aperiodic component in dB.

    fs : int
        Sampling frequency in Hz.

    prm : SynParam or None
        Control parameters for the synthesis.

    Returns
    -------
    out : np.ndarray [shape=(nsample,)]
        Synthesized waveform.

    """
    if prm is None:
        prm = SynParam()  # pragma: no cover

    # Set parameters.
    shiftm = prm.spectral_update_interval
    pconv = prm.pitch_stretching_factor
    fconv = prm.frequency_axis_stretching_factor
    sconv = prm.time_axis_stretching_factor
    gdbw = prm.group_delay_spatial_bandwidth
    delfrac = prm.ratio_to_fundamental_period
    delsp = prm.group_delay_standard_deviation
    cornf = prm.group_delay_randomize_corner_frequency
    delfracind = prm.ratio_mode_indicator
    idcv = prm.frequency_axis_mapping_table
    imap = prm.time_axis_mapping_table
    lowestF0 = prm.lowest_f0

    return straightSynthTB07ca(
        n3sgram,
        f0raw,
        shiftm,
        fs,
        pconv,
        fconv,
        sconv,
        gdbw,
        delfrac,
        delsp,
        cornf,
        delfracind,
        ap,
        idcv,
        imap,
        lowestF0,
    )


def straightSynthTB07ca(
    n2sgram: np.ndarray,
    f0raw: np.ndarray,
    shiftm: float,
    fs: int,
    pcnv: float,
    fconv: float,
    sconv: float,
    gdbw: float,
    delfrac: float,
    delsp: float,
    cornf: float,
    delfracind: int,
    ap: np.ndarray,
    idcv: np.ndarray | None,
    imap: np.ndarray | None,
    lowestF0: float,
) -> np.ndarray:
    """Synthesize waveform with all-pass filter design based on TEMPO analysis result.

    Parameters
    ----------
    n2sgram : np.ndarray [shape=(nframe, nfreq)]
        Amplitude spectrogram.

    f0raw : np.ndarray [shape=(nframe,)]
        Fundamental frequency in Hz.

    shiftm : float
        Frame shift in msec.

    fs : int
        Sampling frequency in Hz.

    pcnv : float
        Pitch stretching factor.

    fconv : float
        Frequency stretching factor.

    sconv : float
        Speaking duration stretching factor.

    gdbw : float
        Einest resolution in group delay in Hz.

    delfrac : float
        Ratio of standard deviation of group delay in terms of F0.

    delsp : float
        Standard deviation of group delay.

    cornf : float
        Lower corner frequency for phase randomization in Hz.

    delfracind : int
        Selector of fixed and proportional group delay.

    ap : np.ndarray [shape=(nframe, nfreq)]
        Aperiodicity measure.

    idcv : np.ndarray [shape=(nfreq,)] or None
        Arbitrary mapping of frequency axis.

    imap : np.ndarray or None
        Arbitrary mapping from new time (sample) to old time (frame).

    lowestF0 : float
        Lower limit of the resynthesized fundamental frequency in Hz.

    Returns
    -------
    out : np.ndarray [shape=(nsample,)]
        Synthesized waveform.

    """
    njj, nii = n2sgram.shape
    njj = min(njj, len(f0raw))
    f0l = f0raw[:njj]

    fftLengthForLowestF0 = 2 ** math.ceil(np.log2(2 * mround(fs / lowestF0)))
    fftl = 2 * nii - 2
    if fftl < fftLengthForLowestF0:
        niiNew = fftLengthForLowestF0 // 2 + 1
        inx = np.arange(nii)
        inxq = np.arange(niiNew) * ((nii - 1) / (niiNew - 1))
        n2sgram = interp1(inx, n2sgram, inxq)
        fftl = fftLengthForLowestF0
        nii = niiNew

    if ap.shape[1] != nii:
        inx = np.arange(ap.shape[1])
        inxq = np.arange(nii) * ((ap.shape[1] - 1) / (nii - 1))
        ap = interp1(inx, ap, inxq)

    aprms = 10 ** (ap / 20)
    aprm = np.clip(aprms * 1.6 - 0.015, 0.001, 1)

    fftl2 = fftl // 2
    if idcv is None:
        idcv = np.minimum(np.arange(fftl2 + 1) / fconv, fftl2)
    idcv = mround(idcv)

    if imap is None:
        rate = shiftm / 1000 * fs * sconv
        sy = np.zeros(mround(njj * rate + 3 * fftl + 1))
        imap_ = np.arange(len(sy))
        imap_ = np.minimum(imap_ / rate, njj - 1)
    else:
        sy = np.zeros(len(imap) + 3 * fftl)
        imap_ = imap
    imap_ = np.pad(imap_, (0, mround(fs * 0.2)))
    ix = np.where(njj - 1 <= imap_)[0][0] + 1
    rmap = interp1(imap_[:ix], np.arange(ix), np.arange(njj))
    imap_: np.ndarray = mround(imap_)

    phs, t = fractpitch2(fftl)

    nsyn = len(sy)

    adjd = sigmoid(20 * t)
    gw = np.exp(-0.25 * np.pi * (fs / gdbw / 2 * t) ** 2)
    gw /= np.sum(gw)
    fgw = np.real(np.fft.fft(np.fft.fftshift(gw)))
    df = fs / fftl * TAU
    fw = np.arange(fftl2 + 1) * (fs / fftl)

    trbw = 300
    rho = sigmoid((fw - cornf) / trbw)

    w = hanning(fftl)
    lft = sigmoid((0.5 - w) * 60)
    ww = sigmoid((w - 0.3) * 23)

    idxs = []
    iis = []
    f0s = []

    idx = 0.0
    while idx < nsyn - fftl - 11:
        iix = imap_[mround(idx)]
        ii = np.clip(iix, 0, njj - 1)

        f0 = 200 if f0l[ii] == 0 else max(lowestF0 / pcnv, f0l[ii])
        f0 *= pcnv

        tnf0 = fs / f0
        tidx = idx + tnf0
        tiix = imap_[mround(tidx)]
        tii = np.clip(tiix, 0, njj - 1)
        tf0 = f0l[tii]
        if 0 < tf0 and 0 < f0l[ii] and 0 < f0l[mround((ii + tii) / 2)]:
            f0 = max(lowestF0 / pcnv, f0l[mround((ii + tii) / 2)])
            f0 *= pcnv

        idxs.append(idx)
        iis.append(ii)
        f0s.append(f0)

        nf0 = fs / f0
        idx += nf0
        iin = np.clip(imap_[mround(idx)], 0, njj - 1)
        if iin == njj - 1:
            break

        if f0raw[ii] == 0 and 0 < f0raw[iin]:
            idxo = idx
            ipos = np.where(0 < f0raw[ii : iin + 1])[0]
            if len(ipos) == 0:
                idx = idxo
            else:
                ipos = ipos[0] + ii
                idx = max(idxo - nf0 + 1, rmap[ipos])

    # Voiced part synthesis.
    if 0 < len(iis):
        idx = np.array(idxs)
        ii = np.array(iis)
        f0 = np.array(f0s)

        ff = np.pad(n2sgram[ii][:, idcv], ((0, 0), (0, fftl2 - 1)), mode="reflect")
        dmx = np.max(n2sgram)
        ccp = np.real(np.fft.fft(np.log(ff + dmx * 1e-6)))
        ccp2 = np.pad(2 * ccp[:, : fftl2 + 1], ((0, 0), (0, fftl2 - 1)))
        ccp2[:, 0] /= 2
        ffx = np.fft.fft(ccp2 * lft) / fftl
        nidx: np.ndarray = mround(idx)

        nf0 = fs / f0
        frt = idx - nidx
        frtz = np.exp(1j * phs * frt[:, None])
        nz = randn((len(idx), fftl2 + 1)) * rho
        nz = np.pad(nz, ((0, 0), (0, fftl2 - 1)), mode="reflect")
        nz = np.real(np.fft.ifft(np.fft.fft(nz) * fgw))
        nz *= np.sqrt(fftl * gdbw / fs)
        if delfracind:
            delsp = delfrac * 1000 / f0[:, None]
        nz *= delsp * (df / 1000)
        mz = np.cumsum(nz, axis=1) - nz[:, :1]
        mmz = -(mz - adjd * (np.remainder(mz[:, -1:] + mz[:, 1:2], TAU) - TAU))
        pzr = np.exp(-1j * mmz)
        wnz = aprm[ii][:, idcv]
        wpr = np.sqrt(np.maximum(0, 1 - wnz**2))
        wnz = np.pad(wnz, ((0, 0), (0, fftl2 - 1)), mode="reflect")
        wpr = np.pad(wpr, ((0, 0), (0, fftl2 - 1)), mode="reflect")

        zt0 = nf0 / fs
        ztc = 0.01
        wf = np.zeros_like(mmz)
        mnf0: np.ndarray = mround(nf0)
        for jj, nf0n in enumerate(mnf0):
            ztp = np.arange(nf0n) / fs
            zt0c = 2 * zt0[jj] / ztc
            ztpc = ztp / ztc
            nev = np.sqrt(zt0c / (1 - np.exp(-zt0c))) * np.exp(-ztpc)
            rx = randn(nf0n)
            wf[jj, :nf0n] = (rx - np.mean(rx)) * nev
        wfv = np.fft.fft(wf)

        ep = np.zeros_like(mmz)
        for jj, nf0n in enumerate(mnf0):
            gh = hanning(nf0n * 2)
            ep[jj, :nf0n] = np.flip(gh[:nf0n])
            ep[jj, -(nf0n - 1) :] = gh[: nf0n - 1]
        ep /= -np.sum(ep, axis=1, keepdims=True)
        ep[:, 0] += 1
        epf = np.fft.fft(ep)

        efx = np.exp(ffx) * frtz
        tx = np.fft.fftshift(np.real(np.fft.ifft(epf * efx * pzr * wpr)), axes=1) * ww
        tx2 = np.fft.fftshift(np.real(np.fft.ifft(wfv * efx * wnz)), axes=1) * ww

        txx = (tx * np.sqrt(nf0)[:, None] + tx2) * (0 < f0raw[ii, None])
        for jj, begin in enumerate(nidx):
            sy[begin : begin + fftl] += txx[jj]

    nidx = []
    iis = []
    f0 = 1000
    nf0 = fs / f0

    idx = 0.0
    while idx < nsyn - fftl:
        ii = imap_[mround(idx)]
        if ii == njj - 1:
            break
        if f0raw[ii] == 0:
            nidx.append(mround(idx))
            iis.append(ii)
        idx += nf0

    # Unvoiced part synthesis.
    if 0 < len(iis):
        ii = np.array(iis)
        ff = np.pad(n2sgram[ii][:, idcv], ((0, 0), (0, fftl2 - 1)), mode="reflect")
        dmx = np.max(n2sgram)
        ccp = np.real(np.fft.fft(np.log(ff + dmx * 1e-6)))
        ccp2 = np.pad(2 * ccp[:, : fftl2 + 1], ((0, 0), (0, fftl2 - 1)))
        ccp2[:, 0] /= 2
        ffx = np.fft.fft(ccp2 * lft) / fftl
        tx = np.fft.fftshift(np.real(np.fft.ifft(np.exp(ffx))), axes=1)
        rx = randn((len(ii), mround(nf0)))
        tnx = fftfilt(rx - np.mean(rx), tx)
        for jj, begin in enumerate(nidx):
            sy[begin : begin + fftl] += tnx[jj] * ww

    shiftl = mround(shiftm / 1000 * fs * sconv)
    return cut(sy, fftl2, ix - 1 + shiftl)


def fractpitch2(fftl: int) -> tuple[np.ndarray, np.ndarray]:
    """Generate phase rotator for fractional pitch.

    Parameters
    ----------
    fftl : int
        FFT length.

    Returns
    -------
    phs : np.ndarray [shape=(fftl,)]
        The phase rotator.

    t : np.ndarray [shape=(fftl,)]
        Auxiliary variable.

    """
    amp = 15
    t = (np.arange(fftl) - fftl // 2) * (2 / fftl)
    phs = t - np.tanh(0.5 * amp * t) + (np.tanh(0.5 * amp) - 1) * t
    phs[0] = 0
    phs *= np.pi
    return phs, t
