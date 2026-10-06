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
import os

import numpy as np


def is_debug_mode() -> bool:
    """Return True if the current environment is debug mode.

    Returns
    -------
    out : bool
        True if the current environment is debug mode.

    """
    return 0 < int(os.getenv("PYLSTRAIGHT_DEBUG", "0"))


def get_fft_length(
    fs: int, frame_length_in_msec: float = 80.0, mode: str = "full"
) -> int:
    """Calculate FFT length from the sampling frequency and frame length.

    Parameters
    ----------
    fs : int
        Th sampling frequency in Hz.

    frame_length_in_msec : float
        The frame length in msec.

    mode : ['full', 'one-sided']
        The spectrum mode.

    Returns
    -------
    out : int
        The FFT length.

    """
    frame_length = frame_length_in_msec * fs / 1000
    fft_length = max(1024, 2 ** math.ceil(np.log2(frame_length)))
    if mode == "full":
        pass
    elif mode == "one-sided":
        fft_length = fft_length // 2 + 1
    else:
        msg = f"Unsupported mode: {mode}"
        raise ValueError(msg)
    return fft_length


def normalize_waveform(x: np.ndarray) -> tuple[np.ndarray, float]:
    """Normalize waveform.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,) or (nchannel, nsample)]
        The waveform.

    Returns
    -------
    z : np.ndarray [shape=(nsample,)]
        The normalized waveform.

    scalar : float
        The scalar value.

    """
    if x.shape[-1] == 0:
        msg = "Input signal must be non-empty."
        raise ValueError(msg)

    if x.dtype == np.int16:
        scalar = 32768.0
        z = x / scalar
    elif x.dtype.type in {np.float32, np.float64}:
        scalar = 1.0
        z = x
    else:
        msg = f"Unsupported data type: {x.dtype}"
        raise ValueError(msg)

    if z.ndim == 2:
        z = z.mean(axis=0)
    if z.ndim != 1:
        msg = "Input signal must be 1 or 2 dimensional."
        raise ValueError(msg)

    return z, scalar
