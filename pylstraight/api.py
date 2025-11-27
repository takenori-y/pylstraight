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

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from types import SimpleNamespace

import math

import numpy as np
import soundfile as sf

from .core.ap import ApParam, exstraightAPind
from .core.f0 import F0Param, MulticueF0v14
from .core.sp import SpParam, exstraightspec
from .core.syn import SynParam, exstraightsynth
from .core.utils.misc import get_fft_length, normalize_waveform

magic_number: float = -1e10


def f0_to_f0(
    f0: np.ndarray, in_format: str, out_format: str, fs: int = 0
) -> np.ndarray:
    """Convert F0 between different formats.

    Parameters
    ----------
    f0 : np.ndarray [shape=(nframe,)]
        The input F0.

    in_format : ['inverse', 'linear', 'log']
        The format of the input F0.

    out_format : ['inverse', 'linear', 'log']
        The format of the output F0.

    fs : int
        The sampling frequency in Hz. This is required when the input or output format
        is *inverse*.

    Returns
    -------
    out : np.ndarray [shape=(nframe,)]
        The converted F0.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> import numpy as np
    >>> f0 = np.array([100, 200, 0, 400])
    >>> pyls.f0_to_f0(f0, "linear", "log")
    array([ 4.60517019e+00,  5.29831737e+00, -1.00000000e+10,  5.99146455e+00])
    >>> pyls.f0_to_f0(f0, "linear", "inverse", fs=16000)
    array([160.,  80.,   0.,  40.])

    """
    if not isinstance(f0, np.ndarray):
        msg = "F0 must be an instance of numpy.ndarray."
        raise TypeError(msg)

    f0 = f0.astype(np.float64)
    if in_format == out_format:
        return f0

    if (in_format == "inverse" or out_format == "inverse") and fs <= 0:
        msg = "Sampling frequency is required."
        raise ValueError(msg)

    voiced = f0 != (magic_number if in_format == "log" else 0)
    voiced_func = {
        "inverse": {
            "linear": lambda x: fs / x,
            "log": lambda x: np.log(fs / x),
        },
        "linear": {
            "inverse": lambda x: fs / x,
            "log": np.log,
        },
        "log": {
            "inverse": lambda x: fs / np.exp(x),
            "linear": np.exp,
        },
    }

    unvoiced = ~voiced
    unvoiced_value = magic_number if out_format == "log" else 0

    new_f0 = np.empty_like(f0)
    try:
        new_f0[voiced] = voiced_func[in_format][out_format](f0[voiced])
        new_f0[unvoiced] = unvoiced_value
    except KeyError as err:
        msg = f"Invalid input/output format: {in_format}/{out_format}."
        raise ValueError(msg) from err
    return new_f0


def ap_to_ap(ap: np.ndarray, in_format: str, out_format: str) -> np.ndarray:
    """Convert aperiodicity between different formats.

    Parameters
    ----------
    ap : np.ndarray [shape=(nframe, nfreq)]
        The input aperiodicity.

    in_format : ['a', 'p', 'a/p', 'p/a']
        The format of the input aperiodicity.

    out_format : ['a', 'p', 'a/p', 'p/a']
        The format of the output aperiodicity.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        The converted aperiodicity.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> import numpy as np
    >>> ap = np.array([[0.3, 0.5, 0.9]])
    >>> pyls.ap_to_ap(ap, "a", "p")
    array([[0.7, 0.5, 0.1]])
    >>> pyls.ap_to_ap(ap, "a", "a/p")
    array([[0.42857143, 1.        , 9.        ]])

    """
    if not isinstance(ap, np.ndarray):
        msg = "Aperiodicity must be an instance of numpy.ndarray."
        raise TypeError(msg)

    ap = ap.astype(np.float64)
    if in_format == out_format:
        return ap

    func = {
        "a": {
            "p": lambda x: 1 - x,
            "a/p": lambda x: x / (1 - x),
            "p/a": lambda x: (1 - x) / x,
        },
        "p": {
            "a": lambda x: 1 - x,
            "a/p": lambda x: (1 - x) / x,
            "p/a": lambda x: x / (1 - x),
        },
        "a/p": {
            "a": lambda x: x / (1 + x),
            "p": lambda x: 1 / (1 + x),
            "p/a": lambda x: 1 / x,
        },
        "p/a": {
            "a": lambda x: 1 / (1 + x),
            "p": lambda x: x / (1 + x),
            "a/p": lambda x: 1 / x,
        },
    }

    try:
        new_ap = func[in_format][out_format](ap)
    except KeyError as err:
        msg = f"Invalid input/output format: {in_format}/{out_format}."
        raise ValueError(msg) from err
    return new_ap


def sp_to_sp(sp: np.ndarray, in_format: str, out_format: str) -> np.ndarray:
    """Convert spectrum between different formats.

    Parameters
    ----------
    sp : np.ndarray [shape=(nframe, nfreq)]
        The input spectrum.

    in_format : ['db', 'log', 'linear', 'power']
        The format of the input spectrum.

    out_format : ['db', 'log', 'linear', 'power']
        The format of the output spectrum.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        The converted spectrum.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> import numpy as np
    >>> sp = np.array([[-10, 60, 0]])
    >>> pyls.sp_to_sp(sp, "db", "linear")
    array([[3.16227766e-01, 1.00000000e+03, 1.00000000e+00]])
    >>> pyls.sp_to_sp(sp, "db", "log")
    array([[-1.15129255,  6.90775528,  0.        ]])

    """
    if not isinstance(sp, np.ndarray):
        msg = "Spectrum must be an instance of numpy.ndarray."
        raise TypeError(msg)

    sp = sp.astype(np.float64)
    if in_format == out_format:
        return sp

    func = {
        "db": {
            "log": lambda x: x * (np.log(10) / 20),
            "linear": lambda x: 10 ** (x / 20),
            "power": lambda x: 10 ** (x / 10),
        },
        "log": {
            "db": lambda x: x * (20 / np.log(10)),
            "linear": np.exp,
            "power": lambda x: np.exp(2 * x),
        },
        "linear": {
            "db": lambda x: 20 * np.log10(x),
            "log": np.log,
            "power": np.square,
        },
        "power": {
            "db": lambda x: 10 * np.log10(x),
            "log": lambda x: np.log(x) / 2,
            "linear": np.sqrt,
        },
    }

    try:
        new_sp = func[in_format][out_format](sp)
    except KeyError as err:
        msg = f"Invalid input/output format: {in_format}/{out_format}."
        raise ValueError(msg) from err
    return new_sp


def init_f0_param() -> F0Param:
    """Make a default F0 parameter set.

    Returns
    -------
    out : F0Param
        Control parameter set for the F0 extraction.

    """
    return F0Param()


def init_ap_param() -> ApParam:
    """Make a default aperiodicity parameter set.

    Returns
    -------
    out : ApParam
        Control parameter set for the aperiodicity extraction.

    """
    return ApParam()


def init_sp_param() -> SpParam:
    """Make a default spectrum parameter set.

    Returns
    -------
    out : SpParam
        Control parameter set for the spectrum extraction.

    """
    return SpParam()


def init_syn_param() -> SynParam:
    """Make a default synthesis parameter set.

    Returns
    -------
    out : SynParam
        Control parameter set for the synthesis.

    """
    return SynParam()


def _extract_f0(
    x: np.ndarray,
    fs: int,
    *,
    frame_shift: float = 5.0,
    f0_range: tuple[float, float] = (40.0, 400.0),
    f0_format: str = "linear",
    f0_param: F0Param | None = None,
    return_aux: bool = False,
) -> np.ndarray:
    """Extract F0 from waveform.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,) or (nchannel, nsample)]
        The input waveform. If multi-channel, it will be averaged.

    fs : int
        The sampling frequency in Hz.

    frame_shift : float
        The frame shift in msec.

    f0_range : tuple[float, float]
        The lower and upper bounds of F0 search in Hz.

    f0_format : ['inverse', 'linear', 'log']
        The output format.

    f0_param : F0Param or None
        Control parameters for the F0 extraction. If given, override the other
        parameters. You have full control and responsibility.

    return_aux : bool
        Whether to return the auxiliary outputs.

    Returns
    -------
    f0 : np.ndarray [shape=(nframe,)]
        The fundamental frequency.

    auxouts : SimpleNamespace (optional)
        The auxiliary outputs.

    """
    x, _ = normalize_waveform(x)

    f0_floor, f0_ceil = f0_range

    if fs < 8000:
        msg = "Minimum sampling frequency is 8000 Hz."
        raise ValueError(msg)

    if frame_shift < 1:
        msg = "Minimum frame shift is 1 ms."
        raise ValueError(msg)

    if f0_floor < 40:
        msg = "Minimum F0 floor is 40 Hz."
        raise ValueError(msg)

    if fs / 2 < f0_ceil:
        msg = "F0 ceil exceeds the Nyquist frequency."
        raise ValueError(msg)

    if f0_param is None:
        f0_param = init_f0_param()
        f0_param.f0_frame_update_interval = frame_shift
        f0_param.f0_search_lower_bound = f0_floor
        f0_param.f0_search_upper_bound = f0_ceil

    nvo = f0_param.number_of_channels_in_octave
    if math.ceil(np.log2(f0_ceil / f0_floor) * nvo) < 3:
        msg = "F0 search range is too narrow."
        raise ValueError(msg)

    f0, vuv, auxouts = MulticueF0v14(x, fs, f0_param)
    f0 *= vuv
    f0[f0_ceil < f0] = f0_ceil
    f0 = f0_to_f0(f0, "linear", f0_format, fs=fs)

    if return_aux:
        return f0, auxouts

    return f0


def extract_f0(
    x: np.ndarray,
    fs: int,
    *,
    frame_shift: float = 5.0,
    f0_range: tuple[float, float] = (40.0, 400.0),
    f0_format: str = "linear",
    f0_param: F0Param | None = None,
    return_aux: bool = False,
    refine_f0_range: bool = False,
    gamma: float = 3.0,
) -> np.ndarray:
    """Extract F0 from waveform.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,) or (nchannel, nsample)]
        The input waveform. If multi-channel, it will be averaged.

    fs : int
        The sampling frequency in Hz.

    frame_shift : float
        The frame shift in msec.

    f0_range : tuple[float, float]
        The lower and upper bounds of F0 search in Hz.

    f0_format : ['inverse', 'linear', 'log']
        The output format.

    f0_param : F0Param or None
        Control parameters for the F0 extraction. If given, override the other
        parameters. You have full control and responsibility.

    return_aux : bool
        Whether to return the auxiliary outputs.

    refine_f0_range : bool
        Whether to refine the F0 search range based on the initial F0 statistics.

    gamma : float
        The width factor for the F0 range refinement.

    Returns
    -------
    f0 : np.ndarray [shape=(nframe,)]
        The fundamental frequency.

    auxouts : SimpleNamespace (optional)
        The auxiliary outputs.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> import numpy as np
    >>> f, fs = 200, 16000
    >>> x = np.mod(2 * np.pi * f / fs * np.arange(600), 2 * np.pi) - np.pi
    >>> pyls.extract_f0(x, fs)
    array([193.86025508, 198.46273078, 199.65873847, 199.98109482,
           199.99103883, 199.99119509])

    """
    f0 = _extract_f0(
        x,
        fs,
        f0_range=f0_range,
        frame_shift=frame_shift,
        f0_format=f0_format,
        f0_param=f0_param,
        return_aux=return_aux,
    )
    if not refine_f0_range:
        return f0

    lf0 = f0_to_f0(f0[0] if return_aux else f0, f0_format, "log", fs=fs)
    voiced_f0 = lf0[lf0 != magic_number]
    if len(voiced_f0) <= 1:
        return f0

    f0_mean = np.mean(voiced_f0)
    f0_sdev = np.std(voiced_f0)
    width = gamma * f0_sdev
    refined_f0_range = (
        max(np.exp(f0_mean - width), f0_range[0]),
        min(np.exp(f0_mean + width), f0_range[1]),
    )

    try:
        return _extract_f0(
            x,
            fs,
            f0_range=refined_f0_range,
            frame_shift=frame_shift,
            f0_format=f0_format,
            f0_param=f0_param,
            return_aux=return_aux,
        )
    except (ValueError, RuntimeError):
        return f0


def extract_ap(
    x: np.ndarray,
    fs: int,
    f0: np.ndarray,
    aux: SimpleNamespace | None = None,
    *,
    frame_shift: float = 5.0,
    ap_floor: float = 0.001,
    f0_format: str = "linear",
    ap_format: str = "a",
    ap_param: ApParam | None = None,
) -> np.ndarray:
    """Extract aperiodicity from waveform.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,) or (nchannel, nsample)]
        The input waveform. If multi-channel, it will be averaged.

    fs : int
        The sampling frequency in Hz.

    f0 : np.ndarray [shape=(nframe,)]
        The fundamental frequency.

    aux : SimpleNamespace or None
        The auxiliary outputs from `extract_f0`.

    frame_shift : float
        The frame shift in msec.

    ap_floor : float
        The minimum value of aperiodicity.

    f0_format : ['inverse', 'linear', 'log']
        The input format of the F0.

    ap_format : ['a', 'p', 'a/p', 'p/a']
        The output format.

    ap_param : ApParam or None
        Control parameters for the aperiodicity extraction. If given, override the other
        parameters. You have full control and responsibility.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        The aperiodicity.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> import numpy as np
    >>> f, fs = 200, 16000
    >>> x = np.sin(2 * np.pi * f / fs * np.arange(640))
    >>> f0 = np.ones(8) * f
    >>> ap = pyls.extract_ap(x, fs, f0)
    >>> ap.shape
    (8, 1025)
    >>> ap.mean(0)
    array([0.17473351, 0.1747341 , 0.17473478, ..., 0.66521378, 0.66521439,
           0.66521501])

    """
    x, _ = normalize_waveform(x)

    if fs < 8000:
        msg = "Minimum sampling frequency is 8000 Hz."
        raise ValueError(msg)

    if frame_shift < 1:
        msg = "Minimum frame shift is 1 ms."
        raise ValueError(msg)

    if ap_param is None:
        ap_param = init_ap_param()
        ap_param.periodicity_frame_update_interval = frame_shift
        ap_param.f0_frame_update_interval = frame_shift

    if len(f0) <= 1:
        msg = "The length of F0 is too short."
        raise ValueError(msg)

    frame_shift_in_sample = fs * ap_param.f0_frame_update_interval / 1000
    expected_f0_length = math.ceil(len(x) / frame_shift_in_sample)
    if not (expected_f0_length - 2 <= len(f0) <= expected_f0_length + 2):
        msg = (
            "The length of F0 is not consistent with the input waveform: "
            f"input {len(f0)} vs expected {expected_f0_length}."
        )
        raise ValueError(msg)

    ap = exstraightAPind(
        x,
        fs,
        f0_to_f0(f0, f0_format, "linear", fs=fs),
        None if aux is None else aux.refined_cn,
        ap_param,
    )
    ap = sp_to_sp(ap, "db", "linear")
    ap = np.clip(ap, ap_floor, 1 - ap_floor)
    return ap_to_ap(ap, "a", ap_format)


def extract_sp(
    x: np.ndarray,
    fs: int,
    f0: np.ndarray,
    *,
    frame_shift: float = 5.0,
    f0_format: str = "linear",
    sp_format: str = "linear",
    sp_param: SpParam | None = None,
) -> np.ndarray:
    """Extract spectrum from waveform.

    Parameters
    ----------
    x : np.ndarray [shape=(nsample,) or (nchannel, nsample)]
        The input waveform. If multi-channel, it will be averaged.

    fs : int
        The sampling frequency in Hz.

    f0 : np.ndarray [shape=(nframe,)]
        The fundamental frequency.

    frame_shift : float
        The frame shift in msec.

    f0_format : ['inverse', 'linear', 'log']
        The input format of the F0.

    sp_format : ['db', 'log', 'linear', 'power']
        The output format.

    sp_param : SpParam or None
        Control parameters for the spectrum extraction. If given, override the other
        parameters. You have full control and responsibility.

    Returns
    -------
    out : np.ndarray [shape=(nframe, nfreq)]
        The spectrum.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> import numpy as np
    >>> f, fs = 200, 16000
    >>> x = np.sin(2 * np.pi * f / fs * np.arange(640))
    >>> f0 = np.ones(8) * f
    >>> sp = pyls.extract_sp(x, fs, f0, sp_format='db')
    >>> sp.shape
    (8, 1025)
    >>> sp.mean(0)
    array([  9.3790045 ,   9.3959332 ,   9.44756499, ..., -74.57928489,
           -74.57916264, -74.57908274])

    """
    x, scaler = normalize_waveform(x)

    if fs < 8000:
        msg = "Minimum sampling frequency is 8000 Hz."
        raise ValueError(msg)

    if frame_shift < 1:
        msg = "Minimum frame shift is 1 ms."
        raise ValueError(msg)

    if sp_param is None:
        sp_param = init_sp_param()
        sp_param.spectral_update_interval = frame_shift

    if len(f0) <= 0:
        msg = "The length of F0 is too short."
        raise ValueError(msg)

    frame_shift_in_sample = fs * sp_param.spectral_update_interval / 1000
    expected_f0_length = math.ceil(len(x) / frame_shift_in_sample)
    if not (expected_f0_length - 2 <= len(f0) <= expected_f0_length + 2):
        msg = (
            "The length of F0 is not consistent with the input waveform: "
            f"input {len(f0)} vs expected {expected_f0_length}."
        )
        raise ValueError(msg)

    sp = exstraightspec(x, f0_to_f0(f0, f0_format, "linear", fs=fs), fs, sp_param)
    if scaler != 1:
        sp *= scaler
    return sp_to_sp(sp, "linear", sp_format)


def synthesize(
    f0: np.ndarray,
    ap: np.ndarray,
    sp: np.ndarray,
    fs: int,
    *,
    frame_shift: float = 5.0,
    f0_format: str = "linear",
    ap_format: str = "a",
    sp_format: str = "linear",
    syn_param: SynParam | None = None,
) -> np.ndarray:
    """Synthesize waveform from F0, aperiodicity, and spectrum.

    Parameters
    ----------
    f0 : np.ndarray [shape=(nframe,)]
        The fundamental frequency.

    ap : np.ndarray [shape=(nframe, nfreq)]
        The aperiodicity.

    sp : np.ndarray [shape=(nframe, nfreq)]
        The spectrum.

    fs : int
        The sampling frequency in Hz.

    frame_shift : float
        The frame shift in msec.

    f0_format : ['inverse', 'linear', 'log']
        The format of F0.

    ap_format : ['a', 'p', 'a/p', 'p/a']
        The format of aperiodicity.

    sp_format : ['db', 'log', 'linear', 'power']
        The format of spectrum.

    syn_param : SynParam or None
        Control parameters for the synthesis. If given, override the other parameters.
        You have full control and responsibility.

    Returns
    -------
    out : np.ndarray [shape=(nsample,)]
        The synthesized waveform.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> import numpy as np
    >>> f, fs = 200, 16000
    >>> x = np.sin(2 * np.pi * f / fs * np.arange(2400)) / 2
    >>> f0 = np.ones(30) * f
    >>> ap = pyls.extract_ap(x, fs, f0)
    >>> sp = pyls.extract_sp(x, fs, f0)
    >>> y = pyls.synthesize(f0, ap, sp, fs)
    >>> y.shape
    (2400,)

    """
    if fs < 8000:
        msg = "Minimum sampling frequency is 8000 Hz."
        raise ValueError(msg)

    if frame_shift < 1:
        msg = "Minimum frame shift is 1 ms."
        raise ValueError(msg)

    if len(f0) <= 0:
        msg = "The length of F0 is too short."
        raise ValueError(msg)

    if len(ap) != len(sp):
        msg = (
            "The lengths of aperiodicity and spectrum are not consistent: "
            f"ap {len(ap)} vs sp {len(sp)}."
        )
        raise ValueError(msg)

    if syn_param is None:
        syn_param = init_syn_param()
        syn_param.spectral_update_interval = frame_shift

    return exstraightsynth(
        f0_to_f0(f0, f0_format, "linear", fs=fs),
        sp_to_sp(sp, sp_format, "linear"),
        sp_to_sp(ap_to_ap(ap, ap_format, "a"), "linear", "db"),
        fs,
        syn_param,
    )


def fromfile(
    filename: str,
    fs: int = 0,
    *,
    frame_length: float | None = None,
    dtype: np.dtype = np.float64,
) -> np.ndarray:
    """Read a STRAIGHT parameter from file.

    Parameters
    ----------
    filename : str
        The file to read.

    fs : int
        The sampling frequency in Hz. If non-positive, the data is assumed to be F0,
        otherwise spectrum.

    frame_length : float
        The frame length in msec. If None, the default value is used.

    dtype : np.dtype
        The data type of the file.

    Returns
    -------
    out : np.ndarray [shape=(nframe,)] or [shape=(nframe, nfreq)]
        The STRAIGHT parameter.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> f0 = pyls.fromfile("tests/references/data.f0")
    >>> f0.shape
    (239,)

    """
    data = np.fromfile(filename, dtype=dtype)
    if 0 < fs:
        if frame_length is None:
            frame_length = SpParam().default_frame_length
        dim = get_fft_length(fs, frame_length, "one-sided")
        data = data.reshape(-1, dim)
    return data


def read(filename: str, **kwargs: Any) -> tuple[np.ndarray, int]:
    """Read audio file. This is the simple wrapper of `soundfile.read`.

    Parameters
    ----------
    filename : str
        The path to the input file.

    **kwargs : Any
        Keyword arguments passed to `soundfile.read`.

    Returns
    -------
    x : np.ndarray [shape=(nsample,) or (nchannel, nsample)]
        The audio signal.

    fs : int
        The sampling frequency in Hz.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> x, fs = pyls.read("assets/data.wav")
    >>> x.shape, fs
    ((19200,), 16000)

    """
    x, fs = sf.read(filename, **kwargs)
    return x, fs


def write(filename: str, x: np.ndarray, fs: int, **kwargs: Any) -> None:
    """Write audio file. This is the simple wrapper of `soundfile.write`.

    Parameters
    ----------
    filename : str
        The path to the output file.

    x : np.ndarray [shape=(nsample,) or (nchannel, nsample)]
        The audio signal.

    fs : int
        The sampling frequency in Hz.

    **kwargs : Any
        Keyword arguments passed to `soundfile.write`.

    Examples
    --------
    >>> import pylstraight as pyls
    >>> x, fs = pyls.read("assets/data.wav")
    >>> pyls.write("copy.wav", x, fs)

    """
    sf.write(filename, x, fs, **kwargs)
