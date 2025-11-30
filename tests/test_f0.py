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

import os

import numpy as np
import pytest

import pylstraight as pyls


def test_sample_with_debug_mode(sample_data: tuple[np.ndarray, int]) -> None:
    """Test sample with debug mode."""
    x, fs = sample_data
    os.environ["PYLSTRAIGHT_DEBUG"] = "1"
    ref_f0 = pyls.fromfile("tests/reference/data.f0")
    hyp_f0 = pyls.extract_f0(x, fs)
    assert np.allclose(ref_f0, hyp_f0)


def test_sample_without_debug_mode(sample_data: tuple[np.ndarray, int]) -> None:
    """Test sample without debug mode."""
    x, fs = sample_data
    os.environ["PYLSTRAIGHT_DEBUG"] = "0"
    hyp_f0 = pyls.extract_f0(x, fs)
    ref_f0 = pyls.fromfile("tests/reference/data.f0")
    # The error arises from the decimation function.
    assert np.allclose(hyp_f0, ref_f0, atol=2.5, rtol=0.0)


def test_f0_range_refinement(sample_data: tuple[np.ndarray, int]) -> None:
    """Test f0 range refinement."""
    x, fs = sample_data
    hyp_f0 = pyls.extract_f0(x, fs, refine_f0_range=True)
    ref_f0 = pyls.extract_f0(x, fs, refine_f0_range=False)
    mask = np.logical_and(hyp_f0 > 0, ref_f0 > 0)
    assert np.mean(np.abs(hyp_f0[mask] - ref_f0[mask])) < 0.5


def test_all_zero_input() -> None:
    """Test all zero input."""
    x, fs = np.zeros(1000), 8000
    f0 = pyls.extract_f0(x, fs)
    assert np.all(f0 == 0)
    f0 = pyls.extract_f0(x, fs, refine_f0_range=True)
    assert np.all(f0 == 0)


def test_very_short_input() -> None:
    """Test very short input."""
    x, fs = np.zeros(10), 8000
    f0 = pyls.extract_f0(x, fs)
    assert len(f0) == 1


def test_empty_input() -> None:
    """Test empty input."""
    x, fs = np.empty(0), 8000
    try:
        pyls.extract_f0(x, fs)
        pytest.fail("Reached unexpected code.")
    except ValueError:
        pass


def test_masked_input(sample_data: tuple[np.ndarray, int]) -> None:
    """Test masked input."""
    x, fs = sample_data
    y = x.copy()
    y[:4000] = 0
    y[5000:] = 0
    f0 = pyls.extract_f0(y, fs)
    assert 10 < len(f0[0 < f0])


def test_stereo_input(sample_data: tuple[np.ndarray, int]) -> None:
    """Test stereo input."""
    x, fs = sample_data
    x2 = np.stack([x, x])
    f0 = pyls.extract_f0(x, fs)
    f02 = pyls.extract_f0(x2, fs)
    assert np.allclose(f0, f02)


def test_very_long_frame_shift() -> None:
    """Test very long frame shift."""
    x, fs = np.zeros(8000), 8000
    f0 = pyls.extract_f0(x, fs, frame_shift=1000)
    assert len(f0) == 1


def test_ac_induction_removal(sample_data: tuple[np.ndarray, int]) -> None:
    """Test removal of AC induction."""
    x, fs = sample_data
    t = np.arange(len(x)) / fs
    ac_frequency = 50
    ac_amplitude = 1e-3
    ac_noise = ac_amplitude * np.sin(2 * np.pi * ac_frequency * t)
    f0 = pyls.extract_f0(x, fs)
    f02 = pyls.extract_f0(x + ac_noise, fs)
    assert len(f0[0 < f0]) == len(f02[0 < f02])


def test_if_number_of_harmonic(sample_data: tuple[np.ndarray, int]) -> None:
    """Test if_number_of_harmonic parameter."""
    x, fs = sample_data
    f0_param = pyls.init_f0_param()
    f0_param.f0_search_lower_bound = 40
    f0_param.f0_search_upper_bound = 400
    f0_param.f0_frame_update_interval = 5
    f0_param.if_number_of_harmonic_for_initial_estimate = 1
    f01 = pyls.extract_f0(x, fs, f0_param=f0_param)
    f0_param.if_number_of_harmonic_for_initial_estimate = 2
    f02 = pyls.extract_f0(x, fs, f0_param=f0_param)
    f0_param.if_number_of_harmonic_for_initial_estimate = 3
    f03 = pyls.extract_f0(x, fs, f0_param=f0_param)
    voiced = (0 < f01) & (0 < f02) & (0 < f03)
    assert abs(len(f01[0 < f01]) - len(f02[0 < f02])) <= 1
    assert abs(len(f01[0 < f01]) - len(f03[0 < f03])) <= 1
    assert np.max(np.abs(f01[voiced] - f02[voiced])) < 5
    assert np.max(np.abs(f01[voiced] - f03[voiced])) < 5


def test_conversion() -> None:
    """Test conversion between different f0 formats."""
    fs = 8000

    def check_reversibility(x: np.ndarray, in_format: str, out_format: str) -> bool:
        """Check if the conversion is identity.

        Parameters
        ----------
        x : np.ndarray
            The input.

        in_format : str
            The input format.

        out_format : str
            The output format.

        Returns
        -------
        out : bool
            True if the conversion is identity.

        """
        y = pyls.f0_to_f0(x, in_format, out_format, fs)
        z = pyls.f0_to_f0(y, out_format, in_format, fs)
        return np.allclose(x, z)

    f0 = np.array([100, 0, 200])
    assert check_reversibility(f0, "linear", "log")
    assert check_reversibility(f0, "linear", "inverse")
    log_f0 = pyls.f0_to_f0(f0, "linear", "log")
    assert check_reversibility(log_f0, "log", "inverse")
