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

import pylstraight as pyls


def test_sample_with_debug_mode(sample_data: tuple[np.ndarray, int]) -> None:
    """Test sample with debug mode."""
    x, fs = sample_data
    os.environ["PYLSTRAIGHT_DEBUG"] = "1"
    f0 = pyls.fromfile("tests/reference/data.f0")
    ref_sp = pyls.fromfile("tests/reference/data.sp", fs)
    hyp_sp = pyls.extract_sp(x, fs, f0)
    assert np.allclose(ref_sp, hyp_sp, rtol=1e-4)


def test_sample_without_debug_mode(sample_data: tuple[np.ndarray, int]) -> None:
    """Test sample without debug mode."""
    x, fs = sample_data
    os.environ["PYLSTRAIGHT_DEBUG"] = "0"
    f0 = pyls.fromfile("tests/reference/data.f0")
    ref_sp = pyls.fromfile("tests/reference/data.sp", fs)
    hyp_sp = pyls.extract_sp(x, fs, f0)
    r = 3
    assert np.allclose(ref_sp[r:-r], hyp_sp[r:-r], rtol=1e-4)


def test_short_type_input(sample_data: tuple[np.ndarray, int]) -> None:
    """Test short type input."""
    x, fs = sample_data
    x2 = (x * 32768).astype(np.int16)
    f0 = pyls.fromfile("tests/reference/data.f0")
    sp = pyls.extract_sp(x, fs, f0)
    sp2 = pyls.extract_sp(x2, fs, f0) / 32768
    r = 3
    assert np.allclose(sp[r:-r], sp2[r:-r], rtol=1e-4)


def test_all_zero_input() -> None:
    """Test all zero input."""
    x, fs = np.zeros(8000), 8000
    f0 = np.zeros(200)
    sp = pyls.extract_sp(x, fs, f0)
    assert np.all(sp == sp[0, 0])


def test_short_f0_input() -> None:
    """Test short f0 input."""
    x, fs = np.zeros(8000), 8000
    f0 = np.zeros(199)
    sp = pyls.extract_sp(x, fs, f0)
    assert len(sp) == 199


def test_long_f0_input() -> None:
    """Test long f0 input."""
    x, fs = np.zeros(8000), 8000
    f0 = np.zeros(201)
    sp = pyls.extract_sp(x, fs, f0)
    assert len(sp) == 200


def test_conversion() -> None:
    """Test conversion between different aperiodicity formats."""

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
        y = pyls.sp_to_sp(x, in_format, out_format)
        z = pyls.sp_to_sp(y, out_format, in_format)
        return np.allclose(x, z)

    sp = np.array([-60, 0, 60])
    assert check_reversibility(sp, "db", "log")
    assert check_reversibility(sp, "db", "linear")
    assert check_reversibility(sp, "db", "power")
    sp = pyls.sp_to_sp(sp, "db", "log")
    assert check_reversibility(sp, "log", "linear")
    assert check_reversibility(sp, "log", "power")
    sp = pyls.sp_to_sp(sp, "log", "linear")
    assert check_reversibility(sp, "linear", "power")
