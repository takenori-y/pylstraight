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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np
import soundfile as sf

import pylstraight as pyls


def test_stereo(tmp_path: Path) -> None:
    """Test reading and writing stereo audio."""
    fs = 16000
    x = np.stack([np.zeros(fs), np.ones(fs) * 0.5])
    path = tmp_path / "stereo.wav"
    pyls.write(path, x, fs)
    assert sf.info(path).channels == 2
    y, fs2 = pyls.read(path)
    assert fs2 == fs
    assert y.shape == (2, fs)
    np.testing.assert_allclose(y, x, atol=1e-4)


def test_mono(tmp_path: Path) -> None:
    """Test reading and writing mono audio."""
    fs = 16000
    x = np.ones(fs) * 0.5
    path = tmp_path / "mono.wav"
    pyls.write(path, x, fs)
    y, _ = pyls.read(path)
    assert y.shape == (fs,)


def test_stereo_read_to_extract(tmp_path: Path) -> None:
    """Test that the read stereo audio can be passed to the extraction directly."""
    x, fs = pyls.read("assets/data.wav")
    path = tmp_path / "stereo.wav"
    sf.write(path, np.stack([x, x], axis=1), fs)
    y, _ = pyls.read(path)
    np.testing.assert_allclose(
        pyls.extract_f0(y, fs, seed=0), pyls.extract_f0(x, fs, seed=0), atol=1e-3
    )
