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

import numpy as np

import pylstraight as pyls


def analyze(seed: int | None) -> tuple[np.ndarray, ...]:
    """Run the whole analysis and synthesis with the given seed.

    Parameters
    ----------
    seed : int or None
        The random seed.

    Returns
    -------
    out : tuple[np.ndarray, ...]
        The F0, aperiodicity, spectrum, and synthesized waveform.

    """
    x, fs = pyls.read("tools/straight/src/vaiueo2d.wav")
    x = x[: fs // 2]
    f0 = pyls.extract_f0(x, fs, seed=seed)
    ap = pyls.extract_ap(x, fs, f0, seed=seed)
    sp = pyls.extract_sp(x, fs, f0, seed=seed)
    syn = pyls.synthesize(f0, ap, sp, fs, seed=seed)
    return f0, ap, sp, syn


def test_same_seed_gives_same_results() -> None:
    """Test that the same seed reproduces the results."""
    for a, b in zip(analyze(0), analyze(0)):
        np.testing.assert_array_equal(a, b)


def test_no_seed_gives_different_results() -> None:
    """Test that the results vary without a seed."""
    *_, syn1 = analyze(None)
    *_, syn2 = analyze(None)
    assert not np.array_equal(syn1, syn2)
