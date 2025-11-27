# pylstraight

This is an unofficial Python reimplementation of the [legacy-STRAIGHT](https://github.com/HidekiKawahara/legacy_STRAIGHT), which was originally written in MATLAB.

[![Latest Manual](https://img.shields.io/badge/docs-latest-blue.svg)](https://takenori-y.github.io/pylstraight/latest/)
[![Stable Manual](https://img.shields.io/badge/docs-stable-blue.svg)](https://takenori-y.github.io/pylstraight/0.1.1/)
[![Downloads](https://static.pepy.tech/badge/pylstraight)](https://pepy.tech/project/pylstraight)
[![Python Version](https://img.shields.io/pypi/pyversions/pylstraight.svg)](https://pypi.python.org/pypi/pylstraight)
[![PyPI](https://img.shields.io/pypi/v/pylstraight.svg)](https://pypi.python.org/pypi/pylstraight)
[![Anaconda](https://anaconda.org/conda-forge/pylstraight/badges/version.svg)](https://anaconda.org/conda-forge/pylstraight)
[![Codecov](https://codecov.io/gh/takenori-y/pylstraight/branch/master/graph/badge.svg)](https://app.codecov.io/gh/takenori-y/pylstraight)
[![License](https://img.shields.io/github/license/takenori-y/pylstraight.svg)](https://github.com/takenori-y/pylstraight/blob/master/LICENSE)
[![GitHub Actions](https://github.com/takenori-y/pylstraight/workflows/package/badge.svg)](https://github.com/takenori-y/pylstraight/actions)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

## Requirements

- Python 3.8+

## Documentation

- See [this page](https://takenori-y.github.io/pylstraight/0.1.1/) for the reference manual.

## Installation

The latest stable release can be installed from PyPI by running the command below.

```sh
pip install pylstraight
```

The development release can be installed from the master branch using the following command:

```sh
pip install git+https://github.com/takenori-y/pylstraight.git@master
```

## Supported Features

- Fundamental frequency extraction
- Aperiodicity measure extraction
- Spectral envelope extraction
- Synthesis using the extracted parameters

## Unsupported Features

- Batch processing
- GPU acceleration
- JIT compilation

## Example

```py
import pylstraight as pyls

# Read an example waveform.
x, fs = pyls.read("assets/data.wav")

# Set control parameters.
fp = 5                # Frame shift in msec
f0_range = (60, 240)  # F0 search range in Hz

# Extract the STRAIGHT parameters.
f0 = pyls.extract_f0(x, fs, f0_range=f0_range, frame_shift=fp)
ap = pyls.extract_ap(x, fs, f0, frame_shift=fp)
sp = pyls.extract_sp(x, fs, f0, frame_shift=fp)

# Synthesize a waveform using the parameters.
sy = pyls.synthesize(f0, ap, sp, fs, frame_shift=fp)

# Write the synthesized waveform.
pyls.write("data.syn.wav", sy, fs)
```

## License

The original code is licensed under the Apache License 2.0.
