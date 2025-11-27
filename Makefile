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

PROJECT := pylstraight

PYTHON_VERSION := 3.11

venv:
	test -d .venv || python$(PYTHON_VERSION) -m venv .venv
	. ./.venv/bin/activate && python -m pip install --upgrade pip
	. ./.venv/bin/activate && python -m pip install --upgrade wheel
	. ./.venv/bin/activate && python -m pip install -e .[dev]

dist:
	. ./.venv/bin/activate && python -m build
	. ./.venv/bin/activate && python -m twine check dist/*

dist-clean:
	rm -rf dist

doc:
	. ./.venv/bin/activate && cd docs && make html

doc-clean:
	rm -rf docs/build

check: tool
	. ./.venv/bin/activate && python -m ruff check $(PROJECT) tests
	. ./.venv/bin/activate && python -m ruff format --check $(PROJECT) tests docs/source
	. ./.venv/bin/activate && python -m mdformat --check *.md
	./.venv/bin/codespell
	./tools/taplo/taplo fmt --check *.toml
	./tools/yamlfmt/yamlfmt --lint *.yml .github/workflows/*.yml

format: tool
	. ./.venv/bin/activate && python -m ruff check --fix $(PROJECT) tests
	. ./.venv/bin/activate && python -m ruff format $(PROJECT) tests docs/source
	. ./.venv/bin/activate && python -m mdformat *.md
	./tools/taplo/taplo fmt *.toml
	./tools/yamlfmt/yamlfmt *.yml .github/workflows/*.yml

test:
	. ./.venv/bin/activate && python -m pytest

test-doc:
	. ./.venv/bin/activate && python -m pytest --doctest-modules --no-cov $(PROJECT)/api.py

test-f0:
	. ./.venv/bin/activate && python -m pytest tests/test_f0.py

test-ap:
	. ./.venv/bin/activate && python -m pytest tests/test_ap.py

test-sp:
	. ./.venv/bin/activate && python -m pytest tests/test_sp.py

test-syn:
	. ./.venv/bin/activate && python -m pytest tests/test_syn.py

test-clean:
	rm -rf tests/output

tool:
	cd tools && make

tool-clean:
	cd tools && make clean

update: tool
	. ./.venv/bin/activate && python -m pip install --upgrade pip
	@for package in $$(./tools/taplo/taplo get -f pyproject.toml project.optional-dependencies.dev); do \
		. ./.venv/bin/activate && python -m pip install --upgrade $$package; \
	done

clean: dist-clean doc-clean test-clean tool-clean
	rm -rf .venv
	find . -name "__pycache__" -type d | xargs rm -rf

.PHONY: venv dist doc check format test tool update clean
