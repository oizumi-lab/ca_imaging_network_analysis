# Full-neuron graphical-lasso investigation

This directory is an isolated investigation of how graphical-lasso estimation
changes the functional-network pipeline for two representative recordings:

- `mouse01_sleep`: awake versus NREM, 1,500-frame windows;
- `mouse05_ane`: awake versus anesthesia, 2,900-frame windows.

The analysis uses every neuron in the dataset's `nonzero_ROI` network population
and never subsamples neurons. Reusable code, interactive scripts, generated
artifacts, and reports remain inside this directory.

## Layout

- `configs/` — frozen initial-analysis settings.
- `src/glasso_analysis/` — preprocessing, exact graphical-lasso estimation,
  fixed-density graph construction, and network summaries.
- `scripts/` — executable, interactive `# %%` analysis scripts.
- `tests/` — small numerical regression tests that do not require recording data.
- `results/` — generated matrices, tables, figures, and run diagnostics.
- `documents/` — protocol and results report.

## Solver installation

The standard scikit-learn graphical-lasso solver is used for small reference
checks. Full-neuron fitting uses the QUIC extension distributed by `skggm`
0.2.5. Its published Python wrapper is Python-2-era code, so this project loads
only the compiled `pyquic` extension and supplies a tested Python 3 wrapper.

From the repository root and its existing virtual environment:

```bash
./.venv/bin/pip install setuptools Cython
./.venv/bin/pip install --no-deps --no-build-isolation skggm==0.2.5
```

The unusual `--no-deps` is intentional: `skggm==0.2.5` pins obsolete versions
of scikit-learn, while its QUIC extension itself works with the current NumPy
ABI after local compilation. `graphical_lasso/tests/test_estimation.py`
cross-checks QUIC against scikit-learn before any recording result is accepted.

## Run order

Both scripts are `# %%` cell scripts and can be run cell by cell in VS Code or
Spyder. They are also executable end to end:

```bash
PYTHONPATH=graphical_lasso/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
  poetry run python graphical_lasso/scripts/00_estimate_full_neuron_matrices.py

PYTHONPATH=graphical_lasso/src MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig \
  poetry run python graphical_lasso/scripts/01_compare_full_neuron_networks.py

PYTHONPATH=graphical_lasso/src poetry run python \
  graphical_lasso/scripts/02_summarize_full_neuron_results.py
```

To run one recording at a time, set `GLASSO_RECORDING` to
`mouse01_sleep` or `mouse05_ane`. Per-alpha checkpoints make interrupted fits
resumable.

To rebuild only the summary figures from existing CSV tables, without loading
the full matrices, set `GLASSO_SUMMARY_FIGURES_ONLY=1` for script `01`.

Tests:

```bash
PYTHONPATH=graphical_lasso/src poetry run python -m unittest discover \
  -s graphical_lasso/tests -v
```
