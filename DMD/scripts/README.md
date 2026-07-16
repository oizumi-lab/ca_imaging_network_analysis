# DMD scripts

- `02_systematic_window_tuning_mouse02_sleep.py` is the current step-by-step
  `# %%` tutorial. It separates fit-window duration, delay memory, retained
  rank, forecast horizon, and POD capacity; performs development-only tuning,
  common-reference and local-reference forecast scoring, low-overlap mode
  recurrence, a separate internal Residual-DMD-style diagnostic, warning
  audits, reconstruction/spectral checks, and within-window drift checks. It
  keeps the outer tail score-locked unless all gates pass. The primary fitted
  estimator is PyDMD `HankelDMD`; PyDMD does not currently provide the
  Colbrook--Townsend Residual DMD algorithm, and its `RDMD` class means
  randomized DMD.

```bash
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/mplconfig poetry run python \
  DMD/scripts/02_systematic_window_tuning_mouse02_sleep.py
```

Outputs are written to `DMD/results/06_window_parameter_tuning/`; the detailed
report is `DMD/documents/06_dmd_window_parameter_tuning_report.pdf`.

The synchronized verification run does not select a production window. The
180-s, delay-1, rank-2 model is retained only as a one-SE fallback after no
configuration passed the gate; the 45-s, delay-1, rank-4 model is retained only
as a local-increment stress test.

`01_resdmd_mouse02_sleep.py` is the preceding single-block analysis. It is written as
an executable `# %%` tutorial but belongs in this scripts directory. It uses
PyDMD 2025.8.1 to compare Hankel delay orders on one NREM block, selects the
delay using calibration data, measures untouched-test activity capture,
computes Residual-DMD diagnostics for the PyDMD candidates, and maps candidate
modes back to all neuron positions.

```bash
poetry run python DMD/scripts/01_resdmd_mouse02_sleep.py
```

Its outputs are written to `DMD/results/05_pydmd_resdmd_mouse02/`.

## Historical multi-recording smoke test

`00_initial_dmd_validation.py` is the thin, configuration-driven entry point for
the private DMD smoke test. Run stages in order; a completed stage is immutable.

```bash
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage data
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage preprocessing
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage precedent
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage simulation
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage forecast
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage stability
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage nulls
PYTHONPATH=DMD/src poetry run python DMD/scripts/00_initial_dmd_validation.py --stage decision
```

Reusable code lives in `DMD/src/dmd_validation/`; tests live in `DMD/tests/`.

```bash
PYTHONPATH=DMD/src poetry run python -m unittest discover -s DMD/tests -v
poetry run ruff check DMD/src DMD/scripts DMD/tests
```
