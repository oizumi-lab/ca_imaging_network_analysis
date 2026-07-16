# DMD documents

- `01_dmd_tracking_literature_survey.pdf` — critical review of Raut, Germain, and related methods.
- `02_dmd_brain_state_analysis_plan.pdf` — preregistration-style research and validation plan.
- `03_dmd_initial_validation_plan.pdf` — narrow Pachitariu-inspired feasibility test before implementation and full-cohort DMD work.
- `04_dmd_initial_validation_results.pdf` — complete stepwise smoke-test report, diagnostics, controls, audit trail, and frozen gate decision.
- `05_resdmd_mouse02_pipeline_plan.md` — current one-recording PyDMD/Hankel and
  Residual-DMD diagnostic plan corresponding to
  `DMD/scripts/01_resdmd_mouse02_sleep.py`.
- Matching `.tex` files are the editable sources.
- `dmd_tracking_references.bib` is the shared bibliography.

Build from this directory:

```bash
latexmk -pdf 01_dmd_tracking_literature_survey.tex
latexmk -pdf 02_dmd_brain_state_analysis_plan.tex
latexmk -pdf 03_dmd_initial_validation_plan.tex
latexmk -pdf 04_dmd_initial_validation_results.tex
latexmk -pdf 06_dmd_window_parameter_tuning_report.tex
```

## Current verification report

- `06_dmd_window_parameter_tuning_report.pdf` --- systematic window, delay,
  rank, POD-capacity, optimized-DMD convergence, low-overlap mode recurrence,
  within-window drift, and an internal stochastic Residual-DMD-style diagnostic for the
  focused `mouse02_sleep` NREM analysis.
- `06_dmd_window_parameter_tuning_report.tex` --- editable LaTeX source.

The report's decision is deliberately non-promotional: 180 s minimizes the
fair common-reference loss but still has negative predictive R², while 45 s
maximizes only a small window-specific local-mean increment. Neither is a
verified working DMD window, so the outer tail and Grassmann-tracking phase
remain unopened.
