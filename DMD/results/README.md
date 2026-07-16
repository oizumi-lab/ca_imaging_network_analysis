# DMD results

Generated DMD figures, tables, manifests, caches, and model outputs belong here. Results are ignored by Git by default; retain only small, deliberately selected final artifacts when a separate private DMD repository is created.

The initial-validation workflow stores immutable timestamped runs under
`03_initial_validation/runs/`. The machine-readable
`03_initial_validation/active_run.json` points to the accepted run; earlier
stepwise inspection runs are intentionally preserved for the audit trail.

The current focused PyDMD analysis writes its reproducible figures, delay
sweep, mode table, and JSON verdict to `05_pydmd_resdmd_mouse02/`.

The systematic, development-only window/delay/rank verification writes its
figures, complete origin/horizon tables, POD oracle, warning audit, mode-match
graph, surrogate residuals, drift diagnostics, frozen decision, and run
manifest to `06_window_parameter_tuning/`. In the current run,
`outer_origin_scores.csv` and `outer_horizon_scores.csv` intentionally contain
headers only because the development gates failed and the outer tail remained
score-locked and unscored. `tuning_decision.json` contains the strict
machine-readable verdict; `run_manifest.json` and `run_status.json` bind the
results to the completed tutorial-script hash.

The key current result is negative: every guarded fixed-reference predictive
R² is below zero. The lowest-loss configuration is 180 s/delay 1/rank 12, and
the descriptive one-SE fallback is 180 s/delay 1/rank 2. The 45-s/delay 1/rank
4 candidate has the largest local-mean increment but remains negative against
the common reference. `finalist_conditioning.csv`,
`training_reconstruction_spectral_audit.csv`, and the modal-shortlist tables
record why none of these candidates is accepted for sliding-window tracking.
