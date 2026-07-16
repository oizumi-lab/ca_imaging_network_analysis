# Full analysis-suite run report

Date: 2026-07-13
Environment: Poetry project environment, non-interactive Matplotlib backend

## Scope and execution status

All analysis entry points in `scripts/` and `verification/` were executed against
the local v2 recordings. The acquisition-only `scripts/download_data.py` was not
run because all eleven expected MAT files were already present with the published
byte sizes.

Every executed entry point exited successfully:

- Teaching: `00_inspect_data.py`, `01_reproduce_example.py`,
  `10_functional_connectivity.py`, `20_modularity.py`,
  `30_state_comparison.py`, `40_small_world.py`,
  `50_coarse_grain_modularity.py`, `60_module_spatial_distribution.py`.
- Verification: `50_verify_modularity.py`, `51_verify_smallworld.py`,
  `52_sparse_firing_robustness.py`, `shuffle_investigation.py`,
  `shuffle_null_control.py`, `smallworld_shuffle_corrected.py`,
  `state_difference_cause.py`, `why_QL_robust_C_confounded.py`, and
  `sparsity_clustering_mechanism.py`.

The heavy teaching runs took about 20 minutes for scripts 00--30, 13 minutes for
script 40, 9.5 minutes for script 50, and 12 minutes for script 60. Scripts 50
and 60 were rerun after visual-QC wording/axis fixes.

## Main results

### Raw modularity

At K=5%, with windows and repeated sessions averaged before the mouse summary:

| Protocol | Awake Q | Unconscious Q | Paired mouse mean difference |
|---|---:|---:|---:|
| Sleep, n=5 | 0.289 | 0.323 | +0.034 +/- 0.008 SE |
| Anaesthesia, n=4 | 0.247 | 0.340 | +0.093 +/- 0.015 SE |

These are raw estimated-graph measures, not sparsity-corrected coupling estimates.

### K=5% within-bout temporal-null benchmark

The 20-surrogate, 2000-neuron verification gave:

| Protocol | Metric | Raw difference | Null difference | Excess over null | Nominal p |
|---|---|---:|---:|---:|---:|
| Sleep | Q | +0.0136 | -0.0021 | +0.0157 | 0.269 |
| Sleep | Mean local C | +0.0307 | +0.0072 | +0.0235 | 0.0201 |
| Sleep | L | +0.0200 | -0.0080 | +0.0280 | 0.146 |
| Anaesthesia | Q | +0.0983 | +0.0203 | +0.0779 | 0.0384 |
| Anaesthesia | Mean local C | +0.1390 | +0.0903 | +0.0487 | 0.0186 |
| Anaesthesia | L | +0.1061 | +0.0075 | +0.0986 | 0.0175 |

Thus the anaesthesia null reproduced about 65% of the raw mean-local-C contrast,
versus about 21% of Q and 7% of L. These ratios are descriptive and not causal
confound percentages. Sleep Q/L were uncertain and changed more across analysis
settings.

The independent 600-neuron audit confirmed that the clustering definition is
material: in anaesthesia the within-bout null reproduced about 68% of mean local
C but only about 26% of transitivity. Anaesthesia transitivity excess was +0.0995;
sleep transitivity excess was +0.0130 and uncertain at n=5.

### Activity sensitivity and mechanism

- Near-silent neurons (<5 events) increased from 3.7% to 10.9% in sleep and from
  12.3% to 46.7% in anaesthesia at the mouse-summary level.
- The event-count-matched independent model reproduced the null-C pattern with
  r=0.90 in sleep and r=0.98 in anaesthesia, but its magnitude is approximate.
- Requiring at least 5 or 10 events in both states sharply reduced the anaesthesia
  mean-local-C null shift while positive excess remained.
- In the shuffled mouse05 anaesthesia graph, event count versus node clustering
  had Spearman rho=-0.93; the sparsest-to-busiest quartile C values were
  0.385, 0.284, 0.216, and 0.149.

These results establish a sparsity-dependent temporal-null baseline and
mechanistic sufficiency in a simplified model. They do not validate null
subtraction as an unbiased cross-sparsity correction.

### Small-world and spatial-scale analyses

- At K=1%, the temporal null reproduced roughly 48--49% of raw C/SWP state
  contrasts in both protocols. L was less null-sensitive (23% sleep, 21%
  anaesthesia), although the anaesthesia L excess was imprecise at n=4.
- In coarse-graining, the sleep awake-minus-NREM Q interval first included zero
  at nnei=10; anaesthesia first included zero at nnei=5. This is not an
  equivalence result. At nnei=160 the anaesthesia contrast reversed to +0.354
  [0.180, 0.529], where only a very small parcel graph remains.
- Single-cell same-module proportions were broadly flat with distance, whereas
  mesoscale curves declined substantially. Far-distance bins sometimes rebounded
  and remain descriptive without pair-count and partition-stability uncertainty.

## Quality control

- Project clustering/path calculations matched NetworkX and bctpy; fixed-partition
  modularity agreed to numerical precision.
- The sparse-firing audit produced 700 state rows, 100 contrasts, and 20 summary
  rows with no missing core metrics or edge-count mismatches.
- Analyzed largest-component fractions were 0.90--1.00.
- All generated figures were nonempty; the principal state, null, sparse-firing,
  coarse-graining, and distance-profile figures were inspected visually.
- Python compilation, focused Ruff checks, and `git diff --check` passed.
- The shared temporal-null cache was rebuilt with the within-bout manifest, so
  the legacy concatenated global-roll cache was not reused.

## Interpretation boundary

All p values above are small-sample, nominal, and uncorrected. The completed
suite supports reporting raw, null, and excess-over-null values separately. It
does not replace the still-unimplemented injection--recovery experiment needed
to show that a corrected metric has the same response to identical coupling at
awake-, sleep-, and anaesthesia-like sparsity.
