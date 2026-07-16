# Independent code and sparse-firing audit

Date: 2026-07-13
Companion analysis: `verification/52_sparse_firing_robustness.py`

Maintenance update (2026-07-13): the minimal script fixes recommended by this
audit have since been applied. Script 30 now uses 1500/2900-frame protocol
windows and mouse-level summaries; the legacy verification null shifts within
contiguous bouts and validates its cache manifest; protocol inference is kept
separate; and output wording uses conditional null excess rather than a causal
correction. The findings below document the audit that motivated those changes;
the larger injection--recovery validation remains future work.

## Executive conclusion

The central qualitative observation is real: **sparse event trains create a high
clustering baseline in fixed-density correlation graphs**, even after genuine
cross-neuron timing has been disrupted. In the present recordings, this effect is
largest for the arithmetic mean of nodewise clustering coefficients and is much
smaller for transitivity, the degree-weighted triangle measure used in the 2026
paper.

The present repository nevertheless overstates what has been established. A
circular-shift benchmark does not split a nonlinear graph metric into an additive
"confound percentage" and "genuine coupling percentage." The existing global
roll is also not an exchangeable randomization for concatenated sleep bouts, and
the headline statistics pool sessions/protocols at the wrong level. The strongest
defensible wording is:

> The observed state contrast exceeds a specified, state-specific,
> per-neuron-preserving temporal-null benchmark by a reported amount.

It is not defensible to say that 56% of clustering is causally confounded and the
remaining 44% is genuine neuronal coupling.

My practical judgment is:

- **Raw mean local clustering and raw SWP should not be compared across states
  with strongly different firing sparsity.**
- **Transitivity is less vulnerable than mean local clustering in these data, but
  it still requires a state-specific temporal null and activity sensitivity
  checks.**
- **Q and L are more promising population summaries, not automatically reliable
  biomarkers.** Under a small independent reanalysis their excess contrasts were
  clear for anaesthesia, while sleep Q/L remained uncertain at only five mice.
- **Individual edges and modules are not reliable for near-silent neurons.** The
  overlap of top-5% edges in adjacent windows was only slightly above chance.
- Awake--NREM and awake--anaesthesia can be compared as separate paired
  experiments. A direct raw ranking of wake, sleep, and anaesthesia is not fair:
  protocols, window lengths, sessions, active-cell populations, and sparsity all
  differ.

## What is sound

Several foundations are correct and useful:

1. All ten full v2 recordings load successfully. The loader gives aligned neural
   arrays and valid state-frame indices for the current files.
2. On an already-constructed symmetric binary graph, the nodewise clustering
   coefficient and connected-graph path length in `src/funcnet/smallworld.py`
   match NetworkX and bctpy to numerical precision.
3. Given a graph and a partition, modularity Q matches independent library
   implementations to numerical precision. Independent Louvain implementations
   obtain close best-Q values on the verification graph.
4. At K=5% in the first correctly sized real windows, the correlation cutoff was
   unique and the achieved density was exactly 5%. Ties are therefore not the
   explanation for the principal observed sparsity effect.
5. Strong selected edges in the checked real and shifted graphs were almost all
   positive. Folding correlation sign is not the principal explanation for the
   K=5% result.
6. The association between near-silent neurons and the temporal-null clustering
   baseline is qualitatively credible and robust to several event-count cutoffs.

These statements validate graph mathematics, not the biological interpretation
of graphs estimated from sparse calcium signals.

## Highest-priority code and analysis findings

### 1. The main state-comparison script uses the wrong anaesthesia window

`scripts/30_state_comparison.py:58,82-86` hard-codes 1500 frames for both
experiments. The paper and the other relevant scripts use 1500 frames for sleep
and **2900 frames for anaesthesia**. The `nonzero_ROI` filter was constructed for
those study windows.

In the independent QC (up to 3000 filtered neurons, two windows per state), the
wrong 1500-frame anaesthesia windows contained:

| State | Zero-variance neurons, range | Median |
|---|---:|---:|
| Awake | 29--270 | 83 |
| Anaesthesia | 233--594 | 295 |

Every tested 2900-frame window had zero zero-variance neurons. Silent rows become
isolates and also make the retained-edge density among estimable neurons higher
than intended. Spot checks showed that the awake--anaesthesia Q direction
survived, but its magnitude changed materially. Script 30 is therefore a light
demonstration, not a faithful paper reproduction.

### 2. Existing inference uses the wrong observational hierarchy

`scripts/30_state_comparison.py:152-178,223-235,266-271` pools windows and/or
densities as observations. `verification/why_QL_robust_C_confounded.py:92-101`
and `verification/state_difference_cause.py:130-180` pool six sleep sessions and
four anaesthesia sessions into one nominal n=10 analysis. These choices ignore:

- paired states within a recording;
- windows nested within session and mouse;
- mouse 4's two sleep days;
- different 1500/2900-frame protocols;
- protocol-specific effect sizes.

The paper first averages windows within mouse/state and then analyzes the paired
mouse differences (sleep n=5, anaesthesia n=4). Using the current cache and first
pooling mouse 4's days, the nominal one-sample p values of the cached
shuffle-excess contrasts are:

| Protocol | Q | Mean local C | L |
|---|---:|---:|---:|
| Sleep (n=5) | 0.415 | 0.0558 | 0.0944 |
| Anaesthesia (n=4) | 0.0249 | 0.0227 | 0.0277 |

These are small-n, uncorrected parametric tests. None of the three anaesthesia
values survives a simple three-measure Bonferroni threshold. The combined pooled
p values do not establish that each protocol has a reliable corrected effect.

### 3. The temporal null is applied across concatenated sleep-bout boundaries

The current null rolls the activity matrix after selecting and concatenating
`used_frame` (`verification/shuffle_investigation.py:62-69,131-139`). In the first
1500 selected sleep frames, the original-frame span averaged 6736 frames for
awake and 3514 for NREM; one awake window spanned 13,132 frames. Each window
contained roughly 2--4 separate bouts.

A global roll moves artificial bout-boundary discontinuities to different
locations in each neuron. It is exact only under cyclic stationarity/exchangeability,
which is not plausible across those gaps. The new audit shifts independently
within each contiguous state bout.

This choice changed the sleep null contrast substantially:

| Metric (sleep, >=1 event) | Global concatenated roll | Within-bout roll |
|---|---:|---:|
| Mean local C null difference | +0.0107 | +0.0075 |
| Transitivity null difference | +0.0090 | +0.0049 |

Anaesthesia values were nearly unchanged because those windows were largely
contiguous. A null-model ladder, not one surrogate family, should be used for
final claims.

### 4. Null subtraction is not a causal decomposition

The interpretation at `verification/shuffle_null_control.py:16-18`,
`verification/why_QL_robust_C_confounded.py:83-101`, and
`documents/04_shuffle_null_confound.md:21-31` assumes that a nonlinear graph
measure can be additively divided into a marginal part and a coupling part.
Correlation ranking, thresholding, component selection, Louvain optimization,
clustering, path length, and SWP are nonlinear and can contain interactions
between sparsity and real shared structure.

The ratio `mean(delta_null)/mean(delta_real)` is a descriptive
"shuffle-reproduced fraction under this pipeline," not an identified causal
confound fraction. Circular shifts also cannot distinguish neuron-neuron coupling
from common drive, motion/neuropil contamination, or slow population modulation.

### 5. Mean local clustering and transitivity answer different questions

The verification uses `sw.avg_clustering`, an equal-weight average over nodewise
clustering. The 2026 paper's principal triangle measure is transitivity:

`sum_i 2*t_i / sum_i k_i*(k_i-1)`.

Transitivity weights nodes by their number of connected triples, so sparse
low-degree nodes do not receive the same leverage as high-degree nodes. In the
new audit (600 seeded common neurons, K=5%, correct window lengths, eight
within-bout shifts), the mouse-level means were:

| Protocol | Metric | Raw state difference | Null difference | Excess over null | Nominal p |
|---|---|---:|---:|---:|---:|
| Sleep, n=5 | Mean local C | +0.0287 | +0.0075 | +0.0212 | 0.031 |
| Sleep, n=5 | Transitivity | +0.0179 | +0.0049 | +0.0130 | 0.148 |
| Anaesthesia, n=4 | Mean local C | +0.1355 | +0.0926 | +0.0429 | 0.0157 |
| Anaesthesia, n=4 | Transitivity | +0.1346 | +0.0350 | +0.0995 | 0.0352 |

For anaesthesia, about 68% of the raw mean-local-C contrast was reproduced by
this null, versus about 26% of the transitivity contrast. These are descriptive
ratios, not causal percentages. The difference shows that the choice of
"clustering coefficient" is scientifically material.

### 6. Activity-threshold sensitivity changes the answer, but does not erase it

The new audit repeated C and transitivity after requiring each selected neuron to
have at least 1, 5, or 10 detected events in **both** paired states. The cap was
600 neurons; one anaesthesia minimum-10 set fell to 210, so this is a sensitivity
analysis rather than a fully size-matched primary result.

| Protocol | Minimum events | Metric | Raw difference | Null difference | Excess |
|---|---:|---|---:|---:|---:|
| Sleep | 1 | Mean local C | +0.0287 | +0.0075 | +0.0212 |
| Sleep | 5 | Mean local C | +0.0352 | +0.0066 | +0.0285 |
| Sleep | 10 | Mean local C | +0.0435 | +0.0066 | +0.0370 |
| Sleep | 1 | Transitivity | +0.0179 | +0.0049 | +0.0130 |
| Sleep | 5 | Transitivity | +0.0265 | +0.0051 | +0.0213 |
| Sleep | 10 | Transitivity | +0.0460 | +0.0049 | +0.0411 |
| Anaesthesia | 1 | Mean local C | +0.1355 | +0.0926 | +0.0429 |
| Anaesthesia | 5 | Mean local C | +0.1166 | +0.0282 | +0.0884 |
| Anaesthesia | 10 | Mean local C | +0.1056 | +0.0240 | +0.0816 |
| Anaesthesia | 1 | Transitivity | +0.1346 | +0.0350 | +0.0995 |
| Anaesthesia | 5 | Transitivity | +0.1671 | +0.0268 | +0.1403 |
| Anaesthesia | 10 | Transitivity | +0.1556 | +0.0242 | +0.1314 |

Removing near-silent cells sharply reduced the anaesthesia null shift while the
real excess remained. This supports a state-related triangle/topology difference
among adequately active neurons. It does **not** show that the full recorded
population has the same estimand: selecting active cells conditions on a
state-dependent biological response. Both full-common-population and
common-active-population results should be reported.

### 7. Q and L look promising for anaesthesia, uncertain for sleep

At the same audit settings, using ordinary singleton Louvain initialization and
L on the largest component:

| Protocol | Metric | Raw difference | Null difference | Excess | Nominal p |
|---|---|---:|---:|---:|---:|
| Sleep, n=5 | Q | +0.0257 | +0.0006 | +0.0251 | 0.0753 |
| Sleep, n=5 | L | +0.0221 | +0.0001 | +0.0219 | 0.164 |
| Anaesthesia, n=4 | Q | +0.0790 | +0.0209 | +0.0581 | 0.0230 |
| Anaesthesia, n=4 | L | +0.1225 | +0.0088 | +0.1137 | 0.0251 |

This supports "less null-sensitive than mean local C," especially for
anaesthesia. It does not justify the categorical label "genuine coupling," and
the sleep evidence is not precise at n=5. Only eight surrogates and three Louvain
runs were used here; the table is an independent sensitivity check, not final
inference.

### 8. Individual functional edges are poorly repeatable

For recordings with a second full window, the top-5% edge-set Jaccard overlap was:

- sleep awake/NREM: mean about 0.036/0.040;
- anaesthesia awake/anaesthesia: mean about 0.035/0.053.

Two independent 5%-density edge sets have expected Jaccard approximately
`0.05/(2-0.05) = 0.0256`. Thus edge identity was only modestly above chance.
Aggregate C was more repeatable across the 16 available window pairs (correlation
0.91; median absolute change 0.019), and transitivity had correlation 0.81
(median absolute change 0.034), although these pooled correlations are inflated
by between-state/recording heterogeneity and do not constitute an ICC.

Population summaries may therefore be usable after calibration even when
individual edges/modules are not. Claims about specific neuron pairs or module
membership require bootstrap/consensus stability and minimum-event QC.

### 9. Cache provenance is unsafe

`verification/shuffle_investigation.py:192-199` accepts an existing NPZ without a
configuration, schema, code hash, data fingerprint, or selected-neuron IDs. The
current cache visibly retains deleted `kurtosis` fields. Moreover, the helper uses
1500 neurons/10 shifts, while `shuffle_null_control.py` uses 2000/20 and the
teaching pipeline uses other counts/windows. The document's claim of one unified
pipeline is therefore inaccurate.

Every cache should include and validate a manifest containing all settings,
software/code version, data identity, row indices, window indices, and schema.

### 10. The autocorrelation claim is contradicted by its own cache

`verification/state_difference_cause.py:123-142` computes a paired test while its
prose says autocorrelation is unchanged. Cached lag-1 means are 0.97553 awake and
0.97334 unconscious, paired t(9)=-3.12, p=0.0123. The absolute difference is
small, but "not different" and "equivalent" are not interchangeable. One averaged
lag cannot rule out differences in full ACF, event width, slow nonstationarity,
or heterogeneous neuronwise effective sample size.

The preprocessing kernel is known to be common by design. That fact should be
stated directly; empirical trace autocorrelation should be assessed with a full
ACF/integrated-time analysis and an equivalence margin.

### 11. The event-count simulation is only approximate

`verification/state_difference_cause.py:89-101` and
`verification/sparsity_clustering_mechanism.py:65-74` draw event indices with
replacement and use advanced-index `+=`; repeated indices do not accumulate as
the comments imply. The model therefore does not place exactly k distinct events.
It also uses one seed, unit events, and SciPy Gaussian sigma=5, whereas fitting the
provided smoothed traces indicates sigma near 3 for this data representation.

Correcting the kernel strengthened rather than removed the event-count prediction
in a spot check, so the qualitative sparsity mechanism remains credible. A final
model should use empirical amplitudes/templates, correct accumulation or
without-replacement sampling, bout-varying rates, many repetitions, and confidence
intervals. Model sufficiency is not proof that sparsity is the sole causal
preserved marginal.

### 12. Density thresholding is not robust to ties or invalid K

`src/funcnet/network.py:88-102` finds the m-th value and keeps every value
`>= cutoff`. On an all-zero 4-node matrix at K=50%, the target is three edges but
the legacy function returns all six. K<=0 still returns at least one edge (or all
tied edges), K>1 is silently clamped, and N=1 fails indirectly.

Real K=5% cutoff ties were absent in the new audit, so this is not the observed
effect's cause. It remains a serious correctness bug precisely for quantized or
extremely sparse data. Select exact top-m indices and use a documented,
repeated/random boundary-tie policy; always record achieved density.

### 13. The giant-component Louvain warm start can lower Q

`src/funcnet/network.py:108-134,169-180` assigns all nodes outside the largest
component--including nodes in distinct nontrivial components--to one initial
community. BCT Louvain need not split that forced group. On a graph containing a
K4 and two disconnected K2 components, the warm start returned Q=0.375 while
ordinary singleton initialization returned Q=0.40625. Non-isolated nodes outside
the giant component also occur in real low-density graphs.

Isolates can be marked separately for module counts; each nontrivial connected
component should remain separable. Q itself should be maximized from multiple
appropriate initializations.

### 14. Small-world/SWP verification is incomplete

The production graph-level random/lattice null is applied **after** correlation
thresholding. It answers whether an estimated graph differs from an edge-random
graph; it cannot control event sparsity, smoothing, or correlation-estimation
noise.

Additional problems in `src/funcnet/smallworld.py` and script 40 are:

- `sw_summary` selects a different largest connected component for every state.
  At K=1%, this can remove more sparse-state nodes and raise the induced density
  and C. Node count, component fraction, edges, and post-LCC density are not
  reported.
- Only one random and one lattice realization are used. Some random realizations
  disconnect, forcing delta-L to 1 and SMN to NaN; SWP can change dramatically.
- Zero/reversed `reg-rand` denominators are not handled. Tests produced division
  errors or SWP values outside the documented range.
- `method="bin"` does not binarize weighted W before path/null calculations as the
  MATLAB function does.
- The Python lattice's partial-ring/weight placement is not an exact MATLAB port.
- `verification/51_verify_smallworld.py` validates binary C and L, but its
  composite check only asks whether different packages qualitatively call one
  small graph small-world. It does not validate SWP, lattice/random invariants,
  disconnected cases, or null variability.
- Higher delta-L makes SWP smaller, not larger. Calling longer paths, inverse
  delta-C, SMN, and SWP collectively "higher small-world metrics" is misleading.

Use a temporal null before graph construction, an ensemble of internal graph
nulls, fixed-node global efficiency/harmonic path length for disconnected graphs,
and explicit validity flags for degenerate SWP cases.

## Other codebase findings

- `scripts/01_reproduce_example.py` uses a full mouse recording rather than the
  shipped official `example_data`. Its assertions verify internal invariants and
  rescore a BCT-derived partition; they do not establish exact equality to saved
  MATLAB arrays/partitions. The "exact reproduction" wording is too strong.
- `src/funcnet/dataio.py` uses `assert` for file validation, so checks disappear
  under `python -O`. It should explicitly validate `data_info`, centroid shape,
  finite/binary filters, sorted unique indices, state agreement, and boundaries.
- `scripts/download_data.py` writes directly to final destinations and validates
  only size when supplied. Atomic partial files, resume/retry, and published hashes
  are important for roughly 11 GB of inputs.
- `scripts/50_coarse_grain_modularity.py:323-336` calls the first CI crossing zero
  the scale where the gap "vanishes." Failure to reject is not equivalence.
- The light defaults in `scripts/50_coarse_grain_modularity.py:96-123` and
  `scripts/60_module_spatial_distribution.py:85-118` randomly subsample neurons
  **before** making `nnei`-neuron parcels. Forty neighbors in a sparse 2000-cell
  subsample cover a larger physical footprint than forty neighbors in the full
  population, so `nnei` is no longer the paper's spatial scale and varies with
  sampling density. Panel maps use all cells while distance curves use the
  subsample, which further changes the estimand.
- `scripts/50_coarse_grain_modularity.py:284-291` uses `sum((n_c/N)^2)` as the
  chance that a node's distinct nearest neighbor shares its module. The exact
  without-replacement baseline is `sum(n_c*(n_c-1))/(N*(N-1))`. The difference is
  modest at large N but can matter for the coarse parcel graphs.
- Same-module-versus-distance curves depend on module count/size and Louvain
  degeneracy as well as spatial localization. Script 60 supplies no label/spatial
  permutation null or partition-consensus uncertainty, so its maps and curves are
  descriptive reproductions rather than calibrated evidence.
- The repository has no automated test suite. The verification scripts execute
  large top-level analyses and do not cover invalid shapes/K, threshold ties,
  component initialization, degenerate SWP, cache manifests, or fixture-level
  MATLAB outputs.
- All Python files compile. Ruff reports style issues in older scripts, but those
  are not scientific-validity failures.

## Recommended primary analysis

1. **Define the estimand.** Distinguish raw population topology, topology beyond
   neuronwise marginals, and topology beyond the population-rate envelope.
2. **Use the correct paired windows and rows.** Sleep 1500, anaesthesia 2900;
   same neurons within a session; exact top-m edges; record zero variance, event
   counts, cutoff, achieved density, degree, components, and LCC fraction.
3. **Use a temporal-null ladder.** At minimum: within-bout circular shifts; local
   event jitter/block permutation; an inhomogeneous independent-event model that
   preserves local rates; optionally a null preserving the population envelope.
4. **Use enough surrogates.** Hundreds for mean/interval estimation and 999+ for
   empirical tail probabilities. Separate temporal-null, Louvain, path-sampling,
   and internal-SWP randomness.
5. **Report null-calibrated measures.** Report real, null distribution, excess,
   percentile/rank, and uncertainty. Avoid causal "confound percentages."
6. **Use complementary triangle metrics.** Report both mean local C and
   transitivity; consider degree-stratified nodewise C. For disconnected graphs,
   use global efficiency on the fixed node set as primary and LCC-L as sensitivity.
7. **Test activity sensitivity without hiding biology.** Primary analysis on the
   paper's common neuron set, plus common-active thresholds/rate-matched empirical
   thinning. State clearly that these target different cell populations.
8. **Measure reliability.** Recompute across windows, neuron bootstraps, density,
   correlation type (Pearson/Spearman), positive-only versus absolute edges, and
   Louvain seeds. Report edge/module consensus and metric ICC/cluster bootstrap.
9. **Infer at mouse level.** Average windows within session/mouse, keep sleep and
   anaesthesia contrasts separate, nest repeated sessions, and use paired
   animal-level confidence intervals or a hierarchical model. A combined model
   needs a protocol interaction and clustered animal inference.
10. **Version every result.** Save settings, code/data hashes, row/window indices,
    package versions, seeds, and cache schema with every numeric table.

## Files produced by the independent audit

- Code: `verification/52_sparse_firing_robustness.py`
- This report: `verification/52_sparse_firing_robustness_report.md`
- Raw state/null values: `results/verification/52_sparse_firing_state_values.csv`
- Recording contrasts: `results/verification/52_sparse_firing_contrasts.csv`
- Animal-level summaries: `results/verification/52_sparse_firing_summary.csv`
- Anaesthesia window QC: `results/verification/52_anesthesia_window_qc.csv`
- Window reliability: `results/verification/52_window_reliability.csv`
- Figure: `results/figures/52_sparse_firing_robustness.png`

Generated result files are under the gitignored `results/` tree; the executable
analysis and report are retained in `verification/`.
