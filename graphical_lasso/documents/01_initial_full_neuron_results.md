# Initial full-neuron graphical-lasso investigation

## Executive conclusion

Graphical lasso materially changes these functional networks; it is not a mild
denoising of the current Pearson pipeline.

Two distinct effects must be separated:

1. The graphical-lasso **covariance** rescaled to marginal correlation becomes
   almost entirely positive and concentrates fixed-density edges within a much
   more degree-heterogeneous, partly disconnected topology. At the current 1%
   and 5% densities, this often raises clustering and path length, lowers
   modularity, and reverses the awake-to-unconscious modularity contrast.
2. The graphical-lasso **partial-correlation** matrix is genuinely sparse:
   99.53%--99.83% of pairs are exactly zero at the densest validated penalties.
   It therefore cannot supply the current 1% or 5% graph without promoting
   exact zeros to edges. At the jointly feasible 0.1% density it produces a
   substantially rewired graph and, especially under anesthesia, a qualitatively
   different state contrast.

These results are an estimator-sensitivity finding from one sleep and one
anesthesia recording, not evidence that graphical lasso is more biologically
correct. Pearson is not ground truth, and the current graphical-lasso penalties
were selected as the densest numerically validated full-neuron solutions rather
than by blocked predictive validation.

## 1. Scope and fixed inputs

| Recording | Comparison | Window | Full network population | Primary alpha |
|---|---|---:|---:|---:|
| `mouse01_sleep` | awake vs NREM | 1,500 frames/state | 7,693/7,693 active neurons | 0.4 |
| `mouse05_ane` | awake vs anesthesia | 2,900 frames/state | 3,210/3,210 active neurons | 0.3 |

No neuron was subsampled or removed: every `nonzero_ROI` neuron was finite and
nonconstant in both paired windows. The identical row set was used for both
states and all estimators.

Each neuron was centered and standardized to unit population variance within
its state window. The empirical covariance is consequently the Pearson
correlation matrix. Three matrices were retained:

- `pearson`: empirical marginal correlation;
- `glasso_marginal`: fitted covariance rescaled to marginal correlation;
- `glasso_partial`: `-precision_ij / sqrt(precision_ii * precision_jj)`.

Graphical lasso estimates a sparse precision matrix under a penalized Gaussian
likelihood ([Friedman, Hastie & Tibshirani, 2008](https://hastie.su.domains/Papers/graph.pdf)).
Exact covariance-threshold component screening was used to split large fits
without changing the optimizer or solution
([Mazumder & Hastie, 2012](https://jmlr.org/papers/v13/mazumder12a.html)).

The alpha path was 0.7, 0.6, 0.5, 0.4, and 0.3. A recording's two states always
used the same primary alpha. `mouse01_sleep` alpha 0.3 terminated without
returning a validated 7,693-node solution, so alpha 0.4 is its densest accepted
fit. `mouse05_ane` converged at alpha 0.3. This means absolute results should be
compared within a recording, not between the two recordings.

## 2. Numerical validation and native sparsity

All reported primary solutions passed the declared duality-gap and
precision-times-covariance inverse gates.

| Recording/state | Edges | Native density | Components | Largest component | Max gap | Max inverse error |
|---|---:|---:|---:|---:|---:|---:|
| sleep awake, alpha 0.4 | 50,451 | 0.1705% | 75 | 7,618 | 1.75e-5 | 3.22e-15 |
| sleep NREM, alpha 0.4 | 54,585 | 0.1845% | 52 | 7,640 | 2.03e-6 | 2.25e-15 |
| anesthesia-session awake, alpha 0.3 | 11,283 | 0.2191% | 187 | 3,016 | 9.13e-6 | 3.28e-15 |
| anesthesia, alpha 0.3 | 24,269 | 0.4712% | 7 | 3,204 | 3.37e-6 | 4.19e-15 |

The unconscious state has more native conditional edges under the same
within-recording penalty: +8.2% for NREM and +115% for anesthesia. This is why
native-support graphs cannot be used for state comparisons without a density
control.

More importantly, the maximum common fixed density is only 0.1705% for the
sleep pair and 0.2191% for the anesthesia pair. A 1% or 5% partial-correlation
graph would contain mostly zero-weight ties and has therefore been marked
**unavailable**, not manufactured.

## 3. Matrix-level effects

| Recording/state | Matrix | Mean | SD | Zero pairs | 99.9th percentile of abs(weight) |
|---|---|---:|---:|---:|---:|
| sleep awake | Pearson | 0.00472 | 0.07919 | 0% | 0.5145 |
|  | GL marginal | 0.01081 | 0.01465 | 1.94% | 0.1383 |
|  | GL partial | 0.000080 | 0.00275 | 99.8295% | 0.0261 |
| sleep NREM | Pearson | 0.00606 | 0.08059 | 0% | 0.5327 |
|  | GL marginal | 0.01605 | 0.01874 | 1.37% | 0.1576 |
|  | GL partial | 0.000086 | 0.00285 | 99.8155% | 0.0302 |
| anesthesia-session awake | Pearson | 0.00278 | 0.05729 | 0% | 0.3862 |
|  | GL marginal | 0.00114 | 0.00767 | 11.72% | 0.0998 |
|  | GL partial | 0.000127 | 0.00428 | 99.7809% | 0.0419 |
| anesthesia | Pearson | 0.01068 | 0.08114 | 0% | 0.7852 |
|  | GL marginal | 0.06350 | 0.06154 | 0.37% | 0.5163 |
|  | GL partial | 0.000245 | 0.00572 | 99.5288% | 0.0775 |

Three features matter:

- The fitted marginal matrix is not simply a compressed Pearson matrix. Its
  signed Pearson agreement with Pearson is only 0.22--0.44, and its absolute
  Spearman agreement is negative (-0.17 to -0.31). Thus mid-ranked absolute
  pairs are substantially reordered.
- The fitted marginal off-diagonal coefficients are nonnegative wherever they
  are nonzero in all four matrices. The sign agrees with Pearson on only
  15.5%--34.1% of supported pairs because Pearson's bulk is slightly negative.
- Almost all nonzero partial edges preserve their corresponding Pearson sign:
  100% in sleep and awake anesthesia-session data and 98.5% under anesthesia.
  The main partial-correlation effect is edge deletion and reranking, not broad
  sign reversal.

## 4. Strong-pair retention at exact fixed density

Jaccard overlap with the Pearson edge set:

| Recording/state | Density | GL marginal | GL partial |
|---|---:|---:|---:|
| sleep awake | 0.05% | 0.857 | 0.402 |
|  | 0.10% | 0.689 | 0.495 |
| sleep NREM | 0.05% | 0.870 | 0.378 |
|  | 0.10% | 0.667 | 0.467 |
| anesthesia-session awake | 0.05% | 0.887 | 0.523 |
|  | 0.10% | 0.767 | 0.584 |
| anesthesia | 0.05% | 0.985 | 0.169 |
|  | 0.10% | 0.738 | 0.208 |

The most extreme marginal pairs are mostly retained, particularly under
anesthesia, but overlap falls rapidly as density grows. At 5%, marginal-vs-
Pearson Jaccard is only 0.084 (sleep awake), 0.084 (NREM), 0.151
(anesthesia-session awake), and 0.167 (anesthesia).

Partial correlation retains only 17%--58% Jaccard overlap at the feasible
high-pair densities. The anesthesia partial graph is the strongest departure:
only 16.9% overlap at 0.05% and 20.8% at 0.1%. Thus conditioning on all other
neurons does not merely prune the weakest marginal correlations; it replaces a
large fraction of the selected strongest-pair graph.

## 5. Network measures at the common feasible density (0.1%)

`Q` is maximum modularity over three seeded Louvain runs. `L` is sampled-source
path length within the largest component; `GC` is that component's fraction of
all neurons.

| Recording/state | Method | Q | C | L | GC |
|---|---|---:|---:|---:|---:|
| sleep awake | Pearson | 0.8562 | 0.3513 | 5.438 | 0.756 |
|  | GL marginal | 0.8292 | 0.3368 | 5.394 | 0.678 |
|  | GL partial | 0.7043 | 0.2239 | 5.104 | 0.962 |
| sleep NREM | Pearson | 0.8775 | 0.3663 | 5.805 | 0.762 |
|  | GL marginal | 0.8162 | 0.3540 | 5.317 | 0.681 |
|  | GL partial | 0.7110 | 0.2231 | 5.255 | 0.969 |
| anesthesia-session awake | Pearson | 0.7849 | 0.1893 | 5.950 | 0.628 |
|  | GL marginal | 0.7796 | 0.1859 | 6.116 | 0.573 |
|  | GL partial | 0.6983 | 0.1133 | 5.807 | 0.780 |
| anesthesia | Pearson | 0.7232 | 0.2237 | 6.277 | 0.102 |
|  | GL marginal | 0.6872 | 0.2016 | 6.635 | 0.092 |
|  | GL partial | 0.8649 | 0.2448 | 9.480 | 0.882 |

### Sleep contrast

At 0.1%, Pearson gives NREM-minus-awake changes of `DeltaQ=+0.0213`,
`DeltaC=+0.0150`, and `DeltaL=+0.3669`. Partial correlation retains the
directions of modularity and path length but attenuates them to `+0.0067` and
`+0.1509`; it removes the clustering increase (`DeltaC=-0.0008`).

Thus, in this window, the partial graph suggests that much of the ultra-sparse
sleep contrast in modularity/path length is carried by marginal relationships,
while the clustering contrast disappears. This is compatible with the
repository's prior sparsity-confound warning for clustering, but no
graphical-lasso shuffle-null was run here, so it is not a confound correction.

### Anesthesia contrast

At 0.1%, Pearson modularity decreases under anesthesia (`DeltaQ=-0.0617`),
whereas partial-correlation modularity increases strongly (`+0.1667`). Partial
correlation also amplifies `DeltaC` from `+0.0344` to `+0.1314` and `DeltaL`
from `+0.327` to `+3.673`.

The accompanying connectedness change is essential: the Pearson anesthesia
graph's largest component contains only 10.2% of neurons, so its `L` describes
326 neurons. The partial graph contains 88.2% (2,830 neurons). The partial
network is therefore not merely a larger value of the same path statistic; it
is a much more globally connected but internally long-path topology.

Partition agreement reinforces the rewiring result. At 0.1%, Pearson-vs-partial
adjusted Rand index is 0.175/0.152 for sleep awake/NREM, 0.421 for the
anesthesia-session awake state, and only 0.0004 under anesthesia.

## 6. What happens at the current 1% and 5% densities?

Only Pearson and `glasso_marginal` can be compared at these densities. The
partial matrices do not contain enough nonzero edges.

Awake-to-unconscious contrasts:

| Recording | Density | Method | DeltaQ | DeltaC | DeltaL |
|---|---:|---|---:|---:|---:|
| sleep | 1% | Pearson | +0.0243 | +0.0252 | +0.0398 |
|  |  | GL marginal | -0.0170 | +0.0165 | +0.0914 |
| sleep | 5% | Pearson | +0.0193 | +0.0161 | +0.0028 |
|  |  | GL marginal | -0.0062 | -0.0039 | +0.0173 |
| anesthesia | 1% | Pearson | +0.2343 | +0.2866 | +1.1327 |
|  |  | GL marginal | -0.0380 | +0.0479 | +2.7714 |
| anesthesia | 5% | Pearson | +0.1840 | +0.1919 | +0.1037 |
|  |  | GL marginal | -0.0372 | +0.0271 | +1.0919 |

The positive Pearson modularity contrast reverses under the graphical-lasso
marginal estimator at both 1% and 5% in both recordings. The clustering
contrast is attenuated or removed, especially for anesthesia, while path-length
contrasts become larger.

This does **not** mean that graphical lasso disproves the published modularity
result. `glasso_marginal` is the inverse of a sparse fitted precision matrix and
is not the conditional-dependence graph for which graphical lasso is normally
used. It is included specifically to isolate covariance-regularization effects.

## 7. Why the marginal network measures move so much

At 5% fixed density, graphical-lasso marginal networks have much higher degree
heterogeneity and acquire isolates even though edge count is held constant:

| Recording/state | Method | Degree CV | Isolates | GC | Q | C | L |
|---|---|---:|---:|---:|---:|---:|---:|
| sleep awake | Pearson | 0.244 | 0 | 1.000 | 0.236 | 0.232 | 1.951 |
|  | GL marginal | 1.746 | 327 | 0.955 | 0.159 | 0.708 | 2.455 |
| sleep NREM | Pearson | 0.303 | 0 | 1.000 | 0.256 | 0.248 | 1.954 |
|  | GL marginal | 1.755 | 330 | 0.955 | 0.152 | 0.704 | 2.472 |
| anesthesia-session awake | Pearson | 0.269 | 0 | 1.000 | 0.213 | 0.182 | 1.968 |
|  | GL marginal | 1.509 | 206 | 0.930 | 0.167 | 0.568 | 2.328 |
| anesthesia | Pearson | 0.720 | 0 | 1.000 | 0.398 | 0.374 | 2.072 |
|  | GL marginal | 1.708 | 373 | 0.872 | 0.129 | 0.595 | 3.420 |

The sparse precision solution has many connected components. Its inverse
covariance is dense within those blocks but zero between blocks. Fixed-density
ranking therefore concentrates marginal edges within selected blocks and on
high-degree neurons. The resulting triangle-rich, heterogeneous graph explains
why clustering and path length rise despite a lower modularity score. This is an
estimator-induced topological change, not an edge-count artifact.

At 5%, Pearson-vs-GL-marginal adjusted Rand indices are only 0.002--0.062,
confirming that the inferred module assignments are nearly unrelated at the
current reference density.

## 8. Interpretation boundaries

1. **Only two recordings and one window per state were analyzed.** There is no
   mouse-level inference and no claim of cohort generalization.
2. **Alpha was constrained by validated all-neuron computation, not selected by
   held-out likelihood or EBIC.** Different recording-level primary alphas were
   necessary; state pairs still share alpha exactly.
3. **Calcium event traces violate ideal i.i.d. Gaussian observations.** They are
   sparse, nonnegative, temporally smoothed, and autocorrelated. A nonzero
   partial edge is conditional association under this fitted model, not direct
   synaptic or causal coupling.
4. **No temporal-shuffle or window-stability refits were run.** In particular,
   the analysis does not establish that graphical lasso removes the known
   firing-sparsity clustering confound.
5. **Louvain used three runs, and path length used 256 sampled sources.** These
   are initial-analysis settings. Q-run SDs are retained in the exact table;
   path-length comparisons involving strongly different giant-component sizes
   must be read with caution.
6. **The 1%/5% partial comparison is scientifically undefined at these alphas.**
   Padding exact zeros would make the result depend on arbitrary tie ordering.

## 9. Recommended next investigation

The next step should not immediately expand to every mouse. First determine a
principled alpha regime on these two recordings using continuity-aware blocked
held-out likelihood or EBIC, then test whether that regime yields stable native
support at densities relevant to the current network analysis. The primary
sensitivity targets are:

- penalty and edge-selection stability across nonoverlapping windows;
- circular-shift nulls refit with the same state-paired alpha;
- common-density partial graphs over the full feasible density interval;
- whether lower-alpha scalable fits can validly reach 1% support;
- more Louvain runs and exact/more-source path validation for retained settings.

Until then, the defensible conclusion is: **full-neuron graphical lasso changes
both which pairs are called strong and the resulting network topology enough to
alter or reverse state effects; the partial graph is too sparse to be inserted
unchanged into the existing 1%/5% pipeline.**

## 10. Reproducible outputs

- Matrix summaries and validated alpha path:
  `graphical_lasso/results/01_matrices/`
- Exact matrix agreement: `matrix_agreement.csv`
- Fixed-density edge overlap: `edge_overlap.csv`
- Full network measures: `network_measures.csv`
- State contrasts: `state_contrasts.csv`
- Method-by-state interactions: `method_state_interactions.csv`
- Louvain partition agreement: `partition_agreement.csv`
- Figures: `graphical_lasso/results/02_network_comparison/figures/`

The scripts and numerical regression tests are described in
`graphical_lasso/README.md`.
