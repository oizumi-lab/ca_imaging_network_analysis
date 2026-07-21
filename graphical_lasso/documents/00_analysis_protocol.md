# Initial full-neuron graphical-lasso protocol

## Scope

The initial investigation is deliberately limited to `mouse01_sleep` and
`mouse05_ane`, but it retains the complete paper-defined network population in
each recording. It asks how replacing empirical Pearson connectivity with a
graphical-lasso estimate changes:

1. the coefficient matrices;
2. the strongest pairs selected at fixed density;
3. modularity, clustering, path length, connectedness, and degree structure.

## Estimands

Each state window yields three matrices on exactly the same neurons:

- empirical Pearson correlation, `pearson`;
- graphical-lasso covariance rescaled to correlation, `glasso_marginal`;
- partial correlation derived from the sparse precision matrix,
  `glasso_partial`.

The second object isolates regularized marginal-covariance effects. The third
changes the estimand to conditional association. These are not interchangeable.

## Population and preprocessing

The starting population is every `nonzero_ROI` neuron. A neuron is removed only
if it is non-finite or constant in either member of the paired state windows,
because graphical lasso requires a valid variance and both states must retain
identical rows. Each retained trace is centered and scaled to unit population
variance within its state window. The resulting empirical covariance is
therefore the Pearson correlation matrix up to floating-point precision.

The inputs are sparse, non-negative, Gaussian-smoothed calcium-event traces.
Graphical-lasso conditional-independence language is therefore a model-based
description, not evidence for direct synaptic or causal connections.

## Regularization path and exact large-scale solution

The frozen descending penalty path is `0.7, 0.6, 0.5, 0.4, 0.3`. The primary
estimate is the densest full-neuron solution that passes the numerical gates in
both states of a recording: `0.4` for `mouse01_sleep` and `0.3` for
`mouse05_ane`. All variables are standardized, and a recording's two states
always use the same numerical penalty. The path is not presented as
likelihood-optimal; its purpose is to locate a converged full-neuron regime and
quantify penalty sensitivity without state-specific tuning. The 7,693-node
sleep fit at `0.3` terminated without returning a solution and is excluded,
whereas its `0.4` fit passed the declared gates.

At penalty alpha, the connected components of the graph
`abs(sample_correlation) > alpha` are exactly the connected components of the
graphical-lasso precision solution. The implementation uses this identity to
solve independent components and warm-starts each lower penalty from the
preceding solution. This changes computation, not the objective or estimator.

Every accepted component must have a finite QUIC duality gap and inverse
consistency. The scripts retain iteration counts, component sizes, objective,
duality gap, inverse error, native edge count, and native density. A failed gate
stops the analysis instead of returning a partial matrix.

The numerical solver tolerance is `1e-7`. Because the reported duality gap is
an unnormalized objective gap whose scale grows with matrix dimension, the
acceptance gate requires both an absolute gap below `1e-3` and a gap per full
network node below `1e-8`. This avoids applying a dimension-dependent criterion
to the 3,210- and 7,693-node problems while remaining much stricter than the
precision needed for edge ranking and network summaries.

## Fixed-density rule

Edges are ranked by absolute coefficient, matching the repository's published
pipeline. Selection retains exactly `floor(K * p * (p - 1) / 2)` pairs, with
deterministic handling of boundary ties. A partial-correlation density is
reported as unavailable when the requested edge count exceeds its native
nonzero support; zero coefficients are never promoted to edges.

The broad edge-overlap grid is 0.05%, 0.1%, 0.2%, 0.5%, 1%, 2%, and 5%.
Network measures are evaluated at 0.1%, 0.5%, 1%, and 5% whenever the matrix
supports that density. This intentionally includes the current pipeline's 1%
small-world and 5% modularity reference points while resolving much sparser
partial-correlation graphs.

## Network summaries

For each feasible graph the analysis reports exact realized density, threshold,
positive-edge fraction, mean/SD degree, degree coefficient of variation,
isolates, largest-component fraction, mean clustering, sampled-source
characteristic path length on the largest component, maximum-over-three-run
Louvain modularity, module count, and spatial edge length.

Agreement is descriptive, not an accuracy score: Pearson is not ground truth.
The main interpretation is whether covariance regularization or conditionalizing
changes the retained high-weight pairs and whether it changes the direction or
magnitude of awake-to-unconscious network contrasts.
