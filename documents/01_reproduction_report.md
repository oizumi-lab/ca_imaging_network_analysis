# Reproduction report — modularity pipeline (MATLAB → Python, v1 → v2 data)

**Goal.** Re-implement the published modularity analysis
(`oizumi-lab/mouse_network_2P`, written for dataset v1.0) in Python and confirm
it reproduces the reference behaviour on the current **v2.0** dataset, before
building teaching materials on top.

## What we ported

| MATLAB (repo / example) | Python (`scripts/lib/network.py`) |
|---|---|
| `corr(spike')`, zero diagonal | `correlation_matrix()` |
| `densityBasedThresh(C, K, option)` | `density_threshold()` |
| `community_louvain(W, gamma)` (BCT) | `louvain_modularity()` (via `bctpy`) |
| `repeat_modularity_analysis` + `get_maxQ` | `repeat_louvain()` |
| `perform_consensus_clustering` (`agreement`, `consensus_und`) | `consensus_partition()` |

Data loading was rewritten for v2.0 in `scripts/lib/dataio.py` (see
`.claude/rules/dataset-v2-format.md`). The two substantive changes versus v1:

1. **File format.** v2.0 `.mat` files are MATLAB **v7.3 (HDF5)**; `scipy.io.loadmat`
   fails on them. We read with `pymatreader`. (Verified from the file header:
   `MATLAB 7.3 MAT-file ... HDF5 schema 1.00`.)
2. **Variable renames + 1-based indices** (e.g. `smoothed_spike`→`spike_smoothed`,
   `frame.used_frame` 1-based → 0-based). Handled in the loader.

## Validation — `scripts/01_reproduce_example.py` on `example_data.mat`

Reproducing the dataset's official `example_network_analysis.m`
(`spike_smoothed → corr → densityBasedThresh(K=0.05) → community_louvain(γ=1)`),
N = 1000 neurons:

**Deterministic steps match exactly.**
- Threshold correlation r = 0.0729.
- Edges kept = **24,975** = `floor(0.05 · N(N−1)/2)` ✓
- Adjacency is symmetric, binary {0,1}, zero-diagonal ✓

**Modularity is internally consistent.**
- Louvain (seed 1): **Q = 0.1856**, 9 modules.
- Independent Newman–Girvan Q on the same partition = 0.185575;
  `|Q − Q_indep| ≈ 3 × 10⁻¹⁷` ✓ (confirms the BCT wrapper computes the
  modularity it claims).
- Q over 30 random seeds: mean 0.191, sd 0.005 — the expected mild stochasticity
  of Louvain, which the full pipeline removes by taking max-Q over ~200 runs.

### On "exact" reproduction

Louvain is a stochastic optimiser; MATLAB and Python use different RNGs, so
partitions are **not** bit-identical across languages — nor are repeated runs
within one language. We therefore validate at the right level:

- **Deterministic** parts (correlation, density threshold, graph) — exact match.
- **Stochastic** part (Louvain) — the reported Q equals an independent modularity
  computation to ~1e-16, the Q distribution is tight, and module structure is
  stable (block-diagonal sorted adjacency; spatially intermixed module map).

This is the appropriate standard for "reproducing the same results" with a
randomised community-detection algorithm.

## State-dependent result — `scripts/30_state_comparison.py`

Applying the validated pipeline to real recordings (1500-frame windows,
activity-filtered neurons, max-Q over runs) reproduces the paper's single-cell
finding: **modularity is higher during unconsciousness.** Spot check at K = 5 %:

- `mouse01_sleep`: Q(awake) = 0.239 → Q(NREM) = 0.253 (**+0.014**)
- `mouse07_ane`:  Q(awake) = 0.288 → Q(anesthesia) = 0.353 (**+0.065**)

The full script aggregates this across densities and animals for both the sleep
and anesthesia datasets.

## Known limitation

`ROIs.atlas` (per-neuron region acronyms) is a MATLAB *string-class* object that
neither `pymatreader` nor `h5py` decode; it is exposed as `None`. It is not
needed for single-cell modularity. Region/mesoscale analyses that need it would
require re-exporting the field as `cellstr`/char from MATLAB, or an MCOS string
decoder.
