# Hands-on: functional-network modularity across brain states

A guided walkthrough for course attendees. Work through the `# %%` scripts in
`scripts/` in order; this document explains the *why* behind each step and ties
it back to the paper.

> Kiyooka & Oomoto et al. (2026), *Single-cell resolution functional networks
> during unconsciousness are segregated into spatially intermixed modules*,
> Cell Reports.

## Big picture

Neurons that fire together can be linked into a **functional network**. We ask
how that network's *organisation* changes when the brain goes from awake to
unconscious (sleep or anesthesia). The key quantity is **modularity** — the
degree to which neurons split into well-separated functional modules.

The whole analysis is four steps:

```
activity (spikes)  →  correlation matrix  →  thresholded graph  →  modularity Q
```

## Setup

```bash
poetry install
poetry run python scripts/download_data.py --example   # 84 MB, enough to start
```

Open scripts in VS Code (Python extension) or Spyder and run cell by cell
(`# %%` markers). Each script saves its figures to `results/figures/`.

## 1 · Inspect the data — `00_inspect_data.py`

Learn what a recording contains: `spike_smoothed` (N neurons × T frames),
per-frame brain `state`, neuron `centroid`s, and the `used_frame` epochs chosen
for analysis. Calcium imaging at 7.65 Hz; states are labelled per frame
(awake / quiet-awake / NREM / REM, or awake / anesthesia).

**Concept to land:** the data are large-scale (thousands of neurons) and
single-cell resolution — that's what makes single-cell network analysis possible.

## 2 · Reproduce the reference — `01_reproduce_example.py`

Before trusting our tools, we reproduce the dataset's own MATLAB example exactly
and check each step (edge count, symmetry, modularity self-consistency). This is
good scientific hygiene: *validate the pipeline on a known case first.* See
`documents/01_reproduction_report.md`.

## 3 · Build a functional network — `10_functional_connectivity.py`

- **Functional connectivity** = Pearson correlation between every neuron pair,
  computed *separately per state* on matched 1500-frame windows.
- Compare the correlation matrices and their distributions between awake and
  unconscious states.

**Concept to land:** "connectivity" here is statistical (co-activity), not
anatomical wiring.

## 4 · Modularity — `20_modularity.py`

Three practical ideas, each a common pitfall if ignored:

1. **Fix the density.** Keep the strongest *K %* of edges so two networks have
   the same number of links. Otherwise a denser network looks more modular for
   trivial reasons. The paper sweeps K from 0.8 % to 30 %.
2. **Louvain is stochastic.** Run it many times and take the **max-Q** partition
   (or a consensus). A single run is noisy.
3. **Resolution γ.** Controls module size; report results across γ to show they
   aren't a parameter artefact.

The **spatial module map** colours each neuron by module on the cortex. The
paper's striking observation: at single-cell scale, modules are **spatially
intermixed** — adjacent neurons often belong to different functional modules.

## 5 · The result — `30_state_comparison.py`

Put it together: compute max-Q modularity for each state across densities, for
the sleep dataset (awake vs NREM) and the anesthesia dataset (awake vs
anesthesia). The unconscious curve sits **above** the awake curve:
**modularity increases during unconsciousness** at single-cell resolution.

Interpretation: during sleep/anesthesia the cortical functional network
fragments into more sharply segregated modules — a candidate signature of
reduced information integration when consciousness is lost.

## Where to go next (future course modules)

- **Spatial scale (coarse-graining).** Group neurons spatially and recompute Q;
  the paper finds the state difference is specific to the single-cell scale and
  vanishes at the mesoscale (Fig. 7).
- **Per-neuron contribution $Q_i$** and its relation to node degree (Fig. 4).
- **Temporal module stability** and consensus partitions (Fig. 6).

## Cheat-sheet (the library API)

```python
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.funcnet import dataio, network as net

rec  = dataio.load_recording("mouse07_ane")      # v2.0 loader
X    = dataio.activity(rec, "anesthesia",        # (N, n_frames) for a state
                       nonzero_only=True)
C    = net.correlation_matrix(X)                 # functional connectivity
adj, thr = net.density_threshold(C, K=0.05)      # fixed-density graph
res  = net.repeat_louvain(adj, gamma=1.0, n_runs=200)
print(res["Q_max"], res["n_modules_max"])        # robust modularity
```
