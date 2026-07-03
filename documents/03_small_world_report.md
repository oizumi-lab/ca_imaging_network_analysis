# Small-world analysis — port & replication report

**Goal.** Add path length, clustering coefficient, small-world-ness and
Small-World Propensity (SWP) to the hands-on, reproducing the "Small-worldness"
and "path length / clustering" slides of the Neuro2026 talk
(`documents/20260730_Neuro2026_Talk_Awake-Sleep_v2.pdf`).

## What we ported

Source: `oizumi-lab/mouse_network`, folder `kiyooka/SWP/` plus
`kiyooka/networkComparison/sliding2/characteristic_path_length_w.m`. The method is
Muldoon, Bridgeford & Bassett (2016), *Small-World Propensity in Weighted,
Real-World Networks*.

Python port → `src/funcnet/smallworld.py`:

| MATLAB | Python (`smallworld.py`) |
|---|---|
| `clustering_coef_matrix.m` (O/Z/B/bin) | `clustering_coef` (fast sparse triangle count for binary) |
| `characteristic_path_length_w.m` | `characteristic_path_length` (`1/W` distances; optional source sampling) |
| `regular_matrix_generator.m` | `regular_lattice` (ring lattice) |
| `randomize_matrix.m` | `randomize_matrix` |
| `small_world_propensity.m` | `small_world_propensity` → `SWResult` |
| `sw_summary.m` | `sw_summary` (|corr| → density threshold → largest CC → SWP) |

Definitions (network vs a ring-**lattice** and a **randomized** null):

```
SMN (small-world-ness) = (C_net/C_rand) / (L_net/L_rand)
ΔC = (C_reg − C_net)/(C_reg − C_rand)      ΔL = (L_net − L_rand)/(L_reg − L_rand)
SWP = 1 − sqrt(ΔC² + ΔL²)/sqrt(2)
```

## Pipeline (matches `sw_summary.m` + the driver `script_20251218_calc_small_world.m`)

Per 1500-frame (sleep) / 2900-frame (ane) window, for each state:
`spike_smoothed → corr → |r| → density_threshold(K=0.01, binary) →
largest connected component → small_world_propensity`.

**One deviation from the v1 driver, required for v2 data:** we restrict to
**active neurons** (`nonzero_ROI`). The v1 `smoothMat` was effectively the active
set; v2 ships all QC-passed neurons plus `nonzero_ROI` (README §2.9) to recover
the paper's set. Without it, the ~75% of neurons silent under anesthesia
fragment the 1%-density graph (largest component collapses from ~6000 to ~3000
nodes), which inflates `C_rand` and wrongly drags small-world-ness *down*. With
the filter, anesthesia small-world-ness rises into the paper's range and exceeds
wakefulness, as reported. (Sleep is nearly unchanged — ~98% of sleep neurons are
active.)

## Verification of the port

An adversarial verification pass (one checker per function, each comparing the
Python to the MATLAB **and** to an independent reference) returned **all
components correct**:

- `clustering_coef` — matches `networkx.clustering` per-node on 20 random graphs +
  hand triangle/path; confirmed Onnela == binary on binary inputs; weighted
  Onnela matches MATLAB to 1e-8.
- `characteristic_path_length` — equals `networkx.average_shortest_path_length`
  exactly (binary) and a brute-force Dijkstra to 1e-16 (weighted); source
  sampling is unbiased and convergent; disconnected → inf.
- `randomize_matrix` — preserves node count, exact weight multiset, edge count,
  symmetry, zero diagonal; uniform coverage.
- `regular_lattice` — preserves binary edge count and ring adjacency at radii
  1..r; radius matches MATLAB `mod(i+z-1,n)+1`.
- `small_world_propensity` / `sw_summary` — ΔC/ΔL/SWP/SMN formulas are
  self-consistent with the returned C/L fields; `avg_rad_eff` and the
  |corr|→threshold→largest-CC pipeline match the MATLAB.

## Replication results

Single window of `mouse01_sleep` (active neurons, K=1%): C_net ≈ 0.30 vs
C_rand ≈ 0.01 and L_net ≈ L_rand ≈ 2.7 → a clear small-world network (SMN ≈ 27).

Across all recordings (per-state pooled means), all three measures are **higher
during unconsciousness**, reproducing the talk:

| Dataset | Measure | Awake | Unconscious | PDF target (awake → unconscious) |
|---|---|---|---|---|
| Sleep | Small-world-ness | 28.2 | **34.5** | ~28 → ~34 ✓ |
| Sleep | Path length ΔL | 0.008 | **0.015** | ~0.007 → ~0.015 ✓ |
| Sleep | Clustering 1/ΔC | 1.78 | **2.62** | ~1.8 → ~2.6 ✓ |
| Anesthesia | Small-world-ness | 20.2 | **30.1** | ~20 → ~31 ✓ |
| Anesthesia | Path length ΔL | 0.006 | **0.016** | ~0.006 → ~0.016 ✓ |
| Anesthesia | Clustering 1/ΔC | 1.44 | **2.30** | ~1.5 → ~2.3 ✓ |

(Pooled across recordings/windows; per-mouse the same ordering holds — see the
figure. Values are not bit-identical to the talk because the lattice/random
nulls are stochastic and v2 keeps a superset of neurons, but the magnitudes and
the awake < unconscious ordering reproduce the slides for **both** datasets.)

Figure: `results/figures/40_small_world.png` — per-mouse scatter, awake vs
unconscious, for small-world-ness, ΔL, and 1/ΔC (sleep and anesthesia).

## Performance

The cost is all-pairs shortest paths on the largest component (thousands of
nodes, ×3 for net/lattice/random). `characteristic_path_length(n_sources=…)`
estimates the average path length from a random sample of source nodes — an
unbiased estimator that is ~4× faster and accurate to ~0.1% at 800 sources
(verified against the exact value). The tutorial uses this; set
`n_sources=None` for the exact computation.
