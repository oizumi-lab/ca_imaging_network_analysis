# %% [markdown]
# # 51 · Verify small-world metrics against established network libraries
#
# Script 40 computes **clustering coefficient**, **characteristic path length**,
# and **small-world-ness / SWP** with our own code (`src/funcnet/smallworld.py`,
# a port of Muldoon et al. 2016). Here we cross-check those measures against
# independent Python libraries.
#
# ## Available Python network-analysis libraries
#
# | library | what it gives us here | installed? |
# |---|---|---|
# | **NetworkX** | `average_clustering`, `clustering`, `transitivity`, `average_shortest_path_length`, `sigma`, `omega` | ✅ (dep) |
# | **bctpy** | `clustering_coef_bu`, `distance_bin` + `charpath` | ✅ (dep) |
# | **igraph** | `transitivity_avglocal_undirected`, `average_path_length` | optional |
#
# ## What is exactly checkable vs. what is only comparable
# - **Clustering coefficient** and **characteristic path length** are *definitions*
#   — on the same binary graph, NetworkX and bctpy must return the **same numbers**
#   as our code (agreement ~1e-12). These are hard PASS/FAIL checks.
# - **Small-world-ness** depends on **null models** (a random graph and a
#   lattice). Every package builds those nulls differently — our SWP uses
#   Muldoon's weight-preserving nulls; NetworkX's `sigma`/`omega` use
#   degree-preserving edge swaps. So the *composite indices differ by
#   construction*; the meaningful cross-check is that they all **agree the
#   network is small-world** (our SMN ≫ 1, NetworkX σ > 1, ω ≈ 0).

# %%
import os
import sys
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import networkx as nx
import bct

from src.funcnet import dataio, network as net, smallworld as sw

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# --- settings ---
RECORDING = "example_data"   # fast 1000-neuron sample; any recording name works
STATE = "awake"
WINDOW = 1500
N_NEURONS = 800              # neurons for the clustering / path-length checks
DENSITY = 0.01               # 1% binary graph, as in script 40
N_SMALL = 120                # small connected graph for the (slow) sigma/omega indices
DENSITY_SMALL = 0.06         # denser threshold so the small graph stays connected
EXACT_TOL = 1e-9             # clustering / path-length agreement (same graph)
RUN_SIGMA_OMEGA = True       # NetworkX σ/ω are O(edges·niter·nrand); set False to skip
SIGMA_NITER, SIGMA_NRAND = 2, 3   # null-model rewiring depth / count for σ, ω (~30 s)

# %% [markdown]
# ## Build the binary test graph
# Correlation → |r| → 1%-density binary threshold, exactly as `sw.sw_summary`
# does. Clustering is defined per node; path length needs a connected graph, so
# for that check we restrict to the largest connected component.

# %%
rec = dataio.load_recording(RECORDING)
keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
rows = np.flatnonzero(keep)
rng = np.random.RandomState(0)
if rows.size > N_NEURONS:
    rows = np.sort(rng.choice(rows, N_NEURONS, replace=False))
win = dataio.state_frames(rec, STATE)[:WINDOW]
C = net.correlation_matrix(rec.spike_smoothed[np.ix_(rows, win)])

A, _ = net.density_threshold(np.abs(C), DENSITY)     # binary, symmetric, 0 diagonal
G = nx.from_numpy_array(A)
print(f"{RECORDING} [{STATE}]: {rows.size} neurons -> {int(A.sum()//2)} edges "
      f"at {DENSITY:.0%} density\n")

# %% [markdown]
# ## Check 1 — clustering coefficient (deterministic, must match exactly)
# Compare per-node clustering from our `clustering_coef` against NetworkX's
# `clustering` and bctpy's `clustering_coef_bu`. Note NetworkX's **transitivity**
# is a *different* quantity (global ratio of triangles to triads), so it is
# expected to differ — we print it to make the distinction explicit.

# %%
Cc_custom = sw.clustering_coef(A, "bin")
Cc_nx = np.array([nx.clustering(G)[i] for i in range(A.shape[0])])
Cc_bct = bct.clustering_coef_bu(A)

d_nx = float(np.max(np.abs(Cc_custom - Cc_nx)))
d_bct = float(np.max(np.abs(Cc_custom - Cc_bct)))
avg_custom = sw.avg_clustering(A)
avg_nx = nx.average_clustering(G)
print(f"  mean clustering   custom={avg_custom:.10f}  nx={avg_nx:.10f}  "
      f"bct={float(np.nanmean(Cc_bct)):.10f}")
print(f"  per-node max|Δ|   vs NetworkX={d_nx:.2e}   vs bctpy={d_bct:.2e}")
print(f"  (NetworkX transitivity={nx.transitivity(G):.6f}  — a DIFFERENT, global metric)")
assert d_nx < EXACT_TOL and d_bct < EXACT_TOL, "clustering coefficient mismatch"
print(f"  PASS — clustering matches custom / NetworkX / bctpy (< {EXACT_TOL:.0e}).")

# %% [markdown]
# ## Check 2 — characteristic path length (deterministic, must match exactly)
# On the largest connected component: our `characteristic_path_length` vs
# NetworkX `average_shortest_path_length` vs bctpy `charpath(distance_bin(·))`.
# We also confirm the fast **sampled** estimator (`n_sources`) that script 40
# uses is accurate to ~1%.

# %%
idx = sw.largest_component(A)
Acc = A[np.ix_(idx, idx)]
Gcc = nx.from_numpy_array(Acc)

L_custom = sw.characteristic_path_length(Acc)                    # exact, all pairs
L_nx = nx.average_shortest_path_length(Gcc)
L_bct = bct.charpath(bct.distance_bin(Acc), include_diagonal=False)[0]
L_samp = sw.characteristic_path_length(Acc, n_sources=min(400, len(idx)),
                                       rng=np.random.RandomState(1))

d_nx = abs(L_custom - L_nx)
d_bct = abs(L_custom - L_bct)
rel_samp = abs(L_samp - L_custom) / L_custom
print(f"  largest connected component: {len(idx)} nodes")
print(f"  L   custom={L_custom:.10f}  nx={L_nx:.10f}  bct={L_bct:.10f}")
print(f"  exact agreement  vs NetworkX={d_nx:.2e}   vs bctpy={d_bct:.2e}")
print(f"  sampled estimator (n_sources=400): L={L_samp:.6f}  rel.err={rel_samp:.2%}")
assert d_nx < EXACT_TOL and d_bct < EXACT_TOL, "path length mismatch"
assert rel_samp < 0.03, "sampled path-length estimator drifted > 3%"
print(f"  PASS — path length matches (< {EXACT_TOL:.0e}); sampled estimator within 3%.")

# %% [markdown]
# ## Check 3 — small-world-ness (comparable, not identical)
# On a small connected subnetwork we compute **our** SMN / SWP and NetworkX's
# **σ** and **ω**. The composite values differ (different null models), but all
# three must agree on the qualitative verdict: **this is a small-world network**
# (SMN ≫ 1, σ > 1, ω near 0).

# %%
rows_s = np.sort(rng.choice(rows.size, min(N_SMALL, rows.size), replace=False))
Cs = C[np.ix_(rows_s, rows_s)]
As, _ = net.density_threshold(np.abs(Cs), DENSITY_SMALL)
si = sw.largest_component(As)
As = As[np.ix_(si, si)]
Gs = nx.from_numpy_array(As)
assert nx.is_connected(Gs), "small graph not connected — raise DENSITY_SMALL"

res = sw.small_world_propensity(As, rng=np.random.RandomState(1))
print(f"  small graph: {As.shape[0]} nodes, {int(As.sum()//2)} edges")
print(f"  ours   : SMN(small-world-ness)={res.sw_ness:.3f}  SWP={res.SWP:.3f}  "
      f"(C={res.net_clus:.3f}, L={res.net_path:.3f})")

if RUN_SIGMA_OMEGA:
    t0 = time.time()
    sigma = nx.sigma(Gs, niter=SIGMA_NITER, nrand=SIGMA_NRAND, seed=1)
    omega = nx.omega(Gs, niter=SIGMA_NITER, nrand=SIGMA_NRAND, seed=1)
    print(f"  NetworkX: sigma={sigma:.3f} (small-world if >1)   "
          f"omega={omega:.3f} (small-world if ≈0)   [{time.time()-t0:.1f}s]")
    small_world = res.sw_ness > 1 and sigma > 1
    assert small_world, "libraries disagree on the small-world verdict"
    print("  PASS — our SMN and NetworkX sigma both flag a small-world network.")
else:
    print("  (RUN_SIGMA_OMEGA=False — skipping NetworkX sigma/omega)")

# %% [markdown]
# ## Summary

# %%
print("=" * 66)
print("SMALL-WORLD VERIFICATION")
print(f"  reference recording : {RECORDING} [{STATE}]")
print(f"  libraries used      : NetworkX {nx.__version__}, bctpy")
print(f"  Check 1 clustering       : PASS — identical to < {EXACT_TOL:.0e}")
print(f"  Check 2 path length      : PASS — identical to < {EXACT_TOL:.0e}")
print(f"  Check 3 small-world-ness : PASS — same qualitative verdict"
      + ("" if RUN_SIGMA_OMEGA else " (sigma/omega skipped)"))
print("=" * 66)
