# %% [markdown]
# # 50 · Verify modularity against established network libraries
#
# The project computes modularity with **our own code** (`src/funcnet/network.py`):
# `density_threshold` builds the graph, and `repeat_louvain` runs BCT's Louvain
# and reports Q. This script **cross-checks those numbers** against independent,
# widely-used Python graph libraries, so we can trust the pipeline.
#
# ## Available Python network-analysis libraries
#
# | library | what it gives us here | installed? |
# |---|---|---|
# | **NetworkX** | `community.modularity` (Q of a partition), `community.louvain_communities` (independent Louvain) | ✅ (dep) |
# | **bctpy** | Brain Connectivity Toolbox port; `community_louvain` is what our code calls | ✅ (dep) |
# | **python-igraph** | `Graph.modularity`, `community_multilevel` (Louvain) | optional |
# | **python-louvain** (`community`) | `best_partition`, `modularity` (the original Louvain package) | optional |
#
# We use **NetworkX** as the primary independent reference (a completely separate
# implementation from our NumPy/SciPy code), and light up **igraph** /
# **python-louvain** too *if* they are installed (`poetry add --group dev
# python-igraph python-louvain`).
#
# ## Two kinds of check
# 1. **Modularity *value* (deterministic).** Given one fixed partition, the
#    Newman-Girvan Q is a formula — every library must return the *same number*
#    (agreement to machine precision; we assert < 1e-9). This validates our
#    `modularity_value` and the Q that Louvain reports.
# 2. **Louvain *detection* (stochastic).** Different Louvain implementations find
#    different partitions, but all maximise the same Q, so their best Q should
#    **agree within a small tolerance**. This validates the community detection.

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import networkx as nx

from src.funcnet import dataio, network as net

warnings.filterwarnings("ignore", message="invalid value encountered in divide")

# --- settings ---
RECORDING = "example_data"   # fast 1000-neuron sample; any recording name works
STATE = "awake"
N_NEURONS = 600              # subsample for a quick, well-connected test graph
WINDOW = 1500
DENSITIES = [0.02, 0.05, 0.10]
GAMMA = 1.0
N_RUNS = 20                  # Louvain runs for our (bctpy) max-Q
N_SEEDS_NX = 10              # independent NetworkX Louvain restarts
VALUE_TOL = 1e-9             # deterministic Q agreement (same partition)
DETECT_TOL = 0.03            # best-Q agreement across independent Louvain impls

# %% [markdown]
# ## Build a test graph
# Correlation → density threshold, exactly as in scripts 10/20/30. We reuse the
# same `net` functions the pipeline uses, then hand the *adjacency matrix* to the
# external libraries — so the only thing under test is the graph analysis.

# %%
rec = dataio.load_recording(RECORDING)
keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
rows = np.flatnonzero(keep)
rng = np.random.RandomState(0)
if rows.size > N_NEURONS:
    rows = np.sort(rng.choice(rows, N_NEURONS, replace=False))
win = dataio.state_frames(rec, STATE)[:WINDOW]
C = net.correlation_matrix(rec.spike_smoothed[np.ix_(rows, win)])
print(f"{RECORDING} [{STATE}]: {rows.size} neurons, {win.size} frames\n")


def communities_from_ci(ci):
    """1-based community-label vector -> list of node-index sets (NetworkX form)."""
    groups = {}
    for node, c in enumerate(ci):
        groups.setdefault(int(c), set()).add(node)
    return list(groups.values())


# optional libraries -----------------------------------------------------------
try:
    import igraph as _ig
    HAVE_IGRAPH = True
except ImportError:
    HAVE_IGRAPH = False
try:
    import community as _louvain      # the "python-louvain" package
    HAVE_PYLOUVAIN = True
except ImportError:
    HAVE_PYLOUVAIN = False

# %% [markdown]
# ## Check 1 — modularity *value* for a fixed partition (deterministic)
# Take the max-Q partition our code finds, then re-score that **same partition**
# with (a) our `modularity_value`, (b) the Q that BCT's Louvain reported, and
# (c) NetworkX's `modularity`. All must match to machine precision (assert < 1e-9).

# %%
value_rows = []
partitions = {}
for K in DENSITIES:
    A, _ = net.density_threshold(C, K, negative=True)   # rank by |r|, as in scripts 20/30/40
    res = net.repeat_louvain(A, gamma=GAMMA, n_runs=N_RUNS)
    ci, Q_bct = res["ci_max"], res["Q_max"]
    partitions[K] = (A, ci, Q_bct)

    Q_custom = net.modularity_value(A, ci, gamma=GAMMA)          # our formula
    G = nx.from_numpy_array(A)
    Q_nx = nx.community.modularity(G, communities_from_ci(ci), resolution=GAMMA)

    spread = max(abs(Q_custom - Q_bct), abs(Q_custom - Q_nx), abs(Q_bct - Q_nx))
    ok = spread < VALUE_TOL
    value_rows.append((K, Q_bct, Q_custom, Q_nx, spread, ok))
    print(f"  K={K:>4.0%}  Q(bct)={Q_bct:.10f}  Q(custom)={Q_custom:.10f}  "
          f"Q(nx)={Q_nx:.10f}  max|Δ|={spread:.2e}  {'PASS' if ok else 'FAIL'}")

assert all(r[-1] for r in value_rows), "modularity VALUE disagreement exceeds tolerance"
print(f"\n✔ modularity value matches across bctpy / custom / NetworkX (< {VALUE_TOL:.0e}).")

# %% [markdown]
# ## Check 2 — Louvain *community detection* (stochastic)
# Run **NetworkX's own Louvain** (`louvain_communities`) from several seeds and
# keep its best Q. It is a different implementation, so it won't return an
# identical partition — but because both maximise the same objective, the best Q
# should land within `DETECT_TOL` of our (bctpy) max-Q. igraph / python-louvain
# join in when available.

# %%
detect_rows = []
for K in DENSITIES:
    A, ci_bct, Q_bct = partitions[K]
    G = nx.from_numpy_array(A)

    q_nx = max(nx.community.modularity(
        G, nx.community.louvain_communities(G, resolution=GAMMA, seed=s), resolution=GAMMA)
        for s in range(N_SEEDS_NX))

    extras = {}
    if HAVE_IGRAPH:
        src, dst = np.nonzero(np.triu(A, 1))
        g = _ig.Graph(n=A.shape[0], edges=list(zip(src.tolist(), dst.tolist())))
        extras["igraph"] = g.community_multilevel(resolution=GAMMA).modularity
    if HAVE_PYLOUVAIN:
        part = _louvain.best_partition(G, resolution=GAMMA, random_state=0)
        extras["pylouvain"] = _louvain.modularity(part, G)

    cand = [Q_bct, q_nx, *extras.values()]
    spread = max(cand) - min(cand)
    ok = spread < DETECT_TOL
    detect_rows.append((K, Q_bct, q_nx, extras, spread, ok))
    extra_str = "".join(f"  Q({k})={v:.4f}" for k, v in extras.items())
    print(f"  K={K:>4.0%}  Q(bct max)={Q_bct:.4f}  Q(nx best)={q_nx:.4f}{extra_str}"
          f"  spread={spread:.4f}  {'PASS' if ok else 'FAIL'}")

assert all(r[-1] for r in detect_rows), "independent Louvain best-Q disagreement too large"
print(f"\n✔ independent Louvain implementations agree on best Q (spread < {DETECT_TOL}).")

# %% [markdown]
# ## Summary

# %%
print("=" * 66)
print("MODULARITY VERIFICATION")
print(f"  reference recording : {RECORDING} [{STATE}], {rows.size} neurons")
print(f"  libraries used      : NetworkX {nx.__version__}, bctpy" +
      (", igraph" if HAVE_IGRAPH else "") + (", python-louvain" if HAVE_PYLOUVAIN else ""))
if not (HAVE_IGRAPH and HAVE_PYLOUVAIN):
    missing = [n for n, h in [("python-igraph", HAVE_IGRAPH),
                              ("python-louvain", HAVE_PYLOUVAIN)] if not h]
    print(f"  (optional, not installed: {', '.join(missing)} — "
          f"`poetry add --group dev {' '.join(missing)}` to enable)")
print(f"  Check 1 (value, det.)  : PASS  — Q identical to < {VALUE_TOL:.0e}")
print(f"  Check 2 (Louvain, stoch): PASS  — best Q agrees to < {DETECT_TOL}")
print("=" * 66)
