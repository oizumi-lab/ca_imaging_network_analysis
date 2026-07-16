# %% [markdown]
# # 20 · Modularity: finding functional modules
#
# A network is **modular** when neurons split into groups that are densely
# connected inside and sparsely connected between. Modularity **Q** measures
# how strong that division is:
#
# $$ Q = \frac{1}{2m}\sum_{ij}\Big(A_{ij} - \gamma\frac{k_i k_j}{2m}\Big)\,\delta(c_i, c_j) $$
#
# - $A$ = adjacency matrix, $k_i$ = degree, $m$ = number of edges.
# - $\gamma$ = **resolution**: $\gamma>1$ → smaller modules, $\gamma<1$ → larger.
# - $c_i$ = the module neuron $i$ is assigned to; $\delta=1$ if same module.
#
# The **Louvain algorithm** searches assignments $c$ to maximise $Q$. This script
# covers the three things you must get right in practice: (1) thresholding at a
# **fixed density**, (2) the **stochasticity** of Louvain and how to tame it, and
# (3) the **resolution** parameter.

# %%
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt

from src.funcnet import dataio, network as net
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Window length follows the paper: 2900 frames (379 s) for wakefulness-anesthesia
# recordings, 1500 frames (196 s) for wakefulness-sleep. mouse07_ane is an
# anesthesia recording, so we use the 2900-frame window here.
WINDOW = 2900
rec = dataio.load_recording("mouse07_ane")
keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
idx = dataio.state_frames(rec, "awake")[:WINDOW]
X = rec.spike_smoothed[keep][:, idx]
coords = rec.centroid[keep]
C = net.correlation_matrix(X)
print(f"{rec.name}: {X.shape[0]} active neurons, one awake window of {len(idx)} frames")

# %% [markdown]
# ## Why a *fixed density*?
# A denser graph trivially has more within-module edges, which inflates Q. To
# compare networks fairly we fix the **connection density** K — the fraction of
# possible edges we keep — so any Q difference reflects *organisation*, not edge
# count. Following the paper, we rank pairs by **absolute** correlation
# (``negative=True``): a neuron pair is connected if its ``|r|`` is in the top K.
# Below: the same correlation matrix thresholded at three densities.

# %%
fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))
for ax, K in zip(axes, [0.02, 0.05, 0.10]):
    adj, thr = net.density_threshold(C, K, negative=True)
    ci, Q = net.louvain_modularity(adj, gamma=1.0, seed=1, ci0=net.giant_component_init(adj))
    order = np.argsort(ci)
    ax.imshow(adj[np.ix_(order, order)], cmap="Greys", interpolation="nearest", aspect="equal")
    ax.set_title(f"K = {K:.0%}  (r≥{thr:.2f})\nQ = {Q:.3f}, {net.n_modules(ci)} modules")
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Module-sorted adjacency at increasing density", y=1.02)
fig.tight_layout()
fig.savefig(FIG_DIR / "20_density_blocks.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Louvain is stochastic — run it many times
# Each Louvain run starts from a random order, so Q and the partition wobble.
# The published pipeline runs Louvain **200×** and keeps the **max-Q** partition
# (`repeat_louvain`), optionally fusing runs into a **consensus** partition.
#
# One subtlety matters for the module *count*: after thresholding some neurons
# have **degree 0**, and Louvain can never move a degree-0 node. With BCT's
# default init (every node its own community) each isolated neuron is frozen as
# its own singleton *module*, wildly inflating the count. Following the paper's
# `modularity_analysis.m`, `repeat_louvain` warm-starts from a
# **giant-component partition** (`net.giant_component_init`) that collapses all
# isolated neurons into one community — so the reported number of modules
# reflects real structure, not stray singletons.

# %%
K = 0.05
adj, thr = net.density_threshold(C, K, negative=True)
res = net.repeat_louvain(adj, gamma=1.0, n_runs=100, seed=12345)
print(f"Q over 100 runs: mean={res['Q_all'].mean():.4f}  sd={res['Q_all'].std():.4f}")
print(f"max-Q = {res['Q_max']:.4f}  with {res['n_modules_max']} modules")

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(res["Q_all"], bins=25, color="steelblue", alpha=0.8)
ax.axvline(res["Q_max"], color="crimson", lw=2, label=f"max-Q = {res['Q_max']:.3f}")
ax.set_xlabel("modularity Q")
ax.set_ylabel("count (of 100 runs)")
ax.set_title("Stochasticity of Louvain — why we take the maximum")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "20_louvain_distribution.png", dpi=140)
plt.show()

# %% [markdown]
# ## The resolution parameter γ
# Sweeping γ traces a resolution profile: larger γ → more, smaller modules.
# Reporting Q across γ (and across density) shows a result is not an artefact of
# one parameter choice.

# %%
gammas = [0.5, 1.0, 1.5, 2.0]
prof = [(g, *(lambda r: (r["Q_max"], r["n_modules_max"]))(
            net.repeat_louvain(adj, gamma=g, n_runs=20, seed=12345)))
        for g in gammas]
for g, Qm, nm in prof:
    print(f"  γ={g:<4}  max-Q={Qm:.3f}  modules={nm}")

# %% [markdown]
# ## The spatial module map
# Colour each neuron by its (max-Q) module on the cortical surface. The paper's
# headline observation: at single-cell resolution the modules are **spatially
# intermixed** — neighbouring neurons often belong to *different* functional
# modules.

# %%
ci = res["ci_max"]
fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.scatter(coords[:, 0], coords[:, 1], c=ci, s=16, cmap="tab20")
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title(f"{rec.name} — awake, K={K:.0%}\n{net.n_modules(ci)} spatially intermixed modules")
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
fig.tight_layout()
fig.savefig(FIG_DIR / "20_spatial_modules.png", dpi=140)
plt.show()

# %% [markdown]
# ## Takeaway
# We can now quantify modular organisation of a functional network with a single
# robust number (max-Q over many runs) at a fixed density. The final script,
# ``30_state_comparison.py``, applies this to **compare states** and reproduces
# the paper's finding that modularity is higher during unconsciousness.
