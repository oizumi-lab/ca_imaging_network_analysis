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
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle

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


def module_display_layout(ci):
    """Build a stable, size-ranked layout for displaying one partition.

    Louvain's integer labels are arbitrary.  For a readable adjacency plot we
    relabel modules from largest to smallest *for display only*, then keep each
    neuron's original order within its module.  This does not change Q or the
    underlying partition.
    """
    labels, sizes = np.unique(ci, return_counts=True)
    rank = np.argsort(-sizes, kind="stable")
    labels = labels[rank]
    sizes = sizes[rank]

    display_ci = np.empty(ci.size, dtype=int)
    for display_id, label in enumerate(labels):
        display_ci[ci == label] = display_id

    order = np.argsort(display_ci, kind="stable")
    boundaries = np.concatenate(([0], np.cumsum(sizes)))
    colors = plt.colormaps["tab20"].resampled(labels.size)(np.arange(labels.size))
    return order, display_ci, boundaries, colors


def outline_modules(ax, boundaries, colors, linewidth=1.6):
    """Outline the within-module blocks of a module-sorted matrix."""
    for start, stop, color in zip(boundaries[:-1], boundaries[1:], colors):
        ax.add_patch(
            Rectangle(
                (start - 0.5, start - 0.5),
                stop - start,
                stop - start,
                fill=False,
                edgecolor=color,
                linewidth=linewidth,
            )
        )

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
    order, _, boundaries, colors = module_display_layout(ci)
    ax.imshow(adj[np.ix_(order, order)], cmap="Greys", interpolation="nearest", aspect="equal")
    outline_modules(ax, boundaries, colors, linewidth=1.3)
    ax.set_title(f"K = {K:.0%}  (|r|≥{thr:.2f})\nQ = {Q:.3f}, {net.n_modules(ci)} modules")
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

# %% [markdown]
# ## From correlations to modules, step by step
# The complete fixed-density pipeline is shown below.  The first two panels
# retain the original neuron order.  Louvain then assigns a module to every
# neuron; in the last panel we reorder neurons by that assignment so dense
# within-module connections become diagonal blocks.  The coloured strip and
# matching outlines explicitly show the assignments.  Because Louvain's raw
# integer labels are arbitrary, display module 1 is simply the largest module,
# module 2 the next largest, and so on; this relabelling does not alter Q.

# %%
ci = res["ci_max"]
order, display_ci, boundaries, module_colors = module_display_layout(ci)
n_display_modules = len(boundaries) - 1
module_cmap = ListedColormap(module_colors)

fig, axes = plt.subplots(
    1,
    3,
    figsize=(16, 5.7),
    gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]},
)
ax_corr, ax_binary, ax_sorted = axes

# Use a robust symmetric colour range so a handful of extreme correlations do
# not hide the correlation structure that drives thresholding.
upper_triangle = C[np.triu_indices_from(C, k=1)]
corr_limit = max(float(np.percentile(np.abs(upper_triangle), 99.5)), 1e-6)
corr_image = ax_corr.imshow(
    C,
    cmap="RdBu_r",
    vmin=-corr_limit,
    vmax=corr_limit,
    interpolation="nearest",
    aspect="equal",
)
colorbar = fig.colorbar(corr_image, ax=ax_corr, fraction=0.046, pad=0.04)
colorbar.set_label("Pearson correlation, r")
ax_corr.set_title("1 | Correlation matrix\nall neuron pairs")
ax_corr.set_xlabel("neuron ID")
ax_corr.set_ylabel("neuron ID")

binary_cmap = ListedColormap(["white", "#202020"])
ax_binary.imshow(
    adj,
    cmap=binary_cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="equal",
)
ax_binary.set_title(f"2 | Fixed-density binarization\nK={K:.0%}, keep top |r| (threshold={thr:.3f})")
ax_binary.set_xlabel("neuron ID (original order)")
ax_binary.set_ylabel("neuron ID")

ax_sorted.imshow(
    adj[np.ix_(order, order)],
    cmap=binary_cmap,
    vmin=0,
    vmax=1,
    interpolation="nearest",
    aspect="equal",
)
outline_modules(ax_sorted, boundaries, module_colors, linewidth=1.8)
ax_sorted.set_title(
    f"3 | Louvain module assignments\nmax Q={res['Q_max']:.3f}, "
    f"{n_display_modules} modules"
)
ax_sorted.set_xlabel("neuron (grouped by module)")
ax_sorted.set_ylabel("")

# A row-aligned categorical strip makes the assignment explicit even for
# modules whose within-module block is visually sparse.
module_strip = ax_sorted.inset_axes((-0.105, 0.0, 0.025, 1.0))
module_strip.imshow(
    display_ci[order, None],
    cmap=module_cmap,
    vmin=-0.5,
    vmax=n_display_modules - 0.5,
    interpolation="nearest",
    aspect="auto",
)
module_centres = (boundaries[:-1] + boundaries[1:] - 1) / 2
module_strip.set_xticks([])
module_strip.set_yticks(module_centres)
module_strip.set_yticklabels(np.arange(1, n_display_modules + 1))
module_strip.tick_params(axis="y", length=0, labelsize=8, pad=2)
module_strip.set_ylabel("display module", fontsize=9, labelpad=6)
for spine in module_strip.spines.values():
    spine.set_visible(False)

for ax in axes:
    ax.tick_params(labelsize=8)

fig.suptitle("Computing modularity from population activity", fontsize=17, y=0.98)
fig.text(
    0.5,
    0.025,
    "Thresholding fixes the edge count; repeated Louvain searches assignments "
    "that maximize within-module connectivity relative to the degree-matched null model.",
    ha="center",
    fontsize=10,
)
fig.subplots_adjust(left=0.055, right=0.985, bottom=0.14, top=0.86, wspace=0.30)
fig.savefig(FIG_DIR / "20_modularity_pipeline.png", dpi=180, bbox_inches="tight")
plt.show()

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
fig, ax = plt.subplots(figsize=(6.5, 6.5))
scatter = ax.scatter(
    coords[:, 0],
    coords[:, 1],
    c=display_ci,
    s=16,
    cmap=module_cmap,
    vmin=-0.5,
    vmax=n_display_modules - 0.5,
)
ax.set_aspect("equal")
ax.invert_yaxis()
ax.set_title(
    f"{rec.name} — awake, K={K:.0%}\n"
    f"{n_display_modules} spatially intermixed modules"
)
ax.set_xlabel("x (px)")
ax.set_ylabel("y (px)")
module_ticks = np.arange(n_display_modules)
module_colorbar = fig.colorbar(scatter, ax=ax, ticks=module_ticks, fraction=0.046, pad=0.04)
module_colorbar.ax.set_yticklabels(module_ticks + 1)
module_colorbar.set_label("display module (size-ranked)")
fig.tight_layout()
fig.savefig(FIG_DIR / "20_spatial_modules.png", dpi=140)
plt.show()

# %% [markdown]
# ## Takeaway
# We can now quantify modular organisation of a functional network with a single
# robust number (max-Q over many runs) at a fixed density. The final script,
# ``30_state_comparison.py``, applies this to **compare states** and reproduces
# the paper's finding that modularity is higher during unconsciousness.
