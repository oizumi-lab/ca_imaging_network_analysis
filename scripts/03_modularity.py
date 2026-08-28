# %% [markdown]
# # 03 · Modularity: finding functional modules
#
# ## Where this script fits
# Script 02 converted activity into two equal-density graphs. A graph tells us
# which neuron pairs are linked, but it does not yet summarize the graph's
# organization. We now ask whether those links form groups.
#
# Imagine a social network. If most connections lie within a few groups and
# relatively few connect the groups, the network is **modular**. Here,
# the nodes are neurons and the edges are the strongest functional relationships.
# A module is therefore a group of neurons whose retained relationships occur
# more often within the group than expected from their edge counts.
#
# The global workflow is now:
#
# ```text
# activity → correlation → equal-density graph → modules and modularity Q
# ```
#
# We need to examine three analysis choices because each can change the answer:
# graph density, the random starting point of the Louvain search, and the module-
# size resolution. This script introduces them one at a time.
#
# Modularity **Q** measures how strongly the network separates into groups:
#
# $$ Q = \frac{1}{2m}\sum_{ij}\Big(A_{ij} - \gamma\frac{k_i k_j}{2m}\Big)\,\delta(c_i, c_j) $$
#
# - $A$ = adjacency matrix, $k_i$ = degree, $m$ = number of edges.
# - $\gamma$ = **resolution**: $\gamma>1$ → smaller modules, $\gamma<1$ → larger.
# - $c_i$ = the module neuron $i$ is assigned to; $\delta=1$ if same module.
#
# To follow the tutorial, read this equation as a comparison between the
# observed within-module edges and the number expected from a degree-matched
# reference model. A larger Q means that the proposed partition separates the
# graph more strongly under those settings.
#
# The **Louvain algorithm** searches assignments $c$ to maximise $Q$. This script
# introduces three practical choices: (1) thresholding at a
# **fixed density**, (2) the **stochasticity** of Louvain and how to tame it, and
# (3) the **resolution** parameter.
#
# ## Code map
#
# This script is longer because it both performs the analysis and explains it
# with several figures. Run it from top to bottom. Important recurring names:
#
# - ``X``: activity with shape ``(neurons, frames)``;
# - ``C``: correlation matrix;
# - ``adj``: binary adjacency matrix;
# - ``ci``: one integer module label per neuron (community index);
# - ``Q``: the modularity score for that partition;
# - ``K``: retained graph density; and
# - ``gamma``: Louvain's module-size resolution parameter.
#
# A dictionary such as ``state_results[label]`` groups all arrays and results for
# one state. Functions introduced below package repeated display operations.
# Their docstrings describe inputs and returned values so the same functions can
# be reused without copying their internal code.

# %%
import argparse
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

# %% [markdown]
# ## Choose the sleep or anesthesia dataset
# Set ``DEFAULT_DATASET`` when running this file cell-by-cell, or pass
# ``--dataset sleep`` / ``--dataset anesthesia`` on the command line. Each
# selection compares the two states from one recording:
#
# - sleep: Awake versus NREM, using 1500-frame windows (about 196 s)
# - anesthesia: Awake versus Anesthesia, using 2900-frame windows (about 379 s)
#
# These are the state-specific window lengths in Kiyooka et al.'s modularity
# code. ``MAX_NEURONS`` keeps this explanatory figure responsive; set it to
# ``None`` to use every activity-filtered neuron, as in the full analysis. The
# identical neuron rows are used for both states.
#
# Runtime settings:
#
# - ``MAX_NEURONS`` caps the graph size; ``None`` uses all active neurons.
# - ``N_RUNS`` controls repeated Louvain searches for the main result. More runs
#   improve the chance of finding a high-Q solution but increase runtime roughly
#   in direct proportion.
# - ``PROFILE_RUNS`` is the smaller repeat count used only for the gamma sweep.
# - ``DEFAULT_RECORDING`` and ``WINDOW_FRAMES`` are dictionaries keyed by the
#   selected dataset, so the sleep and anesthesia choices stay matched.

# %%
DEFAULT_DATASET = "sleep"  # used when no --dataset command-line option is given
DEFAULT_RECORDING = {
    "sleep": "mouse02_sleep",
    "anesthesia": "mouse07_ane",
}
WINDOW_FRAMES = {"sleep": 1500, "anesthesia": 2900}
MAX_NEURONS = 2000  # interactive cap; None means all activity-filtered neurons
N_RUNS = 30         # preview default; use 200 for paper-scale optimization
PROFILE_RUNS = 10   # repetitions for each value in the short gamma sweep

# ``argparse`` lets terminal users override settings without editing this file,
# for example: ``poetry run python scripts/03_modularity.py --dataset anesthesia``.
# In a notebook, leave the command-line options alone and edit DEFAULT_DATASET.

parser = argparse.ArgumentParser(
    description="Explain modularity for one brain-state dataset"
)
parser.add_argument(
    "--dataset",
    choices=("sleep", "anesthesia", "ane"),
    default=DEFAULT_DATASET,
    help="state pair to plot: sleep (Awake/NREM) or anesthesia (Awake/Anesthesia)",
)
parser.add_argument(
    "--recording",
    default=None,
    help="optional recording name; it must belong to the selected dataset",
)
# ``parse_known_args`` ignores notebook/editor arguments that are unrelated to
# this tutorial. The returned ``options`` object exposes ``options.dataset`` and
# ``options.recording``.
options, _unknown = parser.parse_known_args()

DATASET = "anesthesia" if options.dataset == "ane" else options.dataset
recording_name = options.recording or DEFAULT_RECORDING[DATASET]
WINDOW = WINDOW_FRAMES[DATASET]
rec = dataio.load_recording(recording_name)
expected_data_info = "sleep" if DATASET == "sleep" else "ane"
if rec.data_info != expected_data_info:
    raise ValueError(
        f"{recording_name!r} is a {rec.data_info!r} recording, but "
        f"--dataset {DATASET!r} was selected"
    )

rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
coords = rec.centroid[rows]
state_results = {}
for label in rec.state_labels:
    available_frames = dataio.state_frames(rec, label)
    if available_frames.size < WINDOW:
        raise ValueError(
            f"{rec.name} has only {available_frames.size} usable {label} frames; "
            f"the {DATASET} analysis requires {WINDOW}"
        )
    idx = available_frames[:WINDOW]
    X = rec.spike_smoothed[np.ix_(rows, idx)]
    state_results[label] = {
        "frames": idx,
        "activity": X,
        "correlation": net.correlation_matrix(X),
    }
    print(
        f"{rec.name}: {label:<10} | {X.shape[0]} shared active neurons | "
        f"{len(idx)} frames ({len(idx) / rec.fs:.1f} s)"
    )

STATE_TITLES = {"awake": "Awake", "nrem": "NREM", "anesthesia": "Anesthesia"}
output_suffix = DATASET


def module_display_layout(ci):
    """Build a readable plotting order for one module partition.

    Parameters
    ----------
    ci : one-dimensional NumPy array
        One integer community/module label per neuron.

    Returns
    -------
    order : NumPy array
        Neuron indices sorted so members of a module appear together.
    display_ci : NumPy array
        Size-ranked labels starting at zero, used only for colors and display.
    boundaries : NumPy array
        Start/stop positions of module blocks in the sorted matrix.
    colors : NumPy array
        One RGBA plotting color per displayed module.

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
    """Draw colored rectangles around a module-sorted adjacency matrix.

    ``ax`` is the Matplotlib panel to modify. ``boundaries`` and ``colors`` are
    outputs from :func:`module_display_layout`. The function draws on ``ax`` and
    therefore does not need to return a new object.
    """
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
# A denser graph changes the number of possible within-module edges and can
# change Q even if the underlying organization is otherwise similar. To compare
# networks fairly we fix the **connection density** K — the fraction of possible
# edges we keep — so an edge-count difference does not decide the comparison.
# Following the paper, we rank pairs by **absolute** correlation
# (``negative=True``): a neuron pair is connected if its ``|r|`` is in the top K.
# Below: each state's correlation matrix thresholded at three densities.

# %%
fig, axes = plt.subplots(2, 3, figsize=(14, 9.0), squeeze=False)
for row, label in enumerate(rec.state_labels):
    C = state_results[label]["correlation"]
    for ax, K_density in zip(axes[row], [0.02, 0.05, 0.10]):
        adj_density, thr_density = net.density_threshold(C, K_density, negative=True)
        ci_density, Q_density = net.louvain_modularity(
            adj_density,
            gamma=1.0,
            seed=1,
            ci0=net.giant_component_init(adj_density),
        )
        order_density, _, boundaries_density, colors_density = module_display_layout(
            ci_density
        )
        ax.imshow(
            adj_density[np.ix_(order_density, order_density)],
            cmap="Greys",
            interpolation="nearest",
            aspect="equal",
        )
        outline_modules(ax, boundaries_density, colors_density, linewidth=1.3)
        ax.set_title(
            f"K = {K_density:.0%}  (|r|≥{thr_density:.2f})\n"
            f"Q = {Q_density:.3f}, {net.n_modules(ci_density)} modules"
        )
        ax.set_xticks([])
        ax.set_yticks([])
    axes[row, 0].set_ylabel(STATE_TITLES[label], fontsize=12, fontweight="bold")
fig.suptitle(
    f"{rec.name}: module-sorted adjacency at increasing density\n"
    f"one {WINDOW}-frame window per state",
    y=0.995,
)
fig.tight_layout(rect=(0, 0, 1, 0.96))
fig.savefig(
    FIG_DIR / f"03_density_blocks_{output_suffix}.png",
    dpi=140,
    bbox_inches="tight",
)
plt.show()

# %% [markdown]
# ## Louvain is stochastic — run it many times
# Each Louvain run explores the graph in a different random order, so it can
# return a slightly different Q and partition. This is a property of the search,
# not evidence that the recording itself changed.
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
#
# ``net.repeat_louvain`` returns a dictionary. The most-used entries are
# ``Q_all`` (one score per run), ``Q_max`` (the best score), ``ci_max`` (module
# labels from that best run), and ``n_modules_max`` (its number of modules).

# %%
K = 0.05  # keep 5% of all possible undirected edges
for label in rec.state_labels:
    analysis = state_results[label]
    adj, thr = net.density_threshold(analysis["correlation"], K, negative=True)
    res = net.repeat_louvain(adj, gamma=1.0, n_runs=N_RUNS, seed=12345)
    order, display_ci, boundaries, module_colors = module_display_layout(res["ci_max"])
    analysis.update(
        {
            "adjacency": adj,
            "threshold": thr,
            "louvain": res,
            "order": order,
            "display_ci": display_ci,
            "boundaries": boundaries,
            "module_colors": module_colors,
        }
    )
    print(
        f"{STATE_TITLES[label]:<10} Q over {N_RUNS} runs: "
        f"mean={res['Q_all'].mean():.4f}  sd={res['Q_all'].std():.4f}  |  "
        f"max-Q={res['Q_max']:.4f}, {res['n_modules_max']} modules"
    )

# %% [markdown]
# ## From correlations to modules, state by state
# The complete fixed-density pipeline is shown as one separate row per state:
# Awake plus NREM for the sleep dataset, or Awake plus Anesthesia for the
# anesthesia dataset. The first two columns retain the original neuron order.
# Louvain then assigns a module to every neuron; in the last column we reorder
# neurons by that assignment so dense within-module connections become diagonal
# blocks. The coloured strip and matching outlines explicitly show the
# assignments. Because Louvain's raw integer labels are arbitrary, display
# module 1 is simply the largest module, module 2 the next largest, and so on;
# this relabelling does not alter Q.

# %%
fig, axes = plt.subplots(
    2,
    3,
    figsize=(16, 10.6),
    squeeze=False,
    gridspec_kw={"width_ratios": [1.12, 1.0, 1.0]},
)
binary_cmap = ListedColormap(["white", "#202020"])
for row, label in enumerate(rec.state_labels):
    analysis = state_results[label]
    C = analysis["correlation"]
    adj = analysis["adjacency"]
    thr = analysis["threshold"]
    res = analysis["louvain"]
    order = analysis["order"]
    display_ci = analysis["display_ci"]
    boundaries = analysis["boundaries"]
    module_colors = analysis["module_colors"]
    n_display_modules = len(boundaries) - 1
    module_cmap = ListedColormap(module_colors)
    ax_corr, ax_binary, ax_sorted = axes[row]

    # Use a robust symmetric colour range so a handful of extreme correlations
    # do not hide the correlation structure that drives thresholding.
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
    ax_corr.set_ylabel(f"{STATE_TITLES[label]}\nneuron ID", fontweight="bold")

    ax_binary.imshow(
        adj,
        cmap=binary_cmap,
        vmin=0,
        vmax=1,
        interpolation="nearest",
        aspect="equal",
    )
    ax_binary.set_title(
        f"2 | Fixed-density binarization\n"
        f"K={K:.0%}, keep top |r| (threshold={thr:.3f})"
    )
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

    for ax in axes[row]:
        ax.tick_params(labelsize=8)

fig.suptitle(
    f"Computing modularity from population activity | {rec.name}\n"
    f"{WINDOW} frames per state ({WINDOW / rec.fs:.1f} s)",
    fontsize=17,
    y=0.985,
)
fig.text(
    0.5,
    0.02,
    "Thresholding fixes the edge count; repeated Louvain searches assignments "
    "that maximize within-module connectivity relative to the degree-matched null model.",
    ha="center",
    fontsize=10,
)
fig.subplots_adjust(
    left=0.065,
    right=0.985,
    bottom=0.085,
    top=0.88,
    hspace=0.38,
    wspace=0.30,
)
fig.savefig(
    FIG_DIR / f"03_modularity_pipeline_{output_suffix}.png",
    dpi=180,
    bbox_inches="tight",
)
plt.show()

state_colors = {"awake": "steelblue", "nrem": "crimson", "anesthesia": "goldenrod"}
fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), squeeze=False)
for ax, label in zip(axes[0], rec.state_labels):
    res = state_results[label]["louvain"]
    ax.hist(res["Q_all"], bins=25, color=state_colors[label], alpha=0.8)
    ax.axvline(
        res["Q_max"],
        color="black",
        lw=2,
        label=f"max-Q = {res['Q_max']:.3f}",
    )
    ax.set_xlabel("modularity Q")
    ax.set_ylabel(f"count (of {N_RUNS} runs)")
    ax.set_title(STATE_TITLES[label])
    ax.legend()
fig.suptitle(f"{rec.name}: Louvain stochasticity — why we take the maximum")
fig.tight_layout()
fig.savefig(FIG_DIR / f"03_louvain_distribution_{output_suffix}.png", dpi=140)
plt.show()

# %% [markdown]
# ## The resolution parameter γ
# The resolution parameter changes the size of groups favored by the objective:
# larger γ usually produces more, smaller modules. Sweeping γ shows whether the
# qualitative partition depends on a single module-size setting. Compare module
# counts across γ; Q values computed with different γ values do not use the same
# penalty and should not be ranked as interchangeable scores.

# %%
gammas = [0.5, 1.0, 1.5, 2.0]
for label in rec.state_labels:
    print(f"{STATE_TITLES[label]} resolution profile:")
    adj = state_results[label]["adjacency"]
    for gamma in gammas:
        gamma_result = net.repeat_louvain(
            adj,
            gamma=gamma,
            n_runs=PROFILE_RUNS,
            seed=12345,
        )
        print(
            f"  γ={gamma:<4}  max-Q={gamma_result['Q_max']:.3f}  "
            f"modules={gamma_result['n_modules_max']}"
        )

# %% [markdown]
# ## The spatial module map
# Colour each neuron by its (max-Q) module on the cortical surface. The paper's
# headline observation: at single-cell resolution the modules are **spatially
# intermixed** — neighbouring neurons often belong to *different* functional
# modules.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.8), squeeze=False)
for ax, label in zip(axes[0], rec.state_labels):
    analysis = state_results[label]
    display_ci = analysis["display_ci"]
    boundaries = analysis["boundaries"]
    module_colors = analysis["module_colors"]
    n_display_modules = len(boundaries) - 1
    module_cmap = ListedColormap(module_colors)
    scatter = ax.scatter(
        coords[:, 0],
        coords[:, 1],
        c=display_ci,
        s=12,
        cmap=module_cmap,
        vmin=-0.5,
        vmax=n_display_modules - 0.5,
    )
    ax.set_aspect("equal")
    ax.invert_yaxis()
    ax.set_title(
        f"{STATE_TITLES[label]}, K={K:.0%}\n"
        f"{n_display_modules} spatially intermixed modules"
    )
    ax.set_xlabel("x (px)")
    ax.set_ylabel("y (px)")
    module_ticks = np.arange(n_display_modules)
    module_colorbar = fig.colorbar(
        scatter,
        ax=ax,
        ticks=module_ticks,
        fraction=0.046,
        pad=0.04,
    )
    module_colorbar.ax.set_yticklabels(module_ticks + 1)
    module_colorbar.set_label("display module (size-ranked)")
fig.suptitle(f"{rec.name}: spatial module assignments from matched state windows")
fig.tight_layout()
fig.savefig(FIG_DIR / f"03_spatial_modules_{output_suffix}.png", dpi=140)
plt.show()

# %% [markdown]
# ## Takeaway
# We can now quantify modular organisation of a functional network with a single
# robust number (max-Q over many runs) at a fixed density. Here the complete
# procedure is displayed independently for one Awake window and one
# NREM/Anesthesia window. Script ``04_sample_state_comparison.py`` repeats this
# across all complete windows of the example recording. Script
# ``07_all_mice_modularity.py`` then tests robustness across animals.
