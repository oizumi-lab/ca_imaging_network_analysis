# %% [markdown]
# # 02 · From neural activity to a functional network
#
# ## Where this script fits
# In script 01 you inspected the recorded signals. The remaining tutorial asks
# whether the organization of the neural population changes between Awake and
# NREM sleep. The complete analysis has four steps:
#
# ```text
# neural activity over time
#          ↓
# correlation between every neuron pair       ← this script
#          ↓
# keep the same fraction of strongest pairs    ← this script
#          ↓
# identify modules and calculate modularity Q  ← scripts 03–04
# ```
#
# This script stops when one graph has been constructed for each state. Before
# calculating modularity, we need to understand exactly what those graphs mean
# and make sure that they can be compared fairly.
#
# By the end of this script, you should be able to explain:
#
# 1. why correlation is called *functional* rather than anatomical connectivity;
# 2. why a dense correlation matrix must be reduced to a graph;
# 3. why Awake and NREM graphs must contain the same number of edges; and
# 4. why equal graph density can require different correlation thresholds.
#
# ## Beginner's code map
#
# Run the cells from top to bottom. The main variables deliberately keep the
# same meaning in scripts 02--09:
#
# - ``rec``: one loaded recording.
# - ``rows``: neuron-row indices shared by both brain states.
# - ``frames``: time indices belonging to one state.
# - ``activity`` or ``X``: a ``(neurons, frames)`` activity array.
# - ``correlation`` or ``C``: a square pairwise-correlation matrix.
# - ``adjacency`` or ``adj``: a square binary graph matrix (1=edge, 0=no edge).
# - ``K`` or ``density``: the fraction of possible edges retained.
#
# ``state_results`` is a dictionary. Think of it as labeled storage:
# ``state_results["awake"]["activity"]`` retrieves the Awake activity matrix.
# Each loop adds another named result so later cells do not need to recompute it.

# %%
import os
import sys
import warnings

# Add the repository root so this file works from VS Code, Spyder, or a terminal.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.funcnet import dataio, network as net
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Step 1 — choose matched data from the two conditions
#
# A correlation can change simply because we used more neurons or more time
# points. To avoid those obvious differences, both states use:
#
# - exactly the same neuron rows;
# - exactly 1,500 frames; and
# - frames belonging to stable Awake or NREM periods.
#
# ``MAX_NEURONS`` keeps the course exercise responsive. The paper-scale scripts
# can use every activity-filtered neuron.
#
# Settings to try:
#
# - change ``RECORDING`` to another downloaded recording;
# - reduce ``MAX_NEURONS`` for a faster exercise, or use ``None`` for all active
#   neurons (which requires much more memory and time);
# - change ``REFERENCE_DENSITY`` between 0 and 1. For example, ``0.05`` means
#   retain 5% of all possible undirected neuron pairs.

# %%
RECORDING = "mouse02_sleep"  # downloaded dataset name
MAX_NEURONS = 2500           # course-size neuron cap; None means no cap
REFERENCE_DENSITY = 0.05     # 0.05 = retain the strongest 5% of pairs

rec = dataio.load_recording(RECORDING)
window_frames = {"sleep": 1500, "ane": 2900}[rec.data_info]
rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)

# Start with an empty dictionary, then create one nested dictionary per state.
state_results = {}
for state in rec.state_labels:
    available_frames = dataio.state_frames(rec, state)
    if available_frames.size < window_frames:
        raise ValueError(
            f"{state} has only {available_frames.size} stable frames; "
            f"{window_frames} are required"
        )
    frames = available_frames[:window_frames]
    # ``np.ix_`` selects every combination of the chosen neuron rows and frames,
    # giving an array with shape (len(rows), len(frames)).
    activity = rec.spike_smoothed[np.ix_(rows, frames)]
    state_results[state] = {
        "frames": frames,
        "activity": activity,
    }
    print(
        f"{state:<8}: {activity.shape[0]:,} shared neurons × "
        f"{activity.shape[1]:,} frames"
    )

# %% [markdown]
# ## Step 2 — measure statistical co-activity
#
# For every neuron pair, Pearson correlation asks whether the two activity traces
# rise and fall together. The result ranges from −1 to +1:
#
# - positive correlation: the two traces tend to increase together;
# - correlation near zero: no consistent linear relationship;
# - negative correlation: one tends to increase when the other decreases.
#
# Correlation does **not** show a synapse or a direct causal connection. Two
# neurons may be correlated because they receive common input. We therefore call
# this a *functional-connectivity* matrix: it summarizes relationships in the
# recorded activity.
#
# ``net.correlation_matrix`` is a project helper function. It expects a
# ``(nodes, time)`` array and returns a square ``(nodes, nodes)`` array. The
# following loop calculates it once per state and stores it under a new key.

# %%
for state, result in state_results.items():
    result["correlation"] = net.correlation_matrix(result["activity"])

state_colors = {"awake": "royalblue", "nrem": "crimson", "anesthesia": "goldenrod"}
state_titles = {"awake": "Awake", "nrem": "NREM", "anesthesia": "Anesthesia"}

all_upper_values = np.concatenate(
    [
        result["correlation"][np.triu_indices_from(result["correlation"], k=1)]
        for result in state_results.values()
    ]
)
color_limit = max(float(np.percentile(np.abs(all_upper_values), 99.5)), 1e-6)

fig, axes = plt.subplots(1, len(rec.state_labels), figsize=(12, 5.3), squeeze=False)
for ax, state in zip(axes[0], rec.state_labels):
    image = ax.imshow(
        state_results[state]["correlation"],
        cmap="RdBu_r",
        vmin=-color_limit,
        vmax=color_limit,
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(f"{state_titles[state]} correlation matrix")
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
fig.colorbar(image, ax=axes, shrink=0.72, label="Pearson correlation r")
fig.suptitle(f"{rec.name}: one value for every neuron pair")
correlation_path = FIG_DIR / "02_connectivity_matrices.png"
fig.savefig(correlation_path, dpi=150, bbox_inches="tight")
plt.show()
print("saved ->", correlation_path)

# %% [markdown]
# ## Step 3 — turn each dense matrix into a graph
#
# A graph contains **nodes** and **edges**. Here, each neuron is a node and an
# edge marks a neuron pair that we choose to retain. With 2,500 neurons there are
# more than three million possible pairs, so connecting every pair would not
# reveal useful network structure.
#
# We rank pairs by absolute correlation ``|r|`` and retain only the strongest
# fraction. The retained fraction is the **connection density K**. At ``K=5%``,
# the graph contains the strongest 5% of all possible neuron pairs.
#
# Why use absolute correlation? The paper's pipeline treats both strong positive
# and strong negative relationships as functional edges. In
# ``density_threshold``, this choice is requested with ``negative=True``.
#
# Thresholding creates a binary adjacency matrix: ``1`` means that the pair was
# retained as an edge and ``0`` means that it was not. The matrix is symmetric
# because this tutorial treats the relationship as undirected, and its diagonal
# is zero because a neuron is not connected to itself.
#
# ``net.density_threshold(C, K, negative=True)`` returns two objects:
#
# 1. the binary adjacency matrix; and
# 2. the absolute-correlation cutoff required to obtain density ``K``.
#
# The name ``negative=True`` means rank edges by ``|r|`` so strong negative
# correlations can also be retained. It does not create negative graph edges.

# %%
n_neurons = rows.size
# ``//`` is integer division. An undirected graph has n(n-1)/2 unique pairs
# because the lower and upper matrix triangles describe the same edges.
n_possible_edges = n_neurons * (n_neurons - 1) // 2
print(f"Possible undirected neuron pairs: {n_possible_edges:,}")

target_edges = int(np.floor(REFERENCE_DENSITY * n_possible_edges))
print(f"Reference density: K={REFERENCE_DENSITY:.0%}")
print(f"Target edges per state: {target_edges:,}")
for state, result in state_results.items():
    adjacency, threshold = net.density_threshold(
        result["correlation"],
        REFERENCE_DENSITY,
        negative=True,
    )
    retained_edges = int(adjacency[np.triu_indices_from(adjacency, k=1)].sum())
    result["adjacency"] = adjacency
    result["threshold"] = threshold
    result["retained_edges"] = retained_edges
    print(
        f"  {state_titles[state]:<6}: |r| cutoff={threshold:.4f}; "
        f"retained edges={retained_edges:,}"
    )

# %% [markdown]
# ## Step 4 — check that the comparison is fair
#
# The graph density—not the numerical correlation cutoff—is held constant.
# Awake and NREM can therefore have different ``|r|`` thresholds while retaining
# the same number of edges. This is intentional:
#
# - **same density** controls the number of opportunities to form modules;
# - **different thresholds** report how strong correlations had to be to enter
#   each equally sized graph.
#
# The left panel below shows how a horizontal 5% line intersects each state's
# correlation curve at a different threshold. The two right panels show the
# resulting binary graphs. Every dark pixel is a retained edge.

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw={"width_ratios": [1.15, 1, 1]})

absolute_grid = np.linspace(0, min(0.55, color_limit * 2.2), 240)
for state in rec.state_labels:
    correlation = state_results[state]["correlation"]
    upper = np.abs(correlation[np.triu_indices_from(correlation, k=1)])
    sorted_upper = np.sort(upper)
    retained_fraction = 1 - np.searchsorted(sorted_upper, absolute_grid) / upper.size
    axes[0].plot(
        absolute_grid,
        100 * retained_fraction,
        color=state_colors[state],
        lw=2,
        label=state_titles[state],
    )
    threshold = state_results[state]["threshold"]
    axes[0].scatter(
        threshold,
        100 * REFERENCE_DENSITY,
        color=state_colors[state],
        s=50,
        zorder=4,
    )
    label_offset = (-18, 12) if state == "awake" else (18, 12)
    axes[0].annotate(
        f"{threshold:.3f}",
        (threshold, 100 * REFERENCE_DENSITY),
        xytext=label_offset,
        textcoords="offset points",
        color=state_colors[state],
        ha="center",
        fontsize=9,
    )
axes[0].axhline(100 * REFERENCE_DENSITY, color="0.4", lw=1, ls="--")
axes[0].set(
    ylim=(0, 20),
    xlabel="absolute correlation threshold |r|",
    ylabel="neuron pairs retained (%)",
)
axes[0].set_title(f"Choose the |r| cutoff at K={REFERENCE_DENSITY:.0%}")
axes[0].legend(frameon=False)
axes[0].grid(color="0.92")

for ax, state in zip(axes[1:], rec.state_labels):
    result = state_results[state]
    ax.imshow(
        result["adjacency"],
        cmap="Greys",
        interpolation="nearest",
        aspect="equal",
    )
    ax.set_title(
        f"{state_titles[state]} graph\n"
        f"|r| ≥ {result['threshold']:.3f}; "
        f"{result['retained_edges']:,} edges"
    )
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")

fig.suptitle(f"{rec.name}: equal density gives equal edge counts")
fig.tight_layout()
graph_path = FIG_DIR / "02_fixed_density_graphs.png"
fig.savefig(graph_path, dpi=150, bbox_inches="tight")
plt.show()
print("saved ->", graph_path)

# %% [markdown]
# ## Takeaway
#
# We began with matched neural-activity windows, measured pairwise correlations,
# and retained the strongest 5% of pairs in each state. The Awake and NREM
# thresholds are different, but the graphs contain the same number of edges.
# Script 03 can therefore compare their modular organization without an edge-
# count difference deciding the result in advance.

# %% [markdown]
# ## Exercise 2 — change the graph density
#
# Choose ``K=2%`` or ``K=10%`` and construct a new graph for both states. Report
# the correlation threshold and retained edge count for each state.
#
# Before running your code, predict whether the threshold will be higher or lower
# than at 5%. Then verify that Awake and NREM retain the same number of edges even
# when their thresholds differ.
#
# **Where to start:** Step 3 already calls
# ``net.density_threshold``. Reuse that pattern for one density, and use the
# upper triangle of the adjacency matrix when counting undirected edges. Put the
# result in a new cell and leave ``REFERENCE_DENSITY`` unchanged so that you can
# compare your result with the supplied 5% analysis.
