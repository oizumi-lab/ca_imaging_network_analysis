# %% [markdown]
# # 05 · Coarse-grain the example recording across spatial scales
#
# ## Where this script fits
# Scripts 02--04 treated every neuron as one network node. Many other recording
# methods cannot resolve individual neurons: a measurement may average signals
# from a local population. We therefore ask whether the modularity result depends
# on the spatial scale of observation.
#
# Nearby neurons are grouped into parcels containing 2, 5, 10, 20, or 40 cells.
# For every scale we must restart from the activity signals:
#
# ```text
# average nearby activity → correlate parcels → keep 5% of edges → calculate Q
# ```
#
# We do **not** simply merge nodes in the original graph, because averaging the
# signals changes their correlations. This step tells us whether the state
# difference is a single-cell phenomenon or remains visible after local signals
# are combined.
#
# ## Beginner's code map
#
# The outer loop changes spatial scale. Inside it, the code rebuilds parcel
# signals once, then analyzes every state and time window at that same scale.
# Important array shapes are:
#
# - ``coords``: ``(neurons, 2)`` x/y cortical positions;
# - ``activity``: ``(neurons, frames)`` smoothed activity;
# - ``parcel_index``: one parcel label per neuron;
# - ``parcel_activity``: ``(parcels, frames)`` averaged activity; and
# - ``parcel_coords``: ``(parcels, 2)`` mean x/y positions.
#
# The ``records`` list again acts as a table. Each dictionary in it stores one
# state × window × scale result and is later written to CSV.

# %%
import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.funcnet import coarsegrain as cg, dataio, network as net, timeseries as ts
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings you may modify
#
# - ``SCALES`` gives neurons per spatial parcel. ``1`` is the ungrouped network.
# - ``MAX_WINDOWS`` and ``MAX_NEURONS`` are preview-size runtime limits.
# - ``K``, ``GAMMA``, and ``N_RUNS`` control graph thresholding and Louvain in
#   exactly the same way as scripts 03--04.
# - Keep ``WINDOW_FRAMES`` matched to the dataset unless the analysis explicitly
#   studies a different temporal scale.

# %%
RECORDING = "mouse02_sleep"      # dataset used for this worked example
SCALES = (1, 2, 5, 10, 20, 40)  # target neurons per parcel; 1 = single cell
WINDOW_FRAMES = 1500             # frames in one stable-state window
MAX_WINDOWS = 2                  # maximum windows per state; None means all
MAX_NEURONS = 1000              # maximum active neurons; None means all
K = 0.05                         # keep the strongest 5% of parcel pairs
GAMMA = 1.0                      # Louvain module-size resolution
N_RUNS = 10                      # Louvain repeats per scale/state/window graph

# %% [markdown]
# ## Step 1 — build parcels and recompute modularity
#
# ``scale=1`` keeps neurons as individual nodes. At larger scales,
# ``close_clustering`` assigns nearby neurons to one parcel and ``coarse_grain``
# averages their activity. The parcel definition depends only on coordinates,
# so Awake and NREM are compared using the same spatial grouping at a given
# scale.
#
# Notice that the correlation matrix is calculated *after* averaging. This is
# essential: averaging can strengthen or weaken relationships, so thresholding
# or merging the single-cell graph would answer a different question.
#
# Function guide:
#
# - ``pdist`` measures every pairwise cortical distance and ``squareform`` turns
#   those distances into a reusable neuron-by-neuron matrix.
# - ``cg.close_clustering`` returns one spatial-parcel label per neuron.
# - ``cg.coarse_grain`` averages signals and coordinates within each parcel.
# - ``net.modularity_from_activity`` performs correlation, density thresholding,
#   repeated Louvain, and returns all results in one dictionary.

# %%
rec = dataio.load_recording(RECORDING)
rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
coords = rec.centroid_um[rows]
activity = rec.spike_smoothed[rows]
distances = squareform(pdist(coords))  # computed once and reused at every scale
records = []

for scale in SCALES:
    if scale == 1:
        # At scale 1, each neuron is already one parcel; no averaging is needed.
        parcel_activity = activity
        parcel_coords = coords
    else:
        parcel_index = cg.close_clustering(
            coords[:, 0],
            coords[:, 1],
            scale,
            D=distances,
        )
        parcel_activity, parcel_x, parcel_y = cg.coarse_grain(
            activity,
            coords[:, 0],
            coords[:, 1],
            parcel_index,
        )
        parcel_coords = np.column_stack([parcel_x, parcel_y])

    print(f"nnei={scale:>2}: {parcel_activity.shape[0]:,} network nodes", flush=True)
    for state in rec.state_labels:
        windows = ts.frame_windows(
            dataio.state_frames(rec, state),
            WINDOW_FRAMES,
            max_windows=MAX_WINDOWS,
        )
        for window_index, frames in enumerate(windows, start=1):
            result = net.modularity_from_activity(
                parcel_activity[:, frames],
                density=K,
                gamma=GAMMA,
                n_runs=N_RUNS,
                negative=True,
            )
            records.append(
                {
                    "recording": rec.name,
                    "state": state,
                    "window": window_index,
                    "nnei": scale,
                    "n_nodes": parcel_coords.shape[0],
                    "q_max": result["Q_max"],
                    "n_modules": result["n_modules_max"],
                }
            )

csv_path = RESULTS_DIR / "05_sample_coarse_grain_modularity.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print("saved ->", csv_path)

# %% [markdown]
# ## Step 2 — plot modularity and the state contrast versus scale
#
# The left panel retains the state-specific Q values. The right panel subtracts
# NREM from Awake, so zero means no difference at that scale, negative values
# mean NREM is higher, and positive values mean Awake is higher.
#
# ``window_curves`` becomes a list of one-dimensional arrays. ``np.vstack``
# stacks them into a ``(windows, scales)`` matrix, allowing ``mean(axis=0)`` to
# average windows separately at every scale.

# %%
state_colors = {"awake": "royalblue", "nrem": "crimson"}
state_titles = {"awake": "Awake", "nrem": "NREM"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
means_by_state = {}

for state in rec.state_labels:
    window_curves = []
    window_indices = sorted(
        {row["window"] for row in records if row["state"] == state}
    )
    for window_index in window_indices:
        curve = np.asarray(
            [
                next(
                    row["q_max"]
                    for row in records
                    if row["state"] == state
                    and row["window"] == window_index
                    and row["nnei"] == scale
                )
                for scale in SCALES
            ],
            dtype=float,
        )
        window_curves.append(curve)
        axes[0].plot(SCALES, curve, color=state_colors[state], alpha=0.28, lw=1)
    matrix = np.vstack(window_curves)
    means_by_state[state] = matrix.mean(axis=0)
    axes[0].plot(
        SCALES,
        matrix.mean(axis=0),
        "-o",
        color=state_colors[state],
        lw=2.2,
        label=state_titles[state],
    )

contrast = means_by_state["awake"] - means_by_state["nrem"]
axes[1].plot(SCALES, contrast, "-o", color="black", lw=2)
axes[1].axhline(0, color="0.55", lw=1)

for ax in axes:
    ax.set_xscale("log")
    ax.set_xticks(SCALES)
    ax.set_xticklabels(SCALES)
    ax.set_xlabel("neurons per spatial parcel (nnei)")
    ax.grid(color="0.9")
axes[0].set_ylabel("max-Q modularity")
axes[0].set_title("Rebuild the network at every spatial scale")
axes[0].legend(frameon=False)
axes[1].set_ylabel("ΔQ (Awake − NREM)")
axes[1].set_title(f"State contrast within {rec.name}")

fig.suptitle("05 · Coarse-grained modularity in the example recording", fontsize=14)
fig.tight_layout()
figure_path = FIG_DIR / "05_sample_coarse_grain_modularity.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", figure_path)

# %% [markdown]
# ## Takeaway
#
# In this recording, NREM modularity is higher from the single-cell scale through
# ``nnei=10``, and the state contrast approaches zero as neurons are grouped.
# The contrast then reverses at the coarsest settings, where only tens of network
# nodes remain. That instability is a useful warning against treating one mouse
# or a very small parcel graph as the population result. Script 08 repeats the
# analysis across all mice with mouse-level uncertainty.
