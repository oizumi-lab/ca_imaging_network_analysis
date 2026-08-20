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
# ## Settings

# %%
RECORDING = "mouse02_sleep"
SCALES = (1, 2, 5, 10, 20, 40)
WINDOW_FRAMES = 1500
MAX_WINDOWS = 2
MAX_NEURONS = 2500
K = 0.05
GAMMA = 1.0
N_RUNS = 10

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

# %%
rec = dataio.load_recording(RECORDING)
rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
coords = rec.centroid_um[rows]
activity = rec.spike_smoothed[rows]
distances = squareform(pdist(coords))
records = []

for scale in SCALES:
    if scale == 1:
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

# %% [markdown]
# ## Exercise 5 — find where the state contrast becomes smallest
#
# Calculate the mean ``Awake − NREM`` modularity contrast at every value in
# ``SCALES``. Identify the scale with the smallest *absolute* contrast and report
# how many network nodes remain there.
#
# Make a two-column result table containing parcel size and state contrast. Then
# explain why a very small graph at a coarse scale may give a less stable result.
#
# **Where to start:** ``means_by_state`` and ``contrast`` were constructed
# for the figure above. Node counts are stored in each row of ``records``. You
# will need to combine those two pieces of information.
