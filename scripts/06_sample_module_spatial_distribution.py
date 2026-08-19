# %% [markdown]
# # 06 · Where are the modules? Single cells versus spatial parcels
#
# ## Where this script fits
# Modularity Q tells us how strongly a graph separates into groups, but it does
# not tell us where those groups are located. Two networks can have similar Q
# while having very different spatial arrangements.
#
# We therefore add the neurons' cortical coordinates and ask:
#
# 1. Do neighboring single neurons usually belong to the same functional module?
# 2. Does that spatial relationship change after nearby signals are averaged?
#
# A functional module is defined by correlated activity, not by anatomical
# borders. We should therefore inspect the spatial map rather than assume that a
# module forms one compact patch. This script compares single-cell modules with
# modules obtained after averaging 40 neighboring neurons.

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.funcnet import (
    coarsegrain as cg,
    dataio,
    network as net,
    visualization as viz,
)
from src.funcnet.paths import FIG_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings

# %%
RECORDING = "mouse02_sleep"
WINDOW_FRAMES = 1500
MAX_NEURONS = 2500
MESO_NNEI = 40
K = 0.05
GAMMA = 1.0
N_RUNS_SINGLE = 20
N_RUNS_MESO = 100
DISTANCE_EDGES_UM = (500.0, 1000.0, 1500.0, 2000.0, 2500.0)
DISTANCE_LABELS = ("0–500", "500–1k", "1k–1.5k", "1.5k–2k", "2k–2.5k", "2.5k+")

# %% [markdown]
# ## Estimate single-cell and mesoscale partitions

# %%
rec = dataio.load_recording(RECORDING)
rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
coords = rec.centroid_um[rows]
activity = rec.spike_smoothed[rows]

parcel_index = cg.close_clustering(coords[:, 0], coords[:, 1], MESO_NNEI)
parcel_activity, parcel_x, parcel_y = cg.coarse_grain(
    activity,
    coords[:, 0],
    coords[:, 1],
    parcel_index,
)
parcel_coords = np.column_stack([parcel_x, parcel_y])

partitions = {}
profiles = {}
for state in rec.state_labels:
    available = dataio.state_frames(rec, state)
    if available.size < WINDOW_FRAMES:
        raise ValueError(f"{state} has fewer than {WINDOW_FRAMES} stable frames")
    frames = available[:WINDOW_FRAMES]

    single = net.modularity_from_activity(
        activity[:, frames],
        density=K,
        gamma=GAMMA,
        n_runs=N_RUNS_SINGLE,
        negative=True,
    )
    meso = net.modularity_from_activity(
        parcel_activity[:, frames],
        density=K,
        gamma=GAMMA,
        n_runs=N_RUNS_MESO,
        negative=True,
    )
    partitions[state] = {"single": single, "meso": meso}
    profiles[state] = {
        "single": cg.same_module_by_distance(
            coords,
            single["ci_max"],
            DISTANCE_EDGES_UM,
        ),
        "meso": cg.same_module_by_distance(
            parcel_coords,
            meso["ci_max"],
            DISTANCE_EDGES_UM,
        ),
    }
    print(
        f"{state:<6}: single Q={single['Q_max']:.3f}, "
        f"{single['n_modules_max']} modules; "
        f"nnei={MESO_NNEI} Q={meso['Q_max']:.3f}, "
        f"{meso['n_modules_max']} modules",
        flush=True,
    )

# %% [markdown]
# ## Figure — maps and distance dependence
#
# Single-cell modules should look spatially intermixed. After coarse-graining,
# parcels belonging to the same module become spatially localized, and the
# same-module proportion decreases with cortical distance.

# %%
state_titles = {"awake": "Awake", "nrem": "NREM"}
state_colors = {"awake": "royalblue", "nrem": "crimson"}
fig, axes = plt.subplots(2, 3, figsize=(14.2, 8.6))

for row, state in enumerate(rec.state_labels):
    viz.plot_spatial_modules(
        axes[row, 0],
        coords,
        partitions[state]["single"]["ci_max"],
        title=f"{state_titles[state]} · single-cell\nspatially intermixed",
        node_size=7,
    )
    viz.plot_spatial_modules(
        axes[row, 1],
        parcel_coords,
        partitions[state]["meso"]["ci_max"],
        title=f"{state_titles[state]} · {MESO_NNEI} neurons/parcel\nspatially localized",
        node_size=55,
    )

distance_x = np.arange(len(DISTANCE_LABELS))
for state in rec.state_labels:
    axes[0, 2].plot(
        distance_x,
        profiles[state]["single"],
        "-o",
        color=state_colors[state],
        label=state_titles[state],
    )
    axes[1, 2].plot(
        distance_x,
        profiles[state]["meso"],
        "-o",
        color=state_colors[state],
        label=state_titles[state],
    )

for ax, scale_title in zip(axes[:, 2], ("Single-cell", f"{MESO_NNEI} neurons/parcel")):
    ax.set_xticks(distance_x)
    ax.set_xticklabels(DISTANCE_LABELS, rotation=38, ha="right", fontsize=8)
    ax.set_xlabel("cortical distance (µm)")
    ax.set_ylabel("proportion in same module")
    ax.set_ylim(0, 1)
    ax.set_title(f"{scale_title}: module similarity vs distance")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(color="0.9")

fig.suptitle(
    "06 · Functional modules are intermixed at single-cell scale and localized after coarse-graining",
    fontsize=14,
)
fig.tight_layout()
figure_path = FIG_DIR / "06_sample_module_spatial_distribution.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", figure_path)

# %% [markdown]
# ## Takeaway
#
# Functional segregation at single-cell resolution does not mean local spatial
# segregation. Spatially localized modules emerge after averaging nearby cells.
# Script 10 checks the same result across all recordings.

# %% [markdown]
# ## Exercise 6 — create a module-localization contrast (intermediate–advanced)
#
# Define a simple localization contrast as:
#
# ``same-module probability at 0–500 µm − probability at the farthest valid bin``.
#
# Calculate this quantity for both states at the single-cell and 40-neuron-parcel
# scales. A larger positive value indicates stronger spatial localization.
# Present the four results in a table and explain which scale appears more
# localized.
#
# **Where to look for help:** the required arrays are already stored in
# ``profiles[state]["single"]`` and ``profiles[state]["meso"]``. Check for
# non-finite values before choosing the farthest bin. If handling missing bins is
# unfamiliar, this is a reasonable point to ask AI for guidance—but verify that
# the selected bin and subtraction direction match the definition above.
