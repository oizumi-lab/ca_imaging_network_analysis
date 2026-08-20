# %% [markdown]
# # 04 · State-dependent modularity within the example recording
#
# ## Where this script fits
# Scripts 02--03 produced one modularity value from one Awake window and one NREM
# window. A single window may be unusual, so it cannot tell us whether a state
# difference is repeatable over the recording. We now repeat the complete
# activity-to-modularity pipeline across several non-overlapping stable-state
# windows from ``mouse02_sleep``.
#
# We also repeat the calculation at several graph densities. If a conclusion
# appears only at one density, it may depend on an arbitrary threshold choice.
# The two checks answer different questions:
#
# - windows: does the result recur at different times?
# - densities: does the result survive a reasonable graph-building choice?
#
# Each point is a time window from one recording, not an independent mouse. This
# lets us examine within-recording consistency, but it is not a population test.
# Script 07 performs the biological-replicate comparison across mice.

# %%
import csv
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np

from src.funcnet import dataio, network as net, timeseries as ts
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings

# %%
RECORDING = "mouse02_sleep"
WINDOW_FRAMES = 1500
DENSITIES = (0.02, 0.05, 0.10)
REFERENCE_DENSITY = 0.05
MAX_WINDOWS = 4
MAX_NEURONS = 2500
N_RUNS = 10
GAMMA = 1.0

# %% [markdown]
# ## Step 1 — divide each state into comparable windows
#
# A window is a fixed-length sample of one stable state. Non-overlapping windows
# let us check whether the result recurs at different times without counting the
# same frames twice. Both states use the same selected neuron rows, window
# length, density values, resolution, and number of Louvain runs.
#
# For each window, the script repeats the entire pipeline from script 03. Each
# row appended to ``records`` is one combination of state, window, and density;
# keeping this tidy table makes the later plotting and practice analysis easier.

# %%
rec = dataio.load_recording(RECORDING)
rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
records = []

for state in rec.state_labels:
    windows = ts.frame_windows(
        dataio.state_frames(rec, state),
        WINDOW_FRAMES,
        max_windows=MAX_WINDOWS,
    )
    print(f"{state}: {len(windows)} complete windows", flush=True)
    for window_index, frames in enumerate(windows, start=1):
        activity = rec.spike_smoothed[np.ix_(rows, frames)]
        correlation = net.correlation_matrix(activity)
        for density in DENSITIES:
            adjacency, threshold = net.density_threshold(
                correlation,
                density,
                negative=True,
            )
            result = net.repeat_louvain(
                adjacency,
                gamma=GAMMA,
                n_runs=N_RUNS,
                seed=12345,
            )
            records.append(
                {
                    "recording": rec.name,
                    "state": state,
                    "window": window_index,
                    "density": density,
                    "q_max": result["Q_max"],
                    "n_modules": result["n_modules_max"],
                    "correlation_threshold": threshold,
                    "n_neurons": rows.size,
                }
            )
            print(
                f"  window {window_index} K={density:.0%}: "
                f"Q={result['Q_max']:.3f}, modules={result['n_modules_max']}",
                flush=True,
            )

csv_path = RESULTS_DIR / "04_sample_modularity.csv"
with csv_path.open("w", newline="", encoding="utf-8") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(records[0]))
    writer.writeheader()
    writer.writerows(records)
print("saved ->", csv_path)

# %% [markdown]
# ## Step 2 — visualize recurrence across windows and densities
#
# The left panel shows all selected windows at K=5%. The right panel shows the
# same comparison across densities. Look first for the direction of the state
# difference, then ask whether it is consistent across windows and density
# choices. Exact values vary because the data windows and Louvain runs vary.

# %%
colors = {"awake": "royalblue", "nrem": "crimson"}
titles = {"awake": "Awake", "nrem": "NREM"}
fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

for state_index, state in enumerate(rec.state_labels):
    values = np.asarray(
        [
            row["q_max"]
            for row in records
            if row["state"] == state and row["density"] == REFERENCE_DENSITY
        ],
        dtype=float,
    )
    jitter = np.linspace(-0.055, 0.055, values.size) if values.size > 1 else np.zeros(1)
    axes[0].scatter(
        state_index + jitter,
        values,
        s=45,
        color=colors[state],
        edgecolor="black",
        linewidth=0.4,
        zorder=3,
    )
    axes[0].plot(
        [state_index - 0.14, state_index + 0.14],
        [values.mean(), values.mean()],
        color="black",
        lw=2,
    )
axes[0].set_xticks(range(len(rec.state_labels)))
axes[0].set_xticklabels([titles[state] for state in rec.state_labels])
axes[0].set_ylabel("max-Q modularity")
axes[0].set_title(f"One point per {WINDOW_FRAMES}-frame window (K=5%)")
axes[0].grid(axis="y", color="0.9")

for state in rec.state_labels:
    means = []
    errors = []
    for density in DENSITIES:
        values = np.asarray(
            [
                row["q_max"]
                for row in records
                if row["state"] == state and row["density"] == density
            ],
            dtype=float,
        )
        means.append(values.mean())
        errors.append(values.std(ddof=1) if values.size > 1 else 0.0)
    axes[1].errorbar(
        np.asarray(DENSITIES) * 100,
        means,
        yerr=errors,
        marker="o",
        capsize=3,
        color=colors[state],
        label=titles[state],
    )
axes[1].set_xlabel("connection density K (%)")
axes[1].set_ylabel("max-Q modularity (mean ± SD across windows)")
axes[1].set_title("Within-recording density sensitivity")
axes[1].legend(frameon=False)
axes[1].grid(color="0.9")

fig.suptitle(f"04 · State-dependent modularity in {rec.name}", fontsize=14)
fig.tight_layout()
figure_path = FIG_DIR / "04_sample_state_comparison.png"
fig.savefig(figure_path, dpi=160, bbox_inches="tight")
plt.show()
print("saved ->", figure_path)

# %% [markdown]
# ## Takeaway
#
# NREM modularity is higher in this full example recording across windows and
# several density choices. Because all points come from one mouse, use script 07
# before making a population-level claim.

# %% [markdown]
# ## Exercise 4 — calculate the state contrast
#
# Using the completed ``records`` list, calculate mean Q separately for Awake and
# NREM at each density. Then calculate ``NREM − Awake`` and plot that contrast
# against density with a horizontal zero line.
#
# Answer two questions in words:
#
# 1. Does the contrast keep the same sign across the tested densities?
# 2. Why can these window-level values not be treated as independent mice?
#
# **Where to start:** the figure above already demonstrates how to filter
# ``records`` by state and density. Extend that pattern to calculate the
# difference requested here.
