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
#
# ## Beginner's code map
#
# This script uses three nested loops. Read them from the outside inward:
# ``state`` → ``window`` → ``density``. One pass through the innermost loop
# produces one modularity result. That result is stored as a dictionary in the
# ``records`` list, so ``records`` acts like an in-memory table whose rows have
# named columns such as ``state``, ``density``, and ``q_max``.
#
# Names reused from earlier scripts are ``rows`` (neuron indices), ``frames``
# (time indices), ``activity`` (neurons × frames), ``correlation`` (nodes ×
# nodes), and ``adjacency`` (the binary graph). Follow one pass through the loops
# before trying to understand the plotting code.

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
# ## Settings you may modify
#
# - ``WINDOW_FRAMES`` is the number of consecutive stable-state frames per
#   estimate. Shorter windows run faster but estimate correlations from less data.
# - ``DENSITIES`` is a tuple of graph densities to test. ``0.05`` means 5%.
# - ``REFERENCE_DENSITY`` chooses which one appears in the window scatter plot;
#   it must also occur in ``DENSITIES``.
# - ``MAX_WINDOWS`` limits windows per state and ``MAX_NEURONS`` limits graph
#   nodes. Use ``None`` for either limit only when a longer run is acceptable.
# - ``N_RUNS`` repeats stochastic Louvain optimization for each graph.
# - ``GAMMA`` controls module-size resolution; script 03 explains this choice.

# %%
RECORDING = "mouse02_sleep"       # dataset used for this worked example
WINDOW_FRAMES = 1500              # frames in one non-overlapping state window
DENSITIES = (0.02, 0.05, 0.10)   # graph densities evaluated for every window
REFERENCE_DENSITY = 0.05          # density shown as individual window points
MAX_WINDOWS = 4                   # maximum windows per state; None means all
MAX_NEURONS = None                # maximum active neurons; None means all
N_RUNS = 10                       # Louvain repeats for each graph
GAMMA = 1.0                       # Louvain module-size resolution

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
#
# Functions used in this step:
#
# - ``ts.frame_windows(indices, width, max_windows=...)`` returns a list of
#   non-overlapping, fixed-length frame-index arrays.
# - ``net.correlation_matrix(activity)`` returns the node-by-node correlations.
# - ``net.density_threshold(C, density)`` returns ``(adjacency, cutoff)``.
# - ``net.repeat_louvain(adj, ...)`` returns a dictionary containing ``Q_max``,
#   ``ci_max``, and other repeated-search results.

# %%
rec = dataio.load_recording(RECORDING)
rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
records = []  # one dictionary per state × window × density result

for state in rec.state_labels:
    windows = ts.frame_windows(
        dataio.state_frames(rec, state),
        WINDOW_FRAMES,
        max_windows=MAX_WINDOWS,
    )
    print(f"{state}: {len(windows)} complete windows", flush=True)
    # ``enumerate(..., start=1)`` supplies a human-readable window number while
    # also giving us the actual array of frame indices.
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
# The ``with`` block closes the file automatically, even if writing fails.
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
#
# The list comprehensions below filter the table. Read
# ``[row["q_max"] for row in records if ...]`` as “collect q_max from every row
# that satisfies these conditions.” ``np.asarray`` converts the resulting Python
# list into a NumPy array so ``mean`` and ``std`` can be calculated directly.

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
