# %% [markdown]
# # 07 · State-dependent modularity across all mice
#
# ## Where this script fits
# Scripts 01--06 followed one recording in detail. That worked example shows how
# the measurements become a modularity result, but one mouse cannot establish a
# result that is reproducible across animals. This script repeats the same
# single-cell analysis for every version-3 sleep and anesthesia recording.
#
# The biological question is: **does the single-cell functional network become
# more modular when the brain loses consciousness?**
#
# Kiyooka & Oomoto et al. (2026) report that at single-cell resolution,
# **modularity Q is higher during NREM sleep and anesthesia than during
# wakefulness**, robustly across connection densities. Here we reproduce that
# comparison from the version-3 data.
#
# The analysis unit changes as we move through the data hierarchy:
#
# ```text
# frames → windows → recording days → biological mice → cohort summary
# ```
#
# Method (matching the paper): for each state, estimate the functional network
# from **1,500-frame sleep windows** or **2,900-frame anesthesia windows** of
# activity-filtered neurons, threshold at a **fixed density** (ranking pairs by
# ``|r|``, as in script 03), take the **max-Q** Louvain partition, and compare
# states across a range of densities.
#
# **This script produces the talk-style comparison figure**: one modularity-Q
# dot per complete time window for each recording, separately for Wakefulness vs
# NREM and Wakefulness vs anesthesia. The "Average" column contains one pair of
# window-averaged values per recording, joined across states. A second figure
# shows the same result as mean±SE curves **vs connection density**, to make the
# "robust across densities" point explicit.

# %%
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt

from src.funcnet import dataio, network as net, timeseries as ts
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
# The default settings limit neuron count and Louvain repetitions so attendees
# can inspect the workflow without launching the full paper-scale computation.
# Louvain cost
# grows with the number of neurons, so we randomly subsample to ``MAX_NEURONS``
# active neurons per recording. **To reproduce the paper more fully:** set
# ``MAX_NEURONS = None`` (use all neurons), widen ``DENSITIES`` (the paper spans
# 0.008–0.3), and raise ``N_RUNS`` toward 200. Subsampling can change both the
# absolute Q values and the estimated state contrast, so treat the defaults as a
# teaching-sized descriptive run.
#
# Each small **dot** in the first comparison figures is one complete time-window
# estimate at ``REF_DENSITY``. All available complete windows are shown; they
# are repeated observations within a recording, **not independent biological
# replicates**. The later density curves remain deliberately lightweight by
# using at most ``DENSITY_CURVE_WINDOWS`` windows per state. State effects and
# error bars below are summarized at the biological-mouse level after averaging
# windows (and, for sleep mouse 4, its two recording days).
# Fixed graph density equalizes edge count, not temporal firing sparsity. Treat
# raw Q as an estimated-network statistic and add per-neuron-preserving temporal
# nulls when extending the workflow to new biological claims.

# %%
PAPER_MODE = False
SLEEP_WINDOW = 1500
ANE_WINDOW = 2900
DENSITIES = (
    [0.008, 0.01, 0.02, 0.03, 0.05, 0.08, 0.10, 0.20, 0.30]
    if PAPER_MODE
    else [0.02, 0.03, 0.05, 0.08, 0.10]
)
REF_DENSITY = 0.05  # reference density for the numeric summary at the end
N_RUNS = 200 if PAPER_MODE else 5
DENSITY_CURVE_WINDOWS = None if PAPER_MODE else 2
GAMMA = 1.0
MAX_NEURONS = None if PAPER_MODE else 3000

# All recordings: 5 sleep mice (mouse 4 recorded on two days) and 4 anesthesia mice.
SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]

# Repeated recording days are nested within their biological mouse. Keeping this
# hierarchy explicit prevents mouse 4 from receiving twice the inferential weight.
SLEEP_MOUSE_GROUPS = [
    ("Mouse1", ("mouse01_sleep",)),
    ("Mouse2", ("mouse02_sleep",)),
    ("Mouse3", ("mouse03_sleep",)),
    ("Mouse4", ("mouse04_day1_sleep", "mouse04_day2_sleep")),
    ("Mouse5", ("mouse05_sleep",)),
]
ANE_MOUSE_GROUPS = [
    ("Mouse1", ("mouse03_ane",)),
    ("Mouse2", ("mouse05_ane",)),
    ("Mouse3", ("mouse06_ane",)),
    ("Mouse4", ("mouse07_ane",)),
]

# Display every recording separately in the fixed-density window figures. The
# two Mouse4 sleep days are separate dataset columns, matching the attached
# figure; they are pooled only in the later biological-mouse summaries.
SLEEP_RECORDING_COLUMNS = [
    ("Mouse1", "mouse01_sleep"),
    ("Mouse2", "mouse02_sleep"),
    ("Mouse3", "mouse03_sleep"),
    ("Mouse4\n(day1)", "mouse04_day1_sleep"),
    ("Mouse4\n(day2)", "mouse04_day2_sleep"),
    ("Mouse5", "mouse05_sleep"),
]
ANE_RECORDING_COLUMNS = [
    ("Mouse1", "mouse03_ane"),
    ("Mouse2", "mouse05_ane"),
    ("Mouse3", "mouse06_ane"),
    ("Mouse4", "mouse07_ane"),
]

UNCONSCIOUS_COLOR = {"nrem": "crimson", "anesthesia": "goldenrod"}


# %%
def state_measures(rec, label, rows, width):
    """max-Q and module count per (window, density) for one state.

    Every complete window is evaluated at ``REF_DENSITY`` for the per-window
    scatter. To keep the density sweep tractable, only the first
    ``DENSITY_CURVE_WINDOWS`` windows are evaluated at the other densities.
    """
    out = {K: {"Q": [], "nmod": []} for K in DENSITIES}
    windows = ts.frame_windows(
        dataio.state_frames(rec, label),
        width,
        max_windows=None,
    )
    print(f"    {len(windows)} complete {width}-frame windows", flush=True)
    for window_index, win in enumerate(windows):
        # Correlation is shared across densities; compute this quadratic matrix
        # only once per window, then vary the threshold below.
        C = net.correlation_matrix(rec.spike_smoothed[np.ix_(rows, win)])
        window_densities = (
            DENSITIES
            if DENSITY_CURVE_WINDOWS is None
            or window_index < DENSITY_CURVE_WINDOWS
            else (REF_DENSITY,)
        )
        for K in window_densities:
            adj, _ = net.density_threshold(C, K, negative=True)  # rank by |r|
            r = net.repeat_louvain(adj, gamma=GAMMA, n_runs=N_RUNS)
            out[K]["Q"].append(r["Q_max"])
            out[K]["nmod"].append(r["n_modules_max"])
    return out


def dataset_measures(recs, width):
    """Per-recording measures → {recording: {state_label: {K: {"Q":[], "nmod":[]}}}}."""
    data = {}
    for name in recs:
        rec = dataio.load_recording(name)
        # Select once per recording so both states use the identical active-neuron
        # subset. ``dataio`` preserves the tutorial's seeded RandomState behavior.
        rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
        data[name] = {}
        for label in rec.state_labels:
            print(f"  {name} ({rows.size} neurons): {label} ...", flush=True)
            data[name][label] = state_measures(rec, label, rows, width)
    return data


# %% [markdown]
# ## Step 1 — compute measures for both datasets
# This cell may take several minutes. Progress messages identify the recording,
# state, and number of complete windows. Results remain nested by recording and
# state; aggregation is deliberately postponed so the data hierarchy stays
# visible.

# %%
print("SLEEP dataset (awake vs NREM):")
sleep_data = dataset_measures(SLEEP_RECS, SLEEP_WINDOW)
print("ANESTHESIA dataset (awake vs anesthesia):")
ane_data = dataset_measures(ANE_RECS, ANE_WINDOW)


# %% [markdown]
# ## Step 2 — inspect window estimates before averaging
# Each recording has one vertical column. Every small point is max-Q from one
# complete state-specific window at the same fixed density, ``REF_DENSITY``.
# The **Average** column contains one point per recording and state (the mean of
# that recording's windows), with a line joining Awake to NREM/Anesthesia. Sleep
# Mouse4 day1 and day2 remain separate here because they are separate datasets.
# They are combined only for the later biological-mouse summary.

# %%
def recording_summary(rec_states, state, measure, density=None, max_windows=None):
    """Mean over windows, optionally after also averaging across densities."""
    densities = DENSITIES if density is None else [density]
    vals = []
    for K in densities:
        window_values = rec_states[state][K][measure]
        if max_windows is not None:
            window_values = window_values[:max_windows]
        vals.append(np.mean(window_values))
    return float(np.mean(vals))


def mouse_summary(data, rec_names, state, measure, density=None, max_windows=None):
    """Equal-weight mean of recording/day summaries for one biological mouse."""
    vals = [
        recording_summary(
            data[name],
            state,
            measure,
            density,
            max_windows=max_windows,
        )
        for name in rec_names
    ]
    return float(np.mean(vals))


def window_comparison_figure(data, recording_columns, unconscious_state, title):
    """Plot one Q point per window and one paired mean per recording."""
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    color_un = UNCONSCIOUS_COLOR[unconscious_state]
    state_offset = 0.08
    awake_means = []
    unconscious_means = []
    for x, (_, recording_name) in enumerate(recording_columns):
        awake = np.asarray(
            data[recording_name]["awake"][REF_DENSITY]["Q"],
            dtype=float,
        )
        unconscious = np.asarray(
            data[recording_name][unconscious_state][REF_DENSITY]["Q"],
            dtype=float,
        )
        ax.scatter(
            np.full(awake.size, x - state_offset),
            awake,
            s=19,
            color="royalblue",
            zorder=3,
            label="Wakefulness" if x == 0 else None,
        )
        ax.scatter(
            np.full(unconscious.size, x + state_offset),
            unconscious,
            s=19,
            color=color_un,
            zorder=3,
            label=unconscious_state.upper() if x == 0 else None,
        )
        awake_means.append(float(awake.mean()))
        unconscious_means.append(float(unconscious.mean()))

    average_x = len(recording_columns)
    left_x, right_x = average_x - state_offset, average_x + state_offset
    ax.scatter(
        np.full(len(awake_means), left_x),
        awake_means,
        s=24,
        color="royalblue",
        edgecolor="black",
        linewidth=0.35,
        zorder=4,
    )
    ax.scatter(
        np.full(len(unconscious_means), right_x),
        unconscious_means,
        s=24,
        color=color_un,
        edgecolor="black",
        linewidth=0.35,
        zorder=4,
    )
    for awake_mean, unconscious_mean in zip(awake_means, unconscious_means):
        ax.plot(
            [left_x, right_x],
            [awake_mean, unconscious_mean],
            color="black",
            linewidth=0.8,
            zorder=2,
        )

    ax.set_xticks(list(range(len(recording_columns))) + [average_x])
    ax.set_xticklabels(
        [label for label, _ in recording_columns] + ["Average"],
        fontsize=8,
    )
    ax.set_ylabel("Modularity Q (max over runs)")
    ax.set_title(f"{title} at K={REF_DENSITY:.0%}")
    ax.legend(loc="best", fontsize=8, framealpha=0.95)
    ax.grid(axis="y", color="#dddddd", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.margins(x=0.045, y=0.08)
    fig.tight_layout()
    return fig


fig_sleep = window_comparison_figure(
    sleep_data,
    SLEEP_RECORDING_COLUMNS,
    "nrem",
    "Wakefulness vs NREM",
)
fig_sleep.savefig(FIG_DIR / "07_modularity_per_mouse_sleep.png", dpi=140, bbox_inches="tight")

fig_ane = window_comparison_figure(
    ane_data,
    ANE_RECORDING_COLUMNS,
    "anesthesia",
    "Wakefulness vs Anesthesia",
)
fig_ane.savefig(FIG_DIR / "07_modularity_per_mouse_ane.png", dpi=140, bbox_inches="tight")
plt.show()
print("saved ->", FIG_DIR / "07_modularity_per_mouse_sleep.png")
print("saved ->", FIG_DIR / "07_modularity_per_mouse_ane.png")


# %% [markdown]
# ## Step 3 — compare modularity across densities
# The same data are now aggregated to one value per biological mouse and shown
# as mean ± standard error curves. The scientific question is whether the
# direction of the state difference is consistent across densities, rather than
# appearing only at one threshold choice.

# %%
def aggregate_curve(data, mouse_groups, state):
    """Mean±SE of biological-mouse Q values using a matched window cap."""
    m = np.empty(len(DENSITIES))
    se = np.empty(len(DENSITIES))
    for j, K in enumerate(DENSITIES):
        vals = np.asarray([
            mouse_summary(
                data,
                rec_names,
                state,
                "Q",
                density=K,
                max_windows=DENSITY_CURVE_WINDOWS,
            )
            for _, rec_names in mouse_groups
        ], float)
        m[j] = vals.mean()
        se[j] = vals.std(ddof=1) / np.sqrt(vals.size) if vals.size > 1 else np.nan
    return m, se


def curve_panel(ax, data, mouse_groups, unconscious_state, title):
    x = [K * 100 for K in DENSITIES]
    for state, color in [("awake", "royalblue"),
                         (unconscious_state, UNCONSCIOUS_COLOR[unconscious_state])]:
        m, se = aggregate_curve(data, mouse_groups, state)
        ax.errorbar(x, m, yerr=se, marker="o", capsize=3, lw=2,
                    color=color, label="Wakefulness" if state == "awake" else state.upper())
    ax.set_xlabel("connection density K (%)")
    ax.set_ylabel("modularity  Q (max over runs)")
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
curve_panel(axes[0], sleep_data, SLEEP_MOUSE_GROUPS, "nrem", "Wakefulness vs NREM sleep")
curve_panel(axes[1], ane_data, ANE_MOUSE_GROUPS, "anesthesia", "Wakefulness vs anesthesia")
fig.suptitle("Single-cell functional-network modularity is higher during unconsciousness",
             y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "07_all_mice_state_comparison.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 4 — quantify the paired difference at the reference density
# This is a paired summary at ``REF_DENSITY``. Windows are averaged within recording,
# repeated days within mouse, and inference is based on mouse-level state changes.

# %%
def paired_mouse_effects(data, mouse_groups, other):
    """One awake/unconscious pair per biological mouse at REF_DENSITY."""
    awake = np.asarray([
        mouse_summary(data, rec_names, "awake", "Q", density=REF_DENSITY)
        for _, rec_names in mouse_groups
    ])
    unconscious = np.asarray([
        mouse_summary(data, rec_names, other, "Q", density=REF_DENSITY)
        for _, rec_names in mouse_groups
    ])
    return awake, unconscious, unconscious - awake


for name, data, groups, other in [
        ("SLEEP", sleep_data, SLEEP_MOUSE_GROUPS, "nrem"),
        ("ANESTHESIA", ane_data, ANE_MOUSE_GROUPS, "anesthesia")]:
    awake, unconscious, delta = paired_mouse_effects(data, groups, other)
    qa, qb = awake.mean(), unconscious.mean()
    delta_se = delta.std(ddof=1) / np.sqrt(delta.size) if delta.size > 1 else np.nan
    arrow = "↑ higher" if qb > qa else "↓ lower"
    print(f"{name:<11} K={REF_DENSITY*100:g}%:  awake Q={qa:.3f}   {other} Q={qb:.3f}   "
          f"({other} is {arrow}; paired mouse-level mean ΔQ={delta.mean():+.3f} "
          f"± {delta_se:.3f} SE, n={delta.size} mice)")

# %% [markdown]
# ## Takeaway & extensions
# In this teaching-sized analysis, the raw single-cell graphs show higher
# modularity during sleep and anesthesia. The mouse-level summaries avoid treating
# windows, densities, or repeated days as independent observations, but raw Q is
# still an estimated-network statistic rather than a calibrated measure of
# coupling. Natural next steps for the course:
# - **Spatial scale**: coarse-grain neurons into groups and check the effect
#   disappears at the mesoscale (paper Fig. 7).
# - **Per-neuron contribution** $Q_i$ and how degree relates to modularity (Fig. 4).
# - **Module stability** over time (Fig. 6) and consensus partitions.
# Set ``PAPER_MODE = True`` to use all selected neurons, 200 Louvain runs, the
# wider density range, and every available density-curve window. This is a long
# research run rather than an in-class exercise.

# %% [markdown]
# ## Practice — inspect the effect of biological averaging
#
# At ``REF_DENSITY``, make a table with one row per biological mouse and columns
# for Awake Q, NREM or Anesthesia Q, and the paired difference. For sleep Mouse 4,
# first average its two recording days so it contributes only one row.
#
# Compare this table with the window-level points in Figure 1. Explain why the
# number of plotted windows is not the sample size for a cohort-level statement.
#
# **Where to start:** ``paired_mouse_effects`` already returns the three values
# needed for each dataset. The mouse-group definitions near the settings show
# which recordings belong to the same biological mouse.
