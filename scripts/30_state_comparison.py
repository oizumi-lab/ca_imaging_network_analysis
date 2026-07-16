# %% [markdown]
# # 30 · State-dependent modularity (the lecture result)
#
# Everything so far feeds into one question: **does the single-cell functional
# network become more modular when the brain loses consciousness?**
#
# Kiyooka & Oomoto et al. (2026) report that at single-cell resolution,
# **modularity Q is higher during NREM sleep and anesthesia than during
# wakefulness**, robustly across connection densities. Here we reproduce that
# comparison from the v2.0 data.
#
# Method (matching the paper): for each state, estimate the functional network
# from **1500-frame sleep windows** or **2900-frame anesthesia windows** of
# activity-filtered neurons, threshold at a
# **fixed density** (ranking pairs by ``|r|``, ``negative=True``, as in script
# 20), take the **max-Q** Louvain partition, and compare states across a range of
# densities.
#
# **This script produces the talk figure** (`20260730_Neuro2026_Talk` ·
# "Modularity (Functional segregation)"): a **per-mouse scatter** of the number
# of modules and of modularity Q, Wakefulness vs NREM (and Wakefulness vs
# anesthesia), with an "Average" column joining each mouse's two states. A second
# figure shows the same result as mean±SE curves **vs connection density**, to
# make the "robust across densities" point explicit.

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
# Defaults are kept light so the whole script runs in a few minutes. Louvain cost
# grows with the number of neurons, so we randomly subsample to ``MAX_NEURONS``
# active neurons per recording. **To reproduce the paper more fully:** set
# ``MAX_NEURONS = None`` (use all neurons), widen ``DENSITIES`` (the paper spans
# 0.008–0.3), and raise ``N_RUNS`` toward 200. Subsampling can change both the
# absolute Q values and the estimated state contrast, so treat the defaults as a
# teaching-sized descriptive run.
#
# Each small **dot** in the per-mouse scatter is one descriptive estimate: a
# (window × density) pair. These dots show sensitivity to analysis choices;
# they are **not independent replicates**. State effects and error bars below
# are summarized at the biological-mouse level after averaging windows (and,
# for sleep mouse 4, its two recording days).
# Fixed graph density does not equalize temporal firing sparsity. See
# ``verification/52_sparse_firing_robustness.py`` for state-specific temporal-null
# and common-active sensitivity checks.

# %%
SLEEP_WINDOW = 1500
ANE_WINDOW = 2900
DENSITIES = [0.02, 0.03, 0.05, 0.08, 0.10]   # scatter dots + curve x-axis
REF_DENSITY = 0.05  # reference density for the numeric summary at the end
N_RUNS = 5          # Louvain runs per window/density (paper: ~200)
N_WINDOWS = 2       # windows per state (more = more dots / smoother estimates)
GAMMA = 1.0
MAX_NEURONS = 3000  # random subsample for speed; set None to use all (like the paper)

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

MEASURE_LABELS = {"Q": "Modularity  Q", "nmod": "# of Modules"}
UNCONSCIOUS_COLOR = {"nrem": "crimson", "anesthesia": "goldenrod"}


# %%
def state_measures(rec, label, rows, width):
    """max-Q and module count per (window, density) for one state.

    Returns ``{K: {"Q": [...], "nmod": [...]}}`` with one entry per window.
    """
    out = {K: {"Q": [], "nmod": []} for K in DENSITIES}
    windows = ts.frame_windows(
        dataio.state_frames(rec, label),
        width,
        max_windows=N_WINDOWS,
    )
    for win in windows:
        # Correlation is shared across densities; compute this quadratic matrix
        # only once per window, then vary the threshold below.
        C = net.correlation_matrix(rec.spike_smoothed[np.ix_(rows, win)])
        for K in DENSITIES:
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
# ## Compute measures for both datasets
# This is the heavy cell (a few minutes). Each line below prints as it runs.
# Results are kept **per recording** so we can draw the per-mouse scatter.

# %%
print("SLEEP dataset (awake vs NREM):")
sleep_data = dataset_measures(SLEEP_RECS, SLEEP_WINDOW)
print("ANESTHESIA dataset (awake vs anesthesia):")
ane_data = dataset_measures(ANE_RECS, ANE_WINDOW)


# %% [markdown]
# ## Figure 1 — per-mouse scatter (reproduces the talk slide)
# For every biological mouse, each state's small dots show all window/density
# estimates. The gray **Average** column shows one hierarchical mean per mouse,
# joined awake → unconscious. Mouse 4's two sleep days are first averaged
# separately and then combined, so each biological mouse has equal weight.

# %%
def pool(rec_states, state, measure):
    """Flat list over (window, density), used only for descriptive dots."""
    vals = []
    for K in DENSITIES:
        vals.extend(rec_states[state][K][measure])
    return np.asarray(vals, dtype=float)


def recording_summary(rec_states, state, measure, density=None):
    """Mean over windows, optionally after also averaging across densities."""
    densities = DENSITIES if density is None else [density]
    vals = [np.mean(rec_states[state][K][measure]) for K in densities]
    return float(np.mean(vals))


def mouse_summary(data, rec_names, state, measure, density=None):
    """Equal-weight mean of recording/day summaries for one biological mouse."""
    vals = [recording_summary(data[name], state, measure, density) for name in rec_names]
    return float(np.mean(vals))


def scatter_by_mouse(ax, data, mouse_groups, measure, unconscious_state):
    """Per-mouse scatter of a measure, awake vs one unconscious state.

    Each biological mouse is a column; the final column is the across-mouse
    "Average" (per-mouse means, awake and unconscious joined by a line).
    """
    jitter = np.random.RandomState(0)
    color_un = UNCONSCIOUS_COLOR[unconscious_state]
    aw_means, un_means = [], []
    for x, (_, rec_names) in enumerate(mouse_groups):
        aw = np.concatenate([pool(data[name], "awake", measure) for name in rec_names])
        un = np.concatenate([pool(data[name], unconscious_state, measure) for name in rec_names])
        ax.scatter(x + jitter.uniform(-.09, .09, aw.size), aw, s=16,
                   color="royalblue", zorder=3, label="Wakefulness" if x == 0 else None)
        ax.scatter(x + jitter.uniform(-.09, .09, un.size), un, s=16,
                   color=color_un, zorder=3,
                   label=unconscious_state.upper() if x == 0 else None)
        aw_means.append(mouse_summary(data, rec_names, "awake", measure))
        un_means.append(mouse_summary(data, rec_names, unconscious_state, measure))

    xa = len(mouse_groups)  # the "Average" column
    ax.scatter(np.full(len(aw_means), xa) - .06, aw_means, s=26,
               color="royalblue", edgecolor="k", lw=.4, zorder=4)
    ax.scatter(np.full(len(un_means), xa) + .06, un_means, s=26,
               color=color_un, edgecolor="k", lw=.4, zorder=4)
    for a, u in zip(aw_means, un_means):          # join each mouse's two states
        ax.plot([xa - .06, xa + .06], [a, u], color="k", lw=.8, zorder=2)

    ax.set_xticks(list(range(len(mouse_groups))) + [xa])
    ax.set_xticklabels([label for label, _ in mouse_groups] + ["Average"], fontsize=8)
    ax.set_ylabel(MEASURE_LABELS[measure])
    ax.margins(x=0.04)


def per_mouse_figure(data, mouse_groups, unconscious_state, dataset_title):
    """A 1×2 figure: [# of Modules | Modularity Q], as on the talk slide."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, measure in zip(axes, ("nmod", "Q")):
        scatter_by_mouse(ax, data, mouse_groups, measure, unconscious_state)
    axes[0].legend(loc="best", fontsize=9, framealpha=.9)
    fig.suptitle(dataset_title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


fig_sleep = per_mouse_figure(sleep_data, SLEEP_MOUSE_GROUPS, "nrem",
                             "Modularity (functional segregation) — Wakefulness vs NREM")
fig_sleep.savefig(FIG_DIR / "30_modularity_per_mouse_sleep.png", dpi=140, bbox_inches="tight")

fig_ane = per_mouse_figure(ane_data, ANE_MOUSE_GROUPS, "anesthesia",
                           "Modularity (functional segregation) — Wakefulness vs Anesthesia")
fig_ane.savefig(FIG_DIR / "30_modularity_per_mouse_ane.png", dpi=140, bbox_inches="tight")
plt.show()
print("saved ->", FIG_DIR / "30_modularity_per_mouse_sleep.png")
print("saved ->", FIG_DIR / "30_modularity_per_mouse_ane.png")


# %% [markdown]
# ## Figure 2 — modularity vs density (robustness view)
# The same data, aggregated to one value per biological mouse, as mean±SE curves.
# The unconscious state should sit above the awake curve **at every density**.

# %%
def aggregate_curve(data, mouse_groups, state):
    """Mean±SE of biological-mouse Q values at each density."""
    m = np.empty(len(DENSITIES))
    se = np.empty(len(DENSITIES))
    for j, K in enumerate(DENSITIES):
        vals = np.asarray([
            mouse_summary(data, rec_names, state, "Q", density=K)
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
fig.savefig(FIG_DIR / "30_state_comparison.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Quantify the difference at the reference density
# A paired summary at ``REF_DENSITY``. Windows are averaged within recording,
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
# - **Cross-check with libraries**: scripts `50_verify_modularity.py` and
#   `51_verify_smallworld.py` recompute these measures with NetworkX / bctpy.
