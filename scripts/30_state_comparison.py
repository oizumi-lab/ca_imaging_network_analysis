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
# from **1500-frame windows** of activity-filtered neurons, threshold at a
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

from src.funcnet import dataio, network as net
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
# Defaults are kept light so the whole script runs in a few minutes. Louvain cost
# grows with the number of neurons, so we randomly subsample to ``MAX_NEURONS``
# active neurons per recording. **To reproduce the paper more fully:** set
# ``MAX_NEURONS = None`` (use all neurons), widen ``DENSITIES`` (the paper spans
# 0.008–0.3), and raise ``N_RUNS`` toward 200. Subsampling changes the absolute Q
# values but preserves the awake-vs-unconscious *ordering*.
#
# Each **dot** in the per-mouse scatter is one estimate: a (window × density)
# pair. Every state is measured at the same ``DENSITIES``, so the awake-vs-
# unconscious comparison is like-for-like; the vertical spread within a mouse
# reflects window-to-window and density-to-density variation (as in the talk).

# %%
WINDOW = 1500
DENSITIES = [0.02, 0.03, 0.05, 0.08, 0.10]   # scatter dots + curve x-axis
REF_DENSITY = 0.05  # reference density for the numeric summary at the end
N_RUNS = 5          # Louvain runs per window/density (paper: ~200)
N_WINDOWS = 2       # windows per state (more = more dots / smoother estimates)
GAMMA = 1.0
MAX_NEURONS = 3000  # random subsample for speed; set None to use all (like the paper)

# All recordings: 5 sleep mice (mouse 4 recorded on two days) and 4 anesthesia mice.
# The talk's modularity slide keeps mouse 4's two days as SEPARATE columns, so each
# recording is its own column here (unlike script 40, which pools them).
SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]

# Column labels for the per-mouse scatter (order matches the recording lists).
SLEEP_LABELS = ["Mouse1", "Mouse2", "Mouse3", "Mouse4\n(day1)", "Mouse4\n(day2)", "Mouse5"]
ANE_LABELS = ["Mouse1", "Mouse2", "Mouse3", "Mouse4"]

MEASURE_LABELS = {"Q": "Modularity  Q", "nmod": "# of Modules"}
UNCONSCIOUS_COLOR = {"nrem": "crimson", "anesthesia": "goldenrod"}


# %%
def windows_of(rec, label, n_windows=N_WINDOWS, width=WINDOW):
    """Up to ``n_windows`` non-overlapping ``width``-frame windows of a state."""
    idx = dataio.state_frames(rec, label)
    n = idx.size // width
    return [idx[i * width:(i + 1) * width] for i in range(min(n, n_windows))]


def neuron_rows(rec):
    """Row indices of the neurons to analyse: active neurons, optionally subsampled.

    The same neuron set is used for both states of a recording (fair comparison),
    and the subsample is seeded so results are reproducible.
    """
    keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
    rows = np.flatnonzero(keep)
    if MAX_NEURONS is not None and rows.size > MAX_NEURONS:
        rng = np.random.RandomState(0)
        rows = np.sort(rng.choice(rows, MAX_NEURONS, replace=False))
    return rows


def state_measures(rec, label, rows):
    """max-Q and module count per (window, density) for one state.

    Returns ``{K: {"Q": [...], "nmod": [...]}}`` with one entry per window.
    """
    out = {K: {"Q": [], "nmod": []} for K in DENSITIES}
    for win in windows_of(rec, label):
        C = net.correlation_matrix(rec.spike_smoothed[np.ix_(rows, win)])
        for K in DENSITIES:
            adj, _ = net.density_threshold(C, K, negative=True)  # rank by |r|, as in the paper
            r = net.repeat_louvain(adj, gamma=GAMMA, n_runs=N_RUNS)
            out[K]["Q"].append(r["Q_max"])
            out[K]["nmod"].append(r["n_modules_max"])
    return out


def dataset_measures(recs):
    """Per-recording measures → {recording: {state_label: {K: {"Q":[], "nmod":[]}}}}."""
    data = {}
    for name in recs:
        rec = dataio.load_recording(name)
        rows = neuron_rows(rec)
        data[name] = {}
        for label in rec.state_labels:
            print(f"  {name} ({rows.size} neurons): {label} ...", flush=True)
            data[name][label] = state_measures(rec, label, rows)
    return data


# %% [markdown]
# ## Compute measures for both datasets
# This is the heavy cell (a few minutes). Each line below prints as it runs.
# Results are kept **per recording** so we can draw the per-mouse scatter.

# %%
print("SLEEP dataset (awake vs NREM):")
sleep_data = dataset_measures(SLEEP_RECS)
print("ANESTHESIA dataset (awake vs anesthesia):")
ane_data = dataset_measures(ANE_RECS)


# %% [markdown]
# ## Figure 1 — per-mouse scatter (reproduces the talk slide)
# For every recording, each state's dots are pooled over windows and densities.
# The gray **Average** column shows each mouse's per-state mean, joined by a line
# awake → unconscious. If the finding holds, the unconscious mean sits **above**
# the awake mean for (almost) every mouse.

# %%
def pool(rec_states, state, measure):
    """Flat list of one measure over all (window, density) for a state."""
    vals = []
    for K in DENSITIES:
        vals.extend(rec_states[state][K][measure])
    return np.asarray(vals, dtype=float)


def scatter_by_mouse(ax, data, recs, labels, measure, unconscious_state):
    """Per-mouse scatter of a measure, awake vs one unconscious state.

    Each recording is a column; the final column is the across-mouse "Average"
    (per-mouse means, awake and unconscious joined by a line).
    """
    jitter = np.random.RandomState(0)
    color_un = UNCONSCIOUS_COLOR[unconscious_state]
    aw_means, un_means = [], []
    for x, name in enumerate(recs):
        aw = pool(data[name], "awake", measure)
        un = pool(data[name], unconscious_state, measure)
        ax.scatter(x + jitter.uniform(-.09, .09, aw.size), aw, s=16,
                   color="royalblue", zorder=3, label="Wakefulness" if x == 0 else None)
        ax.scatter(x + jitter.uniform(-.09, .09, un.size), un, s=16,
                   color=color_un, zorder=3,
                   label=unconscious_state.upper() if x == 0 else None)
        aw_means.append(aw.mean())
        un_means.append(un.mean())

    xa = len(recs)  # the "Average" column
    ax.scatter(np.full(len(aw_means), xa) - .06, aw_means, s=26,
               color="royalblue", edgecolor="k", lw=.4, zorder=4)
    ax.scatter(np.full(len(un_means), xa) + .06, un_means, s=26,
               color=color_un, edgecolor="k", lw=.4, zorder=4)
    for a, u in zip(aw_means, un_means):          # join each mouse's two states
        ax.plot([xa - .06, xa + .06], [a, u], color="k", lw=.8, zorder=2)

    ax.set_xticks(list(range(len(recs))) + [xa])
    ax.set_xticklabels(labels + ["Average"], fontsize=8)
    ax.set_ylabel(MEASURE_LABELS[measure])
    ax.margins(x=0.04)


def per_mouse_figure(data, recs, labels, unconscious_state, dataset_title):
    """A 1×2 figure: [# of Modules | Modularity Q], as on the talk slide."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, measure in zip(axes, ("nmod", "Q")):
        scatter_by_mouse(ax, data, recs, labels, measure, unconscious_state)
    axes[0].legend(loc="best", fontsize=9, framealpha=.9)
    fig.suptitle(dataset_title, y=1.02, fontsize=13)
    fig.tight_layout()
    return fig


fig_sleep = per_mouse_figure(sleep_data, SLEEP_RECS, SLEEP_LABELS, "nrem",
                             "Modularity (functional segregation) — Wakefulness vs NREM")
fig_sleep.savefig(FIG_DIR / "30_modularity_per_mouse_sleep.png", dpi=140, bbox_inches="tight")

fig_ane = per_mouse_figure(ane_data, ANE_RECS, ANE_LABELS, "anesthesia",
                           "Modularity (functional segregation) — Wakefulness vs Anesthesia")
fig_ane.savefig(FIG_DIR / "30_modularity_per_mouse_ane.png", dpi=140, bbox_inches="tight")
plt.show()
print("saved ->", FIG_DIR / "30_modularity_per_mouse_sleep.png")
print("saved ->", FIG_DIR / "30_modularity_per_mouse_ane.png")


# %% [markdown]
# ## Figure 2 — modularity vs density (robustness view)
# The same data, aggregated across all recordings/windows, as mean±SE curves.
# The unconscious state should sit above the awake curve **at every density**.

# %%
def aggregate_curve(data, state):
    """Mean±SE of Q across all recordings/windows, per density, for one state."""
    m = np.empty(len(DENSITIES))
    se = np.empty(len(DENSITIES))
    for j, K in enumerate(DENSITIES):
        vals = []
        for rec_states in data.values():
            if state in rec_states:
                vals.extend(rec_states[state][K]["Q"])
        vals = np.asarray(vals, float)
        m[j] = vals.mean()
        se[j] = vals.std() / max(1, np.sqrt(vals.size))
    return m, se


def curve_panel(ax, data, unconscious_state, title):
    x = [K * 100 for K in DENSITIES]
    for state, color in [("awake", "royalblue"),
                         (unconscious_state, UNCONSCIOUS_COLOR[unconscious_state])]:
        m, se = aggregate_curve(data, state)
        ax.errorbar(x, m, yerr=se, marker="o", capsize=3, lw=2,
                    color=color, label="Wakefulness" if state == "awake" else state.upper())
    ax.set_xlabel("connection density K (%)")
    ax.set_ylabel("modularity  Q (max over runs)")
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
curve_panel(axes[0], sleep_data, "nrem", "Wakefulness vs NREM sleep")
curve_panel(axes[1], ane_data, "anesthesia", "Wakefulness vs anesthesia")
fig.suptitle("Single-cell functional-network modularity is higher during unconsciousness",
             y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "30_state_comparison.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Quantify the difference at the reference density
# A simple summary: mean Q per state at ``REF_DENSITY``, pooled across recordings,
# and the awake → unconscious change.

# %%
def pooled_at_ref(data, state):
    vals = []
    for rec_states in data.values():
        if state in rec_states:
            vals.extend(rec_states[state][REF_DENSITY]["Q"])
    return float(np.mean(vals))


for name, data, other in [("SLEEP", sleep_data, "nrem"),
                          ("ANESTHESIA", ane_data, "anesthesia")]:
    qa = pooled_at_ref(data, "awake")
    qb = pooled_at_ref(data, other)
    arrow = "↑ higher" if qb > qa else "↓ lower"
    print(f"{name:<11} K={REF_DENSITY*100:g}%:  awake Q={qa:.3f}   {other} Q={qb:.3f}   "
          f"({other} is {arrow} during unconsciousness, ΔQ={qb - qa:+.3f})")

# %% [markdown]
# ## Takeaway & extensions
# We reproduced the paper's core single-cell result: losing consciousness
# (sleep or anesthesia) **increases** functional-network modularity, while the
# **number of modules** is comparable across states — i.e. the unconscious
# network is *more segregated*, not more numerous in its parts. Natural next
# steps for the course:
# - **Spatial scale**: coarse-grain neurons into groups and check the effect
#   disappears at the mesoscale (paper Fig. 7).
# - **Per-neuron contribution** $Q_i$ and how degree relates to modularity (Fig. 4).
# - **Module stability** over time (Fig. 6) and consensus partitions.
# - **Cross-check with libraries**: scripts `50_verify_modularity.py` and
#   `51_verify_smallworld.py` recompute these measures with NetworkX / bctpy.
