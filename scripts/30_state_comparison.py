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
# **fixed density**, take the **max-Q** Louvain partition, and compare states
# across a range of densities.

# %%
import warnings

import numpy as np
import matplotlib.pyplot as plt

from funcnet import dataio
from funcnet import network as net
from funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
# Defaults are kept light so the whole script runs in ~3–4 minutes. Louvain cost
# grows with the number of neurons, so for a snappy demo we randomly subsample to
# ``MAX_NEURONS`` active neurons per recording. **To reproduce the paper more
# fully:** set ``MAX_NEURONS = None`` (use all neurons), expand the recording
# lists to all 5 sleep / 4 anesthesia mice, add higher densities (up to 0.30),
# and raise ``N_RUNS`` toward 200. Subsampling changes the absolute Q values but
# preserves the awake-vs-unconscious *ordering*, which is the teaching point.

# %%
WINDOW = 1500
DENSITIES = [0.01, 0.02, 0.05, 0.10]
N_RUNS = 5          # Louvain runs per window/density (paper: ~200)
N_WINDOWS = 2       # windows per state (more = smoother estimates)
GAMMA = 1.0
MAX_NEURONS = 3000  # random subsample for speed; set None to use all (like the paper)

SLEEP_RECS = ["mouse01_sleep"]   # awake vs NREM   (add more sleep mice to scale up)
ANE_RECS = ["mouse07_ane"]       # awake vs anesthesia (add more ane mice to scale up)


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


def state_modularity(rec, label, rows):
    """max-Q vs density for one state, over windows. Returns {K: [Q,...]}."""
    out = {K: [] for K in DENSITIES}
    for win in windows_of(rec, label):
        C = net.correlation_matrix(rec.spike_smoothed[np.ix_(rows, win)])
        for K in DENSITIES:
            adj, _ = net.density_threshold(C, K)
            out[K].append(net.repeat_louvain(adj, gamma=GAMMA, n_runs=N_RUNS)["Q_max"])
    return out


def dataset_curves(recs):
    """Aggregate max-Q across recordings → {state_label: {K: array of Q}}."""
    agg = {}
    for name in recs:
        rec = dataio.load_recording(name)
        rows = neuron_rows(rec)
        for label in rec.state_labels:
            print(f"  {name} ({rows.size} neurons): {label} ...", flush=True)
            res = state_modularity(rec, label, rows)
            store = agg.setdefault(label, {K: [] for K in DENSITIES})
            for K in DENSITIES:
                store[K].extend(res[K])
    return agg


# %% [markdown]
# ## Compute curves for both datasets
# This is the heavy cell (a few minutes). Each line below prints as it runs.

# %%
print("SLEEP dataset (awake vs NREM):")
sleep_curves = dataset_curves(SLEEP_RECS)
print("ANESTHESIA dataset (awake vs anesthesia):")
ane_curves = dataset_curves(ANE_RECS)

# %% [markdown]
# ## Plot: modularity vs density, by state
# If the finding holds, the unconscious state (NREM / anesthesia) sits **above**
# the awake curve across densities.

# %%
def plot_curves(ax, curves, title):
    colors = {"awake": "tab:gray"}
    for label, store in curves.items():
        m = np.array([np.mean(store[K]) for K in DENSITIES])
        se = np.array([np.std(store[K]) / max(1, np.sqrt(len(store[K]))) for K in DENSITIES])
        ax.errorbar([K * 100 for K in DENSITIES], m, yerr=se, marker="o", capsize=3,
                    lw=2, label=label, color=colors.get(label))
    ax.set_xlabel("connection density K (%)")
    ax.set_ylabel("modularity  Q (max over runs)")
    ax.set_title(title)
    ax.legend()


fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
plot_curves(axes[0], sleep_curves, "Wakefulness vs NREM sleep")
plot_curves(axes[1], ane_curves, "Wakefulness vs anesthesia")
fig.suptitle("Single-cell functional-network modularity is higher during unconsciousness",
             y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "30_state_comparison.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Quantify the difference at K = 5%
# A simple summary: mean Q per state at the reference density, and the awake →
# unconscious change.

# %%
for name, curves in [("SLEEP", sleep_curves), ("ANESTHESIA", ane_curves)]:
    labels = list(curves)
    qa = np.mean(curves["awake"][0.05])
    other = [l for l in labels if l != "awake"][0]
    qb = np.mean(curves[other][0.05])
    arrow = "↑ higher" if qb > qa else "↓ lower"
    print(f"{name:<11} K=5%:  awake Q={qa:.3f}   {other} Q={qb:.3f}   "
          f"({other} is {arrow} during unconsciousness, ΔQ={qb - qa:+.3f})")

# %% [markdown]
# ## Takeaway & extensions
# We reproduced the paper's core single-cell result: losing consciousness
# (sleep or anesthesia) **increases** functional-network modularity. Natural
# next steps for the course:
# - **Spatial scale**: coarse-grain neurons into groups and check the effect
#   disappears at the mesoscale (paper Fig. 7).
# - **Per-neuron contribution** $Q_i$ and how degree relates to modularity (Fig. 4).
# - **Module stability** over time (Fig. 6) and consensus partitions.
