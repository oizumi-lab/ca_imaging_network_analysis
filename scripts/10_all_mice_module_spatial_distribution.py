# %% [markdown]
# # 10 · Module spatial distribution across all mice
#
# This tutorial pulls together the paper's central **multi-scale** message about
# *where* modules sit in cortex:
#
# - **Single-cell scale** (Kiyooka et al. **Fig. 5**): functional modules are
#   **spatially intermixed** — neighbouring neurons often belong to *different*
#   modules, so the chance that two neurons share a module barely depends on how
#   far apart they are.
# - **Mesoscale** (coarse-grained parcels, **Fig. 7**): modules are **spatially
#   localized** — nearby parcels tend to share a module, and that tendency falls
#   off with distance.
#
# We reproduce, in one place:
# - **Fig. 5A–C** & **Fig. 7F** — example **module-assignment maps** (each node
#   drawn at its cortical position, coloured by module) at both scales, for
#   awake / NREM / anesthesia.
# - **Fig. 5G,H** & **Fig. 7G,H** — the **proportion of node pairs in the same
#   module vs. cortical distance**, per mouse, awake vs. NREM (G) and awake vs.
#   anesthesia (H). Flat curves ⇒ intermixed; decreasing curves ⇒ localized.
#
# Method (matching the paper): correlation → ``|r|`` density threshold at
# **K = 0.05** → **max-Q** Louvain (γ = 1) over ``N_RUNS`` iterations. Distances
# are binned in **500-µm** steps and each pair scored 1/0 for same-module
# (``coarsegrain.same_module_by_distance``, a port of ``dist_and_mod.m``).

# %%
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt

from src.funcnet import (
    coarsegrain as cg,
    dataio,
    network as net,
    timeseries as ts,
    visualization as viz,
)
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Mean of empty slice")   # empty distance bins -> NaN
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
# Light defaults so the whole script runs in a few minutes. **To reproduce the
# paper more fully:** set ``MAX_NEURONS = None`` (all active neurons), raise
# ``N_RUNS`` toward 200, and ``N_WINDOWS`` for smoother per-mouse curves.

# %%
PAPER_MODE = False
K = 0.05
GAMMA = 1.0
N_RUNS = 200 if PAPER_MODE else 20
N_WINDOWS = None if PAPER_MODE else 1
MAX_NEURONS = None if PAPER_MODE else 2500
MESO_NNEI = 40         # parcel size for the mesoscale networks (Fig. 7 uses nnei = 40)
# The example MAPS always use ALL active neurons (no subsample) to match the paper's counts.
N_RUNS_MAP = 200 if PAPER_MODE else 30
N_RUNS_MESO = 200      # Louvain runs for the mesoscale example maps (few parcels; cheap, paper: 200)

# 500-µm distance-bin upper edges → bins [0-500, 500-1000, ..., 2000-2500, 2500+].
DIST_EDGES = (500.0, 1000.0, 1500.0, 2000.0, 2500.0)
DIST_LABELS = ["0–500", "500–1k", "1k–1.5k", "1.5k–2k", "2k–2.5k", "2.5k+"]

WIN = {"sleep": 1500, "ane": 2900}

SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
SLEEP_MOUSE = {"mouse01_sleep": "1", "mouse02_sleep": "2", "mouse03_sleep": "3",
               "mouse04_day1_sleep": "4", "mouse04_day2_sleep": "4", "mouse05_sleep": "5"}
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]
ANE_MOUSE = {"mouse03_ane": "1", "mouse05_ane": "2", "mouse06_ane": "3", "mouse07_ane": "4"}

# Example maps (Fig. 5A–C / Fig. 7F): the paper's example is "Mouse 5" (recording
# '71'), recorded in both sessions — mouse05_sleep has 6920 active neurons
# (Awake/NREM) and mouse05_ane has 3210 (Anesthesia), matching the paper exactly.
EX_SLEEP_REC = "mouse05_sleep"   # 6920 active neurons -> Awake + NREM examples
EX_ANE_REC = "mouse05_ane"       # 3210 active neurons -> Anesthesia example


# %%
def prepare(name, subsample=True):
    """Load a recording; return per-neuron and per-parcel (nnei=40) signals+coords (µm)."""
    rec = dataio.load_recording(name)
    max_neurons = MAX_NEURONS if subsample else None
    rows = dataio.select_neuron_rows(rec, max_neurons=max_neurons, seed=0)
    coords = rec.centroid_um[rows]                       # (n, 2) micrometres
    X = rec.spike_smoothed[rows]                         # (n, T)
    idx = cg.close_clustering(coords[:, 0], coords[:, 1], MESO_NNEI)
    res, xp, yp = cg.coarse_grain(X, coords[:, 0], coords[:, 1], idx)
    parcel_coords = np.column_stack([xp, yp])            # (K, 2) parcel centroids (µm)
    return rec, coords, X, res, parcel_coords


# %% [markdown]
# ## Part 1 — example module-assignment maps (Fig. 5A–C and Fig. 7F)
# Each dot is a node (neuron for single-cell, parcel for mesoscale) at its
# cortical position, coloured by its module. Top row: single-cell modules are
# **spatially intermixed**. Bottom row: mesoscale modules form **contiguous
# patches**. These maps use **all active neurons** of the example mouse (Mouse 5),
# so the node counts match the paper: 6920 (Awake/NREM) and 3210 (Anesthesia) at
# single-cell → 173 / 173 / 81 parcels at nnei = 40 with the current remainder
# handling.

# %%
def maps_for(name, labels):
    """Single-cell and mesoscale (coords_µm, module_labels) for each state of one
    recording, using ALL active neurons (so node counts match the paper's Fig. 5/7F).
    Loads the recording once."""
    rec, coords, X, res, pcoords = prepare(name, subsample=False)
    result = {}
    for label in labels:
        win = ts.frame_windows(
            dataio.state_frames(rec, label),
            WIN[rec.data_info],
            max_windows=N_WINDOWS,
        )[0]
        # The shared network workflow keeps correlation, |r| thresholding, and
        # repeated-Louvain settings identical at both spatial scales.
        ci_single = net.modularity_from_activity(
            X[:, win],
            density=K,
            gamma=GAMMA,
            n_runs=N_RUNS_MAP,
            negative=True,
        )["ci_max"]
        ci_meso = net.modularity_from_activity(
            res[:, win],
            density=K,
            gamma=GAMMA,
            n_runs=N_RUNS_MESO,
            negative=True,
        )["ci_max"]
        result[label] = {"single": (coords, ci_single), "meso": (pcoords, ci_meso)}
    return result


sleep_maps = maps_for(EX_SLEEP_REC, ["awake", "nrem"])
ane_maps = maps_for(EX_ANE_REC, ["anesthesia"])
# (state label, column title, maps dict)
EX = [("awake", "Awake", sleep_maps),
      ("nrem", "NREM", sleep_maps),
      ("anesthesia", "Anesthesia", ane_maps)]

fig, axes = plt.subplots(2, 3, figsize=(13, 8.6))
for col, (label, title, mp) in enumerate(EX):
    viz.plot_spatial_modules(
        axes[0, col], *mp[label]["single"], title=title, node_size=6
    )
    viz.plot_spatial_modules(
        axes[1, col], *mp[label]["meso"], title=title, node_size=42
    )
    axes[0, col].title.set_fontsize(9)
    axes[1, col].title.set_fontsize(9)
axes[0, 0].set_ylabel("single-cell\n(Fig. 5A–C)", fontsize=11)
axes[1, 0].set_ylabel(f"mesoscale nnei={MESO_NNEI}\n(Fig. 7F)", fontsize=11)
for ax in (axes[0, 0], axes[1, 0]):   # re-enable the y-label we use as a row header
    ax.set_axis_on()
    ax.set_xticks([])
    ax.set_yticks([])
fig.suptitle("Module assignment maps: single-cell modules are intermixed, "
             "mesoscale modules are localized", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "10_all_mice_module_maps.png", dpi=140, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Part 2 — same-module proportion vs cortical distance (Fig. 5G,H and Fig. 7G,H)
# For every recording and state we compute, at both scales, the fraction of node
# pairs in the same module within each 500-µm distance bin, averaged over windows.
# We then aggregate per mouse (pooling mouse 4's two days).

# %%
def recording_profiles(name, width):
    """Per-window same-module-vs-distance profiles at both scales, for each state."""
    rec, coords, X, res, pcoords = prepare(name)
    out = {label: {"single": [], "meso": []} for label in rec.state_labels}
    for label in rec.state_labels:
        windows = ts.frame_windows(
            dataio.state_frames(rec, label),
            width,
            max_windows=N_WINDOWS,
        )
        for win in windows:
            ci_s = net.modularity_from_activity(
                X[:, win],
                density=K,
                gamma=GAMMA,
                n_runs=N_RUNS,
                negative=True,
            )["ci_max"]
            ci_m = net.modularity_from_activity(
                res[:, win],
                density=K,
                gamma=GAMMA,
                n_runs=N_RUNS,
                negative=True,
            )["ci_max"]
            out[label]["single"].append(cg.same_module_by_distance(coords, ci_s, DIST_EDGES))
            out[label]["meso"].append(cg.same_module_by_distance(pcoords, ci_m, DIST_EDGES))
        print(f"  {name} [{label}]: {len(out[label]['single'])} window(s)", flush=True)
    return out


def run_dataset(recs, kind):
    print(f"{kind.upper()} dataset ({len(recs)} recordings):", flush=True)
    return {name: recording_profiles(name, WIN[kind]) for name in recs}


print("Computing distance–module profiles (single-cell + mesoscale):")
sleep_prof = run_dataset(SLEEP_RECS, "sleep")
ane_prof = run_dataset(ANE_RECS, "ane")


# %%
def per_mouse_profiles(prof, mouse_map, state, scale):
    """List (one per mouse) of mean same-module-vs-distance curves for a state/scale."""
    curves = []
    for mouse in sorted(set(mouse_map.values())):
        recs = [n for n, m in mouse_map.items() if m == mouse and n in prof]
        windows = [w for n in recs if state in prof[n] for w in prof[n][state][scale]]
        if windows:
            curves.append(np.nanmean(np.vstack(windows), axis=0))
    return curves


def plot_profile(ax, prof, mouse_map, unconscious, scale, unc_color, unc_name):
    xs = np.arange(len(DIST_LABELS))
    plotted = []
    for curves, color in [(per_mouse_profiles(prof, mouse_map, "awake", scale), "royalblue"),
                          (per_mouse_profiles(prof, mouse_map, unconscious, scale), unc_color)]:
        for c in curves:
            ax.plot(xs, c, "-o", color=color, ms=4, lw=1, alpha=.85)
            plotted.append(c)
    ax.set_xticks(xs)
    ax.set_xticklabels(DIST_LABELS, rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("cortical distance (µm)")
    ax.set_ylabel("proportion of pairs\nin the same module")
    observed_max = np.nanmax(np.vstack(plotted)) if plotted else 0.8
    ax.set_ylim(0, min(1.0, max(0.8, observed_max * 1.05)))
    ax.plot([], [], "-o", color="royalblue", label="Wakefulness")
    ax.plot([], [], "-o", color=unc_color, label=unc_name)
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(12, 10))
plot_profile(axes[0, 0], sleep_prof, SLEEP_MOUSE, "nrem", "single", "crimson", "NREM")
plot_profile(axes[0, 1], ane_prof, ANE_MOUSE, "anesthesia", "single", "goldenrod", "Anesthesia")
plot_profile(axes[1, 0], sleep_prof, SLEEP_MOUSE, "nrem", "meso", "crimson", "NREM")
plot_profile(axes[1, 1], ane_prof, ANE_MOUSE, "anesthesia", "meso", "goldenrod", "Anesthesia")
axes[0, 0].set_title("(Fig. 5G) single-cell — Awake vs NREM")
axes[0, 1].set_title("(Fig. 5H) single-cell — Awake vs Anesthesia")
axes[1, 0].set_title("(Fig. 7G) mesoscale — Awake vs NREM")
axes[1, 1].set_title("(Fig. 7H) mesoscale — Awake vs Anesthesia")
fig.suptitle("Same-module proportion vs distance: flat at single-cell (intermixed), "
             "decreasing at the mesoscale (localized)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "10_all_mice_same_module_vs_distance.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Takeaway
# The same functional networks look **opposite** at the two scales:
# - **Single-cell** — modules are spatially **intermixed**: the same-module
#   proportion is low and roughly **flat** with distance (Fig. 5G,H). This is the
#   surprising single-cell finding — segregation is *not* spatial at the neuron level.
# - **Mesoscale** — coarse-graining makes modules **localized**: nearby parcels
#   share a module far more than distant ones, so the curve **decreases** with
#   distance (Fig. 7G,H). This matches macro/mesoscale (fMRI/EEG) intuition.
#
# So whether the cortex looks "parcellated into local regions" depends entirely
# on the spatial scale of observation — the paper's core multi-scale conclusion.
# The farthest-distance bins contain fewer pairs and sometimes rebound, so these
# profiles are descriptive until accompanied by pair counts, spatial/label nulls,
# and partition-stability uncertainty.
