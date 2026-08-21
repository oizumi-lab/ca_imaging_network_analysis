# %% [markdown]
# # 09 · Module spatial distribution across all mice
#
# ## Where this script fits
# Script 08 showed that modularity Q changes with spatial scale. Q alone does not
# reveal whether a module occupies one cortical patch or is mixed among other
# modules. This script adds spatial coordinates and pulls together the paper's
# central **multi-scale** message about
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
# The workflow reuses the same network definition as the earlier scripts and
# adds one spatial summary:
#
# ```text
# activity → equal-density graph → modules → cortical coordinates
#          → same-module probability within distance bins
# ```
#
# Method (matching the paper): correlation → ``|r|`` density threshold at
# **K = 0.05** → **max-Q** Louvain (γ = 1) over ``N_RUNS`` iterations. Distances
# are binned in **500-µm** steps and each pair scored 1/0 for same-module
# (``coarsegrain.same_module_by_distance``, a port of ``dist_and_mod.m``).
#
# ## Beginner's code map
#
# This script keeps two spatial scales in parallel. In nested dictionaries,
# ``"single"`` selects individual neurons and ``"meso"`` selects parcels of
# nearby neurons. Important names are:
#
# - ``coords`` / ``pcoords``: neuron / parcel x-y coordinates in µm;
# - ``X`` / ``res``: neuron / parcel activity with time in columns;
# - ``ci_s`` / ``ci_m``: one single-cell / mesoscale module label per node;
# - ``out``: window profiles for one recording; and
# - ``prof``: those profiles for every recording in a cohort.
#
# The script first makes example maps, then calculates distance profiles for all
# mice. Each locally defined function handles one stage and documents the shape
# of what it returns. If you want a new spatial statistic, keep the network-
# building functions unchanged and add it after the module labels are returned.

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
# ## Settings and ``PAPER_MODE``
#
# ``PAPER_MODE`` is one Boolean switch for the expensive cohort calculations.
# ``False`` is the recommended teaching/debugging mode; ``True`` is the
# paper-scale research mode. Changing it automatically makes these choices:
#
# | setting | teaching mode (False) | paper mode (True) |
# | --- | --- | --- |
# | ``N_RUNS`` | 20 Louvain searches | 200 searches |
# | ``N_WINDOWS`` | first window/state | every complete window (``None``) |
# | ``MAX_NEURONS`` | at most 2,500 | every active neuron (``None``) |
# | ``N_RUNS_MAP`` | 30 searches | 200 searches |
#
# This switch does not change the scientific method, ``K``, ``GAMMA``, distance
# bins, or parcel size. It greatly increases runtime because single-cell
# networks contain thousands of nodes; a complete paper-mode run may require
# many hours or days. Develop extensions with ``False`` and use ``True`` only for
# a final unattended run. Numerical results can differ because teaching mode
# samples fewer neurons, windows, and stochastic searches.
#
# Two details are intentionally independent of the switch. Example maps always
# load all active neurons so their node counts match the paper, and mesoscale
# maps always use ``N_RUNS_MESO=200`` because their parcel graphs are small.

# %%
PAPER_MODE = False  # False = teaching run; True = long full-data calculation
K = 0.05            # connection density: retain the strongest 5% of pairs
GAMMA = 1.0         # Louvain module-size resolution
N_RUNS = 200 if PAPER_MODE else 20  # repeats per cohort graph
N_WINDOWS = None if PAPER_MODE else 1  # windows/state; None means every window
MAX_NEURONS = None if PAPER_MODE else 2500  # cohort cap; None means all neurons
MESO_NNEI = 40      # target neurons per mesoscale parcel (paper Fig. 7)
# The example MAPS always use ALL active neurons (no subsample) to match the paper's counts.
N_RUNS_MAP = 200 if PAPER_MODE else 30  # single-cell example-map repeats
N_RUNS_MESO = 200  # mesoscale example-map repeats in both modes (small graphs)

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
    """Load one recording and prepare both spatial resolutions.

    Parameters
    ----------
    name : str
        Recording name understood by ``dataio.load_recording``.
    subsample : bool
        If True, apply ``MAX_NEURONS``; if False, keep every active neuron.

    Returns
    -------
    rec : Recording
        Loaded recording and metadata.
    coords : NumPy array, shape (neurons, 2)
        Selected neuron coordinates in micrometres.
    X : NumPy array, shape (neurons, frames)
        Selected smoothed activity.
    res : NumPy array, shape (parcels, frames)
        Activity averaged within ``MESO_NNEI``-neuron parcels.
    parcel_coords : NumPy array, shape (parcels, 2)
        Corresponding parcel centroids in micrometres.
    """
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
# ## Step 1 — inspect example module-assignment maps (Fig. 5A–C and Fig. 7F)
# Each dot is a node (neuron for single-cell, parcel for mesoscale) at its
# cortical position, coloured by its module. Top row: single-cell modules are
# **spatially intermixed**. Bottom row: mesoscale modules form **contiguous
# patches**. These maps use **all active neurons** of the example mouse (Mouse 5),
# so the node counts match the paper: 6920 (Awake/NREM) and 3210 (Anesthesia) at
# single-cell → 173 / 173 / 81 parcels at nnei = 40 with the current remainder
# handling.

# %%
def maps_for(name, labels):
    """Calculate example-map partitions for selected states of one recording.

    ``labels`` is a sequence such as ``["awake", "nrem"]``. The recording is
    loaded once with every active neuron. The returned dictionary supports
    ``result[state]["single"]`` and ``result[state]["meso"]``; each value is a
    ``(coordinates, module_labels)`` tuple ready for plotting. Only the first
    complete state window is used for these example maps.
    """
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
fig.savefig(FIG_DIR / "09_all_mice_module_maps.png", dpi=140, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Step 2 — quantify same-module proportion versus distance (Fig. 5G,H and Fig. 7G,H)
# For every recording and state we compute, at both scales, the fraction of node
# pairs in the same module within each 500-µm distance bin, averaged over windows.
# We then aggregate per mouse (pooling mouse 4's two days).

# %%
def recording_profiles(name, width):
    """Calculate distance profiles for every selected window of one recording.

    ``name`` selects the recording and ``width`` gives frames per window. The
    return value is ``out[state][scale]``, a list of arrays. Each array contains
    same-module probability for the bins defined by ``DIST_EDGES``.
    """
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
    """Run :func:`recording_profiles` for every recording in one cohort.

    ``kind`` is ``"sleep"`` or ``"ane"`` and selects the matching window length
    from ``WIN``. The return dictionary is indexed by recording name.
    """
    print(f"{kind.upper()} dataset ({len(recs)} recordings):", flush=True)
    return {name: recording_profiles(name, WIN[kind]) for name in recs}


print("Computing distance–module profiles (single-cell + mesoscale):")
sleep_prof = run_dataset(SLEEP_RECS, "sleep")
ane_prof = run_dataset(ANE_RECS, "ane")


# %%
def per_mouse_profiles(prof, mouse_map, state, scale):
    """Average recording windows into one distance curve per biological mouse.

    ``prof`` contains recording-level profiles and ``mouse_map`` assigns those
    recordings to mouse IDs. ``state`` and ``scale`` select one condition. The
    returned list contains one one-dimensional NumPy array per mouse.
    """
    curves = []
    for mouse in sorted(set(mouse_map.values())):
        recs = [n for n, m in mouse_map.items() if m == mouse and n in prof]
        windows = [w for n in recs if state in prof[n] for w in prof[n][state][scale]]
        if windows:
            curves.append(np.nanmean(np.vstack(windows), axis=0))
    return curves


def plot_profile(ax, prof, mouse_map, unconscious, scale, unc_color, unc_name):
    """Draw mouse-level Awake and unconscious-state distance profiles.

    The function selects one scale, obtains per-mouse curves with
    :func:`per_mouse_profiles`, and modifies the supplied Matplotlib ``ax``.
    ``unc_color`` and ``unc_name`` control only the second state's appearance.
    Nothing is returned.
    """
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
fig.savefig(FIG_DIR / "09_all_mice_same_module_vs_distance.png", dpi=140, bbox_inches="tight")
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

# %% [markdown]
# ## Practice — summarize the distance dependence
#
# For each biological mouse, calculate a localization contrast equal to the
# value in the nearest distance bin minus the value in the farthest finite bin.
# Do this separately for the single-cell and parcel scales, then display the
# mouse-level values in a paired plot.
#
# Explain what a positive contrast means and why you should inspect the number of
# node pairs in the farthest bin before interpreting a large value.
#
# **Where to start:** ``per_mouse_profiles`` returns one distance curve per
# biological mouse. Check for finite values before choosing the final bin, and
# keep sleep and anesthesia cohorts separate.
