# %% [markdown]
# # 08 · Mesoscale modularity across all mice (paper Fig. 7)
#
# ## Where this script fits
# Scripts 03 and 07 measured modularity at **single-cell** resolution and found it
# **higher during unconsciousness**, with modules **spatially intermixed**
# (Fig. 5). Figure 7 asks the complementary question: **at what spatial scale
# does that picture change?** We spatially **coarse-grain** the neurons into
# parcels of increasing size, rebuild the functional network between *parcels*,
# and track modularity as a function of scale.
#
# The key rule is to rebuild the network from parcel activity at every scale. The
# analysis pipeline (a port of `oizumi-lab/mouse_network`, Fig. 7) is:
# for each parcel size `nnei ∈ {1, 2, 5, 10, 20, 40, 80, 160}`:
# 1. group every `nnei` neighbouring neurons into one **parcel**
#    (`coarsegrain.close_clustering`, a deterministic greedy spatial grouping);
# 2. **average** the smoothed-spike signal within each parcel
#    (`coarsegrain.coarse_grain`);
# 3. correlation between parcels → `|r|` → **density threshold K = 0.05** →
#    **max-Q Louvain** (γ = 1), exactly as at the single-cell scale.
#
# `nnei = 1` is the single-cell case (no coarse-graining), so the whole sweep is
# one code path. We reproduce:
# - **(B)** modularity vs scale, awake vs NREM (per mouse + mean); **(D)** vs anesthesia.
# - **(C)** the awake−NREM modularity *difference* vs scale, with a 95 % CI
#   (one-sample t-test across mice); **(E)** awake−anesthesia.
# - **(F)** an example spatial map of parcel modules at `nnei = 40` — modules
#   become **spatially localized** at the mesoscale (unlike the intermixed
#   single-cell modules).
#
# The paper's finding: the awake-vs-unconscious modularity gap present at the
# single-cell scale **shrinks toward zero** as coarse-graining increases (the CI
# includes 0 for `nnei ≥ 10` sleep, `≥ 5` anesthesia).
#
# ## Beginner's code map
#
# The main calculation nests recording → spatial scale → state → time window.
# At each spatial scale it creates a new parcel activity matrix and rebuilds the
# network from that matrix. Recurring names are:
#
# - ``X``: neuron-by-frame activity;
# - ``x, y``: one cortical coordinate pair per neuron;
# - ``idx``: one parcel assignment per neuron;
# - ``res``: parcel-by-frame averaged activity (not a statistical result here);
# - ``out``: nested modularity values indexed by state and scale; and
# - ``awake_mat`` / ``unc_mat``: mouse-by-scale result matrices.
#
# Functions divide the work into stages: analyze one recording, analyze a whole
# dataset, aggregate recordings into mice, and plot. Read their docstrings first,
# then inspect their bodies only when you need to change that stage.

# %%
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

from src.funcnet import (
    coarsegrain as cg,
    dataio,
    network as net,
    statistics as stat_utils,
    timeseries as ts,
    visualization as viz,
)
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings and ``PAPER_MODE``
#
# ``PAPER_MODE`` is a Boolean switch. Keep it ``False`` for an interactive
# preview. Set it to ``True`` only for a final, unattended paper-scale run.
# The expression ``200 if PAPER_MODE else 20`` means “choose 200 when the switch
# is True; otherwise choose 20.”
#
# The switch changes three settings together:
#
# | setting | preview mode (False) | paper mode (True) |
# | --- | --- | --- |
# | ``N_RUNS`` | 20 Louvain searches | 200 searches |
# | ``N_WINDOWS`` | first 2 windows/state | every complete window (``None``) |
# | ``MAX_NEURONS`` | at most 2,000 | every active neuron (``None``) |
#
# Paper mode does not change ``SCALES``, ``K``, ``GAMMA``, or the biological
# averaging. The ``nnei=1`` graph is the main bottleneck because it can contain
# thousands of nodes; a complete full-mode run can require many hours or days.
# Preview and paper modes follow the same method but can give different numbers
# because preview mode samples fewer neurons and windows.
#
# The panel-F example maps are a deliberate exception: ``module_map`` always
# uses all active neurons and ``FMAP_N_RUNS=200`` in either mode, because the
# parcel graph is much smaller. ``USE_CONSENSUS`` changes the chosen partition,
# not the amount of input data.

# %%
PAPER_MODE = False  # False = preview; True = long full-data calculation
SCALES = [1, 2, 5, 10, 20, 40, 80, 160]  # neurons/parcel; 1 = single cell
K = 0.05          # connection density: retain the strongest 5% of pairs
GAMMA = 1.0       # Louvain module-size resolution
N_RUNS = 200 if PAPER_MODE else 20  # Louvain repeats per graph
N_WINDOWS = None if PAPER_MODE else 2  # windows/state; None means every window
MAX_NEURONS = None if PAPER_MODE else 2000  # None means every active neuron
FMAP_SCALE = 40    # neurons per parcel for the panel-F example map
FMAP_N_RUNS = 200  # Louvain repetitions for panel F in both modes
USE_CONSENSUS = False  # Fig. 7 (main) uses the max-Q partition; True -> consensus (suppl. DS1-24)

WIN = {"sleep": 1500, "ane": 2900}   # frames per window (paper's per-dataset windows)

# Sleep: 5 mice (mouse 4 has two days, pooled for the n = 5 statistics, as in Fig. 7).
SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
SLEEP_MOUSE = {"mouse01_sleep": "1", "mouse02_sleep": "2", "mouse03_sleep": "3",
               "mouse04_day1_sleep": "4", "mouse04_day2_sleep": "4", "mouse05_sleep": "5"}
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]
ANE_MOUSE = {"mouse03_ane": "1", "mouse05_ane": "2", "mouse06_ane": "3", "mouse07_ane": "4"}

# recordings used for the panel-F example maps. Larger recordings show
# localization most clearly; the smaller anesthesia recordings have fewer parcels.
FMAP_SLEEP_REC = "mouse04_day1_sleep"
FMAP_ANE_REC = "mouse05_ane"


# %%
def recording_measures(name, width):
    """Calculate window-level modularity across all scales for one recording.

    Parameters
    ----------
    name : str
        Recording name understood by ``dataio.load_recording``.
    width : int
        Number of frames in each complete state window.

    Returns
    -------
    out : dict
        ``out[state][nnei]`` is a list containing one maximum-Q value per
        analyzed window.

    The panel-F module maps are computed separately by :func:`module_map`, since they use
    all active neurons and many more Louvain runs. Consensus is an optional
    supplementary-analysis setting rather than the main-figure default.
    """
    rec = dataio.load_recording(name)
    # One seeded selection is reused by both states and every spatial scale.
    rows = dataio.select_neuron_rows(rec, max_neurons=MAX_NEURONS, seed=0)
    X = rec.spike_smoothed[rows]                  # (n, T) smoothed spikes
    x, y = rec.centroid[rows, 0], rec.centroid[rows, 1]
    D = squareform(pdist(np.column_stack([x, y])))   # reuse across scales

    out = {label: {s: [] for s in SCALES} for label in rec.state_labels}
    for s in SCALES:
        idx = cg.close_clustering(x, y, s, D=D)
        res, xp, yp = cg.coarse_grain(X, x, y, idx)
        print(f"  {name}: nnei={s:>3}  ->  {res.shape[0]} parcels", flush=True)
        for label in rec.state_labels:
            windows = ts.frame_windows(
                dataio.state_frames(rec, label),
                width,
                max_windows=N_WINDOWS,
            )
            for win in windows:
                r = net.modularity_from_activity(
                    res[:, win],
                    density=K,
                    gamma=GAMMA,
                    n_runs=N_RUNS,
                    negative=True,
                )
                out[label][s].append(r["Q_max"])
    return out


def run_dataset(recs, kind):
    """Analyze every recording name in ``recs`` with the matching window size.

    ``kind`` is ``"sleep"`` or ``"ane"`` and selects ``WIN[kind]``. The return
    dictionary is indexed first by recording name.
    """
    print(f"{kind.upper()} dataset ({len(recs)} recordings):", flush=True)
    return {name: recording_measures(name, WIN[kind]) for name in recs}


# %% [markdown]
# ## Step 1 — compute every recording at every scale
# This cell may take several minutes. `nnei = 1` dominates the cost; coarser scales
# are quick because the parcel network is small.

# %%
sleep_data = run_dataset(SLEEP_RECS, "sleep")
ane_data = run_dataset(ANE_RECS, "ane")


# %% [markdown]
# ## Step 2 — aggregate recordings into biological mice
# Each mouse's value at a scale is the mean over its windows (and, for mouse 4,
# over both days), matching the paper's per-mouse averaging.

# %%
def per_mouse_means(data, mouse_map, unconscious_label):
    """Pool recording windows into one curve per biological mouse.

    ``mouse_map`` assigns each recording name to a mouse ID; two sleep recording
    days therefore map to the same mouse. Returns the sorted IDs plus Awake and
    unconscious-state arrays, each shaped ``(n_mice, n_scales)``.
    """
    mice = sorted(set(mouse_map.values()))
    awake = np.full((len(mice), len(SCALES)), np.nan)
    unc = np.full((len(mice), len(SCALES)), np.nan)
    for mi, mouse in enumerate(mice):
        recs = [n for n, m in mouse_map.items() if m == mouse and n in data]
        for si, s in enumerate(SCALES):
            aw = [q for n in recs for q in data[n]["awake"][s]]
            un = [q for n in recs for q in data[n][unconscious_label][s]]
            if aw:
                awake[mi, si] = np.mean(aw)
            if un:
                unc[mi, si] = np.mean(un)
    return mice, awake, unc


sleep_mice, sleep_aw, sleep_un = per_mouse_means(sleep_data, SLEEP_MOUSE, "nrem")
ane_mice, ane_aw, ane_un = per_mouse_means(ane_data, ANE_MOUSE, "anesthesia")


# %% [markdown]
# ## Step 3 — plot modularity and its paired state contrast (panels B–E)
# Left column: modularity of both states per scale (awake ×, unconscious ○, with
# mean lines). Right column: the awake−unconscious modularity **difference** with
# its 95% confidence interval. The interval describes uncertainty in the mean
# paired difference across mice. If it includes zero, these data do not resolve
# the sign of the mean difference at that scale; this is not proof that the two
# states are equivalent.

# %%
def plot_modularity_scale(ax, awake_mat, unc_mat, unc_label, unc_color):
    """Plot individual mouse values and cohort means across spatial scales.

    ``awake_mat`` and ``unc_mat`` have mice in rows and scales in columns. The
    supplied Matplotlib ``ax`` is modified in place; nothing is returned.
    """
    xs = np.arange(len(SCALES))
    ax.plot(xs, awake_mat.T, "x", color="royalblue", ms=6, ls="none")
    ax.plot(xs, unc_mat.T, "o", color=unc_color, ms=4, mec="k", mew=.3, ls="none")
    ax.plot(xs, np.nanmean(awake_mat, 0), "-", color="royalblue", lw=2, label="Wakefulness")
    ax.plot(xs, np.nanmean(unc_mat, 0), "-", color=unc_color, lw=2, label=unc_label)
    ax.set_xticks(xs)
    ax.set_xticklabels(SCALES)
    ax.set_xlabel("parcel size  nnei  (neurons/parcel)")
    ax.set_ylabel("modularity  Q  (K = 5%)")
    ax.legend(fontsize=8)


def plot_diff_scale(ax, awake_mat, unc_mat, unc_label):
    """Plot paired Awake-minus-unconscious differences with a 95% CI.

    Subtracting equally shaped matrices preserves mouse pairing. The confidence
    interval is calculated independently at each scale, ignoring missing values.
    This function draws on ``ax`` and returns nothing.
    """
    xs = np.arange(len(SCALES))
    diff = awake_mat - unc_mat
    # Student-t interval with a separate non-NaN mouse count at each scale.
    m, lo, hi = stat_utils.mean_confidence_interval(diff, axis=0)
    ax.fill_between(xs, lo, hi, color="0.85", label="95% CI")
    ax.plot(xs, diff.T, "o", color="k", ms=3, ls="none")
    ax.plot(xs, m, "-", color="k", lw=2, label="mean across mice")
    ax.axhline(0, color="0.5", lw=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(SCALES)
    ax.set_xlabel("parcel size  nnei  (neurons/parcel)")
    ax.set_ylabel(f"ΔQ   (awake − {unc_label})")
    ax.legend(fontsize=8)


fig, axes = plt.subplots(2, 2, figsize=(13, 9))
plot_modularity_scale(axes[0, 0], sleep_aw, sleep_un, "NREM", "crimson")
plot_diff_scale(axes[0, 1], sleep_aw, sleep_un, "NREM")
plot_modularity_scale(axes[1, 0], ane_aw, ane_un, "Anesthesia", "goldenrod")
plot_diff_scale(axes[1, 1], ane_aw, ane_un, "Anesthesia")
axes[0, 0].set_title("(B) Wakefulness vs NREM")
axes[0, 1].set_title(f"(C) modularity difference  (n = {len(sleep_mice)} mice)")
axes[1, 0].set_title("(D) Wakefulness vs Anesthesia")
axes[1, 1].set_title(f"(E) modularity difference  (n = {len(ane_mice)} mice)")
fig.suptitle("Mesoscale modularity across spatial scales — state contrast and uncertainty",
             y=1.01, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "08_all_mice_coarse_grain_modularity.png", dpi=140, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Step 4 — inspect example module maps at nnei = 40 (panel F)
# Each dot is a parcel at its centroid, coloured by its module. At the mesoscale,
# modules are **spatially localized** (contiguous patches) in every state — unlike
# the spatially intermixed single-cell modules of Figure 5.
#
# These maps use the paper's Fig. 7 method: the **max-Q partition among 200
# Louvain iterations** (STAR Methods; "we executed it 200 times ... and selected
# the partition with the highest Q") on **all active neurons** (no subsample).
# The paper's *supplementary* Fig. DS1-24 repeats this with consensus clustering;
# set ``USE_CONSENSUS = True`` to reproduce that variant.
# The **localization index** annotated on each panel is how much more often a
# parcel's spatially nearest neighbour shares its module than chance (> 1 = localized).

# %%
def module_map(name, label, width, nnei=FMAP_SCALE, n_runs=FMAP_N_RUNS):
    """Build one state's coarse-grained module map.

    Parameters are the recording name, state label, window width, target neurons
    per parcel, and number of Louvain searches. ``nnei`` and ``n_runs`` have
    defaults, so ordinary calls need only the first three arguments.

    Uses ALL active neurons (no subsample). Returns the **max-Q partition over
    ``n_runs`` Louvain iterations** (the paper's Fig. 7 method), or the consensus
    partition if ``USE_CONSENSUS`` (the supplementary Fig. DS1-24 variant).
    Returns ``(x_parcel, y_parcel, ci)``—two coordinate arrays plus one module
    label per parcel—or ``None`` if the state is too short.
    """
    rec = dataio.load_recording(name)
    rows = dataio.select_neuron_rows(rec)  # all active rows; no map subsampling
    x, y = rec.centroid[rows, 0], rec.centroid[rows, 1]
    idx = cg.close_clustering(x, y, nnei)
    res, xp, yp = cg.coarse_grain(rec.spike_smoothed[rows], x, y, idx)
    fr = dataio.state_frames(rec, label)
    if fr.size < width:
        return None
    r = net.modularity_from_activity(
        res[:, fr[:width]],
        density=K,
        gamma=GAMMA,
        n_runs=n_runs,
        negative=True,
    )
    ci = net.consensus_partition(r["ci_all"]) if USE_CONSENSUS else r["ci_max"]
    return xp, yp, ci


panels = [
    (module_map(FMAP_SLEEP_REC, "awake", WIN["sleep"]), f"Awake ({FMAP_SLEEP_REC})"),
    (module_map(FMAP_SLEEP_REC, "nrem", WIN["sleep"]), f"NREM ({FMAP_SLEEP_REC})"),
    (module_map(FMAP_ANE_REC, "anesthesia", WIN["ane"]), f"Anesthesia ({FMAP_ANE_REC})"),
]
fig, axes = plt.subplots(1, 3, figsize=(15, 5.2))
for ax, (entry, title) in zip(axes, panels):
    if entry is not None:
        xp, yp, ci = entry
        coords = np.column_stack([xp, yp])
        localization = cg.module_localization_index(coords, ci)
        map_title = (
            f"{title}\n{np.unique(ci).size} modules, {ci.size} parcels, "
            f"localization {localization:.1f}×"
        )
        viz.plot_spatial_modules(
            ax,
            coords,
            ci,
            title=map_title,
            node_size=40,
            show_counts=False,
            edge_linewidth=0.3,
        )
    else:
        ax.set_axis_off()
fig.suptitle(f"(F) Spatial distribution of coarse-grained modules  "
             f"(nnei = {FMAP_SCALE}, K = 5%, "
             f"{'consensus' if USE_CONSENSUS else f'max-Q of {FMAP_N_RUNS} runs'})"
             f" — modules are spatially localized", y=1.02, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "08_all_mice_coarse_grain_modules.png", dpi=140, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Numeric summary: where does the CI first overlap zero?
# This is a descriptive uncertainty threshold, not evidence that the states are
# equivalent or that the effect has vanished.

# %%
for name, aw, un in [("SLEEP (awake−NREM)", sleep_aw, sleep_un),
                     ("ANESTHESIA (awake−anesthesia)", ane_aw, ane_un)]:
    m, lo, hi = stat_utils.mean_confidence_interval(aw - un, axis=0)
    crossed = next((SCALES[i] for i in range(len(SCALES)) if lo[i] <= 0 <= hi[i]), None)
    print(f"{name}:")
    for i, s in enumerate(SCALES):
        star = "  <- CI includes 0" if lo[i] <= 0 <= hi[i] else ""
        print(f"   nnei={s:>3}:  ΔQ={m[i]:+.3f}  [{lo[i]:+.3f}, {hi[i]:+.3f}]{star}")
    print(f"   -> CI first includes zero at nnei = {crossed} (not an equivalence test)\n")

# %% [markdown]
# ## Takeaway
# At the **single-cell** scale the unconscious network is more modular (scripts
# 03/07). At modest coarse-graining the contrast weakens and its interval first
# overlaps zero, while modules become **spatially localized**. At the coarsest
# scales only a handful of parcels/edges remain, and the contrast can reverse;
# those estimates are discrete and unstable. The scale dependence is therefore
# descriptive rather than proof that the state effect vanishes (Fig. 7).
