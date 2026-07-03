# %% [markdown]
# # 40 · Small-worldness: path length, clustering, and small-world propensity
#
# Modularity (script 30) measured **functional segregation**. Here we measure the
# complementary side — **integration** — with three classic graph metrics and
# ask how they change between wakefulness and unconsciousness:
#
# - **Clustering coefficient C** — how often a neuron's neighbours are also
#   connected to each other (local, "cliquey" structure).
# - **Characteristic path length L** — the average shortest-path distance between
#   neurons (global reach; short L = efficient integration).
# - **Small-world-ness** and **Small-World Propensity (SWP)** — how the network
#   sits between a random graph (short L, low C) and a ring lattice (long L,
#   high C).
#
# We reproduce the analysis in `oizumi-lab/mouse_network` (`kiyooka/SWP/`), which
# implements Muldoon, Bridgeford & Bassett (2016). Formulas, comparing the
# network to a **random** null (rand) and a **lattice** null (reg):
#
# ```
# small-world-ness  SMN = (C_net/C_rand) / (L_net/L_rand)
# ΔC = (C_reg − C_net)/(C_reg − C_rand)      1/ΔC grows as the net gets more clustered
# ΔL = (L_net − L_rand)/(L_reg − L_rand)      grows as paths get longer than random
# SWP = 1 − sqrt(ΔC² + ΔL²)/sqrt(2)
# ```
#
# **Pipeline** (from `sw_summary.m`): correlation → |r| → binary threshold at a
# fixed density (K = 1%) → largest connected component → the metrics above.
#
# We restrict to **active neurons** (`nonzero_ROI`, dataset README §2.9) — as in
# scripts 20/30. This matters most for anesthesia: without it, the many neurons
# silent under anesthesia fragment the 1%-density graph, shrinking the connected
# component and distorting the null comparison.

# %%
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt

from src.funcnet import dataio, network as net, smallworld as sw
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# --- analysis settings (match the reference driver) ---
DENSITY = 0.01      # 1% connection density (binary), as in script_20251218_calc_small_world.m
N_SOURCES = 800     # sample this many source nodes for path length (None = exact, ~4x slower)
METHOD = "O"        # Onnela clustering (= binary clustering on a binary graph)

# %% [markdown]
# ## One network, one state: the anatomy of the calculation
# Take a 1500-frame awake window of `mouse01_sleep`, build the 1%-density binary
# graph, and compute the network's C and L alongside its lattice and random
# nulls. `sw_summary` returns everything in one `SWResult`.

# %%
rec = dataio.load_recording("mouse01_sleep")
keep = rec.nonzero_ROI              # active neurons only (the paper's network set)
Xa = rec.spike_smoothed[keep][:, dataio.state_frames(rec, "awake")[:1500]]
res = sw.sw_summary(net.correlation_matrix(Xa), density=DENSITY,
                    n_sources=N_SOURCES, rng=np.random.RandomState(1))

print(f"largest connected component: {res.n} neurons\n")
print(f"                   network    lattice    random")
print(f"  clustering C  :  {res.net_clus:7.3f}   {res.reg_clus:7.3f}   {res.rand_clus:7.3f}")
print(f"  path length L :  {res.net_path:7.3f}   {res.reg_path:7.3f}   {res.rand_path:7.3f}")
print(f"\n  small-world-ness (SMN) = {res.sw_ness:.2f}")
print(f"  ΔL = {res.delta_L:.4f}   1/ΔC = {1/res.delta_C:.3f}   SWP = {res.SWP:.3f}")
print("\nNote C_net ≫ C_rand but L_net ≈ L_rand → a small-world network (SMN ≫ 1).")

# %% [markdown]
# ## Awake vs NREM in one mouse
# The paper's claim: during unconsciousness the network becomes **more
# clustered** and its **paths lengthen** — i.e. more locally segregated, less
# globally integrated. Compare the two states of `mouse01_sleep`.

# %%
for label in rec.state_labels:
    X = rec.spike_smoothed[keep][:, dataio.state_frames(rec, label)[:1500]]
    r = sw.sw_summary(net.correlation_matrix(X), density=DENSITY,
                      n_sources=N_SOURCES, rng=np.random.RandomState(1))
    print(f"  {label:<11}: C={r.net_clus:.3f}  L={r.net_path:.3f}  "
          f"SMN={r.sw_ness:5.2f}  ΔL={r.delta_L:.4f}  1/ΔC={1/r.delta_C:.3f}")
print("\n→ NREM: higher C, longer L, higher SMN — the unconscious signature.")

# %% [markdown]
# ## Reproduce the talk figures across all recordings
# We compute the three summary measures for every 1500-frame window of every
# recording and both states, then plot them per mouse (awake vs unconscious),
# reproducing the "Small-worldness" and "path length / clustering" slides.
#
# This is the heavy cell. With ``QUICK = True`` it runs on a subset (~4 min);
# set ``QUICK = False`` (or run with the env var ``SWP_FULL=1``) for the full
# replication across all 6 sleep + 4 anesthesia recordings (~15 min).

# %%
QUICK = os.environ.get("SWP_FULL") != "1"

SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]
if QUICK:
    SLEEP_RECS, ANE_RECS = SLEEP_RECS[:3], ANE_RECS[:2]
    MAX_WINDOWS = 2
else:
    MAX_WINDOWS = None

MEASURES = ["sw_ness", "delta_L", "inv_delta_C"]
MEASURE_LABELS = {"sw_ness": "Small-world-ness",
                  "delta_L": "Path length  (ΔL)",
                  "inv_delta_C": "Clustering  (1/ΔC)"}


def windows_of(idx, width, cap):
    n = idx.size // width
    if cap is not None:
        n = min(n, cap)
    return [idx[i * width:(i + 1) * width] for i in range(n)]


def recording_measures(name, width):
    """Per-window small-world measures for each state of one recording."""
    rec = dataio.load_recording(name)
    keep = rec.nonzero_ROI          # active neurons only (README §2.9)
    out = {}
    for si, label in enumerate(rec.state_labels):
        vals = {m: [] for m in MEASURES}
        for k, win in enumerate(windows_of(dataio.state_frames(rec, label), width, MAX_WINDOWS)):
            C = net.correlation_matrix(rec.spike_smoothed[keep][:, win])
            r = sw.sw_summary(C, density=DENSITY, n_sources=N_SOURCES,
                              rng=np.random.RandomState(1000 * si + k))
            vals["sw_ness"].append(r.sw_ness)
            vals["delta_L"].append(r.delta_L)
            vals["inv_delta_C"].append(1.0 / r.delta_C)
        out[label] = vals
        print(f"    {name} [{label}]: {len(vals['sw_ness'])} windows", flush=True)
    return out


def run_dataset(recs, width):
    print(f"  dataset ({len(recs)} recordings):", flush=True)
    return {name: recording_measures(name, width) for name in recs}


print("SLEEP (awake vs NREM):")
sleep_data = run_dataset(SLEEP_RECS, 1500)
print("ANESTHESIA (awake vs anesthesia):")
ane_data = run_dataset(ANE_RECS, 2900)

# %% [markdown]
# ### Plot: per-mouse scatter, awake vs unconscious
# Each dot is one window; the black "Average" column shows the per-state means
# joined across states. Unconscious (red/orange) sits above awake (blue).

# %%
def scatter_panel(ax, data, measure, unconscious_color, unconscious_name):
    names = list(data)
    aw_means, un_means = [], []
    for x, name in enumerate(names):
        states = data[name]
        aw = np.array(states["awake"][measure])
        un = np.array(states[list(states)[1]][measure])
        ax.scatter(np.full(aw.size, x) + np.random.uniform(-.08, .08, aw.size), aw,
                   s=18, color="royalblue", zorder=3)
        ax.scatter(np.full(un.size, x) + np.random.uniform(-.08, .08, un.size), un,
                   s=18, color=unconscious_color, zorder=3)
        aw_means.append(aw.mean()); un_means.append(un.mean())
    xa = len(names)
    ax.scatter(np.full(len(aw_means), xa) - .05, aw_means, s=22, color="royalblue", zorder=3)
    ax.scatter(np.full(len(un_means), xa) + .05, un_means, s=22, color=unconscious_color, zorder=3)
    for a, u in zip(aw_means, un_means):  # connect each mouse's means
        ax.plot([xa - .05, xa + .05], [a, u], color="k", lw=.7, zorder=2)
    ax.set_xticks(list(range(len(names))) + [xa])
    ax.set_xticklabels([n.replace("mouse", "M").replace("_sleep", "").replace("_ane", "")
                        for n in names] + ["Avg"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(MEASURE_LABELS[measure])


fig, axes = plt.subplots(len(MEASURES), 2, figsize=(13, 11))
for row, measure in enumerate(MEASURES):
    scatter_panel(axes[row, 0], sleep_data, measure, "crimson", "NREM")
    scatter_panel(axes[row, 1], ane_data, measure, "goldenrod", "Anesthesia")
    axes[row, 0].set_title("Wakefulness vs NREM" if row == 0 else "")
    axes[row, 1].set_title("Wakefulness vs Anesthesia" if row == 0 else "")
fig.suptitle("Small-world metrics are higher during unconsciousness"
             + ("   [QUICK subset]" if QUICK else ""), y=0.995, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "40_small_world.png", dpi=140)
plt.show()
print("saved ->", FIG_DIR / "40_small_world.png")

# %% [markdown]
# ### Summary at a glance
# Mean of each measure per state, pooled across recordings/windows.

# %%
for title, data in [("SLEEP", sleep_data), ("ANESTHESIA", ane_data)]:
    pooled = {"awake": {m: [] for m in MEASURES}, "unconscious": {m: [] for m in MEASURES}}
    for states in data.values():
        for m in MEASURES:
            pooled["awake"][m] += states["awake"][m]
            pooled["unconscious"][m] += states[list(states)[1]][m]
    print(f"\n{title}:")
    for m in MEASURES:
        a, u = np.mean(pooled["awake"][m]), np.mean(pooled["unconscious"][m])
        print(f"  {MEASURE_LABELS[m]:<22}  awake={a:7.3f}   unconscious={u:7.3f}   "
              f"({'↑ higher' if u > a else '↓ lower'} when unconscious)")

# %% [markdown]
# ## Takeaway
# During NREM sleep and anesthesia the single-cell functional network has
# **higher clustering (1/ΔC), longer path length (ΔL), and higher
# small-world-ness** than during wakefulness — communication becomes more
# **localised** and less globally integrated. This complements the modularity
# result (script 30): the unconscious cortex is more segregated at every level
# we measure.
