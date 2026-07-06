# %% [markdown]
# # Shuffle-corrected small-world measures (C, L, SWP)
#
# Companion to ``shuffle_null_control.py``, for the **small-world** pipeline of
# script 40 (correlation → ``|r|`` → **K = 1 %** binary graph → largest connected
# component → clustering C, path length L, small-world propensity SWP).
#
# It reports each measure as **excess over the circular-shift shuffle null**,
# separately for each state:
#
#     M_corrected(state) = M_real(state) − mean(M_shuffle(state))
#
# and asks whether the awake-vs-unconscious difference survives the correction —
# i.e. whether the small-world signatures (higher C, longer L, higher SWP during
# unconsciousness) are genuine or reproduced by time-shuffled data.

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

from src.funcnet import dataio, network as net, smallworld as sw
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %%
DENSITY = 0.01         # small-world density (as in script 40)
N_SHUFFLE = 15         # circular-shift surrogates per state
MAX_NEURONS = 2000     # subsample active neurons (same set across states/shuffles); None = all
N_SOURCES = 400        # sampled sources for path length
SEED = 0

WIN = {"sleep": 1500, "ane": 2900}
SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]

SW_MEASURES = ["C (clustering)", "L (path length)", "SWP"]


# %%
def neuron_rows(rec):
    keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
    rows = np.flatnonzero(keep)
    if MAX_NEURONS is not None and rows.size > MAX_NEURONS:
        rows = np.sort(np.random.RandomState(0).choice(rows, MAX_NEURONS, replace=False))
    return rows


def circular_shuffle(X, rng):
    out = np.empty_like(X)
    T = X.shape[1]
    for i in range(X.shape[0]):
        out[i] = np.roll(X[i], int(rng.integers(1, T)))
    return out


def sw_measures(corr, rng):
    """(clustering, path length, SWP) of the 1%-density small-world network."""
    r = sw.sw_summary(corr, density=DENSITY, n_sources=N_SOURCES, rng=rng)
    return np.array([r.net_clus, r.net_path, r.SWP])


def analyze(name, width):
    rec = dataio.load_recording(name)
    rows = neuron_rows(rec)
    out = {"unc_label": rec.state_labels[1]}
    for label in rec.state_labels:
        win = dataio.state_frames(rec, label)[:width]
        X = rec.spike_smoothed[rows][:, win]
        real = sw_measures(net.correlation_matrix(X), np.random.RandomState(0))
        rng = np.random.default_rng(SEED)
        shuf = np.array([sw_measures(net.correlation_matrix(circular_shuffle(X, rng)),
                                     np.random.RandomState(100 + s)) for s in range(N_SHUFFLE)])
        out[label] = {"orig": real, "shuf": shuf}
        print(f"  {name} [{label}]: C/L/SWP real = {real[0]:.3f}/{real[1]:.3f}/{real[2]:.3f}"
              f"   shuffle = {shuf[:,0].mean():.3f}/{shuf[:,1].mean():.3f}/{shuf[:,2].mean():.3f}", flush=True)
    return out


# %% [markdown]
# ## Compute (heavy cell)

# %%
results = {}
for kind, recs in [("sleep", SLEEP_RECS), ("ane", ANE_RECS)]:
    print(f"{kind.upper()} dataset:", flush=True)
    for name in recs:
        results[name] = analyze(name, WIN[kind])


# %% [markdown]
# ## Raw values and shuffle nulls, per state (no subtraction)
# ★ = real, box = shuffle null, for awake (blue) and unconscious (red) separately.

# %%
def raw_panel(ax, recs, m):
    ticks, labels = [], []
    for i, name in enumerate(recs):
        r = results[name]
        for j, (lab, color) in enumerate([("awake", "royalblue"), (r["unc_label"], "crimson")]):
            d = r[lab]; pos = i * 3 + j
            ax.boxplot([d["shuf"][:, m]], positions=[pos], widths=.7, showfliers=False,
                       patch_artist=True, boxprops=dict(facecolor=color, alpha=.25, edgecolor=color),
                       medianprops=dict(color=color), whiskerprops=dict(color=color),
                       capprops=dict(color=color))
            ax.plot(pos, d["orig"][m], "*", color=color, ms=13, zorder=5,
                    markeredgecolor="k", markeredgewidth=.4)
        ticks.append(i * 3 + .5)
        labels.append(name.replace("_sleep", "").replace("mouse", "m").replace("_", ""))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(SW_MEASURES[m])
    ax.plot([], [], "*", color="royalblue", label="Awake real")
    ax.plot([], [], "s", color="royalblue", alpha=.3, label="Awake shuffle")
    ax.plot([], [], "*", color="crimson", label="Unconscious real")
    ax.plot([], [], "s", color="crimson", alpha=.3, label="Unconscious shuffle")


fig, axes = plt.subplots(3, 2, figsize=(15, 12))
for m in range(3):
    raw_panel(axes[m, 0], SLEEP_RECS, m)
    raw_panel(axes[m, 1], ANE_RECS, m)
axes[0, 0].set_title("SLEEP (awake vs NREM)"); axes[0, 1].set_title("ANESTHESIA (awake vs anesthesia)")
axes[0, 0].legend(fontsize=8, ncol=2, loc="best")
fig.suptitle("Small-world measures (K=1%): raw (★) vs shuffle null (box), per state — no subtraction",
             y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "smallworld_shuffle_raw_by_state.png", dpi=140, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Shuffle-corrected state comparison
# ``ΔM_real`` vs ``ΔM_shuffle`` and the shuffle-corrected difference, per measure.

# %%
def summary(recs, kind_label):
    print("\n" + "=" * 78)
    print(f"Small-world, shuffle-corrected  [{kind_label}]")
    print("=" * 78)
    for m, mname in enumerate(SW_MEASURES):
        # within-state z (real vs shuffle), and the state difference
        z = [(results[n][lab]["orig"][m] - results[n][lab]["shuf"][:, m].mean())
             / results[n][lab]["shuf"][:, m].std() for n in recs for lab in ("awake", results[n]["unc_label"])]
        rd = np.array([results[n][results[n]["unc_label"]]["orig"][m] - results[n]["awake"]["orig"][m] for n in recs])
        sd = np.array([(results[n][results[n]["unc_label"]]["shuf"][:, m]
                        - results[n]["awake"]["shuf"][:, m]).mean() for n in recs])
        excess = rd - sd
        t, p = stats.ttest_1samp(excess, 0.0)
        frac = np.nanmean(sd) / np.nanmean(rd) * 100 if np.nanmean(rd) else np.nan
        print(f"  {mname:16s}: within-state |z| real vs shuffle = {np.nanmean(np.abs(z)):5.1f}")
        print(f"  {'':16s}  ΔM_real={np.mean(rd):+.4f}  ΔM_shuffle={np.mean(sd):+.4f} ({frac:+.0f}% of real)"
              f"  corrected={np.mean(excess):+.4f} (t={t:+.2f}, p={p:.3g}, n={len(rd)})")


summary(SLEEP_RECS, "SLEEP: awake vs NREM")
summary(ANE_RECS, "ANESTHESIA: awake vs anesthesia")


# %% [markdown]
# ## Figure — shuffle-corrected difference per measure
# ★ = ΔM_real; box = ΔM_shuffle. ★ well outside the box ⇒ the small-world state
# difference is genuine, not reproduced by time-shuffled data.

# %%
def diff_panel(ax, recs, m, title):
    boxes = [(results[n][results[n]["unc_label"]]["shuf"][:, m] - results[n]["awake"]["shuf"][:, m])
             for n in recs]
    reals = [results[n][results[n]["unc_label"]]["orig"][m] - results[n]["awake"]["orig"][m] for n in recs]
    ax.boxplot(boxes, positions=range(len(recs)), widths=.6, showfliers=False, patch_artist=True,
               boxprops=dict(facecolor="0.85", edgecolor="0.5"), medianprops=dict(color="0.5"))
    ax.plot(range(len(recs)), reals, "*", color="crimson", ms=14, label="ΔM real", zorder=5)
    ax.axhline(0, color="k", lw=.8)
    ax.set_xticks(range(len(recs)))
    ax.set_xticklabels([n.replace("_sleep", "").replace("mouse", "m").replace("_", "") for n in recs],
                       rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"Δ {SW_MEASURES[m]}"); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)


fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for m in range(3):
    diff_panel(axes[m, 0], SLEEP_RECS, m, "SLEEP" if m == 0 else "")
    diff_panel(axes[m, 1], ANE_RECS, m, "ANESTHESIA" if m == 0 else "")
fig.suptitle("Small-world state differences: real (★) vs shuffle null (box)", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "smallworld_shuffle_state_difference.png", dpi=140, bbox_inches="tight")
plt.show()
