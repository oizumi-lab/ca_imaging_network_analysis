# %% [markdown]
# # Shuffle-null control: are the state differences genuine, or a shuffle artifact?
#
# Script 10 shows that the **circular-shift shuffle null** (roll each neuron's
# trace by an independent random lag) has a correlation histogram *similar* to the
# real data. That raises two questions, both tested rigorously here:
#
# 1. **Within a state** — do the network measures (modularity Q, clustering C,
#    path length L) on **shuffled** data resemble those on **real** data? (If so,
#    the measures aren't capturing genuine pairwise structure.)
# 2. **The critical one** — the circular shift **preserves each neuron's own
#    signal** (autocorrelation *and* amplitude/variance). Do the awake-vs-
#    unconscious **differences** in Q/C/L survive shuffling, or are they an
#    artifact of a per-neuron property the shuffle keeps?
#
# **Shuffle-correction** (per state): ``M_corr(state) = M_real(state) −
# mean(M_shuffle(state))`` — the excess over chance. The state comparison of
# corrected values equals ``ΔM_real − ΔM_shuffle``.

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
warnings.filterwarnings("ignore", message="Mean of empty slice")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
# ``MAX_NEURONS`` subsamples active neurons **for every measure, including
# modularity** (so 25×2×10 Louvain fits in a few minutes). It lowers absolute Q
# but not the real-vs-shuffle comparison. Set ``None`` for the full neuron set.

# %%
K = 0.05
GAMMA = 1.0
N_RUNS = 5             # Louvain runs per network (max-Q)
N_SHUFFLE = 20         # circular-shift surrogates per state
MAX_NEURONS = 2000     # subsample active neurons (same set across states/shuffles); None = all
N_SOURCES = 400        # sampled sources for path length
SEED = 0

WIN = {"sleep": 1500, "ane": 2900}
SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]

MEASURES = ["Q (modularity)", "C (clustering)", "L (path length)"]


# %%
def neuron_rows(rec):
    keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
    rows = np.flatnonzero(keep)
    if MAX_NEURONS is not None and rows.size > MAX_NEURONS:
        rows = np.sort(np.random.RandomState(0).choice(rows, MAX_NEURONS, replace=False))
    return rows


def circular_shuffle(X, rng):
    """Roll each neuron's trace by an independent random lag (script-10 null):
    destroys cross-neuron timing, preserves each neuron's own signal."""
    out = np.empty_like(X)
    T = X.shape[1]
    for i in range(X.shape[0]):
        out[i] = np.roll(X[i], int(rng.integers(1, T)))
    return out


def measures_of(C):
    """(Q, clustering, path length) of the K-density binary network from corr matrix C."""
    adj, _ = net.density_threshold(C, K, negative=True)     # rank by |r|, as in the paper
    Q = net.repeat_louvain(adj, gamma=GAMMA, n_runs=N_RUNS)["Q_max"]
    Cc = sw.avg_clustering(adj)
    idx = sw.largest_component(adj)
    if idx.size > 2:
        L = sw.characteristic_path_length(adj[np.ix_(idx, idx)],
                                          n_sources=min(N_SOURCES, idx.size),
                                          rng=np.random.RandomState(1))
    else:
        L = np.nan
    return np.array([Q, Cc, L])


def analyze(name, width):
    """Real + shuffled measures for both states of one recording (same neuron set)."""
    rec = dataio.load_recording(name)
    rows = neuron_rows(rec)
    out = {"kind": rec.data_info, "unc_label": rec.state_labels[1]}
    for label in rec.state_labels:
        win = dataio.state_frames(rec, label)[:width]
        X = rec.spike_smoothed[rows][:, win]
        C = net.correlation_matrix(X)
        rng = np.random.default_rng(SEED)
        shuf = np.array([measures_of(net.correlation_matrix(circular_shuffle(X, rng)))
                         for _ in range(N_SHUFFLE)])
        out[label] = {"orig": measures_of(C), "shuf": shuf}
        o = out[label]
        print(f"  {name} [{label}]: Q/C/L={o['orig'][0]:.3f}/{o['orig'][1]:.3f}/{o['orig'][2]:.2f}",
              flush=True)
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
# ## Q1 — within a state, do real measures exceed the shuffle null?
# ``z = (real − mean(shuffle)) / std(shuffle)``. Large ``|z|`` ⇒ real structure.

# %%
print("\n" + "=" * 78)
print("Q1. Real vs shuffle within each state  (z = (real-shuffle_mean)/shuffle_std)")
print("=" * 78)
for m, mname in enumerate(MEASURES):
    zz = [( r[lab]["orig"][m] - r[lab]["shuf"][:, m].mean() ) / r[lab]["shuf"][:, m].std()
          for r in results.values() for lab in ("awake", r["unc_label"])]
    zz = np.array(zz)
    print(f"  {mname:18s}: mean z = {np.nanmean(zz):+6.2f}   (|z|>2 in {np.mean(np.abs(zz)>2)*100:.0f}%)")


# %% [markdown]
# ## Figure 1 — RAW values and shuffle nulls, per state, no subtraction
# For each recording: real value (★) and its shuffle-null distribution (box), shown
# **separately for awake (blue) and unconscious (red)**. This is the un-subtracted
# view you asked for — you can read off each state's real value, its shuffle
# baseline, and the "excess over null" (★ minus box) directly.

# %%
def raw_panel(ax, recs, m):
    ticks, labels = [], []
    for i, name in enumerate(recs):
        r = results[name]
        for j, (lab, color) in enumerate([("awake", "royalblue"),
                                          (r["unc_label"], "crimson")]):
            d = r[lab]
            pos = i * 3 + j
            ax.boxplot([d["shuf"][:, m]], positions=[pos], widths=.7, showfliers=False,
                       patch_artist=True,
                       boxprops=dict(facecolor=color, alpha=.25, edgecolor=color),
                       medianprops=dict(color=color), whiskerprops=dict(color=color),
                       capprops=dict(color=color))
            ax.plot(pos, d["orig"][m], "*", color=color, ms=13, zorder=5,
                    markeredgecolor="k", markeredgewidth=.4)
        ticks.append(i * 3 + .5)
        labels.append(name.replace("_sleep", "").replace("mouse", "m").replace("_", ""))
    ax.set_xticks(ticks); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(MEASURES[m])
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
fig.suptitle("Raw network measures (★) vs shuffle null (box), per state — no subtraction",
             y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "shuffle_null_raw_by_state.png", dpi=140, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## Q2 — does the state DIFFERENCE survive shuffling? (shuffle-corrected)

# %%
def diff_stats(recs, kind_label):
    print("\n" + "=" * 78)
    print(f"Q2. State difference: real vs shuffle-corrected  [{kind_label}]")
    print("=" * 78)
    for m, mname in enumerate(MEASURES):
        rd, sd = [], []
        for name in recs:
            r = results[name]; aw, un = r["awake"], r[r["unc_label"]]
            rd.append(un["orig"][m] - aw["orig"][m])
            sd.append((un["shuf"][:, m] - aw["shuf"][:, m]).mean())
        rd, sd = np.array(rd), np.array(sd)
        excess = rd - sd
        t, p = stats.ttest_1samp(excess, 0.0)
        frac = np.nanmean(sd) / np.nanmean(rd) * 100 if np.nanmean(rd) else np.nan
        print(f"  {mname:18s}: ΔM_real={np.mean(rd):+.4f}  ΔM_shuffle={np.mean(sd):+.4f} ({frac:+.0f}% of real)"
              f"  corrected={np.mean(excess):+.4f} (t={t:+.2f}, p={p:.3g}, n={len(rd)})")


diff_stats(SLEEP_RECS, "SLEEP: awake vs NREM")
diff_stats(ANE_RECS, "ANESTHESIA: awake vs anesthesia")


# %% [markdown]
# ## Figure 2 — the state difference, real vs shuffle-null difference
# ★ = ΔM_real; box = ΔM_shuffle. ★ outside the box ⇒ genuine.

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
    ax.set_ylabel(f"Δ {MEASURES[m]}"); ax.set_title(title, fontsize=10); ax.legend(fontsize=8)


fig, axes = plt.subplots(3, 2, figsize=(14, 12))
for m in range(3):
    diff_panel(axes[m, 0], SLEEP_RECS, m, "SLEEP" if m == 0 else "")
    diff_panel(axes[m, 1], ANE_RECS, m, "ANESTHESIA" if m == 0 else "")
fig.suptitle("Does the awake-vs-unconscious difference survive shuffling?\n"
             "★ real difference vs box = shuffle-null difference (★ outside box ⇒ genuine)",
             y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "shuffle_null_state_difference.png", dpi=140, bbox_inches="tight")
plt.show()


# %% [markdown]
# ## (b) What drives the clustering confound? — see the dedicated mechanism scripts
#
# **Retraction.** An earlier version of this script attributed the clustering
# confound to per-neuron **variance/amplitude heterogeneity → degree heterogeneity**.
# That explanation is **wrong** and has been removed: Pearson correlation is
# scale-invariant, so a neuron's amplitude/variance cannot affect the correlation
# graph at all (demonstrated directly in ``sparsity_clustering_mechanism.py``,
# where sweeping amplitude heterogeneity leaves clustering flat).
#
# **What actually drives it** (rigorously established, with adversarial
# cross-checks, in three companion scripts):
# * ``sparsity_clustering_mechanism.py`` — the driver is **temporal sparsity**: for
#   independent (zero-coupling) neurons, a single chance coincidence gives a large
#   correlation (r ~ 1/√(n_i n_j)), so strong correlations become per-frame
#   coincidence-cliques → high clustering; amplitude is irrelevant.
# * ``state_difference_cause.py`` (OQ2) — across the 10 recordings the confound's
#   awake→unconscious *difference* is predicted by the **fraction of near-silent
#   neurons** (Spearman ~0.99), and *not* by the mean event rate, while the trace
#   autocorrelation (smoothing) is unchanged across states. An event-count-matched
#   zero-coupling model reproduces the difference pattern (r ~ 0.93) — so coupling
#   is not needed to generate it.
# * ``why_QL_robust_C_confounded.py`` (OQ1) — why this hits clustering (local
#   triangle count) far more than modularity or path length (global measures):
#   C ~56% confounded vs Q ~18% vs L ~4%.

# %% [markdown]
# ## Reading the result
# - **Q1**: large ``|z|`` ⇒ each measure reflects real structure (not the shuffle).
# - **Figure 1 / Q2**: compare each state's ★ to its box, and ΔM_real to ΔM_shuffle.
#   ``ΔM_shuffle ≈ 0`` ⇒ genuine; a large fraction ⇒ shuffle-reproducible confound.
#   Here clustering C is substantially confounded, modularity Q less so, and path
#   length L barely at all — the mechanism is dissected in the scripts named in (b).
