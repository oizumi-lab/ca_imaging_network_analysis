# %% [markdown]
# # OQ2 — What causes the awake -> unconscious clustering difference in the
# #        shuffle null, under *identical* smoothing?
#
# The circular-shift shuffle destroys coupling but keeps each neuron's own trace,
# and its clustering C is still higher for unconscious states (the confound
# quantified in ``why_QL_robust_C_confounded.py``). The 15-frame Gaussian
# smoothing is identical across states, so the difference must come from a
# *marginal* property the shuffle preserves. This script pins it down rigorously:
#
# The essential property is **temporal sparsity**: under unconsciousness most
# neurons fire only a handful of events, and sparse activity inflates the chance
# clustering (``sparsity_clustering_mechanism.py`` derives *why*, and shows that
# per-neuron kurtosis is merely a proxy for sparsity, ~ 1/event-count — not an
# essential variable). This script:
#
# 1. **Which marginals differ by state?** Unconscious states are **sparser**
#    (fewer events), while the trace **autocorrelation is unchanged** — i.e. the
#    smoothing really is identical.
# 2. **Does the sparsity track the confound?** Across 10 recordings the shuffle
#    clustering state-difference is tracked by per-neuron kurtosis (Spearman
#    ~ 0.99), used here only as a *tail-weighted summary of sparsity*: it works
#    where the arithmetic-mean event rate fails, because the confound is set by the
#    near-silent majority of neurons (which kurtosis up-weights and the mean rate
#    washes out).
# 3. **Is sparsity sufficient (causal)?** An **independent-signal** model with zero
#    coupling, matched only to each recording's marginal shape, reproduces the
#    clustering state-difference (r ~ 0.96). It over-predicts the magnitude —
#    independent signals produce *more* clustering difference than the data — so
#    coupling is certainly not needed to explain it.

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from scipy.ndimage import gaussian_filter1d

from src.funcnet import dataio, network as net, smallworld as sw
from src.funcnet.paths import FIG_DIR
import verification.shuffle_investigation as si

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Mean of empty slice")
FIG_DIR.mkdir(parents=True, exist_ok=True)

MARG = ["event_rate", "active_frac", "kurtosis", "concentration", "autocorr1"]
MARG_LBL = ["event\nrate", "active\nfraction", "kurtosis\n(sparsity proxy)",
            "activity\nconcentration", "autocorr\n(smoothing)"]

# %%
R = si.load_or_compute()
SLEEP, ANE, ALL = si.SLEEP_RECS, si.ANE_RECS, si.SLEEP_RECS + si.ANE_RECS
C_IDX = 1  # clustering is column 1 of the [Q, C, L] vectors


def unc(n):
    return R[n][R[n]["unc_label"]]


def aw(n):
    return R[n]["awake"]


def dmarg(recs, key):
    return np.array([unc(n)[key] - aw(n)[key] for n in recs])


def dshufC(recs):
    return np.array([unc(n)["shuf"][:, C_IDX].mean() - aw(n)["shuf"][:, C_IDX].mean() for n in recs])


# %% [markdown]
# ## Step 1 — marginals by state: sparser & burstier, but same smoothing

# %%
print("=" * 74)
print("Marginals: unconscious vs awake (paired, n=10)")
print("=" * 74)
pct, stars = [], []
for key in MARG:
    a = np.array([aw(n)[key] for n in ALL])
    u = np.array([unc(n)[key] for n in ALL])
    t, p = stats.ttest_rel(u, a)
    pc = np.mean((u - a) / a) * 100
    pct.append(pc)
    stars.append("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s.")
    print(f"  {key:14s}: awake={a.mean():.4f}  unc={u.mean():.4f}  "
          f"change={pc:+5.1f}%  (p={p:.2g} {stars[-1]})")

# %% [markdown]
# ## What these statistics ARE (all per-neuron, over time — then averaged)
#
# Every marginal above is a property of **one neuron's own trace across frames**,
# computed for each neuron and then averaged over the population — not a
# cross-neuron or spike-rate summary. On this dataset's signals (``spike_deconv``
# = deconvolved events; ``spike_smoothed`` = the Gaussian-smoothed trace used to
# build the network):
#
# | statistic | signal | per-neuron definition | high value = |
# |---|---|---|---|
# | **event rate**    | spike_deconv   | # event onsets (0→nonzero) / #frames                 | fires often |
# | **active frac**   | spike_deconv   | fraction of frames with an event                     | active much of the time |
# | **kurtosis**      | spike_smoothed | 4th standardised moment of the trace                 | a few big peaks, mostly flat (bursty) |
# | **concentration** | spike_smoothed | share of the neuron's total activity in its top-5% frames | activity packed into few frames (bursty) |
# | **autocorr(lag1)**| spike_smoothed | corr(x_t, x_{t-1})                                    | wide smoothing bumps (set by the filter; ~equal across states) |
#
# Sparser firing (lower event rate), at the *same* smoothing, turns each trace into
# a few isolated bumps on a flat baseline → higher kurtosis and concentration. The
# next figure makes this visible on real neurons.

# %% [markdown]
# ## See it: raster + example neurons + population spread
# One recording (mouse05_ane), awake vs anaesthesia, the **same neurons** in both
# states. Row 1: event raster (each tick = one deconvolved event onset) — anaesthesia
# is visibly sparser. Row 2: three example neurons' smoothed traces (each normalised
# to its own peak, so only the *shape* shows) — the same cell goes from many small
# bumps to a few tall ones, with its own kurtosis / event rate printed. Row 3: the
# per-neuron distribution of each statistic across the whole population, with the
# mean (the number used above) marked.

# %%
EX_REC = "mouse05_ane"
rec = dataio.load_recording(EX_REC)
keep = np.flatnonzero(rec.nonzero_ROI)
LAB_AW, LAB_UN = rec.state_labels                       # 'awake', 'anesthesia'
W = si.WIN[rec.data_info]                                # same window as the analysis (2900 for ane)
frames = {lab: dataio.state_frames(rec, lab)[:W] for lab in (LAB_AW, LAB_UN)}
fs = rec.fs


def per_neuron_stats(rows, lab):
    """The per-neuron marginals, computed exactly as in shuffle_investigation.py."""
    fr = frames[lab]
    sm = rec.spike_smoothed[np.ix_(rows, fr)]
    dc = rec.spike_deconv[np.ix_(rows, fr)]
    onsets = (dc[:, 1:] > 0) & (dc[:, :-1] == 0)
    n_ev = onsets.sum(1) + (dc[:, :1] > 0).ravel()                       # # event onsets
    ev_hz = n_ev / fr.size * fs                                          # events / second
    kurt = stats.kurtosis(sm, axis=1, fisher=True)
    top = int(np.ceil(0.05 * fr.size))
    Xs = np.sort(sm, axis=1)[:, ::-1]
    tot = sm.sum(1)
    conc = Xs[:, :top].sum(1) / np.where(tot > 0, tot, 1)
    return {"ev_hz": ev_hz, "n_ev": n_ev, "kurt": kurt, "conc": conc}


stA = per_neuron_stats(keep, LAB_AW)
stU = per_neuron_stats(keep, LAB_UN)

# raster subset (same neurons both states)
rng = np.random.RandomState(1)
raster_rows = np.sort(rng.choice(keep, 45, replace=False))


def onset_times(rows, lab):
    dc = rec.spike_deconv[np.ix_(rows, frames[lab])]
    on = (dc[:, 1:] > 0) & (dc[:, :-1] == 0)
    return [np.flatnonzero(on[i]) / fs for i in range(rows.size)]


# 3 example neurons: busy when awake, still firing (not silenced) and clearly
# burstier under anaesthesia -- so the panels show fewer/peakier events, not silence
rise = np.nan_to_num(stU["kurt"] - stA["kurt"], nan=-np.inf)
qualifies = (stA["n_ev"] >= 25) & (stU["n_ev"] >= 8) & (rise > 40)
score = np.where(qualifies, stA["n_ev"], -np.inf)      # busiest-awake among qualifying
ex_local = np.argsort(score)[::-1][:3]
ex_rows = keep[ex_local]

print(f"[{EX_REC}] population means (per-neuron, averaged; window={W} frames):")
print(f"  awake       : {stA['ev_hz'].mean():.3f} events/s   kurtosis {np.nanmean(stA['kurt']):5.0f}   "
      f"concentration {stA['conc'].mean():.2f}")
print(f"  anaesthesia : {stU['ev_hz'].mean():.3f} events/s   kurtosis {np.nanmean(stU['kurt']):5.0f}   "
      f"concentration {stU['conc'].mean():.2f}")

# %%
ZOOM_S = 60                                             # seconds shown in the trace panels
fig = plt.figure(figsize=(15, 14))
gs = fig.add_gridspec(4, 6, height_ratios=[1.15, 1.0, 1.0, 0.9], hspace=0.62, wspace=0.75)
AW_C, UN_C = "#4477aa", "#cc6677"

# --- row 0: event rasters (deconvolved onsets) ----------------------------
for j, (lab, col) in enumerate([(LAB_AW, AW_C), (LAB_UN, UN_C)]):
    ax = fig.add_subplot(gs[0, j * 3:(j + 1) * 3])
    rs = per_neuron_stats(raster_rows, lab)
    ax.eventplot(onset_times(raster_rows, lab), colors=col, linelengths=0.8, linewidths=0.7)
    ax.set_xlim(0, frames[lab].size / fs); ax.set_ylim(-1, raster_rows.size)
    ax.set_xlabel("time (s)"); ax.set_ylabel("neuron (same set)")
    ax.set_title(f"{lab.upper()} — event raster ({raster_rows.size} neurons), "
                 f"mean {rs['ev_hz'].mean():.3f} events/s", fontsize=10)

# --- row 1: the spike_smoothed TRACE of 3 example neurons (zoomed) --------
# This is the Gaussian-smoothed signal that kurtosis is computed on. Zoomed to a
# short window (centred on the cell's largest anaesthesia transient) so the smooth
# bumps are visible: awake = many moderate bumps, anaesthesia = a few tall ones.
for k, (loc, gid) in enumerate(zip(ex_local, ex_rows)):
    ax = fig.add_subplot(gs[1, k * 2:(k + 1) * 2])
    un_full = rec.spike_smoothed[gid, frames[LAB_UN]]
    zfr = int(ZOOM_S * fs)
    f0 = int(np.clip(np.argmax(un_full) - zfr // 2, 0, max(0, un_full.size - zfr)))
    sl = slice(f0, f0 + zfr)
    for lab, col, off in [(LAB_AW, AW_C, 0.0), (LAB_UN, UN_C, 1.25)]:
        tr = rec.spike_smoothed[gid, frames[lab]][sl]
        tr = tr / (tr.max() + 1e-9)
        ax.plot(np.arange(tr.size) / fs, tr + off, color=col, lw=0.8)
    ax.text(0.02, 0.28, f"awake  kurtosis {stA['kurt'][loc]:.0f}", transform=ax.transAxes,
            fontsize=8, color=AW_C)
    ax.text(0.02, 0.92, f"anaes  kurtosis {stU['kurt'][loc]:.0f}", transform=ax.transAxes,
            fontsize=8, color=UN_C)
    ax.set_yticks([]); ax.set_xlabel(f"time (s) — {ZOOM_S}s window")
    ax.set_title(f"neuron #{gid} — spike_smoothed (÷ peak)", fontsize=9)

# --- row 2: the VALUE DISTRIBUTION of that smoothed signal = kurtosis ------
# Kurtosis IS the shape of this histogram: high kurtosis = the value sits near
# baseline almost always, with a heavy tail of rare large excursions.
for k, (loc, gid) in enumerate(zip(ex_local, ex_rows)):
    ax = fig.add_subplot(gs[2, k * 2:(k + 1) * 2])
    for lab, col, name in [(LAB_AW, AW_C, "awake"), (LAB_UN, UN_C, "anaes")]:
        v = rec.spike_smoothed[gid, frames[lab]]
        v = v / (v.max() + 1e-9)
        ax.hist(v, bins=np.linspace(0, 1, 50), density=True, histtype="step",
                color=col, lw=1.5, log=True)
    ax.text(0.30, 0.90, f"awake kurtosis {stA['kurt'][loc]:.0f}", transform=ax.transAxes,
            fontsize=8, color=AW_C)
    ax.text(0.30, 0.78, f"anaes kurtosis {stU['kurt'][loc]:.0f}", transform=ax.transAxes,
            fontsize=8, color=UN_C)
    ax.set_xlabel("smoothed value (÷ peak)"); ax.set_ylabel("density (log)")
    ax.set_title(f"neuron #{gid} — value distribution", fontsize=9)

# --- row 3: per-neuron population distributions ---------------------------
def hist_panel(ax, key, label, xclip=None, xlabel=""):
    a = stA[key][np.isfinite(stA[key])]; u = stU[key][np.isfinite(stU[key])]
    hi = np.nanpercentile(np.concatenate([a, u]), xclip) if xclip else max(a.max(), u.max())
    bins = np.linspace(0, hi, 40)
    ax.hist(np.clip(a, None, hi), bins=bins, color=AW_C, alpha=.55, label="awake")
    ax.hist(np.clip(u, None, hi), bins=bins, color=UN_C, alpha=.55, label="anaesthesia")
    ax.axvline(a.mean(), color=AW_C, lw=1.8); ax.axvline(u.mean(), color=UN_C, lw=1.8)
    ax.set_xlabel(xlabel); ax.set_ylabel("# neurons"); ax.set_title(label, fontsize=9)

hist_panel(fig.add_subplot(gs[3, 0:2]), "ev_hz", "per-neuron event rate", None, "events / s")
hist_panel(fig.add_subplot(gs[3, 2:4]), "kurt", "per-neuron kurtosis", 97, "kurtosis (clipped 97%)")
h3 = fig.add_subplot(gs[3, 4:6]); hist_panel(h3, "conc", "per-neuron concentration", None, "top-5% activity share")
h3.legend(fontsize=8)

fig.suptitle(f"{EX_REC}: per-neuron statistics behind 'sparser & burstier', all from spike_smoothed — "
             "same neurons, awake vs anaesthesia\n"
             "row 1: the smoothed trace  •  row 2: its value distribution (peakiness = kurtosis)  •  "
             "row 3: the population (vertical lines = means)", y=1.0, fontsize=11)
fig.savefig(FIG_DIR / "oq2_burstiness_examples.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Step 2 — which marginal tracks the shuffle-clustering state difference?
# Spearman (robust to the two extreme-anaesthesia points). Kurtosis tracks it best
# — as a tail-weighted proxy for sparsity, not as a fundamental variable (see
# ``sparsity_clustering_mechanism.py``); it also holds *within* the sleep group
# alone, so it is not an anaesthesia artefact.

# %%
dC = dshufC(ALL)
rho = {}
print("\nSpearman rho( Δmarginal , ΔshuffleC )  across 10 recordings:")
for key in MARG:
    r, p = stats.spearmanr(dmarg(ALL, key), dC)
    rho[key] = r
    print(f"  {key:14s}: rho={r:+.2f} (p={p:.3g})")
rk_sleep, pk_sleep = stats.spearmanr(dmarg(SLEEP, "kurtosis"), dshufC(SLEEP))
print(f"  kurtosis, SLEEP only (n=6): rho={rk_sleep:+.2f} (p={pk_sleep:.3g})")

# %% [markdown]
# ## Step 3 — causal test: an independent-signal model matched only to the marginal
# For each recording we simulate ``N`` INDEPENDENT neurons (no coupling), Gaussian-
# smoothed, at the recording's own window length ``T`` (window length also affects
# the absolute shuffle clustering), and tune the event rate so the simulated
# marginal shape (its kurtosis, our sparsity proxy) matches the measured one. The
# simulated clustering is the predicted shuffle-C. Compare predicted vs observed.

# %%
NSIM, SIG, KK = 1500, 5, 0.05
LAM = np.array([0.03, 0.02, 0.014, 0.010, 0.007, 0.005, 0.0035, 0.0025, 0.0017, 0.0011, 0.0007])


def sim_kurt_C(lam, T, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((NSIM, T))
    for i in range(NSIM):
        k = rng.poisson(lam * T)
        X[i, rng.integers(0, T, k)] += rng.exponential(1.0, k)
    X = gaussian_filter1d(X, SIG, axis=1)
    kurt = float(np.nanmean(stats.kurtosis(X, axis=1, fisher=True)))
    adj, _ = net.density_threshold(net.correlation_matrix(X), KK, negative=True)
    return kurt, sw.avg_clustering(adj)


grids = {}
for T in sorted(set(si.WIN.values())):
    ks = np.array([sim_kurt_C(l, T, 7) for l in LAM])
    order = np.argsort(ks[:, 0])
    grids[T] = (ks[order, 0], ks[order, 1])   # (kurtosis, C), ascending kurtosis
    print(f"  synthetic grid T={T}: kurt {ks[order,0][0]:.0f}..{ks[order,0][-1]:.0f}  "
          f"C {ks[order,1][0]:.3f}..{ks[order,1][-1]:.3f}")


def predict_C(kurt, T):
    kk, cc = grids[T]
    return float(np.interp(kurt, kk, cc))


obs_dC, pred_dC, rows_tbl = [], [], []
for n in ALL:
    T = si.WIN[R[n]["kind"]]
    ca_o, cu_o = aw(n)["shuf"][:, C_IDX].mean(), unc(n)["shuf"][:, C_IDX].mean()
    ca_p, cu_p = predict_C(aw(n)["kurtosis"], T), predict_C(unc(n)["kurtosis"], T)
    obs_dC.append(cu_o - ca_o)
    pred_dC.append(cu_p - ca_p)
    rows_tbl.append((n, T, aw(n)["kurtosis"], unc(n)["kurtosis"], ca_o, cu_o, ca_p, cu_p))
obs_dC, pred_dC = np.array(obs_dC), np.array(pred_dC)
r_rep, p_rep = stats.pearsonr(pred_dC, obs_dC)
slope = np.polyfit(pred_dC, obs_dC, 1)[0]
print(f"\nreproduction: predicted vs observed ΔshuffleC  r={r_rep:+.2f} (p={p_rep:.2g}), "
      f"slope={slope:.2f}  (model over-predicts magnitude {1/slope:.1f}x)")

# %% [markdown]
# ## Figure

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.32)
SLEEP_C, ANE_C = "#4477aa", "#cc6677"


def dataset_color(n):
    return SLEEP_C if R[n]["kind"] == "sleep" else ANE_C


# (a) marginal % change by state
axa = fig.add_subplot(gs[0, 0])
bars = axa.bar(range(len(MARG)), pct,
               color=["#cc6677" if p > 0 else "#4477aa" for p in pct])
for i, (b, s) in enumerate(zip(bars, stars)):
    axa.text(b.get_x() + b.get_width() / 2, b.get_height() + (2 if b.get_height() >= 0 else -6),
             s, ha="center", fontsize=9)
axa.axhline(0, color="k", lw=.7)
axa.set_xticks(range(len(MARG)))
axa.set_xticklabels(MARG_LBL, fontsize=8)
axa.set_ylabel("% change (unconscious vs awake)")
axa.set_title("(a) Unconscious: sparser (fewer events),\nsmoothing unchanged", fontsize=11)

# (b) which marginal predicts ΔshuffleC
axb = fig.add_subplot(gs[0, 1])
vals = [abs(rho[k]) for k in MARG]
axb.bar(range(len(MARG)), vals,
        color=["#ddaa33" if k == "kurtosis" else "0.7" for k in MARG])
axb.set_xticks(range(len(MARG)))
axb.set_xticklabels(MARG_LBL, fontsize=8)
axb.set_ylabel("|Spearman rho| with Δshuffle-C")
axb.set_ylim(0, 1)
axb.set_title("(b) Kurtosis (a sparsity proxy)\ntracks the confound", fontsize=11)

# (c) scatter: ΔshuffleC vs Δkurtosis
axc = fig.add_subplot(gs[0, 2])
dk = dmarg(ALL, "kurtosis")
for i, n in enumerate(ALL):
    axc.scatter(dk[i], dC[i], s=40, color=dataset_color(n), edgecolor="k", linewidth=.3)
b1, b0 = np.polyfit(dk, dC, 1)
xs = np.linspace(dk.min(), dk.max(), 20)
axc.plot(xs, b1 * xs + b0, "0.4", ls="--", lw=1)
axc.set_xlabel("Δ kurtosis (unconscious − awake)")
axc.set_ylabel("Δ shuffle-clustering")
axc.set_title(f"(c) Predictor: Spearman rho={rho['kurtosis']:+.2f}", fontsize=11)
axc.scatter([], [], color=SLEEP_C, label="sleep")
axc.scatter([], [], color=ANE_C, label="anaesthesia")
axc.legend(fontsize=8)

# (d) parametric synthetic curve (kurtosis -> C) with real points overlaid
axd = fig.add_subplot(gs[1, 0])
for T, style in [(1500, "-"), (2900, "--")]:
    kk, cc = grids[T]
    axd.plot(kk, cc, style, color="0.5", lw=1.4, label=f"independent model (T={T})")
for n in ALL:
    T = si.WIN[R[n]["kind"]]
    for d, mk in [(aw(n), "o"), (unc(n), "s")]:
        axd.scatter(d["kurtosis"], d["shuf"][:, C_IDX].mean(), s=32, marker=mk,
                    color=dataset_color(n), edgecolor="k", linewidth=.3)
axd.set_xlabel("per-neuron kurtosis")
axd.set_ylabel("shuffle-clustering C")
axd.set_title("(d) Real points track the zero-coupling\nmodel at their own T", fontsize=11)
axd.plot([], [], "ko", label="awake"); axd.plot([], [], "ks", label="unconscious")
axd.legend(fontsize=7, loc="upper left")

# (e) matched-sim reproduction: predicted vs observed ΔshuffleC
axe = fig.add_subplot(gs[1, 1])
for i, n in enumerate(ALL):
    axe.scatter(pred_dC[i], obs_dC[i], s=44, color=dataset_color(n), edgecolor="k", linewidth=.3)
lim = max(pred_dC.max(), obs_dC.max()) * 1.1
axe.plot([0, lim], [0, lim], "0.5", ls="--", lw=1, label="y=x")
axe.set_xlim(0, lim); axe.set_ylim(0, lim)
axe.set_xlabel("predicted Δ shuffle-C (independent model)")
axe.set_ylabel("observed Δ shuffle-C")
axe.set_title(f"(e) Sparsity reproduces the difference\nr={r_rep:.2f} (over-predicts {1/slope:.1f}x)", fontsize=11)
axe.legend(fontsize=8)

# (f) per-recording observed vs predicted ΔshuffleC
axf = fig.add_subplot(gs[1, 2])
order = np.argsort(obs_dC)
xx = np.arange(len(ALL))
axf.bar(xx - 0.2, obs_dC[order], 0.4, color="0.35", label="observed")
axf.bar(xx + 0.2, pred_dC[order], 0.4, color="#ddaa33", label="predicted (kurtosis-matched)")
axf.set_xticks(xx)
axf.set_xticklabels([ALL[i].replace("_sleep", "").replace("mouse", "m").replace("_", "")
                     for i in order], rotation=60, ha="right", fontsize=7)
axf.set_ylabel("Δ shuffle-clustering")
axf.set_title("(f) Per recording", fontsize=11)
axf.legend(fontsize=8)

fig.suptitle("OQ2 — the clustering confound and its state difference are driven by temporal "
             "SPARSITY (kurtosis is only a proxy for it),\nnot coupling and not a smoothing difference",
             y=1.0, fontsize=13)
fig.savefig(FIG_DIR / "oq2_state_difference_cause.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Conclusion
# * Under identical smoothing (autocorrelation unchanged, panel a), unconscious
#   states are **sparser** — most neurons fire only a handful of events. Temporal
#   **sparsity** is the cause of the clustering confound.
# * **Kurtosis is not essential — it is a proxy for that sparsity.** Per neuron,
#   kurtosis ~ 1/event-count, so it up-weights the near-silent majority of cells;
#   that is why it tracks the confound across recordings (Spearman ~0.99 overall,
#   ~0.94 within sleep, surviving a partial correlation for window length T) while
#   the arithmetic-mean event rate — dominated by the busy minority — does not.
#   ``sparsity_clustering_mechanism.py`` shows the kurtosis≈1/events relation and
#   the mechanism directly. (Kurtosis also depends on the smoothing width, so it is
#   the operative proxy only because the smoothing is pinned across states.)
# * An **independent-signal model** (zero coupling) matched only to the marginal
#   shape reproduces the clustering state-difference across recordings (r ~ 0.96,
#   panels e/f), **over-predicting** the magnitude ~1.5× — a coupling-free null
#   generating *more* difference than the data, so coupling is not needed to
#   explain it. Window length T shifts the absolute level (panel d), which is why
#   the confound looks larger for the longer anaesthesia windows.
# * Mechanism (see ``sparsity_clustering_mechanism.py`` and OQ1): sparse activity
#   makes a single shared frame dominate each pair's correlation, so chance
#   coincidences at that frame create triangle cliques → inflated *local*
#   clustering, with no genuine modules or shortcuts.
