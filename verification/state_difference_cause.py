# %% [markdown]
# # OQ2 — What causes the awake -> unconscious clustering difference in the
# #        shuffle null, under *identical* smoothing?
#
# The circular-shift shuffle destroys coupling but keeps each neuron's own trace,
# and its clustering C is still higher for unconscious states (the confound
# quantified in ``why_QL_robust_C_confounded.py``). The 15-frame Gaussian smoothing
# is identical across states, so the difference must come from a *marginal*
# property the shuffle preserves. That property is **temporal sparsity**: under
# unconsciousness most neurons fire only a handful of events, and sparse activity
# inflates the chance clustering. ``sparsity_clustering_mechanism.py`` derives *why*
# (single-coincidence law -> per-frame coincidence-cliques). Here we show it drives
# the *state difference*:
#
# 1. **Which marginals differ by state?** Unconscious states are **sparser** — the
#    fraction of near-silent neurons jumps — while the trace **autocorrelation is
#    unchanged**, i.e. the smoothing really is identical.
# 2. **Does the sparsity predict the confound?** Across 10 recordings the shuffle
#    clustering state-difference is predicted by the **fraction of near-silent
#    neurons** (Spearman ~0.99). The *arithmetic-mean* event rate does NOT predict
#    it, because it is dominated by the busy minority, not the near-silent majority
#    that drives the confound.
# 3. **Is sparsity sufficient (causal)?** An **independent-signal** model with zero
#    coupling, in which each neuron fires its *measured* number of events, reproduces
#    the clustering state-difference (r ~ 0.93). It over-predicts the magnitude —
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

AW_C, UN_C = "#4477aa", "#cc6677"
K, SIG = 0.05, 5

# %%
R = si.load_or_compute()          # cached shuffle-null Q/C/L + smoothing-related marginals
SLEEP, ANE, ALL = si.SLEEP_RECS, si.ANE_RECS, si.SLEEP_RECS + si.ANE_RECS
C_IDX = 1                         # clustering is column 1 of the [Q, C, L] vectors


def unc(n):
    return R[n][R[n]["unc_label"]]


def aw(n):
    return R[n]["awake"]


def dshufC(recs):
    return np.array([unc(n)["shuf"][:, C_IDX].mean() - aw(n)["shuf"][:, C_IDX].mean() for n in recs])


# %% [markdown]
# ## Compute the SPARSITY of each recording
# Load each recording, count each neuron's events (``spike_deconv`` onsets), and
# derive: the fraction of near-silent neurons, mean event rate, and an
# **event-count-matched** independent-signal reproduction of the shuffle clustering
# (each simulated neuron fires exactly its measured number of events, zero coupling).

# %%
def recording_sparsity(name):
    kind = R[name]["kind"]
    rec = dataio.load_recording(name)
    rows = si.neuron_rows(rec)
    out = {"unc_label": R[name]["unc_label"]}
    for lab in rec.state_labels:
        fr = dataio.state_frames(rec, lab)[:si.WIN[kind]]
        dc = rec.spike_deconv[np.ix_(rows, fr)]
        ev = dc > 0
        k = (ev[:, 1:] & ~ev[:, :-1]).sum(1) + ev[:, :1].sum(1)          # per-neuron event count
        # event-count-matched independent reproduction
        rng = np.random.default_rng(0)
        Xsim = np.zeros((rows.size, fr.size))
        for i, ki in enumerate(k.astype(int)):
            if ki > 0:
                Xsim[i, rng.integers(0, fr.size, ki)] += 1.0
        Xsim = gaussian_filter1d(Xsim, SIG, axis=1)
        adj, _ = net.density_threshold(net.correlation_matrix(Xsim), K, negative=True)
        out[lab] = {
            "frac_silent": float(np.mean(k < 5)),
            "mean_rate": float(np.mean(k / fr.size)),
            "median_events": float(np.median(k)),
            "C_pred": float(sw.avg_clustering(adj)),
            "events": k,
        }
    return out


print("Loading recordings and computing sparsity ...")
S = {name: recording_sparsity(name) for name in ALL}


def su(n):
    return S[n][S[n]["unc_label"]]


def sa(n):
    return S[n]["awake"]


# %% [markdown]
# ## Step 1 — marginals by state: much sparser, but the smoothing is unchanged

# %%
# sparsity marginals (loaded) + the smoothing marginal (autocorr, from cache)
MARGS = [
    ("frac_silent", "% near-silent\n(fire <5)", lambda n, d: d["frac_silent"], True),
    ("mean_rate", "mean event\nrate", lambda n, d: d["mean_rate"], True),
    ("median_events", "median\nevents", lambda n, d: d["median_events"], True),
    ("autocorr1", "autocorr\n(smoothing)", None, False),
]
print("\nStep 1 — marginals: unconscious vs awake (paired, n=10)")
pct, stars = [], []
for key, _lbl, fn, from_S in MARGS:
    if from_S:
        a = np.array([fn(n, sa(n)) for n in ALL]); u = np.array([fn(n, su(n)) for n in ALL])
    else:
        a = np.array([aw(n)[key] for n in ALL]); u = np.array([unc(n)[key] for n in ALL])
    t, p = stats.ttest_rel(u, a)
    with np.errstate(divide="ignore", invalid="ignore"):
        pc = np.mean(np.where(a > 0, (u - a) / a, np.nan)) * 100
    pct.append(pc)
    stars.append("***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s.")
    print(f"  {key:14s}: awake={a.mean():.4f}  unc={u.mean():.4f}  change={pc:+6.1f}%  (p={p:.2g} {stars[-1]})")

# %% [markdown]
# ## Step 2 — which marginal predicts the shuffle-clustering state difference?
# The fraction of near-silent neurons (a direct sparsity measure) predicts it; the
# arithmetic-mean event rate does not.

# %%
dC = dshufC(ALL)
preds = {
    "frac near-silent": np.array([su(n)["frac_silent"] - sa(n)["frac_silent"] for n in ALL]),
    "mean event rate": np.array([su(n)["mean_rate"] - sa(n)["mean_rate"] for n in ALL]),
    "median events": np.array([su(n)["median_events"] - sa(n)["median_events"] for n in ALL]),
    "autocorr (smoothing)": np.array([unc(n)["autocorr1"] - aw(n)["autocorr1"] for n in ALL]),
}
rho = {}
print("\nStep 2 — Spearman rho( Δmarginal , ΔshuffleC )  across 10 recordings:")
for kkey, dd in preds.items():
    rho[kkey] = stats.spearmanr(dd, dC).correlation
    print(f"  {kkey:22s}: rho={rho[kkey]:+.2f}")
rk_sleep = stats.spearmanr(np.array([su(n)["frac_silent"] - sa(n)["frac_silent"] for n in SLEEP]),
                           dshufC(SLEEP)).correlation
print(f"  frac near-silent, SLEEP only (n=6): rho={rk_sleep:+.2f}")

# %% [markdown]
# ## Step 3 — causal test: an event-count-matched INDEPENDENT model reproduces it
# Each recording's neurons are re-simulated as INDEPENDENT (zero coupling), each
# firing exactly its measured number of events; the simulated clustering is the
# prediction. Note the observed target is itself the *shuffle* clustering — already
# coupling-free — so this asks whether pure sparsity (no coupling) generates the
# same state-difference. It reproduces the **pattern** (r ~ 0.93); the magnitude is
# only approximate (the minimal single-frame-event model over-predicts the small
# sleep differences and under-predicts the deep-anaesthesia ones, where real
# calcium events are wider than single frames).

# %%
obs_dC = dshufC(ALL)
pred_dC = np.array([su(n)["C_pred"] - sa(n)["C_pred"] for n in ALL])
r_rep, p_rep = stats.pearsonr(pred_dC, obs_dC)
print(f"\nStep 3 — event-count-matched reproduction: predicted vs observed ΔshuffleC "
      f"r={r_rep:+.2f} (p={p_rep:.2g}); pred mean {pred_dC.mean():+.4f} vs obs {obs_dC.mean():+.4f} "
      f"(sleep: pred>obs; deep anaesthesia: pred<obs — pattern reproduced, magnitude approximate)")

# %% [markdown]
# ## Figure

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)


def dcol(n):
    return AW_C if False else (AW_C if R[n]["kind"] == "sleep" else UN_C)


# (a) marginals % change by state
axa = fig.add_subplot(gs[0, 0])
labels_a = [m[1] for m in MARGS]
bars = axa.bar(range(len(pct)), pct, color=["#cc6677" if p > 0 else "#4477aa" for p in pct])
for b, s in zip(bars, stars):
    axa.text(b.get_x() + b.get_width() / 2, b.get_height() + (3 if b.get_height() >= 0 else -8),
             s, ha="center", fontsize=8)
axa.axhline(0, color="k", lw=.7)
axa.set_xticks(range(len(pct))); axa.set_xticklabels(labels_a, fontsize=8)
axa.set_ylabel("% change (unconscious vs awake)")
axa.set_title("(a) Unconscious: far more near-silent neurons,\nsmoothing unchanged", fontsize=11)

# (b) which marginal predicts
axb = fig.add_subplot(gs[0, 1])
keys_b = list(rho)
axb.bar(range(len(keys_b)), [abs(rho[k]) for k in keys_b],
        color=["#ddaa33" if k == "frac near-silent" else "0.7" for k in keys_b])
axb.set_xticks(range(len(keys_b)))
axb.set_xticklabels([k.replace(" ", "\n") for k in keys_b], fontsize=7)
axb.set_ylim(0, 1); axb.set_ylabel("|Spearman rho| with Δshuffle-C")
axb.set_title("(b) Sparsity (frac near-silent)\npredicts the confound", fontsize=11)

# (c) scatter ΔshufC vs Δfrac_silent
axc = fig.add_subplot(gs[0, 2])
dfs = preds["frac near-silent"]
for n, x, y in zip(ALL, dfs, dC):
    axc.scatter(x, y, s=40, color=dcol(n), edgecolor="k", linewidth=.3)
b1, b0 = np.polyfit(dfs, dC, 1)
xs = np.linspace(dfs.min(), dfs.max(), 20)
axc.plot(xs, b1 * xs + b0, "0.4", ls="--", lw=1)
axc.set_xlabel("Δ fraction near-silent (unconscious − awake)")
axc.set_ylabel("Δ shuffle-clustering")
axc.set_title(f"(c) Predictor: Spearman rho={rho['frac near-silent']:+.2f}", fontsize=11)
axc.scatter([], [], color=AW_C, label="sleep"); axc.scatter([], [], color=UN_C, label="anaesthesia")
axc.legend(fontsize=8)

# (d) reproduction predicted vs observed
axd = fig.add_subplot(gs[1, 0])
for n, x, y in zip(ALL, pred_dC, obs_dC):
    axd.scatter(x, y, s=44, color=dcol(n), edgecolor="k", linewidth=.3)
lim = max(pred_dC.max(), obs_dC.max()) * 1.1
axd.plot([0, lim], [0, lim], "0.5", ls="--", lw=1, label="y=x")
axd.set_xlim(0, lim); axd.set_ylim(0, lim)
axd.set_xlabel("predicted Δ shuffle-C (event-count-matched model)")
axd.set_ylabel("observed Δ shuffle-C")
axd.set_title(f"(d) Sparsity reproduces the pattern\nr={r_rep:.2f} (magnitude approximate)", fontsize=11)
axd.legend(fontsize=8)

# (e) per-recording observed vs predicted
axe = fig.add_subplot(gs[1, 1])
order = np.argsort(obs_dC)
xx = np.arange(len(ALL))
axe.bar(xx - 0.2, obs_dC[order], 0.4, color="0.35", label="observed")
axe.bar(xx + 0.2, pred_dC[order], 0.4, color="#ddaa33", label="predicted (event-count-matched)")
axe.set_xticks(xx)
axe.set_xticklabels([ALL[i].replace("_sleep", "").replace("mouse", "m").replace("_", "") for i in order],
                    rotation=60, ha="right", fontsize=7)
axe.set_ylabel("Δ shuffle-clustering")
axe.set_title("(e) Per recording", fontsize=11); axe.legend(fontsize=8)

# (f) the data is sparse: per-neuron event-count distribution (mouse05_ane)
axf = fig.add_subplot(gs[1, 2])
ex = S["mouse05_ane"]
for lab, col in [("awake", AW_C), ("anesthesia", UN_C)]:
    k = ex[lab]["events"]
    axf.hist(np.clip(k, 0, 60), bins=np.arange(0, 61, 2), color=col, alpha=.55,
             label=f"{lab} ({ex[lab]['frac_silent']*100:.0f}% fire <5)")
axf.set_xlabel("events the neuron fired"); axf.set_ylabel("# neurons")
axf.set_title("(f) most neurons are near-silent\n(mouse05_ane)", fontsize=11); axf.legend(fontsize=7)

fig.suptitle("OQ2 — the clustering confound and its state difference are driven by temporal SPARSITY "
             "(fraction of near-silent neurons),\nnot coupling and not a smoothing difference", y=1.0, fontsize=12)
fig.savefig(FIG_DIR / "oq2_state_difference_cause.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## The sparseness, seen directly (event raster)

# %%
recr = dataio.load_recording("mouse05_ane")
rowsr = si.neuron_rows(recr)
rng = np.random.RandomState(1)
show = np.sort(rng.choice(rowsr, 45, replace=False))
figr, axes = plt.subplots(1, 2, figsize=(14, 4.5))
for ax, (lab, col) in zip(axes, [("awake", AW_C), ("anesthesia", UN_C)]):
    fr = dataio.state_frames(recr, lab)[:si.WIN["ane"]]
    dc = recr.spike_deconv[np.ix_(show, fr)]
    on = (dc[:, 1:] > 0) & (dc[:, :-1] == 0)
    ax.eventplot([np.flatnonzero(on[i]) / recr.fs for i in range(show.size)],
                 colors=col, linelengths=0.8, linewidths=0.7)
    ax.set_xlim(0, fr.size / recr.fs); ax.set_ylim(-1, show.size)
    ax.set_xlabel("time (s)"); ax.set_ylabel("neuron (same set)")
    ax.set_title(f"{lab.upper()} — event raster ({show.size} neurons)", fontsize=11)
figr.suptitle("mouse05_ane: unconsciousness is visibly sparser — the same neurons fire far fewer events",
              y=1.03, fontsize=12)
figr.savefig(FIG_DIR / "oq2_sparsity_raster.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Conclusion
# * Under identical smoothing (autocorrelation unchanged, panel a), unconscious
#   states are far **sparser** — the fraction of near-silent neurons jumps.
#   Temporal sparsity is the cause of the clustering confound.
# * The **fraction of near-silent neurons** predicts the confound across recordings
#   (Spearman ~0.99; ~0.94 within sleep alone), while the arithmetic-mean event rate
#   does not — because the confound is set by the near-silent majority, not the busy
#   minority.
# * The observed shuffle-C is already coupling-free, so the confound cannot be
#   coupling. An **independent-signal model** (zero coupling) in which each neuron
#   fires its measured number of events reproduces the state-difference **pattern**
#   (r ~ 0.93); the magnitude is approximate (the single-frame-event model
#   over-predicts sleep and under-predicts deep anaesthesia, where real events are
#   wider than one frame). Sparsity alone, no coupling, generates the effect.
# * *Why* sparsity does this (single-coincidence law -> per-frame coincidence-
#   cliques) is derived in ``sparsity_clustering_mechanism.py``; *why* it hits
#   clustering far more than modularity or path length is in
#   ``why_QL_robust_C_confounded.py`` (OQ1).
