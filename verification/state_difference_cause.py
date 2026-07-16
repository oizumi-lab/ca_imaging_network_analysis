# %% [markdown]
# # OQ2 — Can temporal sparsity generate the awake -> unconscious clustering
# #        difference in the temporal null?
#
# The within-bout circular-shift null disrupts cross-neuron timing while keeping
# each neuron's trace, and its clustering C is still higher for unconscious states
# (the benchmark quantified in ``why_QL_robust_C_confounded.py``). The preprocessing
# kernel is identical across states by design. One preserved marginal that differs
# strongly is **temporal sparsity**: under unconsciousness most neurons fire only a
# handful of events, and sparse activity inflates chance clustering.
# ``sparsity_clustering_mechanism.py`` derives the mechanism (single-coincidence law
# -> per-frame coincidence-cliques). Here we ask whether sparsity is associated
# with, and mechanistically sufficient to generate, the temporal-null difference:
#
# 1. **Which marginals differ by state?** Unconscious states are **sparser** — the
#    fraction of near-silent neurons jumps. Lag-1 trace autocorrelation is reported
#    descriptively; it is not used to infer that smoothing is equivalent.
# 2. **Does sparsity track the benchmark?** Sleep and anaesthesia are summarized
#    separately at the biological-mouse level (mouse 4's days are averaged).
# 3. **Is sparsity mechanistically sufficient?** A simplified independent-signal
#    model with zero coupling, in which every neuron fires exactly its measured
#    number of events, can generate the temporal-null clustering difference. This
#    is a sufficiency demonstration, not a causal decomposition of the real data.

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
# The shared helper validates a cache manifest specifying the within-bout null.
R = si.load_or_compute()
SLEEP, ANE, ALL = si.SLEEP_RECS, si.ANE_RECS, si.SLEEP_RECS + si.ANE_RECS
C_IDX = 1  # clustering is column 1 of the [Q, C, L] vectors
PROTOCOLS = {"sleep": SLEEP, "anaesthesia": ANE}


def unc(n):
    return R[n][R[n]["unc_label"]]


def aw(n):
    return R[n]["awake"]


def dshufC(recs):
    return np.array([unc(n)["shuf"][:, C_IDX].mean() - aw(n)["shuf"][:, C_IDX].mean() for n in recs])


def by_mouse(values, recs):
    """Average repeated sessions before protocol-level summaries."""
    return si.aggregate_by_mouse(values, recs)


def mouse_names(recs):
    """Mouse labels in the same insertion order as ``aggregate_by_mouse``."""
    return list(dict.fromkeys(si.mouse_id(name) for name in recs))


# %% [markdown]
# ## Compute the SPARSITY of each recording
# Load each recording, count each neuron's events (``spike_deconv`` onsets), and
# derive: the fraction of near-silent neurons, mean event rate, and an
# **event-count-matched** independent-signal reproduction of the shuffle clustering
# (each simulated neuron fires exactly its measured number of events, zero coupling).

# %%
def recording_sparsity(name, seed):
    kind = R[name]["kind"]
    rec = dataio.load_recording(name)
    rows = si.neuron_rows(rec)
    out = {"unc_label": R[name]["unc_label"]}
    rng = np.random.default_rng(seed)
    for lab in rec.state_labels:
        fr = dataio.state_frames(rec, lab)[:si.WIN[kind]]
        dc = rec.spike_deconv[np.ix_(rows, fr)]
        ev = dc > 0
        k = (ev[:, 1:] & ~ev[:, :-1]).sum(1) + ev[:, :1].sum(1)
        # event-count-matched independent reproduction
        Xsim = np.zeros((rows.size, fr.size))
        for i, ki in enumerate(k.astype(int)):
            if ki > 0:
                event_frames = rng.choice(fr.size, size=ki, replace=False)
                Xsim[i, event_frames] = 1.0
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
S = {name: recording_sparsity(name, si.SEED + i) for i, name in enumerate(ALL)}


def su(n):
    return S[n][S[n]["unc_label"]]


def sa(n):
    return S[n]["awake"]


# %% [markdown]
# ## Step 1 — state marginals, separately by protocol and biological mouse

# %%
# Sparsity marginals (loaded) plus lag-1 trace autocorrelation from the cache.
# The common preprocessing kernel is identical across states by construction;
# equality of trace autocorrelation is neither assumed nor inferred here.
MARGS = [
    ("frac_silent", "% near-silent\n(fire <5)", lambda n, d: d["frac_silent"], True),
    ("mean_rate", "mean event\nrate", lambda n, d: d["mean_rate"], True),
    ("median_events", "median\nevents", lambda n, d: d["median_events"], True),
    ("autocorr1", "lag-1 trace\nautocorrelation", None, False),
]
protocol_marginals = {}
for protocol, recs in PROTOCOLS.items():
    print(f"\nStep 1 — {protocol}: unconscious vs awake (paired mice, n={len(mouse_names(recs))})")
    pct, stars = [], []
    for key, _lbl, fn, from_S in MARGS:
        if from_S:
            awake_values = np.array([fn(n, sa(n)) for n in recs])
            unconscious_values = np.array([fn(n, su(n)) for n in recs])
        else:
            awake_values = np.array([aw(n)[key] for n in recs])
            unconscious_values = np.array([unc(n)[key] for n in recs])
        a = by_mouse(awake_values, recs)
        u = by_mouse(unconscious_values, recs)
        _t, p = stats.ttest_rel(u, a)
        with np.errstate(divide="ignore", invalid="ignore"):
            pc = np.nanmean(np.where(a != 0, (u - a) / a, np.nan)) * 100
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 0.05 else "n.s."
        pct.append(pc)
        stars.append(star)
        print(
            f"  {key:14s}: awake={a.mean():.4f}  unc={u.mean():.4f}  "
            f"change={pc:+6.1f}%  (p={p:.2g} {star})"
        )
    protocol_marginals[protocol] = {"pct": np.array(pct), "stars": stars}

# %% [markdown]
# ## Step 2 — which marginal tracks the temporal-null clustering difference?
# Associations are descriptive and are computed separately within each protocol.

# %%
def marginal_differences(recs):
    return {
        "frac near-silent": by_mouse(
            np.array([su(n)["frac_silent"] - sa(n)["frac_silent"] for n in recs]), recs
        ),
        "mean event rate": by_mouse(
            np.array([su(n)["mean_rate"] - sa(n)["mean_rate"] for n in recs]), recs
        ),
        "median events": by_mouse(
            np.array([su(n)["median_events"] - sa(n)["median_events"] for n in recs]), recs
        ),
        "lag-1 trace autocorr": by_mouse(
            np.array([unc(n)["autocorr1"] - aw(n)["autocorr1"] for n in recs]), recs
        ),
    }


protocol_results = {}
for protocol, recs in PROTOCOLS.items():
    d_c = by_mouse(dshufC(recs), recs)
    predictors = marginal_differences(recs)
    rho = {key: stats.spearmanr(values, d_c).correlation for key, values in predictors.items()}
    protocol_results[protocol] = {"dC": d_c, "preds": predictors, "rho": rho}
    print(f"\nStep 2 — {protocol}: Spearman rho(Δmarginal, Δnull-C), n={d_c.size} mice")
    for key, value in rho.items():
        print(f"  {key:24s}: rho={value:+.2f}")

# %% [markdown]
# ## Step 3 — mechanistic sufficiency: event-count-matched independent signals
# Each recording's neurons are re-simulated as INDEPENDENT (zero coupling), each
# firing exactly its measured number of events; the simulated clustering is the
# prediction. This asks whether sparsity alone can generate a difference in the
# simplified model. It does not identify how much of the real state contrast is
# caused by sparsity. The magnitude is approximate because the model omits
# empirical event amplitudes, widths, local rates, and nonstationarity.

# %%
for protocol, recs in PROTOCOLS.items():
    observed = protocol_results[protocol]["dC"]
    predicted = by_mouse(np.array([su(n)["C_pred"] - sa(n)["C_pred"] for n in recs]), recs)
    r_rep, p_rep = stats.pearsonr(predicted, observed)
    protocol_results[protocol].update({"pred_dC": predicted, "r_rep": r_rep, "p_rep": p_rep})
    print(
        f"\nStep 3 — {protocol}: event-count-matched model, n={observed.size} mice; "
        f"r={r_rep:+.2f} (p={p_rep:.2g}); predicted mean={predicted.mean():+.4f}, "
        f"observed mean={observed.mean():+.4f} (magnitude approximate)"
    )

# %% [markdown]
# ## Figure

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)
protocol_colors = {"sleep": AW_C, "anaesthesia": UN_C}


# (a) marginals % change by state
axa = fig.add_subplot(gs[0, 0])
labels_a = [m[1] for m in MARGS]
x_a = np.arange(len(MARGS))
width = 0.36
for offset, protocol in zip((-width / 2, width / 2), PROTOCOLS, strict=True):
    values = protocol_marginals[protocol]["pct"]
    bars = axa.bar(
        x_a + offset,
        values,
        width,
        color=protocol_colors[protocol],
        alpha=0.85,
        label=protocol,
    )
    for bar, star in zip(bars, protocol_marginals[protocol]["stars"], strict=True):
        y = bar.get_height() + (3 if bar.get_height() >= 0 else -8)
        axa.text(bar.get_x() + bar.get_width() / 2, y, star, ha="center", fontsize=7)
axa.axhline(0, color="k", lw=0.7)
axa.set_xticks(x_a)
axa.set_xticklabels(labels_a, fontsize=8)
axa.set_ylabel("% change (unconscious vs awake)")
axa.set_title("(a) State marginals by protocol\n(common kernel by design)", fontsize=11)
axa.legend(fontsize=7)

# (b) which marginal tracks the null shift
axb = fig.add_subplot(gs[0, 1])
keys_b = list(protocol_results["sleep"]["rho"])
x_b = np.arange(len(keys_b))
for offset, protocol in zip((-width / 2, width / 2), PROTOCOLS, strict=True):
    axb.bar(
        x_b + offset,
        [abs(protocol_results[protocol]["rho"][key]) for key in keys_b],
        width,
        color=protocol_colors[protocol],
        alpha=0.85,
        label=protocol,
    )
axb.set_xticks(x_b)
axb.set_xticklabels([k.replace(" ", "\n") for k in keys_b], fontsize=7)
axb.set_ylim(0, 1)
axb.set_ylabel("|Spearman rho| with Δnull-C")
axb.set_title("(b) Mouse-level associations\n(descriptive, protocol-specific)", fontsize=11)
axb.legend(fontsize=7)

# (c) scatter Δnull-C vs Δfrac_silent
axc = fig.add_subplot(gs[0, 2])
for protocol in PROTOCOLS:
    dfs = protocol_results[protocol]["preds"]["frac near-silent"]
    d_c = protocol_results[protocol]["dC"]
    axc.scatter(
        dfs,
        d_c,
        s=40,
        color=protocol_colors[protocol],
        edgecolor="k",
        linewidth=0.3,
        label=protocol,
    )
    b1, b0 = np.polyfit(dfs, d_c, 1)
    xs = np.linspace(dfs.min(), dfs.max(), 20)
    axc.plot(xs, b1 * xs + b0, color=protocol_colors[protocol], ls="--", lw=1)
axc.set_xlabel("Δ fraction near-silent (unconscious − awake)")
axc.set_ylabel("Δ temporal-null clustering")
sleep_rho = protocol_results["sleep"]["rho"]["frac near-silent"]
ane_rho = protocol_results["anaesthesia"]["rho"]["frac near-silent"]
axc.set_title(f"(c) Near-silent fraction tracks Δnull-C\nρ={sleep_rho:+.2f} sleep, {ane_rho:+.2f} anaesthesia", fontsize=11)
axc.legend(fontsize=8)

# (d) simplified-model prediction vs observed temporal-null shift
axd = fig.add_subplot(gs[1, 0])
all_values = []
for protocol in PROTOCOLS:
    predicted = protocol_results[protocol]["pred_dC"]
    observed = protocol_results[protocol]["dC"]
    all_values.extend((predicted, observed))
    axd.scatter(
        predicted,
        observed,
        s=44,
        color=protocol_colors[protocol],
        edgecolor="k",
        linewidth=0.3,
        label=protocol,
    )
low = min(0.0, *(float(values.min()) for values in all_values))
high = max(float(values.max()) for values in all_values)
pad = max(0.01, 0.1 * (high - low))
axd.plot([low - pad, high + pad], [low - pad, high + pad], "0.5", ls="--", lw=1, label="y=x")
axd.set_xlim(low - pad, high + pad)
axd.set_ylim(low - pad, high + pad)
axd.set_xlabel("model-predicted Δnull-C (event-count matched)")
axd.set_ylabel("observed Δ temporal-null clustering")
sleep_r = protocol_results["sleep"]["r_rep"]
ane_r = protocol_results["anaesthesia"]["r_rep"]
axd.set_title(f"(d) Mechanistic sufficiency model\nr={sleep_r:+.2f} sleep, {ane_r:+.2f} anaesthesia", fontsize=11)
axd.legend(fontsize=8)

# (e) per-mouse observed vs predicted
axe = fig.add_subplot(gs[1, 1])
names = []
observed_values = []
predicted_values = []
for protocol, recs in PROTOCOLS.items():
    names.extend(mouse_names(recs))
    observed_values.extend(protocol_results[protocol]["dC"])
    predicted_values.extend(protocol_results[protocol]["pred_dC"])
observed_values = np.asarray(observed_values)
predicted_values = np.asarray(predicted_values)
order = np.argsort(observed_values)
xx = np.arange(observed_values.size)
axe.bar(xx - 0.2, observed_values[order], 0.4, color="0.35", label="observed")
axe.bar(
    xx + 0.2,
    predicted_values[order],
    0.4,
    color="#ddaa33",
    label="predicted (event-count matched)",
)
axe.set_xticks(xx)
axe.set_xticklabels(
    [names[i].replace("_sleep", "").replace("mouse", "m").replace("_", "") for i in order],
    rotation=60,
    ha="right",
    fontsize=7,
)
axe.set_ylabel("Δ temporal-null clustering")
axe.set_title("(e) Per biological mouse", fontsize=11)
axe.legend(fontsize=8)

# (f) the data is sparse: per-neuron event-count distribution (mouse05_ane)
axf = fig.add_subplot(gs[1, 2])
ex = S["mouse05_ane"]
for lab, col in [("awake", AW_C), ("anesthesia", UN_C)]:
    k = ex[lab]["events"]
    axf.hist(
        np.clip(k, 0, 60),
        bins=np.arange(0, 61, 2),
        color=col,
        alpha=0.55,
        label=f"{lab} ({ex[lab]['frac_silent'] * 100:.0f}% fire <5)",
    )
axf.set_xlabel("events the neuron fired")
axf.set_ylabel("# neurons")
axf.set_title("(f) most neurons are near-silent\n(mouse05_ane)", fontsize=11)
axf.legend(fontsize=7)

fig.suptitle(
    "OQ2 — temporal sparsity is sufficient to generate elevated temporal-null clustering "
    "in a simplified independent-event model",
    y=1.0,
    fontsize=12,
)
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
    ax.eventplot(
        [np.flatnonzero(on[i]) / recr.fs for i in range(show.size)],
        colors=col,
        linelengths=0.8,
        linewidths=0.7,
    )
    ax.set_xlim(0, fr.size / recr.fs)
    ax.set_ylim(-1, show.size)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("neuron (same set)")
    ax.set_title(f"{lab.upper()} — event raster ({show.size} neurons)", fontsize=11)
figr.suptitle(
    "mouse05_ane: unconsciousness is visibly sparser — the same neurons fire far fewer events",
    y=1.03,
    fontsize=12,
)
figr.savefig(FIG_DIR / "oq2_sparsity_raster.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Conclusion
# * The preprocessing kernel is identical across states by design, whereas the
#   measured traces and event-count distributions can still have different
#   autocorrelation and sparsity. Unconscious states contain many more near-silent
#   neurons in these windows.
# * Mouse-level associations between sparsity marginals and temporal-null
#   clustering are reported separately for sleep and anaesthesia. They are
#   descriptive at n=5 and n=4 mice, respectively, not evidence that one marginal
#   is the unique explanation.
# * The **independent-signal model** places exactly each neuron's measured number of
#   events and demonstrates that sparsity is mechanistically sufficient to
#   generate a state-dependent clustering baseline without network interactions in
#   that model. Its approximate agreement does not partition the real contrast or
#   rule out other preserved marginals, common drive, or sparsity-by-coupling
#   interactions.
# * *Why* sparsity does this (single-coincidence law -> per-frame coincidence-
#   cliques) is derived in ``sparsity_clustering_mechanism.py``; *why* it hits
#   clustering far more than modularity or path length is in
#   ``why_QL_robust_C_confounded.py`` (OQ1).
