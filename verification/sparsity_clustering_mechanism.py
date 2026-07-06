# %% [markdown]
# # Is it really "kurtosis"? — sparsity is the cause, kurtosis is a proxy, and the
# #   mechanism that turns sparsity into a high clustering coefficient
#
# `state_difference_cause.py` (OQ2) found that the awake→unconscious clustering
# **confound** is predicted by per-neuron **kurtosis**. Two fair objections:
#
# 1. In the per-neuron kurtosis histogram, the mass **piles up at the far right**.
#    Why? Is kurtosis even the right variable?
# 2. What is the actual **mechanism** — *why* would a high kurtosis (or whatever
#    the real variable is) produce a high clustering coefficient?
#
# This script answers both, with real data and a zero-coupling simulation, and
# **reframes** the OQ2 result accordingly. No cache is used — it recomputes
# everything from the raw recordings and a small synthetic model.
#
# ## Report — what this script shows
#
# **A. Kurtosis is a proxy for temporal SPARSITY.** Per neuron, the kurtosis of
# `spike_smoothed` is ~ 1 / (number of events the neuron fired) — Spearman ≈ −0.9
# to −0.95. Under unconsciousness most neurons go nearly silent (mouse05_ane:
# **median 4 events** in ~6 min, 55 % fire < 5), and a nearly-silent trace has a
# mechanically enormous kurtosis. So the far-right pile-up is just the near-silent
# cells (exaggerated by the 97th-percentile clip used for display), **not** a
# separate phenomenon. The causal variable is **sparsity**; kurtosis is a
# tail-sensitive restatement of it. (This also explains why the *arithmetic-mean*
# event rate — dominated by the busy minority — is a poor predictor while mean
# kurtosis — which up-weights the near-silent majority — is good.)
#
# **B. The mechanism (why sparsity → high clustering).** In independent (zero
# coupling) signals, as neurons get sparser their activity concentrates into fewer
# effective frames, so each pairwise correlation is dominated by the *single* frame
# where both happen to be large. A frame where **three** neurons are coincidentally
# large makes all three pairwise-correlated — a **triangle**. The thresholded graph
# becomes a **union of per-frame coincidence-cliques**, and cliques are maximally
# clustered. Sparser ⇒ higher clustering; at the sparsest, most triangles trace to
# a single shared frame. Amplitude heterogeneity does nothing (Pearson r is
# scale-invariant).
#
# **C. Confirmed on real data.** In the actual circular-shuffle graph, per-node
# clustering falls with the node's event count (mouse05_ane anaesthesia: Spearman
# ≈ −0.94; sparsest event-quartile clustering ≈ 0.38 vs ≈ 0.15 for the busiest).
# The near-silent neurons carry the confound.

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import itertools

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
K = 0.05


def per_neuron_kurt_events(rec, lab, width):
    """Per-neuron (event count, kurtosis of spike_smoothed) for one state."""
    fr = dataio.state_frames(rec, lab)[:width]
    sm = rec.spike_smoothed[np.ix_(si.neuron_rows(rec), fr)]
    dc = rec.spike_deconv[np.ix_(si.neuron_rows(rec), fr)]
    ev = dc > 0
    k = (ev[:, 1:] & ~ev[:, :-1]).sum(1) + ev[:, :1].sum(1)
    kurt = stats.kurtosis(sm, axis=1, fisher=True)
    return k, kurt, fr.size


# %% [markdown]
# ## Part A — kurtosis is a proxy for sparsity (real data)
# Two recordings (one anaesthesia, one sleep) to show it is not anaesthesia-specific.

# %%
REC_A = [("mouse05_ane", "ane"), ("mouse04_day1_sleep", "sleep")]
partA = {}
print("Part A — per-neuron kurtosis vs event count")
for name, kind in REC_A:
    rec = dataio.load_recording(name)
    partA[name] = {}
    for lab in rec.state_labels:
        k, kurt, N = per_neuron_kurt_events(rec, lab, si.WIN[kind])
        v = (k >= 1) & np.isfinite(kurt) & (kurt > 0)
        slope = np.polyfit(np.log(k[v]), np.log(kurt[v]), 1)[0]
        rho = stats.spearmanr(k[v], kurt[v]).correlation
        partA[name][lab] = {"k": k, "kurt": kurt, "N": N}
        print(f"  {name:18s} [{lab:10s}]: N={N}  median_events={np.median(k):.0f}  "
              f"frac(<5 events)={np.mean(k < 5):.2f}  "
              f"log-log slope(kurt~events)={slope:+.2f}  Spearman={rho:+.2f}  "
              f"mean_kurt={np.nanmean(kurt):.0f}")

# %% [markdown]
# ## Part B — the mechanism (zero-coupling simulation)
# Independent neurons (no coupling of any kind), Gaussian-smoothed, thresholded at
# K = 5 %. Vary the events per neuron; measure clustering, the effective number of
# active frames per neuron, and the fraction of triangles whose three edges are all
# dominated by ~one shared frame. Then a control: amplitude heterogeneity.

# %%
Nm, Tm, SIGm = 800, 1500, 5
mrng = np.random.default_rng(0)


def sim_indep(kev, amps=None):
    """N independent neurons, each firing ``kev`` unit events at random frames,
    Gaussian-smoothed. ``amps`` optionally scales each neuron (amplitude test)."""
    X = np.zeros((Nm, Tm))
    for i in range(Nm):
        X[i, mrng.integers(0, Tm, kev)] += 1.0
    if amps is not None:
        X *= amps[:, None]
    return gaussian_filter1d(X, SIGm, axis=1)


def eff_frames(X):
    """Effective # of active frames per neuron (participation ratio of energy)."""
    e = X ** 2
    return float(np.mean(e.sum(1) ** 2 / np.maximum((e ** 2).sum(1), 1e-12)))


def frac_tri_one_frame(X, adj):
    Xc = X - X.mean(1, keepdims=True)
    dfr = lambda i, j: int(np.argmax(Xc[i] * Xc[j]))
    nodes = np.where(adj.sum(1) >= 2)[0]
    shared = total = 0
    for a in mrng.choice(nodes, min(250, nodes.size), replace=False):
        nb = np.where(adj[a])[0]
        for b, c in itertools.islice(itertools.combinations(nb, 2), 40):
            if adj[b, c]:
                total += 1
                f = [dfr(a, b), dfr(a, c), dfr(b, c)]
                shared += (max(f) - min(f) <= 3 * SIGm)
    return shared / max(1, total)


kev_grid = [160, 80, 40, 20, 10, 5]
m_clus, m_eff, m_tri, m_kurt = [], [], [], []
print("\nPart B — mechanism (independent signals, K=5%)")
print(f"  {'events/nrn':>10} {'kurtosis':>9} {'eff_frames':>11} {'clustering':>10} {'tri_1frame':>11}")
for kev in kev_grid:
    X = sim_indep(kev)
    adj, _ = net.density_threshold(net.correlation_matrix(X), K, negative=True)
    m_clus.append(sw.avg_clustering(adj))
    m_eff.append(eff_frames(X))
    m_tri.append(frac_tri_one_frame(X, adj) * 100)
    m_kurt.append(float(np.nanmean(stats.kurtosis(X, axis=1, fisher=True))))
    print(f"  {kev:10d} {m_kurt[-1]:9.0f} {m_eff[-1]:11.0f} {m_clus[-1]:10.3f} {m_tri[-1]:10.0f}%")

# amplitude control: fixed rate, vary per-neuron amplitude heterogeneity
amp_sig = [0.0, 0.5, 1.0, 1.5, 2.0]
m_clus_amp = [sw.avg_clustering(net.density_threshold(
    net.correlation_matrix(sim_indep(20, amps=mrng.lognormal(0, s, Nm))), K, negative=True)[0])
    for s in amp_sig]
print("  amplitude control (fixed rate=20 events): clustering vs lognormal sigma =",
      [round(c, 3) for c in m_clus_amp], "(flat -> amplitude irrelevant)")

# one concrete coincidence-clique for the illustration (sparsest regime)
Xd = sim_indep(5)
adjd, _ = net.density_threshold(net.correlation_matrix(Xd), K, negative=True)
Xc_d = Xd - Xd.mean(1, keepdims=True)
tri = None
for a in np.where(adjd.sum(1) >= 2)[0]:
    nb = np.where(adjd[a])[0]
    for b, c in itertools.combinations(nb, 2):
        if adjd[b, c]:
            tri = (a, b, c)
            break
    if tri is not None:
        break
tstar = int(np.argmax(sum(Xc_d[tri[i]] * Xc_d[tri[j]] for i, j in [(0, 1), (0, 2), (1, 2)])))

# %% [markdown]
# ## Part C — confirmed on real data: sparse neurons carry the clustering
# Build the real circular-shuffle graph (coupling destroyed) and ask whether a
# node's clustering depends on how many events it fired.

# %%
SNAP = "mouse05_ane"
recc = dataio.load_recording(SNAP)
rows = si.neuron_rows(recc)
partC = {}
print(f"\nPart C — per-node clustering vs event count in the real shuffle graph ({SNAP})")
for lab in recc.state_labels:
    fr = dataio.state_frames(recc, lab)[:si.WIN["ane"]]
    Xsm = recc.spike_smoothed[np.ix_(rows, fr)]
    dc = recc.spike_deconv[np.ix_(rows, fr)]
    ev = dc > 0
    k = (ev[:, 1:] & ~ev[:, :-1]).sum(1) + ev[:, :1].sum(1)
    adj, _ = net.density_threshold(
        net.correlation_matrix(si.circular_shuffle(Xsm, np.random.default_rng(0))), K, negative=True)
    cc = sw.clustering_coef(adj)
    deg = adj.sum(1)
    m = deg >= 2
    rho = stats.spearmanr(k[m], cc[m]).correlation
    q = np.quantile(k[m], [0, .25, .5, .75, 1.0])
    quart = [float(cc[m][(k[m] >= q[i]) & (k[m] <= q[i + 1])].mean()) for i in range(4)]
    partC[lab] = {"k": k[m], "cc": cc[m], "rho": rho, "quart": quart}
    print(f"  {lab:11s}: Spearman(events, node-clustering)={rho:+.2f}  "
          f"node-C by event quartile (sparse→busy): " + " ".join(f"{x:.3f}" for x in quart))

# %% [markdown]
# ## Figure

# %%
fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.34)

# (A) kurtosis vs event count (real, 2 recordings)
axA = fig.add_subplot(gs[0, 0])
for (name, kind), mk in zip(REC_A, ["o", "^"]):
    for lab in partA[name]:
        d = partA[name][lab]
        col = UN_C if lab in ("anesthesia", "nrem") else AW_C
        v = (d["k"] >= 1) & np.isfinite(d["kurt"]) & (d["kurt"] > 0)
        axA.scatter(d["k"][v], d["kurt"][v], s=5, marker=mk, color=col, alpha=.25)
kk = np.array([2.0, 300.0])
axA.plot(kk, 1500 / kk, "k--", lw=1, label="slope −1  (∝ 1/events)")
axA.set_xscale("log"); axA.set_yscale("log")
axA.set_xlabel("events the neuron fired"); axA.set_ylabel("kurtosis of spike_smoothed")
axA.set_title("(A) kurtosis ≈ 1 / events → a sparsity proxy\n"
              "○ anaesthesia rec, △ sleep rec; blue awake / red unconscious", fontsize=9)
axA.legend(fontsize=8)

# (B) mechanism: clustering & single-frame triangles vs sparsity
axB = fig.add_subplot(gs[0, 1])
axB.plot(kev_grid, m_clus, "-o", color="#d62728")
axB.axhline(K, color="0.6", ls="--", lw=.8)
axB.text(kev_grid[0], K * 1.15, f"random baseline K={K}", fontsize=7, color="0.4")
axB.invert_xaxis(); axB.set_xscale("log")
axB.set_xlabel("events / neuron  (← sparser)")
axB.set_ylabel("clustering C", color="#d62728")
axBt = axB.twinx()
axBt.plot(kev_grid, m_tri, "-s", color="#7030a0")
axBt.set_ylabel("% triangles from one shared frame", color="#7030a0")
axBt.set_ylim(0, 100)
axB.set_title("(B) sparser → higher C, via single-frame\ncoincidence triangles (zero coupling)", fontsize=9)

# (C) amplitude control
axC = fig.add_subplot(gs[0, 2])
axC.plot(amp_sig, m_clus_amp, "-o", color="goldenrod")
axC.axhline(K, color="0.6", ls="--", lw=.8)
axC.set_ylim(0, max(m_clus_amp) * 1.4)
axC.set_xlabel("amplitude heterogeneity (lognormal σ)")
axC.set_ylabel("clustering C")
axC.set_title("(C) amplitude has NO effect\n(Pearson r is scale-invariant)", fontsize=9)

# (D) one coincidence-clique
axD = fig.add_subplot(gs[1, 0])
lo, hi = max(0, tstar - 110), min(Tm, tstar + 110)
for idx, nm in enumerate("ABC"):
    tr = Xd[tri[idx], lo:hi]
    tr = tr / (tr.max() + 1e-9)
    axD.plot(np.arange(lo, hi), tr + idx * 1.2, lw=1.1)
    axD.text(lo, idx * 1.2 + 0.55, f"neuron {nm}", fontsize=8)
axD.axvline(tstar, color="k", ls=":", lw=1)
axD.text(tstar + 3, 3.3, "shared frame", fontsize=7)
axD.set_xlabel("frame"); axD.set_yticks([])
axD.set_title("(D) 3 independent neurons peak at one chance\nframe → all pairwise-correlated → a triangle", fontsize=9)

# (E) real: per-node clustering vs event count (anaesthesia)
axE = fig.add_subplot(gs[1, 1])
d = partC["anesthesia"]
axE.scatter(d["k"], d["cc"], s=5, color=UN_C, alpha=.25)
axE.set_xscale("log")
axE.set_xlabel("events the neuron fired"); axE.set_ylabel("its clustering in the shuffle graph")
axE.set_title(f"(E) real shuffle graph (anaesthesia):\nsparse neurons cluster more (Spearman {d['rho']:.2f})", fontsize=9)

# (F) mean node-clustering by event quartile, both states
axF = fig.add_subplot(gs[1, 2])
x = np.arange(4)
axF.bar(x - 0.2, partC["awake"]["quart"], 0.4, color=AW_C, label="awake")
axF.bar(x + 0.2, partC["anesthesia"]["quart"], 0.4, color=UN_C, label="anaesthesia")
axF.set_xticks(x); axF.set_xticklabels(["Q1\n(sparsest)", "Q2", "Q3", "Q4\n(busiest)"], fontsize=8)
axF.set_ylabel("mean node clustering (shuffle graph)")
axF.set_title("(F) the sparsest neurons carry the\nclustering confound", fontsize=9)
axF.legend(fontsize=8)

fig.suptitle("Kurtosis is a proxy for temporal sparsity; sparsity makes the correlation graph a union of "
             "single-frame coincidence-cliques → high clustering", y=1.0, fontsize=12)
fig.savefig(FIG_DIR / "sparsity_clustering_mechanism.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Conclusion (reframes OQ2)
# * The clustering confound is driven by **temporal sparsity** — under
#   unconsciousness most neurons fire only a handful of events. **Kurtosis** is a
#   tail-sensitive **proxy** for that sparsity (per neuron ≈ 1 / event-count), which
#   is why it predicts the confound and why its histogram piles up at the right
#   (near-silent cells). It is not a fundamental variable in its own right.
# * **Mechanism:** sparse activity concentrates each neuron into few effective
#   frames, so correlations are dominated by single coincidental shared frames; a
#   shared frame among three neurons is a triangle, so the graph is a union of
#   coincidence-cliques → high clustering. Amplitude/variance is irrelevant.
# * **Real-data check:** in the shuffle graph the sparsest neurons have ~2–3× the
#   clustering of the busiest, confirming they carry the confound.
# * Practical: when comparing clustering / small-worldness across conditions that
#   differ in firing sparsity (states, drugs, cell types), control for event rate /
#   report clustering as excess over a per-neuron-preserving shuffle. See
#   `why_QL_robust_C_confounded.py` (OQ1) for why this hits C far more than Q or L.
