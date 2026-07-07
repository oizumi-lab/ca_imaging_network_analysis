# %% [markdown]
# # OQ1 — Why does the shuffle confound clustering (C) but not modularity (Q)
# #        or path length (L)?
#
# The circular-shift shuffle destroys all cross-neuron coupling but preserves each
# neuron's own trace. Earlier work established that the awake -> unconscious
# **state difference** in C is largely reproduced by shuffled data (a confound),
# while Q and L are not. This script asks *why*, rigorously, and corrects an
# intuitive-but-wrong guess along the way.
#
# **Wrong guess (falsified below):** "chance coincidence-cliques are random, so
# they inflate local triangles (C) but form no communities (Q) and no shortcuts
# (L)" — i.e. Q and L shuffle-values stay flat. In fact **sparsity inflates all
# three** shuffle measures above the random (Erdos-Renyi) baseline. Q is *not*
# flat.
#
# **What is actually true (three independent lines of evidence):**
# 1. **Confound is a matter of degree, set by locality.** Fraction of the real
#    state-difference reproduced by the shuffle: **L 4%  <  Q 18%  <  C 56%**.
#    C — a purely *local* triangle-density measure — is inflated most; L — a
#    *global* integration measure — least.
# 2. **Synthetic sensitivity.** In an independent-signal model (zero coupling),
#    making the signals sparser raises the shuffle C, Q and L above ER, but C
#    rises fastest and L slowest — the local measure is the most marginal-sensitive.
# 3. **Mechanism snapshot.** In a real deep-anaesthesia graph vs its shuffle: the
#    shuffle keeps most of the *local* clustering, but its path length collapses
#    to the random value (no genuine shortcuts) and its modularity comes from many
#    small scattered chance-cliques rather than a few large coherent modules.

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

MEAS = ["Q (modularity)", "C (clustering)", "L (path length)"]
SHORT = ["Q", "C", "L"]
COLORS = {"Q": "#1f77b4", "C": "#d62728", "L": "#2ca02c"}

# %% [markdown]
# ## Load the per-recording cache (real + shuffle Q/C/L, marginals)
# Built by ``shuffle_investigation.py`` (same neuron subsample, window and density
# K=5% across states). Delete ``results/cache/shuffle_investigation.npz`` to force
# a recompute.

# %%
R = si.load_or_compute()
SLEEP, ANE, ALL = si.SLEEP_RECS, si.ANE_RECS, si.SLEEP_RECS + si.ANE_RECS


def unc(n):
    return R[n][R[n]["unc_label"]]


def aw(n):
    return R[n]["awake"]


def dreal(recs, m):
    return np.array([unc(n)["real"][m] - aw(n)["real"][m] for n in recs])


def dshuf(recs, m):
    return np.array([unc(n)["shuf"][:, m].mean() - aw(n)["shuf"][:, m].mean() for n in recs])


# %% [markdown]
# ## Evidence 1 — the confound is set by how *local* the measure is
# For each measure, the fraction of the real awake->unconscious difference the
# shuffle reproduces. C (local) >> Q > L (global).

# %%
print("=" * 76)
print("Confound fraction  =  mean(dM_shuffle) / mean(dM_real)")
print("=" * 76)
frac_tbl = {}
for grp, recs in [("SLEEP", SLEEP), ("ANE", ANE), ("ALL", ALL)]:
    frac_tbl[grp] = {}
    for m, s in enumerate(SHORT):
        dr, ds = dreal(recs, m), dshuf(recs, m)
        exc = dr - ds
        t, p = stats.ttest_1samp(exc, 0.0)
        frac = ds.mean() / dr.mean() * 100      # mean-of-means (headline)
        frac_tbl[grp][s] = frac
        print(f"  [{grp:5s}] {s}: dReal={dr.mean():+.4f}  dShuf={ds.mean():+.4f}"
              f"  confound={frac:4.0f}%   genuine excess p={p:.3g}")

# %% [markdown]
# ### Robustness (the headline is a mean-of-means; check it is not one-recording-driven)
# The clean ``L<Q<C`` ordering is strongest under anaesthesia; per recording it
# holds in most but not all sessions, and the C mean is pulled up by the deepest
# anaesthesia recording. What is fully robust is the *sign* — the shuffle
# over-reproduces C in every recording and L in almost none.

# %%
print("\nRobustness on ALL (n=10):")
for m, s in enumerate(SHORT):
    dr, ds = dreal(ALL, m), dshuf(ALL, m)
    # per-recording fraction, ignoring sessions whose real change is ~0 (fraction undefined)
    ok = np.abs(dr) > 0.01
    per_rec = ds[ok] / dr[ok] * 100
    n_confound_pos = int((ds > 0).sum())
    print(f"  {s}: mean-of-means={ds.mean()/dr.mean()*100:4.0f}%  "
          f"median-per-rec={np.median(per_rec):4.0f}%  "
          f"shuffle over-reproduces (ΔShuf>0) in {n_confound_pos}/10 recordings")
# leave-one-out sensitivity of C to the deepest-anaesthesia recording
loo = [r for r in ALL if r != "mouse03_ane"]
print(f"  C leave-one-out (drop mouse03_ane): {dshuf(loo,1).mean()/dreal(loo,1).mean()*100:.0f}% "
      f"(vs {dshuf(ALL,1).mean()/dreal(ALL,1).mean()*100:.0f}% with all)")
# per-recording ordering L<Q<C
order_ok = sum(1 for n in ALL
               if (unc(n)["shuf"][:,2].mean()-aw(n)["shuf"][:,2].mean())
               <= (unc(n)["shuf"][:,0].mean()-aw(n)["shuf"][:,0].mean())
               <= (unc(n)["shuf"][:,1].mean()-aw(n)["shuf"][:,1].mean()))
print(f"  per-recording ΔShuf ordering L<=Q<=C holds in {order_ok}/10 (cleanest under anaesthesia)")

# %% [markdown]
# ## Evidence 2 — synthetic sensitivity: sparsity inflates C fastest, L slowest
# Independent signals (NO coupling), Gaussian-smoothed, thresholded at K=5%. As
# the event rate falls (sparser), every measure climbs above its Erdos-Renyi
# baseline — but the *local* measure C climbs fastest, the *global* measure L
# barely moves.

# %%
Ns, Ts, KK, SIG = 1000, 1500, 0.05, 5


def synth_graph(rate, seed):
    rng = np.random.default_rng(seed)
    X = np.zeros((Ns, Ts))
    for i in range(Ns):
        k = rng.poisson(rate * Ts)
        X[i, rng.integers(0, Ts, k)] += rng.exponential(1.0, k)
    X = gaussian_filter1d(X, SIG, axis=1)
    adj, _ = net.density_threshold(net.correlation_matrix(X), KK, negative=True)
    return adj


def graph_QCL(adj):
    Q = net.repeat_louvain(adj, n_runs=5)["Q_max"]
    C = sw.avg_clustering(adj)
    idx = sw.largest_component(adj)
    L = sw.characteristic_path_length(adj[np.ix_(idx, idx)], n_sources=400,
                                      rng=np.random.RandomState(1))
    return np.array([Q, C, L])


def er_graph(seed):
    rng = np.random.default_rng(seed)
    m = int(np.floor(KK * Ns * (Ns - 1) / 2))
    iu = np.triu_indices(Ns, 1)
    pick = rng.choice(iu[0].size, m, replace=False)
    adj = np.zeros((Ns, Ns))
    adj[iu[0][pick], iu[1][pick]] = 1
    return adj + adj.T


rates = np.array([0.050, 0.030, 0.020, 0.014, 0.010, 0.007, 0.005])
QCL = np.array([graph_QCL(synth_graph(r, 7)) for r in rates])
ER = graph_QCL(er_graph(3))
print("\nSynthetic ER baseline  Q/C/L =", np.round(ER, 3))
print("excess-over-ER at sparsest rate:", np.round(QCL[-1] - ER, 3))

# %% [markdown]
# ## Evidence 3 — mechanism snapshot on a real deep-anaesthesia graph
# Real vs shuffle for one recording: modules (Louvain), largest-module size, and
# path length vs a randomised null. Chance-cliques inflate local C but leave L at
# the random value and give many small modules instead of a few large ones.

# %%
def module_snapshot(adj):
    res = net.repeat_louvain(adj, n_runs=8)
    ci = res["ci_max"]
    order = np.argsort(ci)
    sizes = np.bincount(ci)
    sizes = np.sort(sizes[sizes > 0])[::-1]
    idx = sw.largest_component(adj)
    L = sw.characteristic_path_length(adj[np.ix_(idx, idx)], n_sources=400,
                                      rng=np.random.RandomState(1))
    Lr = sw.characteristic_path_length(sw.randomize_matrix(adj, np.random.RandomState(2)),
                                       n_sources=400, rng=np.random.RandomState(1))
    return {"Q": res["Q_max"], "C": sw.avg_clustering(adj), "L": L, "Lrand": Lr,
            "order": order, "adj": adj, "n_big": int((sizes >= 10).sum()),
            "largest": sizes[0] / adj.shape[0]}


SNAP = "mouse05_ane"
rec = dataio.load_recording(SNAP)
rows = si.neuron_rows(rec)
win = dataio.state_frames(rec, "anesthesia")[:si.WIN["ane"]]
Xr = rec.spike_smoothed[rows][:, win]
adj_real, _ = net.density_threshold(net.correlation_matrix(Xr), 0.05, negative=True)
adj_shuf, _ = net.density_threshold(
    net.correlation_matrix(si.circular_shuffle(Xr, np.random.default_rng(0))), 0.05, negative=True)
snap_real = module_snapshot(adj_real)
snap_shuf = module_snapshot(adj_shuf)
print(f"\n[{SNAP} anaesthesia]  real   {({k: round(snap_real[k],3) for k in ['Q','C','L','Lrand','n_big','largest']})}")
print(f"[{SNAP} anaesthesia]  shuffle{({k: round(snap_shuf[k],3) for k in ['Q','C','L','Lrand','n_big','largest']})}")

# %% [markdown]
# ## Figure 1 — the three lines of evidence

# %%
fig = plt.figure(figsize=(15, 5.2))
gs = fig.add_gridspec(1, 3, wspace=0.32)

# (a) confound fractions
axa = fig.add_subplot(gs[0, 0])
groups = ["SLEEP", "ANE", "ALL"]
x = np.arange(len(groups))
w = 0.26
for j, s in enumerate(SHORT):
    axa.bar(x + (j - 1) * w, [frac_tbl[g][s] for g in groups], w,
            color=COLORS[s], label=s)
axa.axhline(0, color="k", lw=.6)
axa.axhline(100, color="0.6", ls="--", lw=.8)
axa.text(2.35, 101, "fully confounded", fontsize=7, color="0.4", va="bottom", ha="right")
axa.set_xticks(x)
axa.set_xticklabels(groups)
axa.set_ylabel("% of real state-difference\nreproduced by the shuffle")
axa.set_title("(a) Confound is local: L < Q < C", fontsize=11)
axa.legend(title="measure", fontsize=9)

# (b) real per-recording dReal vs dShuffle
axb = fig.add_subplot(gs[0, 1])
for m, s in enumerate(SHORT):
    dr, ds = dreal(ALL, m), dshuf(ALL, m)
    axb.scatter(dr, ds, s=34, color=COLORS[s], label=s, edgecolor="k", linewidth=.3)
lim = 0.25
axb.plot([0, lim], [0, lim], "0.5", ls="--", lw=.9)
axb.text(lim, lim, " y=x\n(confounded)", fontsize=7, color="0.4", va="top")
axb.axhline(0, color="0.5", lw=.8)
axb.text(lim, 0.002, "y=0 (genuine) ", fontsize=7, color="0.4", ha="right", va="bottom")
axb.set_xlim(-0.02, lim)
axb.set_xlabel("real state-difference  ΔM")
axb.set_ylabel("shuffle state-difference  ΔM")
axb.set_title("(b) C hugs y=x; L hugs y=0", fontsize=11)
axb.legend(fontsize=9)

# (c) synthetic sensitivity (excess over ER)
axc = fig.add_subplot(gs[0, 2])
for m, s in enumerate(SHORT):
    axc.plot(rates * 100, QCL[:, m] - ER[m], "-o", color=COLORS[s], ms=4, label=s)
axc.axhline(0, color="0.6", lw=.8)
axc.invert_xaxis()
axc.set_xlabel("event rate  (% frames)  ← sparser")
axc.set_ylabel("excess over Erdos-Renyi baseline")
axc.set_title("(c) Sparsity inflates C fastest,\nL slowest (independent signals)", fontsize=11)
axc.legend(fontsize=9)

fig.suptitle("OQ1 — why the shuffle confounds the LOCAL measure (C) but not the GLOBAL ones (Q, L)",
             y=1.02, fontsize=13)
fig.savefig(FIG_DIR / "oq1_why_QL_robust_C_confounded.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Figure 2 — mechanism: real modules vs scattered chance-cliques
# Adjacency matrices reordered by Louvain community. Real: a few large dense
# blocks (coherent modules) and L well above random. Shuffle: comparable *local*
# clustering but no large coherent blocks, and L back at the random value.

# %%
fig2, axes = plt.subplots(1, 2, figsize=(12, 6))
for ax, snap, title in [(axes[0], snap_real, "REAL"), (axes[1], snap_shuf, "SHUFFLE")]:
    A = snap["adj"][np.ix_(snap["order"], snap["order"])]
    ax.imshow(A, cmap="Greys", interpolation="nearest", aspect="equal")
    ax.set_title(f"{SNAP} anaesthesia — {title}\n"
                 f"Q={snap['Q']:.2f}  C={snap['C']:.2f}  "
                 f"L={snap['L']:.2f} (Lrand={snap['Lrand']:.2f})\n"
                 f"{snap['n_big']} modules (≥10), largest={snap['largest']*100:.0f}% of nodes",
                 fontsize=10)
    ax.set_xlabel("neuron (ordered by module)")
    ax.set_xticks([]); ax.set_yticks([])
fig2.suptitle("Real coupling → few large coherent modules + genuine integration loss "
              "(L_real 6-14% above random).\n"
              "Shuffle → local clustering survives, but only small scattered chance-cliques "
              "and L within ~2-3% of random (almost no genuine modules/shortcuts).",
              y=1.06, fontsize=11)
fig2.savefig(FIG_DIR / "oq1_mechanism_modules.png", dpi=140, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## Conclusion
# * The naive "Q stays flat / chance-cliques form no communities" guess is
#   **false**: sparsity inflates the shuffle value of Q and L too (both exceed
#   the ER baseline; panel c).
# * What is fully robust: the shuffle **over-reproduces C in 10/10 recordings**, a
#   minority of the Q increase, and almost none of the L increase. Aggregate
#   confound fractions **C ~56% (median-per-recording ~42%, leave-out-mouse03_ane
#   ~46%) > Q ~18% > L ~4%**. The *raw* shuffle-floor shift is ordered L<Q<C in
#   **10/10** recordings; the confound-*fraction* ordering is noisier (~7/10,
#   cleanest under anaesthesia — 4/4 ane vs ~3/6 sleep) only because it divides by
#   per-recording ΔReal, which is near zero for Q in some sleep sessions. L's ~4%
#   is a real, high-effect-size result (ΔL_real ≈ 40× the shuffle sampling noise),
#   not an estimator floor.
# * Interpretation (consistent with panels b/c and the module snapshot; an
#   interpretation the numbers support, not one they prove): the confound tracks
#   how *local* the measure is. Clustering is a local triangle count — exactly what
#   chance coincidence-cliques from sparsity create — so it inherits most of the
#   marginal-driven state difference. Path length is a global integration measure
#   the chance-cliques barely touch (shuffle L stays within ~2-3% of random while
#   real L is 6-14% above), so its awake->unconscious increase is almost entirely
#   genuine. Modularity is intermediate: chance-cliques form small scattered
#   modules, but the real state change is dominated by reorganisation into a few
#   large coherent modules.
# * Defensible one-line summary: *the shuffle reproduces most of the clustering
#   increase (~50%, 10/10 recordings), a minority of the modularity increase
#   (~15-18%), and essentially none of the path-length increase (~3-4%); cleanest
#   under anaesthesia.*
# * Practical takeaway (unchanged): trust **L** most and **Q** for coupling
#   claims; report **C/SWP** as excess-over-shuffle. See ``state_difference_cause.py``
#   (OQ2) for *what* marginal drives the C confound.
