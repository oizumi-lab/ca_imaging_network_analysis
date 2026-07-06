# %% [markdown]
# # Why do INDEPENDENT (shuffled) sparse signals produce high clustering?
#
# The circular-shift shuffle destroys all real coupling, yet the thresholded
# correlation graph has clustering far above the Erdős–Rényi baseline (~K), and
# more so for unconscious states. This script isolates the mechanism with fully
# **independent synthetic neurons** (zero coupling by construction) and confirms
# it in the real data.
#
# **Result (see the four steps below):**
# 1. Pearson correlation is **scale-invariant** → a neuron's amplitude/variance is
#    irrelevant (amplitude heterogeneity changes nothing).
# 2. The driver is **temporal sparsity**: sparse traces make correlations depend
#    on a few coincident large frames.
# 3. A single frame where **≥3 neurons are coincidentally large** makes them all
#    mutually correlated → a **triangle**. Clustering = the prevalence of such
#    chance coincidence-cliques (not hubs / degree heterogeneity).
# 4. Unconscious states are **sparser** (most neurons near-silent), so they produce
#    more coincidence-cliques → the clustering confound. (Per-neuron kurtosis, shown
#    below, rises only as a proxy for that sparsity — see
#    ``sparsity_clustering_mechanism.py``.)

# %%
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings
import itertools

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy import stats as sps

from src.funcnet import dataio, network as net, smallworld as sw
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

N, T, K, SMOOTH = 1000, 1500, 0.05, 5
rng = np.random.default_rng(0)


# %%
def sparse_signals(rates, amps):
    """N INDEPENDENT neurons: sparse positive transients (rate_i), amplitude a_i,
    Gaussian-smoothed like spike_smoothed. No cross-neuron coupling of any kind."""
    X = np.zeros((N, T))
    for i in range(N):
        k = rng.poisson(rates[i] * T)
        X[i, rng.integers(0, T, k)] += amps[i] * rng.exponential(1.0, k)
    return gaussian_filter1d(X, SMOOTH, axis=1)


def clustering_of(X):
    C = net.correlation_matrix(X)
    adj, _ = net.density_threshold(C, K, negative=True)
    return sw.avg_clustering(adj), adj


def frac_triangles_one_frame(X, adj, n_sample=250):
    """Fraction of triangles whose 3 edges are each dominated by ~the same frame."""
    Xc = X - X.mean(1, keepdims=True)
    tf = lambda i, j: int(np.argmax(Xc[i] * Xc[j]))
    nodes = np.where(adj.sum(1) >= 2)[0]
    shared = total = 0
    for a in rng.choice(nodes, min(n_sample, nodes.size), replace=False):
        nb = np.where(adj[a])[0]
        for b, c in itertools.islice(itertools.combinations(nb, 2), 40):
            if adj[b, c]:
                total += 1
                f = [tf(a, b), tf(a, c), tf(b, c)]
                shared += (max(f) - min(f) <= 3 * SMOOTH)
    return shared / max(1, total)


# %% [markdown]
# ## Step 1–3 (simulation): sparsity drives clustering; amplitude does not

# %%
rates = np.array([0.30, 0.15, 0.06, 0.03, 0.015, 0.008])
clus_by_rate, frac_by_rate = [], []
for r in rates:
    X = sparse_signals(np.full(N, r), np.ones(N))      # ONE realization for both
    c, adj = clustering_of(X)
    clus_by_rate.append(c)
    frac_by_rate.append(frac_triangles_one_frame(X, adj) if r <= 0.06 else np.nan)
    print(f"  event rate {r:.3f}: clustering={c:.3f}  "
          f"tri-from-1-frame={frac_by_rate[-1]}", flush=True)

amp_spreads = [0.0, 0.5, 1.0, 1.5, 2.0]
clus_by_amp = [clustering_of(sparse_signals(np.full(N, 0.02), rng.lognormal(0, s, N)))[0]
               for s in amp_spreads]
gauss_clus = clustering_of(rng.standard_normal((N, T)))[0]
print(f"  Gaussian iid (not sparse) clustering={gauss_clus:.3f}  (≈ ER baseline K={K})", flush=True)


# %% [markdown]
# ## Step 4 (real data): unconscious marginals are sparser (kurtosis rises as its proxy)

# %%
REC = [("mouse05_ane", 2900), ("mouse04_day1_sleep", 1500),
       ("mouse02_sleep", 1500), ("mouse06_ane", 2900)]
kurt = {}
for name, width in REC:
    rec = dataio.load_recording(name)
    keep = np.flatnonzero(rec.nonzero_ROI)
    ks = []
    for lab in rec.state_labels:
        X = rec.spike_smoothed[keep][:, dataio.state_frames(rec, lab)[:width]]
        ks.append(float(np.nanmean(sps.kurtosis(X, axis=1, fisher=True))))
    kurt[name] = ks
    print(f"  {name}: awake kurtosis={ks[0]:.0f}  {rec.state_labels[1]}={ks[1]:.0f}", flush=True)


# %% [markdown]
# ## Figure

# %%
fig, ax = plt.subplots(2, 2, figsize=(13, 10))

ax[0, 0].plot(rates, clus_by_rate, "-o", color="crimson")
ax[0, 0].axhline(K, color="0.5", ls="--", label=f"ER baseline (≈K={K})")
ax[0, 0].axhline(gauss_clus, color="steelblue", ls=":", label="Gaussian iid (not sparse)")
ax[0, 0].invert_xaxis()
ax[0, 0].set_xlabel("event rate  (← sparser / burstier)")
ax[0, 0].set_ylabel("clustering of the INDEPENDENT graph")
ax[0, 0].set_title("(1) Sparser signals → higher clustering\n(no coupling; the confound)")
ax[0, 0].legend(fontsize=8)

ax[0, 1].plot(amp_spreads, clus_by_amp, "-o", color="goldenrod")
ax[0, 1].set_ylim(0, 0.25)
ax[0, 1].set_xlabel("amplitude heterogeneity (lognormal σ)")
ax[0, 1].set_ylabel("clustering")
ax[0, 1].set_title("(2) Amplitude/variance has NO effect\n(Pearson r is scale-invariant)")

valid = ~np.isnan(frac_by_rate)
ax[1, 0].plot(np.array(rates)[valid], np.array(frac_by_rate)[valid] * 100, "-o", color="purple")
ax[1, 0].invert_xaxis()
ax[1, 0].set_ylim(0, 100)
ax[1, 0].set_xlabel("event rate  (← sparser)")
ax[1, 0].set_ylabel("% of triangles from ~one shared frame")
ax[1, 0].set_title("(3) Triangles come from single coincidence frames\n(a chance clique, not real structure)")

names = list(kurt)
xs = np.arange(len(names))
ax[1, 1].plot(xs - .12, [kurt[n][0] for n in names], "o", color="royalblue", ms=10, label="Awake")
ax[1, 1].plot(xs + .12, [kurt[n][1] for n in names], "o", color="crimson", ms=10, label="Unconscious")
for i, n in enumerate(names):
    ax[1, 1].plot([i - .12, i + .12], kurt[n], "-", color="0.6")
ax[1, 1].set_xticks(xs)
ax[1, 1].set_xticklabels([n.replace("_sleep", "").replace("mouse", "m").replace("_", "") for n in names],
                         rotation=30, ha="right", fontsize=8)
ax[1, 1].set_ylabel("mean per-neuron kurtosis (sparsity proxy)")
ax[1, 1].set_title("(4) Real data: unconscious is sparser\n→ higher kurtosis → more coincidence-cliques")
ax[1, 1].legend(fontsize=8)

fig.suptitle("Why independent (shuffled) sparse signals produce high clustering — "
             "sparsity/coincidence, not amplitude or hubs", y=1.0, fontsize=13)
fig.tight_layout()
fig.savefig(FIG_DIR / "clustering_confound_mechanism.png", dpi=140, bbox_inches="tight")
plt.show()
