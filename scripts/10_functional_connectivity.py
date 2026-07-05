# %% [markdown]
# # 10 · From calcium signals to a functional network
#
# A **functional network** connects neurons that fire together. The recipe is
# simple and is the foundation of everything that follows:
#
# 1. Take the activity matrix ``spike_smoothed`` (N neurons × T frames).
# 2. Compute the **Pearson correlation** between every pair of neurons → an
#    N × N **functional connectivity** matrix.
# 3. (Next script) keep only the strongest correlations → a graph.
#
# Here we build connectivity matrices **separately for each brain state** and
# compare them, setting up the central question of the lecture: *does the
# functional network reorganise between wakefulness and unconsciousness?*

# %%
import os
import sys

# add the repo root (parent of scripts/) to the path so `src.funcnet` is importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import warnings

import numpy as np
import matplotlib.pyplot as plt

from src.funcnet import dataio, network as net
from src.funcnet.paths import FIG_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Choose a recording
# Use a real recording so the two states are biologically meaningful. We pick an
# anesthesia session (awake vs anesthesia); ``mouse01_sleep`` would give awake
# vs NREM instead. The paper estimates each network from a **1500-frame window**
# (~196 s) and keeps only neurons active in that window (``nonzero_ROI``).

# %%
REC = "mouse01_sleep"          # try "mouse01_sleep" for awake-vs-NREM
WINDOW = 1500                # frames per network (paper's setting)

rec = dataio.load_recording(REC)
print(rec)
awake_label, second_label = rec.state_labels
print(f"Comparing:  '{awake_label}'  vs  '{second_label}'")

# %% [markdown]
# ## Extract one matched window per state
# We take the first ``WINDOW`` frames of each state's stable epochs, restricted
# to the activity-filtered neurons so both states use the **same neuron set**
# (essential for a fair comparison).

# %%
keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)

def window_activity(label):
    idx = dataio.state_frames(rec, label)[:WINDOW]
    return rec.spike_smoothed[keep][:, idx]

X_awake = window_activity(awake_label)
X_second = window_activity(second_label)
print(f"awake window : {X_awake.shape}")
print(f"{second_label} window : {X_second.shape}")

C_awake = net.correlation_matrix(X_awake)
C_second = net.correlation_matrix(X_second)

# %% [markdown]
# ## Visualise the connectivity matrices
# Same neurons, two states. Even by eye, the structure of pairwise correlations
# differs between wakefulness and unconsciousness.

# %%
fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
for ax, Cmat, lab in [(axes[0], C_awake, awake_label), (axes[1], C_second, second_label)]:
    im = ax.imshow(Cmat, cmap="RdBu_r", vmin=-0.3, vmax=0.3, interpolation="nearest")
    ax.set_title(f"correlation — {lab}")
    ax.set_xlabel("neuron")
    ax.set_ylabel("neuron")
fig.colorbar(im, ax=axes, shrink=0.7, label="Pearson r")
fig.savefig(FIG_DIR / "10_connectivity_matrices.png", dpi=140)
plt.show()

# %% [markdown]
# ## Distribution of pairwise correlations
# A compact way to compare states: the histogram of off-diagonal correlations.
# A heavier right tail means more strongly co-active pairs.
#
# We also overlay a **shuffle null** (black outline): each neuron's trace is
# circularly shifted by an independent random lag, which destroys every
# cross-neuron timing relationship (so *no* real correlation remains) while
# keeping each neuron's own signal shape. The gap between the real distribution
# and this null — a heavier right tail and a mean nudged positive — is the
# genuine co-activity; everything the two curves share is a statistical baseline,
# not biology. (See the next cell for why that baseline peaks below zero.)

# %%
def circular_shuffle(X, rng):
    """Roll each neuron's trace by an independent random lag.

    Destroys all cross-neuron timing (→ no true correlation) while preserving
    each neuron's own signal shape (sparsity, amplitude distribution). This is
    the correct null for a correlation distribution: it answers "what would the
    histogram look like if these exact signals were unrelated?".
    """
    out = np.empty_like(X)
    for i in range(X.shape[0]):
        out[i] = np.roll(X[i], rng.integers(1, X.shape[1]))
    return out

rng = np.random.default_rng(0)
C_null = net.correlation_matrix(circular_shuffle(X_awake, rng))

iu = np.triu_indices(C_awake.shape[0], 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(-0.4, 0.6, 80)
ax.hist(C_awake[iu], bins=bins, alpha=0.55, density=True, label=awake_label)
ax.hist(C_second[iu], bins=bins, alpha=0.55, density=True, label=second_label)
ax.hist(C_null[iu], bins=bins, density=True, histtype="step", color="k", lw=1.5,
        label=f"{awake_label} shuffled (null)")
ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("pairwise correlation r")
ax.set_ylabel("density")
ax.set_title(f"{rec.name}: correlation distribution by state")
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "10_correlation_hist.png", dpi=140)
plt.show()

for lab, Cmat in [(awake_label, C_awake), (second_label, C_second), ("null (shuffled)", C_null)]:
    v = Cmat[iu]
    print(f"  {lab:<16}: mean r = {v.mean():+.4f}   |  fraction r>0.1 = {(v>0.1).mean():.3f}")

# %% [markdown]
# ## Why does the peak sit slightly *below* zero?
# Students always ask this: the most common correlation is a small **negative**
# value (~−0.03), not zero — yet most neuron pairs are surely unrelated. Nothing
# is wrong. Notice the shuffle null peaks at the **same** negative value: with
# every real relationship destroyed, the shape is unchanged. So the negative peak
# is a property of the **signal shape**, not of the biology (it is *not* evidence
# of widespread inhibition).
#
# The mechanism, in one chain:
#
# 1. ``spike_smoothed`` is **sparse and non-negative** — a flat baseline with
#    occasional positive transients. Each neuron sits near baseline ~90 % of the
#    time.
# 2. Pearson r subtracts each neuron's **own mean**. The rare transients pull the
#    mean up, so after subtraction the trace is **below its mean ~90 % of the
#    time** (a small negative deviation), spiking positive only during transients.
# 3. Correlation is dominated by the **large deviations = the transients**. When
#    neuron A fires, neuron B is (~90 % likely) at baseline — i.e. *below* its
#    mean → a **negative** deviation. (large +) × (small −) = a **negative**
#    contribution on exactly the highest-leverage frames.
# 4. Only when two transients **coincide** do you get (large +)×(large +). For
#    unrelated neurons that is rare, so a *typical* finite window misses those big
#    positives and the negatives dominate → **r slightly negative** (the bulk).
#    The few pairs that catch coincidences form the **long positive tail**.
# 5. Mean is pinned at ~0 (independence) and the distribution is right-skewed, so
#    ``mode < median < mean`` — the mode lands below zero.
#
# The informative quantity is therefore **not** the peak location but the
# **excess over the null**: the extra right tail and the positive shift of the
# mean. That is the real functional connectivity, and it grows as the cortex
# synchronises in sleep/anesthesia — the subject of the next scripts.

# %% [markdown]
# ## Takeaway
# We now have one functional connectivity matrix per state. But a full
# correlation matrix is dense and noisy. To analyse *network structure*
# (modules, hubs), we first turn it into a graph by keeping only the strongest
# edges — at a **fixed density** so the two states are directly comparable.
# That is the subject of script ``20_modularity.py``.

# %%
