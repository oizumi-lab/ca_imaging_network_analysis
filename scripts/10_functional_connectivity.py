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
import sys
import pathlib
import warnings

_base = pathlib.Path(__file__).resolve().parent if "__file__" in globals() else pathlib.Path.cwd()
_scripts = next((p for p in [_base, *_base.parents] if (p / "lib" / "dataio.py").exists()), None) \
    or next((p / "scripts" for p in [_base, *_base.parents] if (p / "scripts" / "lib" / "dataio.py").exists()), None)
sys.path.insert(0, str(_scripts))

import numpy as np
import matplotlib.pyplot as plt

from lib import dataio
from lib import network as net

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
_figdir = pathlib.Path(_scripts) / "figures"
_figdir.mkdir(exist_ok=True)

# %% [markdown]
# ## Choose a recording
# Use a real recording so the two states are biologically meaningful. We pick an
# anesthesia session (awake vs anesthesia); ``mouse01_sleep`` would give awake
# vs NREM instead. The paper estimates each network from a **1500-frame window**
# (~196 s) and keeps only neurons active in that window (``nonzero_ROI``).

# %%
REC = "mouse07_ane"          # try "mouse01_sleep" for awake-vs-NREM
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
fig.savefig(_figdir / "10_connectivity_matrices.png", dpi=140)
plt.show()

# %% [markdown]
# ## Distribution of pairwise correlations
# A compact way to compare states: the histogram of off-diagonal correlations.
# A heavier right tail means more strongly co-active pairs.

# %%
iu = np.triu_indices(C_awake.shape[0], 1)
fig, ax = plt.subplots(figsize=(7, 4.5))
bins = np.linspace(-0.4, 0.6, 80)
ax.hist(C_awake[iu], bins=bins, alpha=0.55, density=True, label=awake_label)
ax.hist(C_second[iu], bins=bins, alpha=0.55, density=True, label=second_label)
ax.axvline(0, color="k", lw=0.7)
ax.set_xlabel("pairwise correlation r")
ax.set_ylabel("density")
ax.set_title(f"{rec.name}: correlation distribution by state")
ax.legend()
fig.tight_layout()
fig.savefig(_figdir / "10_correlation_hist.png", dpi=140)
plt.show()

for lab, Cmat in [(awake_label, C_awake), (second_label, C_second)]:
    v = Cmat[iu]
    print(f"  {lab:<11}: mean r = {v.mean():+.4f}   |  fraction r>0.1 = {(v>0.1).mean():.3f}")

# %% [markdown]
# ## Takeaway
# We now have one functional connectivity matrix per state. But a full
# correlation matrix is dense and noisy. To analyse *network structure*
# (modules, hubs), we first turn it into a graph by keeping only the strongest
# edges — at a **fixed density** so the two states are directly comparable.
# That is the subject of script ``20_modularity.py``.
