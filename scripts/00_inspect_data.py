# %% [markdown]
# # 00 · Inspecting the dataset
#
# This is an interactive ``# %%`` script (run it cell-by-cell in VS Code or
# Spyder). It opens one recording from the RIKEN v2.0 dataset and shows what is
# inside, so you understand the raw material before doing any network analysis.
#
# Dataset: *Single-cell calcium imaging across wakefulness, sleep, and anesthesia*
# (RIKEN 20260409-001 v2.0; Kiyooka & Oomoto et al., Cell Reports 2026).

# %%
import sys
sys.path.insert(0, ".")  # run from the project root; the package lives in ./src

import numpy as np
import matplotlib.pyplot as plt

from src.funcnet import dataio
from src.funcnet.paths import FIG_DIR

FIG_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load a recording
# We start with ``example_data`` (a 1,000-neuron subsample, ~84 MB) because it
# loads fast. Swap in any real recording, e.g. ``"mouse01_sleep"`` or
# ``"mouse07_ane"`` (see ``dataio.list_recordings()``).

# %%
rec = dataio.load_recording("example_data")
print(rec)
print("Available recordings:", dataio.list_recordings(include_example=True))

# %% [markdown]
# ## What is inside a Recording?
# All time-series are ``(N_neurons, T_frames)`` and **row-aligned** — row *i* is
# the same neuron everywhere. Sampling rate is 7.65 Hz.

# %%
print(f"data_info        : {rec.data_info!r}  (states compared: {rec.state_labels})")
print(f"n_neurons (N)    : {rec.n_neurons}")
print(f"n_frames  (T)    : {rec.n_frames}  (~{rec.n_frames / rec.fs / 60:.1f} min)")
print(f"dFF              : {rec.dFF.shape}  raw fluorescence dF/F")
print(f"spike_deconv     : {rec.spike_deconv.shape}  OASIS-deconvolved spikes")
print(f"spike_smoothed   : {rec.spike_smoothed.shape}  <-- used for all networks")
print(f"centroid (px)    : {rec.centroid.shape}  (x, y)")
print(f"animal_info      : {rec.animal_info}")
if rec.nonzero_ROI is not None:
    print(f"nonzero_ROI      : {int(rec.nonzero_ROI.sum())}/{rec.n_neurons} neurons "
          f"kept in the paper's network analysis")

# %% [markdown]
# ## The brain-state vector
# ``state`` labels every frame. For sleep recordings: 0 = awake, 0.5 = quiet
# awake, 1 = NREM, 2 = REM. For anesthesia: 0 = awake, 1 = anesthesia. The
# analysis only uses long, stable epochs, listed in ``used_frame``.

# %%
codes = dataio.SLEEP_STATE_CODES if rec.data_info == "sleep" else dataio.ANE_STATE_CODES
vals, counts = np.unique(rec.state, return_counts=True)
for v, c in zip(vals, counts):
    print(f"  state {v:>4}  = {codes.get(v, '?'):<12}  {c:6d} frames ({100*c/rec.n_frames:.1f}%)")
print("\nused_frame epochs (0-based indices) selected for analysis:")
for label, idx in rec.used_frame.items():
    print(f"  {label:<11}: {idx.size} frames")

# %% [markdown]
# ## Look at the data
# Top: example spike_smoothed traces for a handful of neurons.
# Bottom: the brain-state timeline (when the animal was awake vs asleep).

# %%
t = np.arange(rec.n_frames) / rec.fs / 60  # minutes
fig, axes = plt.subplots(2, 1, figsize=(11, 6), height_ratios=[3, 1], sharex=True)

for k, n in enumerate(np.linspace(0, rec.n_neurons - 1, 6).astype(int)):
    axes[0].plot(t, rec.spike_smoothed[n] + k * np.nanmax(rec.spike_smoothed[n]) * 1.1,
                 lw=0.6)
axes[0].set_ylabel("spike_smoothed\n(6 example neurons, offset)")
axes[0].set_title(f"{rec.name}  ·  {rec.n_neurons} neurons  ·  {rec.data_info}")

axes[1].plot(t, rec.state, lw=0.8, color="k")
axes[1].set_yticks(sorted(codes))
axes[1].set_yticklabels([codes[v] for v in sorted(codes)])
axes[1].set_xlabel("time (min)")
axes[1].set_ylabel("state")
fig.tight_layout()

fig.savefig(FIG_DIR / "00_inspect_traces.png", dpi=130)
plt.show()
print("saved ->", FIG_DIR / "00_inspect_traces.png")

# %% [markdown]
# ## Spatial layout of the neurons
# Each neuron has an (x, y) centroid in the imaging field (3 mm × 3 mm). This is
# what we will later colour by functional module.

# %%
um = rec.centroid_um
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(um[:, 0], um[:, 1], s=6, alpha=0.6)
ax.set_aspect("equal")
ax.invert_yaxis()  # image convention
ax.set_xlabel("x (µm)")
ax.set_ylabel("y (µm)")
ax.set_title(f"Neuron positions (N = {rec.n_neurons})")
fig.tight_layout()
fig.savefig(FIG_DIR / "00_inspect_positions.png", dpi=130)
plt.show()
