import numpy as np, sys, os
import matplotlib.pyplot as plt
from scipy import stats as sps
sys.path.append(os.path.abspath("."))
from src.funcnet import dataio

def concentration(X):
    """Fraction of each neuron's total activity in its top 5% most active frames."""
    Xs = np.sort(X, axis=1)[:, ::-1]
    top = int(np.ceil(0.05 * X.shape[1]))
    tot = X.sum(1)
    frac = Xs[:, :top].sum(1) / np.where(tot > 0, tot, 1)
    return frac

def lorenz(X):
    """Mean cumulative activity vs cumulative frame fraction (sorted desc)."""
    Xs = np.sort(X, axis=1)[:, ::-1]
    c = np.cumsum(Xs, 1) / np.where(X.sum(1, keepdims=True) > 0, X.sum(1, keepdims=True), 1)
    return np.linspace(0, 1, X.shape[1]), np.nanmean(c, 0)

def make(name, width, unc_name, out):
    rec = dataio.load_recording(name)
    keep = np.flatnonzero(rec.nonzero_ROI)
    rng = np.random.RandomState(1)
    show = np.sort(rng.choice(keep, 160, replace=False))     # same neurons both states
    labels = rec.state_labels
    Xsm, Xdc, pop, kurt = {}, {}, {}, {}
    for lab in labels:
        fr = dataio.state_frames(rec, lab)[:width]
        Xsm[lab] = rec.spike_smoothed[np.ix_(show, fr)]
        Xdc[lab] = rec.spike_deconv[np.ix_(show, fr)]
        pop[lab] = rec.spike_smoothed[np.ix_(keep, fr)].mean(0)
        kurt[lab] = sps.kurtosis(rec.spike_smoothed[np.ix_(keep, fr)], axis=1)
    fs = rec.fs
    t = np.arange(width) / fs
    aw, un = labels[0], labels[1]

    fig = plt.figure(figsize=(15, 15))
    gs = fig.add_gridspec(4, 2, height_ratios=[3, 1, 1.6, 1.6], hspace=.45, wspace=.18)
    # row 1: rasters (deconvolved events), per-neuron normalised so timing is visible
    for j, lab in enumerate([aw, un]):
        ax = fig.add_subplot(gs[0, j])
        D = Xdc[lab].copy()
        D = D / np.where(D.max(1, keepdims=True) > 0, D.max(1, keepdims=True), 1)
        ax.imshow(D, aspect="auto", cmap="Greys", vmin=0, vmax=0.5,
                  extent=[0, t[-1], D.shape[0], 0], interpolation="nearest")
        ax.set_title(f"{name}  —  {lab.upper()}   (raster: {D.shape[0]} neurons)", fontsize=11)
        ax.set_ylabel("neuron"); ax.set_xlabel("time (s)")
    # row 2: population activity
    for j, lab in enumerate([aw, un]):
        ax = fig.add_subplot(gs[1, j])
        ax.plot(t, pop[lab], color="crimson" if j else "royalblue", lw=.7)
        ax.set_title(f"population mean activity — {lab}", fontsize=10)
        ax.set_xlabel("time (s)"); ax.set_ylabel("mean")
        ax.set_ylim(0, max(pop[aw].max(), pop[un].max()) * 1.05)
    # row 3: example single neurons (same 4 neurons both states)
    ex = show[np.argsort(-rec.spike_smoothed[np.ix_(show, dataio.state_frames(rec, un)[:width])].var(1))[:4]]
    exrows = [int(np.where(show == e)[0][0]) for e in ex]
    for j, lab in enumerate([aw, un]):
        ax = fig.add_subplot(gs[2, j])
        for k, r in enumerate(exrows):
            tr = Xsm[lab][r]
            ax.plot(t, tr / (tr.max() + 1e-9) + k, lw=.6,
                    color="crimson" if j else "royalblue")
        ax.set_title(f"example neurons (each normalised) — {lab}", fontsize=10)
        ax.set_xlabel("time (s)"); ax.set_yticks([]); ax.set_ylabel("4 neurons")
    # row 4: burstiness quantification
    ax = fig.add_subplot(gs[3, 0])
    ax.hist(kurt[aw], bins=40, alpha=.6, color="royalblue", label=f"{aw} (mean {np.nanmean(kurt[aw]):.0f})")
    ax.hist(kurt[un], bins=40, alpha=.6, color="crimson", label=f"{un} (mean {np.nanmean(kurt[un]):.0f})")
    ax.set_xlabel("per-neuron kurtosis (burstiness)"); ax.set_ylabel("# neurons")
    ax.set_title("burstiness of each neuron's own trace"); ax.legend(fontsize=8)
    ax.set_xlim(0, np.nanpercentile(np.concatenate([kurt[aw], kurt[un]]), 97))
    ax = fig.add_subplot(gs[3, 1])
    for lab, col in [(aw, "royalblue"), (un, "crimson")]:
        Xa = rec.spike_smoothed[np.ix_(keep, dataio.state_frames(rec, lab)[:width])]
        fx, fy = lorenz(Xa)
        f5 = concentration(Xa)
        ax.plot(fx * 100, fy * 100, color=col,
                label=f"{lab}: top-5% frames hold {100*np.nanmean(f5):.0f}% of activity")
    ax.plot([0, 100], [0, 100], "k:", lw=.8, label="uniform (not bursty)")
    ax.set_xlabel("% of frames (most active first)"); ax.set_ylabel("cumulative % of a neuron's activity")
    ax.set_title("activity concentration (higher = burstier)"); ax.legend(fontsize=8); ax.set_xlim(0, 40)
    fig.suptitle(f"{name}: SAME {len(keep)} neurons, SAME window, SAME density — "
                 f"only the temporal pattern (burstiness) differs", y=1.0, fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    print("saved", out, flush=True)

from src.funcnet.paths import FIG_DIR
make("mouse05_ane", 2900, "anesthesia", FIG_DIR / "burstiness_mouse05_ane.png")
make("mouse04_day1_sleep", 1500, "nrem", FIG_DIR / "burstiness_mouse04_sleep.png")
