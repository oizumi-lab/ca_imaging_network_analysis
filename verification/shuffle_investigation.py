"""Shared compute for the two shuffle-null open questions (OQ1 & OQ2).

Both ``why_QL_robust_C_confounded.py`` (OQ1) and ``state_difference_cause.py``
(OQ2) need the same expensive per-recording quantities:

    * real network measures  Q, C, L   (K = 5 % binary graph, as in script 30)
    * within-bout circular-shift **temporal-null** Q, C, L
      (mean + samples per state)
    * per-neuron **marginal** statistics the shuffle preserves
      (event rate, active-frame fraction, activity concentration,
       lag-1 autocorrelation)

This module computes them **once** for every recording (same active-neuron
subsample, window, and density across states) and caches the result to an
``.npz`` under ``results/cache/`` so the two analysis scripts are fast to
iterate on. It is *investigation* code (a helper for ``verification/``), not
part of the teaching ``src.funcnet`` library.
"""

from __future__ import annotations

import os
import sys
import warnings
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np

from src.funcnet import dataio, network as net, smallworld as sw
from src.funcnet.paths import PROJECT_ROOT

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
warnings.filterwarnings("ignore", message="Mean of empty slice")

# --- settings (kept in step with shuffle_null_control.py) -------------------
K = 0.05               # connection density of the binary graph (paper Fig. 5/script 30)
GAMMA = 1.0
N_RUNS = 5             # Louvain runs per graph (max-Q)
N_SHUFFLE = 10         # circular-shift surrogates per state
MAX_NEURONS = 1500     # active-neuron subsample (same set across states/shuffles)
N_SOURCES = 400        # sampled sources for path length
SEED = 0

WIN = {"sleep": 1500, "ane": 2900}
SLEEP_RECS = ["mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
              "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep"]
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]
ALL_RECS = [(n, "sleep") for n in SLEEP_RECS] + [(n, "ane") for n in ANE_RECS]

CACHE = PROJECT_ROOT / "results" / "cache" / "shuffle_investigation.npz"
CACHE_SCHEMA = 2


# --- primitives -------------------------------------------------------------
def neuron_rows(rec):
    keep = rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
    rows = np.flatnonzero(keep)
    if MAX_NEURONS is not None and rows.size > MAX_NEURONS:
        rows = np.sort(np.random.RandomState(0).choice(rows, MAX_NEURONS, replace=False))
    return rows


def contiguous_runs(frames):
    """Positions of contiguous bouts in a selected original-frame vector."""
    frames = np.asarray(frames)
    cuts = np.flatnonzero(np.diff(frames) != 1) + 1
    return [run for run in np.split(np.arange(frames.size), cuts) if run.size]


def circular_shuffle(X, rng, runs=None):
    """Independently roll each neuron within each contiguous state bout.

    ``runs=None`` treats all columns as one contiguous bout, which is useful for
    synthetic inputs. Real state data should always pass runs derived from the
    original frame indices so events are never rolled across concatenated gaps.
    """
    out = np.array(X, copy=True)
    runs = [np.arange(X.shape[1])] if runs is None else runs
    for i in range(X.shape[0]):
        for run in runs:
            if run.size > 1:
                out[i, run] = np.roll(X[i, run], int(rng.integers(1, run.size)))
    return out


def mouse_id(name):
    """Biological-mouse key; mouse 4's two sleep days are repeated sessions."""
    if name in {"mouse04_day1_sleep", "mouse04_day2_sleep"}:
        return "mouse04_sleep"
    return name


def aggregate_by_mouse(values, recs):
    """Average recording-level values within biological mouse."""
    grouped = {}
    for name, value in zip(recs, np.asarray(values), strict=True):
        grouped.setdefault(mouse_id(name), []).append(value)
    return np.asarray([np.nanmean(group, axis=0) for group in grouped.values()])


def graph_measures(C):
    """(Q, clustering, path length) of the K-density binary graph from corr matrix C."""
    adj, _ = net.density_threshold(C, K, negative=True)     # rank by |r|, as in the paper
    Q = net.repeat_louvain(adj, gamma=GAMMA, n_runs=N_RUNS)["Q_max"]
    Cc = sw.avg_clustering(adj)
    idx = sw.largest_component(adj)
    if idx.size > 2:
        L = sw.characteristic_path_length(adj[np.ix_(idx, idx)],
                                          n_sources=min(N_SOURCES, idx.size),
                                          rng=np.random.RandomState(1))
    else:
        L = np.nan
    return np.array([Q, Cc, L])


def event_rate_lambda(Xdc):
    """Poisson event rate per neuron per frame, estimated as the **onset** rate of
    the deconvolved spikes (0 -> nonzero transitions). This is the generative
    ``lambda`` that a Poisson-then-smooth surrogate would need to match the data's
    marginal sparsity."""
    ev = Xdc > 0
    onsets = ev[:, 1:] & ~ev[:, :-1]
    lam = (onsets.sum(1) + ev[:, :1].sum(1)) / Xdc.shape[1]   # per-neuron onsets/frame
    return float(np.mean(lam))


def concentration_top(Xsm, frac=0.05):
    """Mean fraction of a neuron's total smoothed activity held by its top-``frac``
    most active frames (higher = sparser / more concentrated in time)."""
    Xs = np.sort(Xsm, axis=1)[:, ::-1]
    top = int(np.ceil(frac * Xsm.shape[1]))
    tot = Xsm.sum(1)
    return float(np.mean(Xs[:, :top].sum(1) / np.where(tot > 0, tot, 1)))


def autocorr_lag1(Xsm):
    Xc = Xsm - Xsm.mean(1, keepdims=True)
    num = (Xc[:, 1:] * Xc[:, :-1]).sum(1)
    den = (Xc * Xc).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        return float(np.nanmean(np.where(den > 0, num / den, np.nan)))


def marginal_stats(Xsm, Xdc):
    """Per-neuron marginal properties the circular shuffle preserves."""
    return {
        "event_rate": event_rate_lambda(Xdc),
        "active_frac": float(np.mean((Xdc > 0).mean(1))),
        "concentration": concentration_top(Xsm, 0.05),
        "autocorr1": autocorr_lag1(Xsm),
    }


# --- per-recording driver ---------------------------------------------------
def compute_recording(name, kind):
    width = WIN[kind]
    rec = dataio.load_recording(name)
    rows = neuron_rows(rec)
    out = {"kind": kind, "unc_label": rec.state_labels[1]}
    for label in rec.state_labels:
        win = dataio.state_frames(rec, label)[:width]
        runs = contiguous_runs(win)
        Xsm = rec.spike_smoothed[rows][:, win]
        Xdc = rec.spike_deconv[rows][:, win]
        C = net.correlation_matrix(Xsm)
        real = graph_measures(C)
        rng = np.random.default_rng(SEED)
        shuf = np.array([graph_measures(net.correlation_matrix(circular_shuffle(Xsm, rng, runs)))
                         for _ in range(N_SHUFFLE)])
        marg = marginal_stats(Xsm, Xdc)
        out[label] = {"real": real, "shuf_mean": shuf.mean(0), "shuf_std": shuf.std(0),
                      "shuf": shuf, **marg}
        print(f"  {name} [{label}]: Q/C/L real={real[0]:.3f}/{real[1]:.3f}/{real[2]:.2f}"
              f"  shufC={shuf[:,1].mean():.3f}  rate={marg['event_rate']:.4f}", flush=True)
    return out


def compute_all():
    results = {}
    for name, kind in ALL_RECS:
        results[name] = compute_recording(name, kind)
    return results


# --- cache I/O (npz of a flattened dict) ------------------------------------
_SCALAR = ("event_rate", "active_frac", "concentration", "autocorr1")


def _manifest():
    """Settings that determine cached numeric results."""
    return {
        "schema": CACHE_SCHEMA,
        "null": "within_bout_circular_shift",
        "K": K,
        "gamma": GAMMA,
        "n_runs": N_RUNS,
        "n_shuffle": N_SHUFFLE,
        "max_neurons": MAX_NEURONS,
        "n_sources": N_SOURCES,
        "seed": SEED,
        "windows": WIN,
        "recordings": ALL_RECS,
    }


def _flatten(results):
    flat = {"__manifest__": json.dumps(_manifest(), sort_keys=True)}
    for name, r in results.items():
        flat[f"{name}::kind"] = r["kind"]
        flat[f"{name}::unc_label"] = r["unc_label"]
        for lab in ("awake", r["unc_label"]):
            d = r[lab]
            flat[f"{name}::{lab}::real"] = d["real"]
            flat[f"{name}::{lab}::shuf"] = d["shuf"]
            for s in _SCALAR:
                flat[f"{name}::{lab}::{s}"] = np.float64(d[s])
    return flat


def _unflatten(flat):
    names = sorted({k.split("::")[0] for k in flat if not k.startswith("__")})
    results = {}
    for name in names:
        kind = str(flat[f"{name}::kind"])
        unc = str(flat[f"{name}::unc_label"])
        r = {"kind": kind, "unc_label": unc}
        for lab in ("awake", unc):
            real = flat[f"{name}::{lab}::real"]
            shuf = flat[f"{name}::{lab}::shuf"]
            d = {"real": real, "shuf": shuf,
                 "shuf_mean": shuf.mean(0), "shuf_std": shuf.std(0)}
            for s in _SCALAR:
                d[s] = float(flat[f"{name}::{lab}::{s}"])
            r[lab] = d
        results[name] = r
    return results


def load_or_compute(force=False):
    if CACHE.exists() and not force:
        flat = dict(np.load(CACHE, allow_pickle=False))
        cached_manifest = flat.pop("__manifest__", None)
        cached_manifest = None if cached_manifest is None else str(cached_manifest.item())
        expected_manifest = json.dumps(_manifest(), sort_keys=True)
        if cached_manifest == expected_manifest:
            return _unflatten(flat)
        print("cached shuffle results use different settings; recomputing", flush=True)
    results = compute_all()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(CACHE, **_flatten(results))
    print(f"cached -> {CACHE}", flush=True)
    return results


if __name__ == "__main__":
    load_or_compute(force="--force" in sys.argv)
