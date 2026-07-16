# %% [markdown]
# # 52 · Independent audit: sparse firing, temporal nulls, and network reliability
#
# This verification script addresses a narrow question that the ordinary graph
# null (edge randomisation) cannot answer:
#
# > When awake and unconscious activity have very different event sparsity, how
# > much of a network-measure contrast is already expected from the neuronwise
# > signals, before attributing it to cross-neuron timing?
#
# The analysis deliberately differs from the earlier verification scripts in four
# ways:
#
# 1. Sleep and anaesthesia are never pooled for inference, and mouse 4's two sleep
#    sessions are averaged before the mouse-level test.
# 2. The paper's windows are used: 1500 frames for sleep and 2900 for anaesthesia.
# 3. Circular shifts are performed **within contiguous state bouts**, rather than
#    after rolling a concatenation across large gaps. The older global-roll null is
#    retained as a sensitivity check.
# 4. Mean local clustering is shown beside **transitivity**, the degree-weighted
#    triangle measure used in the 2026 paper. Activity-threshold sensitivity uses
#    neurons meeting the same minimum event count in both paired states.
#
# `real - mean(null)` below means "excess over this conditional null". It is not an
# additive or causal decomposition of the network measure.

# %%
import os
import sys
import warnings

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse import csr_matrix

from src.funcnet import dataio, network as net, smallworld as sw
from src.funcnet.paths import FIG_DIR, RESULTS_DIR

warnings.filterwarnings("ignore", message="invalid value encountered in divide")
FIG_DIR.mkdir(parents=True, exist_ok=True)
AUDIT_DIR = RESULTS_DIR / "verification"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
#
# These defaults are an audit-sized sensitivity analysis, not a full paper
# reproduction. Increase `MAX_NEURONS`, `N_SURROGATES`, and `N_LOUVAIN` for final
# publication inference. Hundreds (ideally 999+) of surrogates are needed for
# empirical tail probabilities; the small ensemble here estimates null means.

# %%
DENSITY = 0.05
MAX_NEURONS = 600
N_SURROGATES = 8
N_LOUVAIN = 3
N_PATH_SOURCES = 200
MIN_EVENTS = (1, 5, 10)
SEED = 20260713

WIN = {"sleep": 1500, "ane": 2900}
SLEEP_RECS = [
    "mouse01_sleep", "mouse02_sleep", "mouse03_sleep",
    "mouse04_day1_sleep", "mouse04_day2_sleep", "mouse05_sleep",
]
ANE_RECS = ["mouse03_ane", "mouse05_ane", "mouse06_ane", "mouse07_ane"]
ALL_RECS = SLEEP_RECS + ANE_RECS


# %% [markdown]
# ## Exact-density graph and graph measures
#
# `src.funcnet.network.density_threshold` keeps every edge tied at the cutoff and
# can therefore exceed K. The helper below always selects exactly floor(K*Npairs)
# edges, randomising only the boundary ties. Achieved density and cutoff-tie count
# are retained as diagnostics.

# %%
def exact_density_graph(C, density, rng):
    C = np.asarray(C, dtype=float)
    if C.ndim != 2 or C.shape[0] != C.shape[1] or C.shape[0] < 2:
        raise ValueError("C must be a square matrix with at least two nodes")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")

    n = C.shape[0]
    iu = np.triu_indices(n, 1)
    values = np.abs(C[iu])
    m = int(np.floor(density * values.size))
    if m == 0:
        raise ValueError("density is too small to retain one edge")

    cutoff = float(np.partition(values, values.size - m)[values.size - m])
    above = np.flatnonzero(values > cutoff)
    tied = np.flatnonzero(values == cutoff)
    need = m - above.size
    boundary = rng.choice(tied, size=need, replace=False) if need else np.empty(0, int)
    chosen = np.concatenate([above, boundary])

    A = np.zeros((n, n), dtype=np.uint8)
    A[iu[0][chosen], iu[1][chosen]] = 1
    A = A + A.T
    info = {
        "cutoff": cutoff,
        "cutoff_ties": int(tied.size),
        "target_edges": m,
        "actual_edges": int(A.sum() // 2),
        "achieved_density": float(A.sum() / (n * (n - 1))),
    }
    return A, info


def transitivity(A):
    """Global fraction of closed triples, matching paper Equation 10."""
    graph = csr_matrix(A, dtype=np.float64)
    degree = np.asarray(graph.sum(axis=1)).ravel()
    triples = np.sum(degree * (degree - 1))
    closed = graph.multiply(graph @ graph).sum()
    return float(closed / triples) if triples else np.nan


def basic_graph_metrics(A):
    idx = sw.largest_component(A)
    return {
        "C_local": sw.avg_clustering(A),
        "transitivity": transitivity(A),
        "lcc_fraction": idx.size / A.shape[0],
    }


def extended_graph_metrics(A):
    """Q and connected-component L, used only for the >=1-event baseline."""
    out = basic_graph_metrics(A)
    # Singleton initialization avoids forcing distinct nontrivial components into
    # one community (the current giant-component warm start can do that).
    out["Q"] = net.repeat_louvain(
        A, n_runs=N_LOUVAIN, seed=SEED, warm_start=False
    )["Q_max"]
    idx = sw.largest_component(A)
    if idx.size > 2:
        out["L_lcc"] = sw.characteristic_path_length(
            A[np.ix_(idx, idx)],
            n_sources=min(N_PATH_SOURCES, idx.size),
            rng=np.random.RandomState(SEED % (2**32 - 1)),
        )
    else:
        out["L_lcc"] = np.nan
    return out


def graph_from_activity(X, rng, extended=False):
    C = net.correlation_matrix(X)
    A, info = exact_density_graph(C, DENSITY, rng)
    measures = extended_graph_metrics(A) if extended else basic_graph_metrics(A)
    return A, {**measures, **info}


# %% [markdown]
# ## State bouts, event counts, and two circular-shift schemes

# %%
def contiguous_runs(frames):
    frames = np.asarray(frames)
    cuts = np.flatnonzero(np.diff(frames) != 1) + 1
    return [r for r in np.split(np.arange(frames.size), cuts) if r.size]


def event_counts(X_deconv):
    active = X_deconv > 0
    return (active[:, 1:] & ~active[:, :-1]).sum(axis=1) + active[:, 0]


def circular_shift(X, runs, rng, within_bouts):
    out = np.empty_like(X)
    if not within_bouts:
        for i in range(X.shape[0]):
            out[i] = np.roll(X[i], int(rng.integers(1, X.shape[1])))
        return out

    out[:] = X
    for i in range(X.shape[0]):
        for run in runs:
            if run.size > 1:
                lag = int(rng.integers(1, run.size))
                out[i, run] = np.roll(X[i, run], lag)
    return out


def stable_rows(base_rows, eligible, priority, cap):
    chosen_pos = priority[eligible[priority]]
    if cap is not None:
        chosen_pos = chosen_pos[:cap]
    return base_rows[chosen_pos]


def edge_jaccard(A, B):
    a = A[np.triu_indices(A.shape[0], 1)] > 0
    b = B[np.triu_indices(B.shape[0], 1)] > 0
    union = np.count_nonzero(a | b)
    return np.count_nonzero(a & b) / union if union else np.nan


# %% [markdown]
# ## Sanity check: ties really can violate the legacy fixed-density promise

# %%
C_tied = np.zeros((4, 4))
A_legacy, _ = net.density_threshold(C_tied, 0.5, negative=True)
A_exact, tied_info = exact_density_graph(C_tied, 0.5, np.random.default_rng(SEED))
print("Tie adversary (N=4, K=50%):")
print(f"  target edges = 3; legacy = {int(A_legacy.sum()//2)}; exact = {int(A_exact.sum()//2)}")
assert int(A_exact.sum() // 2) == 3


# %% [markdown]
# ## Main computation across all recordings
#
# Each minimum-event analysis uses neurons that meet the threshold in **both**
# paired states. A fixed random priority makes the capped samples nested and
# reproducible. The blockwise shift is primary; the older concatenated global roll
# is recomputed at the >=1-event baseline as a null-model sensitivity check.

# %%
rows_out = []
pair_qc = []
window_qc = []

for rec_i, name in enumerate(ALL_RECS):
    print(f"Loading {name} ...", flush=True)
    rec = dataio.load_recording(name)
    kind = rec.data_info
    width = WIN[kind]
    labels = rec.state_labels
    base_rows = np.flatnonzero(
        rec.nonzero_ROI if rec.nonzero_ROI is not None else np.ones(rec.n_neurons, bool)
    )
    priority = np.random.default_rng(SEED + rec_i).permutation(base_rows.size)

    frames = {lab: dataio.state_frames(rec, lab)[:width] for lab in labels}
    counts = {}
    for lab in labels:
        Xdc = rec.spike_deconv[np.ix_(base_rows, frames[lab])]
        counts[lab] = event_counts(Xdc)

    # Retain a regression check showing why the corrected 2900-frame anaesthesia
    # setting matters relative to the legacy 1500-frame implementation.
    if kind == "ane":
        qc_rows = base_rows[priority[: min(3000, base_rows.size)]]
        for lab in labels:
            all_idx = dataio.state_frames(rec, lab)
            for test_width in (1500, 2900):
                for w_i in range(min(2, all_idx.size // test_width)):
                    fr = all_idx[w_i * test_width:(w_i + 1) * test_width]
                    Xq = rec.spike_smoothed[np.ix_(qc_rows, fr)]
                    window_qc.append({
                        "recording": name, "state": lab, "window": w_i,
                        "width": test_width, "n": qc_rows.size,
                        "zero_variance": int(np.sum(np.ptp(Xq, axis=1) == 0)),
                    })

    real_graphs = {}
    for minimum in MIN_EVENTS:
        eligible = (counts[labels[0]] >= minimum) & (counts[labels[1]] >= minimum)
        selected = stable_rows(base_rows, eligible, priority, MAX_NEURONS)
        if selected.size < 50:
            print(f"  skip minimum={minimum}: only {selected.size} neurons", flush=True)
            continue

        # Positions of selected rows in base_rows (base_rows is sorted).
        selected_pos = np.searchsorted(base_rows, selected)
        for state_i, lab in enumerate(labels):
            fr = frames[lab]
            runs = contiguous_runs(fr)
            X = rec.spike_smoothed[np.ix_(selected, fr)]
            k = counts[lab][selected_pos]
            extended = minimum == 1

            A_real, real = graph_from_activity(
                X, np.random.default_rng(SEED + 10_000 * rec_i + state_i), extended=extended
            )
            real_graphs[(minimum, lab)] = A_real
            rows_out.append({
                "recording": name, "kind": kind, "state": lab,
                "minimum_events": minimum, "source": "real", "surrogate": -1,
                "n": selected.size, "n_bouts": len(runs),
                "frame_span": int(fr[-1] - fr[0] + 1),
                "median_events": float(np.median(k)),
                "zero_variance": int(np.sum(np.ptp(X, axis=1) == 0)),
                **real,
            })

            schemes = ["within_bout"]
            if minimum == 1:
                schemes.append("global_concat")
            for scheme_i, scheme in enumerate(schemes):
                for s in range(N_SURROGATES):
                    rng = np.random.default_rng(
                        SEED + 1_000_000 * rec_i + 10_000 * minimum + 100 * scheme_i + s
                    )
                    X0 = circular_shift(X, runs, rng, within_bouts=scheme == "within_bout")
                    _, null = graph_from_activity(
                        X0, rng, extended=extended and scheme == "within_bout"
                    )
                    rows_out.append({
                        "recording": name, "kind": kind, "state": lab,
                        "minimum_events": minimum, "source": scheme, "surrogate": s,
                        "n": selected.size, "n_bouts": len(runs),
                        "frame_span": int(fr[-1] - fr[0] + 1),
                        "median_events": float(np.median(k)),
                        "zero_variance": int(np.sum(np.ptp(X, axis=1) == 0)),
                        **null,
                    })

        # A second-window edge-overlap diagnostic at the baseline threshold.
        if minimum == 1:
            for state_i, lab in enumerate(labels):
                all_idx = dataio.state_frames(rec, lab)
                if all_idx.size >= 2 * width:
                    fr2 = all_idx[width:2 * width]
                    X2 = rec.spike_smoothed[np.ix_(selected, fr2)]
                    A2, met2 = graph_from_activity(
                        X2, np.random.default_rng(SEED + 50_000 + rec_i + state_i), extended=False
                    )
                    pair_qc.append({
                        "recording": name, "kind": kind, "state": lab,
                        "n": selected.size,
                        "edge_jaccard": edge_jaccard(real_graphs[(minimum, lab)], A2),
                        "C_local_window2": met2["C_local"],
                        "transitivity_window2": met2["transitivity"],
                    })

df = pd.DataFrame(rows_out)
df.to_csv(AUDIT_DIR / "52_sparse_firing_state_values.csv", index=False)
pd.DataFrame(window_qc).to_csv(AUDIT_DIR / "52_anesthesia_window_qc.csv", index=False)
pd.DataFrame(pair_qc).to_csv(AUDIT_DIR / "52_window_reliability.csv", index=False)
print(f"Saved raw audit tables -> {AUDIT_DIR}")


# %% [markdown]
# ## Paired contrasts and animal-level inference
#
# For each recording and metric:
#
# `raw contrast = unconscious_real - awake_real`
#
# `null contrast = mean(unconscious_null) - mean(awake_null)`
#
# `excess contrast = raw contrast - null contrast`
#
# Mouse 4's two sleep sessions are averaged before the one-sample test. These tiny
# n t-tests are descriptive and uncorrected; confidence intervals and exact/cluster
# resampling should be preferred for final inference.

# %%
MOUSE = {name: name for name in ALL_RECS}
MOUSE["mouse04_day1_sleep"] = "mouse04_sleep"
MOUSE["mouse04_day2_sleep"] = "mouse04_sleep"


def recording_contrasts(values, metric, minimum, null_source="within_bout"):
    sub = values[values["minimum_events"] == minimum]
    rec_rows = []
    for name in sub["recording"].unique():
        r = sub[sub["recording"] == name]
        labels = ["awake", "nrem" if r["kind"].iloc[0] == "sleep" else "anesthesia"]
        real = {
            lab: r[(r["state"] == lab) & (r["source"] == "real")][metric].iloc[0]
            for lab in labels
        }
        null = {
            lab: r[(r["state"] == lab) & (r["source"] == null_source)][metric].mean()
            for lab in labels
        }
        raw = real[labels[1]] - real[labels[0]]
        null_diff = null[labels[1]] - null[labels[0]]
        rec_rows.append({
            "recording": name, "mouse": MOUSE[name], "kind": r["kind"].iloc[0],
            "minimum_events": minimum, "metric": metric, "null_source": null_source,
            "raw_difference": raw, "null_difference": null_diff,
            "excess_difference": raw - null_diff,
        })
    return pd.DataFrame(rec_rows)


contrast_parts = []
for minimum in MIN_EVENTS:
    for metric in ("C_local", "transitivity"):
        contrast_parts.append(recording_contrasts(df, metric, minimum, "within_bout"))
for metric in ("Q", "L_lcc"):
    contrast_parts.append(recording_contrasts(df.dropna(subset=[metric]), metric, 1, "within_bout"))
for metric in ("C_local", "transitivity"):
    contrast_parts.append(recording_contrasts(df, metric, 1, "global_concat"))

contrasts = pd.concat(contrast_parts, ignore_index=True)
contrasts.to_csv(AUDIT_DIR / "52_sparse_firing_contrasts.csv", index=False)

animal = (
    contrasts.groupby(["kind", "mouse", "minimum_events", "metric", "null_source"], as_index=False)
    [["raw_difference", "null_difference", "excess_difference"]].mean()
)

summary_rows = []
print("\nAnimal-level paired contrasts (nominal one-sample t-test of excess):")
for keys, g in animal.groupby(["kind", "minimum_events", "metric", "null_source"]):
    kind, minimum, metric, null_source = keys
    x = g["excess_difference"].to_numpy()
    test = stats.ttest_1samp(x, 0) if x.size > 1 else None
    row = {
        "kind": kind, "minimum_events": minimum, "metric": metric,
        "null_source": null_source, "n_mice": x.size,
        "mean_raw": g["raw_difference"].mean(),
        "mean_null": g["null_difference"].mean(),
        "mean_excess": x.mean(),
        "t": np.nan if test is None else test.statistic,
        "p_nominal": np.nan if test is None else test.pvalue,
    }
    summary_rows.append(row)
    print(
        f"  {kind:5s} min={minimum:2d} {metric:12s} null={null_source:13s} "
        f"n={x.size}: raw={row['mean_raw']:+.4f} null={row['mean_null']:+.4f} "
        f"excess={row['mean_excess']:+.4f} p={row['p_nominal']:.4g}"
    )

summary = pd.DataFrame(summary_rows)
summary.to_csv(AUDIT_DIR / "52_sparse_firing_summary.csv", index=False)


# %% [markdown]
# ## Compact audit figure

# %%
fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex="col")
for col, kind in enumerate(("sleep", "ane")):
    for row, metric in enumerate(("C_local", "transitivity")):
        ax = axes[row, col]
        s = summary[
            (summary["kind"] == kind)
            & (summary["metric"] == metric)
            & (summary["null_source"] == "within_bout")
        ].sort_values("minimum_events")
        ax.plot(s["minimum_events"], s["mean_raw"], "-o", label="raw state difference")
        ax.plot(s["minimum_events"], s["mean_null"], "-o", label="within-bout null difference")
        ax.plot(s["minimum_events"], s["mean_excess"], "-o", label="excess over null")
        ax.axhline(0, color="0.5", lw=0.8)
        ax.set_title(f"{kind}: {metric}")
        ax.set_ylabel("unconscious - awake")
        ax.set_xticks(MIN_EVENTS)
        if row == 1:
            ax.set_xlabel("minimum events in both states")
axes[0, 0].legend(fontsize=8)
fig.suptitle("Sparse-firing sensitivity: local clustering vs transitivity", y=1.01)
fig.tight_layout()
fig.savefig(FIG_DIR / "52_sparse_firing_robustness.png", dpi=150, bbox_inches="tight")
plt.show()
print(f"Saved figure -> {FIG_DIR / '52_sparse_firing_robustness.png'}")


# %% [markdown]
# ## Reading this audit
#
# - A large raw C difference accompanied by a large null difference is not clean
#   evidence of changed coupling; the state-specific firing marginals already move
#   the estimator.
# - If transitivity is less null-sensitive than mean local C, low-degree sparse
#   neurons are disproportionately affecting the unweighted mean of local C.
# - Stability across minimum-event thresholds supports a population-level result;
#   instability means the claim depends on neurons whose pairwise correlations are
#   estimated from only a few events.
# - Low edge Jaccard across non-overlapping windows warns against interpreting
#   individual edges or modules, even if an aggregate network measure is stable.
# - L is computed on each graph's own largest component here only to audit the
#   existing pipeline. Always inspect `lcc_fraction`; global efficiency on a fixed
#   node set is preferable when components differ by state.
