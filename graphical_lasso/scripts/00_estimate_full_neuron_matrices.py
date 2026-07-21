# %% [markdown]
# # 00 · Estimate full-neuron graphical-lasso matrices
#
# This script fits the complete paper-defined active-neuron populations for
# `mouse01_sleep` and `mouse05_ane`. There is no neuron subsampling. Each state
# uses the first paper-length window and paired states use identical rows.
#
# The descending alpha path is fit with exact connected-component screening.
# Per-alpha checkpoints make this expensive cell resumable. The primary output
# retains three matrices per state: empirical Pearson correlation,
# graphical-lasso-implied marginal correlation, and precision-derived partial
# correlation.

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GLASSO_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GLASSO_ROOT / "src"))

from src.funcnet import dataio  # noqa: E402

from glasso_analysis.config import CONFIG, MATRIX_DIR, alpha_tag  # noqa: E402
from glasso_analysis.estimation import (  # noqa: E402
    common_valid_rows,
    covariance_to_correlation,
    empirical_correlation,
    fit_screened_graphical_lasso,
    precision_to_partial,
    standardize_activity,
)

MATRIX_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Frozen settings
#
# `GLASSO_RECORDING=mouse01_sleep` (or `mouse05_ane`) can be set when the two
# recordings should be run in separate terminal processes. With no environment
# filter, both are processed in config order.

# %%
requested_recording = os.environ.get("GLASSO_RECORDING")
recording_names = list(CONFIG["recordings"])
if requested_recording:
    if requested_recording not in recording_names:
        raise ValueError(f"unknown GLASSO_RECORDING={requested_recording!r}")
    recording_names = [requested_recording]

ALPHAS = [float(value) for value in CONFIG["alpha_path_descending"]]
PRIMARY_ALPHA_BY_RECORDING = {
    name: float(value)
    for name, value in CONFIG["primary_alpha_by_recording"].items()
}
TOL = float(CONFIG["quic_tolerance"])
MAX_ITER = int(CONFIG["quic_max_iterations"])
SUPPORT_TOL = float(CONFIG["precision_support_tolerance"])

print("recordings:", recording_names)
print("alpha path:", ALPHAS, "primary by recording:", PRIMARY_ALPHA_BY_RECORDING)


# %% [markdown]
# ## Small output helpers

# %%
def atomic_save_array(path: Path, array: np.ndarray) -> None:
    """Write a NumPy array completely before publishing the checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    os.replace(temporary, path)


def atomic_write_json(path: Path, payload: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def pair_distribution(matrix: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(matrix[np.triu_indices(matrix.shape[0], 1)], dtype=np.float64)
    absolute = np.abs(values)
    quantiles = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999]
    result: dict[str, float | int] = {
        "n_pairs": int(values.size),
        "mean": float(values.mean()),
        "sd": float(values.std()),
        "positive_fraction": float(np.mean(values > 0)),
        "negative_fraction": float(np.mean(values < 0)),
        "zero_fraction": float(np.mean(values == 0)),
    }
    for quantile in quantiles:
        label = str(quantile).replace(".", "p")
        result[f"signed_q_{label}"] = float(np.quantile(values, quantile))
        result[f"absolute_q_{label}"] = float(np.quantile(absolute, quantile))
    return result


# %% [markdown]
# ## Load each recording and establish its paired common-valid population

# %%
for recording_name in recording_names:
    settings = CONFIG["recordings"][recording_name]
    width = int(settings["window_frames"])
    primary_alpha = PRIMARY_ALPHA_BY_RECORDING[recording_name]
    fit_alphas = [alpha for alpha in ALPHAS if alpha >= primary_alpha]
    expected_states = list(settings["states"])
    recording_dir = MATRIX_DIR / recording_name
    recording_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n=== {recording_name} ===", flush=True)
    recording = dataio.load_recording(recording_name)
    if recording.state_labels != expected_states:
        raise RuntimeError(
            f"state mismatch: data={recording.state_labels}, config={expected_states}"
        )
    active_rows = dataio.select_neuron_rows(recording, max_neurons=None)
    state_frames = {
        state: dataio.state_frames(recording, state)[:width]
        for state in expected_states
    }
    if any(frames.size != width for frames in state_frames.values()):
        raise RuntimeError(f"{recording_name} lacks a complete configured state window")
    raw_activity = {
        state: recording.spike_smoothed[np.ix_(active_rows, frames)]
        for state, frames in state_frames.items()
    }
    valid = common_valid_rows(raw_activity)
    rows = active_rows[valid]
    coordinates_um = recording.centroid_um[rows]
    print(
        f"all active={active_rows.size}; paired common-valid={rows.size}; "
        f"removed={active_rows.size - rows.size}",
        flush=True,
    )
    atomic_save_array(recording_dir / "neuron_rows.npy", rows)
    atomic_save_array(recording_dir / "coordinates_um.npy", coordinates_um)
    atomic_write_json(
        recording_dir / "population.json",
        {
            "recording": recording_name,
            "window_frames": width,
            "states": expected_states,
            "all_recording_neurons": int(recording.n_neurons),
            "all_active_neurons": int(active_rows.size),
            "paired_common_valid_neurons": int(rows.size),
            "removed_nonfinite_or_constant": int(active_rows.size - rows.size),
            "row_selection": "all nonzero_ROI; no subsampling",
        },
    )

    # Release the large complete recording after the paired windows are copied.
    del recording

    # %% [markdown]
    # ## State-wise empirical correlation and descending exact GLASSO path
    #
    # The precision/covariance checkpoints are float64 because each lower alpha
    # uses them as a warm start. Accepted fits have passed the QUIC gap and
    # inverse-consistency gates in `fit_screened_graphical_lasso`.

    # %%
    for state in expected_states:
        print(f"\n--- {recording_name}: {state} ---", flush=True)
        state_dir = recording_dir / state
        state_dir.mkdir(parents=True, exist_ok=True)
        activity = raw_activity[state][valid]
        standardized = standardize_activity(activity)
        pearson_path = state_dir / "pearson.npy"
        if pearson_path.exists():
            pearson = np.load(pearson_path)
            print("loaded Pearson checkpoint", pearson.shape, flush=True)
        else:
            pearson = empirical_correlation(standardized)
            atomic_save_array(pearson_path, pearson)
            print("saved Pearson", pearson.shape, flush=True)
        del standardized, activity

        diagnostics_path = state_dir / "alpha_diagnostics.json"
        if diagnostics_path.exists():
            alpha_diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
        else:
            alpha_diagnostics = []

        warm_precision = None
        warm_covariance = None
        for alpha in fit_alphas:
            tag = alpha_tag(alpha)
            precision_path = state_dir / f"precision_alpha_{tag}.npy"
            covariance_path = state_dir / f"covariance_alpha_{tag}.npy"
            if precision_path.exists() and covariance_path.exists():
                warm_precision = np.load(precision_path)
                warm_covariance = np.load(covariance_path)
                print(f"alpha={alpha:.3f}: loaded checkpoint", flush=True)
                continue

            print(f"alpha={alpha:.3f}: fitting all {rows.size} neurons ...", flush=True)
            precision, covariance, diagnostics = fit_screened_graphical_lasso(
                pearson,
                alpha,
                init_precision=warm_precision,
                init_covariance=warm_covariance,
                tol=TOL,
                max_iter=MAX_ITER,
                support_tol=SUPPORT_TOL,
            )
            atomic_save_array(precision_path, precision)
            atomic_save_array(covariance_path, covariance)
            alpha_diagnostics = [
                item for item in alpha_diagnostics if float(item["alpha"]) != alpha
            ]
            alpha_diagnostics.append(diagnostics.to_dict())
            alpha_diagnostics.sort(key=lambda item: float(item["alpha"]), reverse=True)
            atomic_write_json(diagnostics_path, alpha_diagnostics)
            print(
                f"alpha={alpha:.3f}: edges={diagnostics.n_edges:,} "
                f"density={diagnostics.native_density:.6%}, "
                f"components={diagnostics.n_components:,}, "
                f"largest={diagnostics.max_component_size:,}, "
                f"gap={diagnostics.max_abs_duality_gap:.2e}, "
                f"wall={diagnostics.wall_seconds:.1f}s",
                flush=True,
            )
            warm_precision, warm_covariance = precision, covariance

        # The last path value is the frozen primary alpha.
        if fit_alphas[-1] != primary_alpha:
            raise RuntimeError("primary alpha must be present in the descending path")
        if warm_precision is None or warm_covariance is None:
            raise RuntimeError("primary graphical-lasso matrices were not produced")

        marginal = covariance_to_correlation(warm_covariance)
        partial = precision_to_partial(warm_precision)
        atomic_save_array(state_dir / "glasso_marginal.npy", marginal.astype(np.float32))
        atomic_save_array(state_dir / "glasso_partial.npy", partial.astype(np.float32))

        matrix_summary = {
            "recording": recording_name,
            "state": state,
            "n_nodes": int(rows.size),
            "window_frames": width,
            "primary_alpha": primary_alpha,
            "pearson": pair_distribution(pearson),
            "glasso_marginal": pair_distribution(marginal),
            "glasso_partial": pair_distribution(partial),
        }
        atomic_write_json(state_dir / "matrix_summary.json", matrix_summary)
        print("saved primary matrix products and summaries", flush=True)
        del pearson, marginal, partial, warm_precision, warm_covariance

    del raw_activity

print("\nAll requested full-neuron matrix fits are complete.")
