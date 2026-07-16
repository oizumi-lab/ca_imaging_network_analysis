"""Stage-by-stage orchestration for the approved initial validation."""

from __future__ import annotations

import json
import traceback
from dataclasses import fields
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import scipy.linalg

from .artifacts import (
    RunContext,
    active_run,
    complete_stage,
    create_run,
    fail_stage,
    stage_state,
    start_stage,
)
from .config import load_config
from .data import (
    RecordingMetadata,
    Window,
    centered_long_window,
    load_metadata,
    maximin_windows,
    read_signal,
    state_runs,
    windows_frame,
)
from .io import software_manifest, write_csv, write_json
from .model import eigenvalue_diagnostics, fit_ridge_dmd, snapshot_pairs
from .negative_controls import evaluate_negative_controls, summarize_negative_controls
from .paths import RAW_DIR
from .plotting import (
    plot_all_neuron_heatmaps,
    plot_bootstrap_forecast_intervals,
    plot_development_selection,
    plot_diagonal_gain,
    plot_empirical_spectra,
    plot_gate_summary,
    plot_heldout_skill,
    plot_long_block_comparison,
    plot_neuron_subset_classes,
    plot_null_multivariate_gain,
    plot_null_subspace_similarity,
    plot_pc_traces,
    plot_pca_diagnostics,
    plot_precedent_spectrum,
    plot_representative_predictions,
    plot_simulation_recovery,
    plot_signal_diagnostics,
    plot_smoothing_trim_sensitivity,
    plot_subspace_stability,
    plot_tracking_resolution,
    plot_window_map,
)
from .preprocessing import ArmSpec, FrozenRepresentation, fit_representation, fit_scaler
from .resampling import (
    bootstrap_predictive_windows,
    bootstrap_tracking_windows,
    deterministic_neuron_subset_classes,
    evaluate_tracking_windows,
    fit_development_reference,
    synchronized_cell_intervals,
    tracking_resolution,
)
from .simulation import run_known_systems
from .validation import (
    development_grid,
    evaluate_configuration,
    fit_long_block,
    load_scores,
    representative_prediction,
    select_configuration,
    summarize_grid,
)


PROSPECTIVE_AMENDMENTS = [
    {
        "id": "A1_nontrivial_projector",
        "timing": "before empirical fitting",
        "reason": "A k=4 projector is the identity when q=4 and would falsely report perfect stability.",
        "resolution": "Only q>=8 models are eligible for the k=4 subspace/tracking gates; q<8 remains prediction-only.",
    },
    {
        "id": "A2_global_maximin_windows",
        "timing": "before empirical fitting",
        "reason": "The phrase maximally time-spread did not define an exact optimization or tie rule.",
        "resolution": "Maximize the minimum integer start separation globally; among ties choose the lexicographically earliest sequence.",
    },
    {
        "id": "A3_local_blocked_development_cv",
        "timing": "before empirical fitting",
        "reason": "Pooled leave-one-window fitting would not match the intended local deployment estimator.",
        "resolution": "Fit the first 80% and score the final 20% within every development window, then aggregate anonymously.",
    },
    {
        "id": "A4_smoothed_boundary_sensitivity",
        "timing": "before empirical fitting",
        "reason": "Endpoint windows are label-pure but stored smoothing may include support from up to seven neighboring frames.",
        "resolution": "Keep the approved raw-label primary windows and report a seven-frame trimmed Sz/Sc sensitivity.",
    },
    {
        "id": "A5_failed_simulation_diagnostic_override",
        "timing": "before accepted rerun; after inspection of the first exploratory simulation stage",
        "reason": "Gate 2 failed, but the requested smoke test also requires empirical failure-bound and null diagnostics.",
        "resolution": "Continue only under an explicit configuration flag and label all downstream evidence diagnostic, never confirmatory.",
    },
    {
        "id": "A6_fixed_trimmed_split",
        "timing": "before accepted rerun; prompted by read-only forecast audit",
        "reason": "Recomputing 80/20 after edge trimming shifted the original train/test boundary by five frames.",
        "resolution": "Trim seven frames at both edges while preserving the original frame-240 train/test boundary.",
    },
    {
        "id": "A7_strongest_long_diagonal_baseline",
        "timing": "before accepted rerun; prompted by read-only forecast audit",
        "reason": "The first long-block diagnostic reused the full-DMD ridge for diagonal AR.",
        "resolution": "Tune diagonal-AR ridge independently inside the same nested development split.",
    },
]


def empirical_metadata(config: dict[str, Any]) -> dict[str, RecordingMetadata]:
    metadata: dict[str, RecordingMetadata] = {}
    for name, values in config["recordings"].items():
        if "labels" not in values:
            continue
        metadata[name] = load_metadata(RAW_DIR / f"{name}.mat", str(values["paradigm"]))
    return metadata


def all_metadata(config: dict[str, Any]) -> dict[str, RecordingMetadata]:
    return {
        name: load_metadata(RAW_DIR / f"{name}.mat", str(values["paradigm"]))
        for name, values in config["recordings"].items()
    }


def _window_margin(meta: RecordingMetadata, window: Window) -> tuple[int, int]:
    for start, stop, code, _ in state_runs(meta, window.state_code):
        if np.isclose(code, window.state_code) and start <= window.start and window.stop <= stop:
            return window.start - start, stop - window.stop
    raise ValueError(f"Could not find containing state run for {window.window_id}")


def build_windows(
    config: dict[str, Any], metadata: dict[str, RecordingMetadata]
) -> tuple[list[Window], pd.DataFrame]:
    spec = config["windows"]
    windows: list[Window] = []
    for recording, meta in metadata.items():
        labels = config["recordings"][recording]["labels"]
        for label, state_code in labels.items():
            windows.extend(
                maximin_windows(
                    meta,
                    label=label,
                    state_code=float(state_code),
                    n_frames=int(spec["deployment_frames"]),
                    count=int(spec["per_label"]),
                    development_positions=spec["development_positions"],
                )
            )
            windows.append(
                centered_long_window(
                    meta,
                    label=label,
                    state_code=float(state_code),
                    n_frames=int(spec["long_block_frames"]),
                )
            )
    table = windows_frame(windows)
    left: list[int] = []
    right: list[int] = []
    for window in windows:
        margins = _window_margin(metadata[window.recording], window)
        left.append(margins[0])
        right.append(margins[1])
    margin_map = {window.window_id: pair for window, pair in zip(windows, zip(left, right), strict=True)}
    table["left_label_margin_frames"] = table["window_id"].map(lambda key: margin_map[key][0])
    table["right_label_margin_frames"] = table["window_id"].map(lambda key: margin_map[key][1])
    half_width = int(spec["stored_smoothing_half_width_frames"])
    table["stored_smoothing_boundary_risk"] = (
        (table["left_label_margin_frames"] < half_width)
        | (table["right_label_margin_frames"] < half_width)
    )
    return windows, table


def _geometry_summary(windows: pd.DataFrame) -> dict[str, Any]:
    deployment = windows[windows["kind"] == "deployment"]
    overlap_count = 0
    for _, group in deployment.groupby(["recording", "label"]):
        ordered = group.sort_values("start")
        overlap_count += int(np.sum(ordered["start"].to_numpy()[1:] < ordered["stop"].to_numpy()[:-1]))
    return {
        "deployment_windows": int(len(deployment)),
        "development_windows": int(np.sum(deployment["split"] == "development")),
        "evaluation_windows": int(np.sum(deployment["split"] == "evaluation")),
        "long_diagnostic_windows": int(np.sum(windows["kind"] == "long")),
        "overlapping_deployment_pairs": overlap_count,
        "label_or_boundary_violations": 0,
        "smoothing_boundary_risk_windows": int(np.sum(deployment["stored_smoothing_boundary_risk"])),
    }


def run_data_stage(config_path: str | Path | None = None) -> RunContext:
    config = load_config() if config_path is None else load_config(config_path)
    context = create_run(config)
    start_stage(context, "data")
    try:
        metadata_all = all_metadata(config)
        metadata = {name: value for name, value in metadata_all.items() if "labels" in config["recordings"][name]}
        windows, window_table = build_windows(config, metadata)
        stage_dir = context.stage_dir("data")
        inventory = pd.DataFrame(
            [
                {
                    "recording": meta.name,
                    "paradigm": meta.paradigm,
                    "n_neurons": meta.n_neurons,
                    "n_frames": meta.n_frames,
                    "duration_minutes": meta.n_frames / float(config["fs_hz"]) / 60,
                    "segment_stops": ";".join(str(int(value)) for value in meta.segment_stops),
                    "file_size_bytes": meta.file_size,
                    "file_mtime_ns": meta.file_mtime_ns,
                    "path": str(meta.path),
                }
                for meta in metadata_all.values()
            ]
        )
        write_csv(stage_dir / "recording_inventory.csv", inventory)
        write_csv(stage_dir / "window_manifest.csv", window_table)
        geometry = _geometry_summary(window_table)
        write_json(stage_dir / "geometry_checks.json", geometry)
        frozen_config = {key: value for key, value in config.items() if not key.startswith("_")}
        frozen_config["config_sha256"] = config["_config_sha256"]
        write_json(context.root / "config_frozen.json", frozen_config)
        write_json(context.root / "software_manifest.json", software_manifest(config))
        write_json(context.root / "prospective_implementation_amendments.json", PROSPECTIVE_AMENDMENTS)
        plot_window_map(
            metadata,
            window_table,
            fs_hz=float(config["fs_hz"]),
            output=stage_dir / "00_window_geometry",
        )
        complete_stage(context, "data", geometry)
        return context
    except Exception as error:
        fail_stage(context, "data", f"{type(error).__name__}: {error}")
        (context.root / "data_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def _load_window_table(context: RunContext) -> tuple[pd.DataFrame, list[Window]]:
    table = pd.read_csv(context.stage_dir("data") / "window_manifest.csv")
    field_names = {field.name for field in fields(Window)}
    windows = [Window(**{name: row[name] for name in field_names}) for row in table.to_dict("records")]
    return table, windows


def _pooled_autocorrelation(windows: list[np.ndarray], lag: int) -> float:
    numerator = 0.0
    left_square = 0.0
    right_square = 0.0
    for values in windows:
        if values.shape[1] <= lag:
            continue
        left = values[:, :-lag]
        right = values[:, lag:]
        numerator += float(np.einsum("ij,ij->", left, right, optimize=True))
        left_square += float(np.einsum("ij,ij->", left, left, optimize=True))
        right_square += float(np.einsum("ij,ij->", right, right, optimize=True))
    denominator = np.sqrt(left_square * right_square)
    return float(numerator / denominator) if denominator > 0 else np.nan


def _sample_values(windows: list[np.ndarray], maximum: int = 200_000) -> np.ndarray:
    samples: list[np.ndarray] = []
    per_window = max(1, maximum // len(windows))
    for values in windows:
        flat = values.ravel()
        stride = max(1, flat.size // per_window)
        samples.append(flat[::stride][:per_window])
    return np.concatenate(samples)[:maximum]


def run_preprocessing_stage(config_path: str | Path | None = None) -> RunContext:
    config = load_config() if config_path is None else load_config(config_path)
    context = active_run()
    if stage_state(context, "data") != "completed":
        raise RuntimeError("The data stage must complete before preprocessing")
    start_stage(context, "preprocessing")
    try:
        metadata = empirical_metadata(config)
        window_table, windows = _load_window_table(context)
        stage_dir = context.stage_dir("preprocessing")
        model_dir = stage_dir / "models"
        model_dir.mkdir(parents=True, exist_ok=True)
        signal_rows: list[dict[str, Any]] = []
        acf_rows: list[dict[str, Any]] = []
        pca_rows: list[dict[str, Any]] = []
        audit_rows: list[dict[str, Any]] = []
        representations: dict[tuple[str, str], FrozenRepresentation] = {}
        for recording_index, (recording, meta) in enumerate(metadata.items()):
            development = [
                window
                for window in windows
                if window.recording == recording
                and window.kind == "deployment"
                and window.split == "development"
            ]
            for arm_index, (arm_id, arm_values) in enumerate(config["arms"].items()):
                spec = ArmSpec.from_config(arm_id, arm_values)
                representation, arm_windows, scaled_windows, audit = fit_representation(
                    meta,
                    spec,
                    development,
                    fs_hz=float(config["fs_hz"]),
                    max_rank=int(config["model"]["pca_max_rank"]),
                    oversamples=int(config["model"]["pca_oversamples"]),
                    power_iterations=int(config["model"]["pca_power_iterations"]),
                    seed=int(config["random_seed"]) + 100 * recording_index + arm_index,
                )
                representation.save(model_dir / f"{recording}__{arm_id}.npz")
                representations[(recording, arm_id)] = representation
                raw_count = sum(values.size for values in arm_windows)
                zero_count = sum(int(np.count_nonzero(values == 0)) for values in arm_windows)
                sample = _sample_values(arm_windows)
                signal_rows.append(
                    {
                        "recording": recording,
                        "arm": arm_id,
                        "signal": spec.signal,
                        "bin_frames": spec.bin_frames,
                        "effective_fs_hz": float(config["fs_hz"]) / spec.bin_frames,
                        "zero_fraction": zero_count / raw_count,
                        "raw_mean": float(np.mean(sample)),
                        "raw_sd": float(np.std(sample)),
                        "raw_q01": float(np.quantile(sample, 0.01)),
                        "raw_median": float(np.median(sample)),
                        "raw_q99": float(np.quantile(sample, 0.99)),
                    }
                )
                for lag in (1, 2, 4, 8, 15):
                    if lag < min(values.shape[1] for values in scaled_windows):
                        acf_rows.append(
                            {
                                "recording": recording,
                                "arm": arm_id,
                                "lag_samples": lag,
                                "lag_seconds": lag * spec.bin_frames / float(config["fs_hz"]),
                                "autocorrelation": _pooled_autocorrelation(scaled_windows, lag),
                            }
                        )
                cumulative = np.cumsum(representation.explained_variance_ratio)
                for component, (individual, cumulative_value) in enumerate(
                    zip(representation.explained_variance_ratio, cumulative, strict=True), start=1
                ):
                    pca_rows.append(
                        {
                            "recording": recording,
                            "arm": arm_id,
                            "component": component,
                            "explained_variance": float(individual),
                            "cumulative_explained_variance": float(cumulative_value),
                            "orthogonality_error": representation.orthogonality_error,
                        }
                    )
                audit_rows.append(
                    {
                        "recording": recording,
                        "arm": arm_id,
                        **audit,
                        "pca_rank": representation.components.shape[1],
                        "pca_orthogonality_error": representation.orthogonality_error,
                    }
                )

        signal_table = pd.DataFrame(signal_rows)
        acf_table = pd.DataFrame(acf_rows)
        pca_table = pd.DataFrame(pca_rows)
        audit_table = pd.DataFrame(audit_rows)
        write_csv(stage_dir / "signal_statistics.csv", signal_table)
        write_csv(stage_dir / "autocorrelation.csv", acf_table)
        write_csv(stage_dir / "pca_statistics.csv", pca_table)
        write_csv(stage_dir / "preprocessing_audit.csv", audit_table)
        plot_signal_diagnostics(signal_table, acf_table, stage_dir / "01_signal_diagnostics")
        plot_pca_diagnostics(pca_table, stage_dir / "02_pca_diagnostics")

        for recording, meta in metadata.items():
            heatmap_panels: list[dict[str, object]] = []
            trace_panels: list[dict[str, object]] = []
            labels = list(config["recordings"][recording]["labels"])
            for arm_id, arm_values in config["arms"].items():
                representation = representations[(recording, arm_id)]
                for label in labels:
                    candidate = window_table[
                        (window_table["recording"] == recording)
                        & (window_table["label"] == label)
                        & (window_table["kind"] == "deployment")
                        & (window_table["split"] == "evaluation")
                    ].sort_values("start").iloc[0]
                    raw = read_signal(meta, str(arm_values["signal"]), int(candidate.start), int(candidate.stop))
                    scaled = representation.scaled_window(raw)
                    scores = representation.components[:, :4].T @ scaled
                    heatmap_panels.append({"arm": arm_id, "label": label, "values": scaled})
                    trace_panels.append(
                        {
                            "arm": arm_id,
                            "label": label,
                            "scores": scores,
                            "bin_frames": int(arm_values["bin_frames"]),
                        }
                    )
            plot_all_neuron_heatmaps(
                heatmap_panels,
                recording,
                stage_dir / f"03_all_neuron_heatmaps__{recording}",
            )
            plot_pc_traces(
                trace_panels,
                recording,
                fs_hz=float(config["fs_hz"]),
                output=stage_dir / f"04_pc_traces__{recording}",
            )

        summary = {
            "recording_arm_models": int(len(audit_table)),
            "minimum_eligible_neurons": int(audit_table["eligible_neurons"].min()),
            "maximum_pca_orthogonality_error": float(audit_table["pca_orthogonality_error"].max()),
            "nonfinite_neurons": int(audit_table["nonfinite_neurons"].sum()),
            "all_models_passed_orthogonality_check": bool(
                np.all(audit_table["pca_orthogonality_error"] < 1e-8)
            ),
        }
        write_json(stage_dir / "preprocessing_checks.json", summary)
        complete_stage(context, "preprocessing", summary)
        return context
    except Exception as error:
        fail_stage(context, "preprocessing", f"{type(error).__name__}: {error}")
        (context.root / "preprocessing_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def run_precedent_stage(config_path: str | Path | None = None) -> RunContext:
    config = load_config() if config_path is None else load_config(config_path)
    context = active_run()
    if stage_state(context, "preprocessing") != "completed":
        raise RuntimeError("Preprocessing must complete before the precedent check")
    start_stage(context, "precedent")
    try:
        meta = load_metadata(RAW_DIR / "example_data.mat", "sleep")
        raw = read_signal(meta, "spike_deconv", 0, meta.n_frames)
        mean, scale, rows, scaler_audit = fit_scaler([raw], "rms")
        scaled = (raw[rows] - mean[:, None]) / scale[:, None]
        lag = int(config["model"]["precedent_lag_frames"])
        raw_alpha = float(config["model"]["precedent_raw_ridge"])
        model = fit_ridge_dmd([scaled], lag=lag, raw_alpha=raw_alpha)
        x, y, _ = snapshot_pairs([scaled], lag)
        normal = x @ x.T + raw_alpha * np.eye(x.shape[0])
        cross = y @ x.T
        direct_operator = scipy.linalg.solve(
            normal,
            cross.T,
            assume_a="pos",
            check_finite=False,
        ).T
        comparison_error = float(
            np.linalg.norm(model.operator - direct_operator, ord="fro")
            / np.linalg.norm(direct_operator, ord="fro")
        )
        diagnostics = eigenvalue_diagnostics(
            model,
            fs_effective=float(config["fs_hz"]),
            window_samples=meta.n_frames,
            minimum_modulus=float(config["model"]["rotation_minimum_modulus"]),
            nearly_real_tolerance=float(config["model"]["rotation_nearly_real_tolerance"]),
        )
        stage_dir = context.stage_dir("precedent")
        write_csv(stage_dir / "precedent_eigenvalues.csv", diagnostics)
        summary: dict[str, Any] = {
            **model.diagnostics,
            **scaler_audit,
            "lag_frames": lag,
            "lag_seconds": lag / float(config["fs_hz"]),
            "raw_ridge": raw_alpha,
            "svd_vs_gram_relative_error": comparison_error,
            "v1_code_filter_count": int(diagnostics["v1_code_filter"].sum()),
            "caption_realpart_filter_count": int(diagnostics["caption_realpart_filter"].sum()),
            "interpretable_positive_conjugate_members": int(diagnostics["interpretable_rotation"].sum()),
            "coordinate_note": (
                "All 1000 standardized neuron dimensions are retained. An orthogonal full-rank PCA "
                "rotation gives a similar ridge operator with identical eigenvalues."
            ),
        }
        write_json(stage_dir / "precedent_summary.json", summary)
        np.savez_compressed(
            stage_dir / "precedent_operator.npz",
            operator=model.operator,
            eigenvalues=model.eigenvalues,
            eligible_rows=rows,
            mean=mean,
            scale=scale,
        )
        plot_precedent_spectrum(
            model.eigenvalues,
            diagnostics,
            summary,
            stage_dir / "05_precedent_eigenplane",
        )
        checks = {
            "finite_operator_and_spectrum": bool(
                model.diagnostics["finite_operator"] and model.diagnostics["finite_eigenvalues"]
            ),
            "svd_matches_v1_gram_operator": comparison_error < 1e-8,
            "relative_operator_difference": comparison_error,
        }
        write_json(stage_dir / "precedent_checks.json", checks)
        complete_stage(context, "precedent", checks)
        return context
    except Exception as error:
        fail_stage(context, "precedent", f"{type(error).__name__}: {error}")
        (context.root / "precedent_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def run_simulation_stage(config_path: str | Path | None = None) -> RunContext:
    config = load_config() if config_path is None else load_config(config_path)
    context = active_run()
    if stage_state(context, "precedent") != "completed":
        raise RuntimeError("The precedent check must complete before known-system simulation")
    start_stage(context, "simulation")
    try:
        signal_table = pd.read_csv(context.stage_dir("preprocessing") / "signal_statistics.csv")
        target_zero = float(signal_table.loc[signal_table["arm"] == "P", "zero_fraction"].median())
        metrics = run_known_systems(config, target_event_zero_fraction=target_zero)
        stage_dir = context.stage_dir("simulation")
        write_csv(stage_dir / "known_system_metrics.csv", metrics)
        summary_table = (
            metrics.groupby(["system_class", "observation"], sort=False)
            .agg(
                seeds=("seed", "nunique"),
                classification_accuracy=("classification_correct", "mean"),
                median_subspace_overlap=("latent_embedding_overlap", "median"),
                median_eigenvalue_relative_error=("median_eigenvalue_relative_error", "median"),
                median_skill_persistence=("skill_persistence_near_one_second", "median"),
                finite_fraction=("finite_fit", "mean"),
            )
            .reset_index()
        )
        write_csv(stage_dir / "known_system_summary.csv", summary_table)
        transformed = metrics[metrics["observation"].isin(["event", "smoothed_event", "calcium"])]
        gate_values = {
            "target_empirical_zero_fraction": target_zero,
            "classification_accuracy": float(transformed["classification_correct"].mean()),
            "median_subspace_overlap": float(transformed["latent_embedding_overlap"].median()),
            "median_eigenvalue_relative_error": float(
                transformed["median_eigenvalue_relative_error"].median()
            ),
            "median_skill_persistence_near_one_second": float(
                transformed["skill_persistence_near_one_second"].median()
            ),
            "finite_fraction": float(transformed["finite_fit"].mean()),
        }
        gates = config["gates"]
        gate_values["classification_pass"] = (
            gate_values["classification_accuracy"] >= gates["simulation_classification_fraction"]
        )
        gate_values["subspace_pass"] = (
            gate_values["median_subspace_overlap"] >= gates["simulation_subspace_overlap"]
        )
        gate_values["eigenvalue_error_pass"] = (
            gate_values["median_eigenvalue_relative_error"]
            <= gates["simulation_eigenvalue_relative_error"]
        )
        gate_values["forecast_pass"] = gate_values["median_skill_persistence_near_one_second"] > 0
        gate_values["gate_2_pass"] = bool(
            gate_values["classification_pass"]
            and gate_values["subspace_pass"]
            and gate_values["eigenvalue_error_pass"]
            and gate_values["forecast_pass"]
        )
        write_json(stage_dir / "simulation_gate.json", gate_values)
        plot_simulation_recovery(metrics, stage_dir / "06_known_system_recovery")
        complete_stage(context, "simulation", gate_values)
        return context
    except Exception as error:
        fail_stage(context, "simulation", f"{type(error).__name__}: {error}")
        (context.root / "simulation_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def run_forecast_stage(config_path: str | Path | None = None) -> RunContext:
    config = load_config() if config_path is None else load_config(config_path)
    context = active_run()
    if stage_state(context, "simulation") != "completed":
        raise RuntimeError("Known-system simulation must complete before empirical forecasting")
    simulation_gate = json.loads(
        (context.stage_dir("simulation") / "simulation_gate.json").read_text()
    )
    diagnostic_override = not bool(simulation_gate["gate_2_pass"])
    if diagnostic_override and not bool(
        config.get("execution", {}).get("continue_after_failed_simulation_gate", False)
    ):
        raise RuntimeError(
            "Gate 2 failed; set execution.continue_after_failed_simulation_gate=true "
            "only to complete explicitly diagnostic downstream stages"
        )
    start_stage(context, "forecast")
    try:
        metadata = empirical_metadata(config)
        window_table, windows = _load_window_table(context)
        preprocessing_dir = context.stage_dir("preprocessing")
        stage_dir = context.stage_dir("forecast")
        grid_tables: list[pd.DataFrame] = []
        summary_tables: list[pd.DataFrame] = []
        selected_predictive: dict[str, dict[str, dict[str, Any] | None]] = {}
        selected_tracking: dict[str, dict[str, dict[str, Any] | None]] = {}
        selected_rows: list[dict[str, Any]] = []

        for recording, meta in metadata.items():
            selected_predictive[recording] = {}
            selected_tracking[recording] = {}
            development = [
                window
                for window in windows
                if window.recording == recording
                and window.kind == "deployment"
                and window.split == "development"
            ]
            for arm_id, arm_values in config["arms"].items():
                spec = ArmSpec.from_config(arm_id, arm_values)
                representation = FrozenRepresentation.load(
                    preprocessing_dir / "models" / f"{recording}__{arm_id}.npz"
                )
                grid = development_grid(meta, development, spec, representation, config)
                grid_tables.append(grid)
                fs_effective = float(config["fs_hz"]) / spec.bin_frames
                summary = summarize_grid(grid, fs_effective, config)
                summary_tables.append(summary)
                predictive = select_configuration(
                    summary,
                    precedent_lag=int(config["model"]["precedent_lag_frames"]),
                )
                tracking = select_configuration(
                    summary,
                    precedent_lag=int(config["model"]["precedent_lag_frames"]),
                    minimum_rank=int(config["model"]["tracking_minimum_rank"]),
                )
                selected_predictive[recording][arm_id] = predictive
                selected_tracking[recording][arm_id] = tracking
                if predictive is not None:
                    selected_rows.append(
                        {
                            "recording": recording,
                            "arm": arm_id,
                            "selection_role": "predictive",
                            **predictive,
                        }
                    )
                if tracking is not None:
                    selected_rows.append(
                        {
                            "recording": recording,
                            "arm": arm_id,
                            "selection_role": "tracking",
                            **tracking,
                        }
                    )

        grid_table = pd.concat(grid_tables, ignore_index=True)
        summary_table = pd.concat(summary_tables, ignore_index=True)
        selected_table = pd.DataFrame(selected_rows)
        write_csv(stage_dir / "development_grid_metrics.csv", grid_table)
        write_csv(stage_dir / "development_grid_summary.csv", summary_table)
        write_csv(stage_dir / "selected_configurations.csv", selected_table)

        predictive_arm_scores: dict[str, float] = {}
        tracking_arm_scores: dict[str, float] = {}
        for arm_id in config["arms"]:
            predictive_values = [
                selected_predictive[recording][arm_id]["median_skill_one_second"]
                for recording in metadata
                if selected_predictive[recording][arm_id] is not None
            ]
            if len(predictive_values) == len(metadata):
                predictive_arm_scores[arm_id] = float(np.median(predictive_values))
            tracking_values = [
                selected_tracking[recording][arm_id]["median_skill_one_second"]
                for recording in metadata
                if selected_tracking[recording][arm_id] is not None
            ]
            if len(tracking_values) == len(metadata):
                tracking_arm_scores[arm_id] = float(np.median(tracking_values))
        global_predictive_arm = max(predictive_arm_scores, key=predictive_arm_scores.get)
        global_tracking_arm = (
            max(tracking_arm_scores, key=tracking_arm_scores.get) if tracking_arm_scores else None
        )
        selection_manifest = {
            "selected_predictive": selected_predictive,
            "selected_tracking": selected_tracking,
            "global_predictive_arm": global_predictive_arm,
            "global_tracking_arm": global_tracking_arm,
            "predictive_arm_development_scores": predictive_arm_scores,
            "tracking_arm_development_scores": tracking_arm_scores,
            "labels_used_for_selection": False,
        }
        write_json(stage_dir / "selection_manifest.json", selection_manifest)
        plot_development_selection(
            summary_table,
            selected_table[selected_table["selection_role"] == "predictive"],
            stage_dir / "07_development_selection",
        )

        heldout_tables: list[pd.DataFrame] = []
        spectrum_tables: list[pd.DataFrame] = []
        long_tables: list[pd.DataFrame] = []
        long_selected: list[dict[str, Any]] = []
        representative_panels: list[dict[str, object]] = []
        for recording, meta in metadata.items():
            evaluation = [
                window
                for window in windows
                if window.recording == recording
                and window.kind == "deployment"
                and window.split == "evaluation"
            ]
            long_windows = [
                window for window in windows if window.recording == recording and window.kind == "long"
            ]
            for arm_id, arm_values in config["arms"].items():
                selected = selected_predictive[recording][arm_id]
                if selected is None:
                    continue
                spec = ArmSpec.from_config(arm_id, arm_values)
                representation = FrozenRepresentation.load(
                    preprocessing_dir / "models" / f"{recording}__{arm_id}.npz"
                )
                metrics, spectra = evaluate_configuration(
                    meta,
                    evaluation,
                    spec,
                    representation,
                    selected,
                    config,
                )
                heldout_tables.append(metrics)
                spectrum_tables.append(spectra)
                if arm_id in {"Sz", "Sc"}:
                    sensitivity_metrics, sensitivity_spectra = evaluate_configuration(
                        meta,
                        evaluation,
                        spec,
                        representation,
                        selected,
                        config,
                        trim_samples=int(config["windows"]["stored_smoothing_half_width_frames"]),
                        sensitivity_name="trimmed_seven_frame_edges",
                    )
                    heldout_tables.append(sensitivity_metrics)
                    spectrum_tables.append(sensitivity_spectra)
                for long_window in long_windows:
                    long_metrics, long_choice = fit_long_block(
                        meta,
                        long_window,
                        spec,
                        representation,
                        config,
                    )
                    long_tables.append(long_metrics)
                    long_selected.append(
                        {
                            "recording": recording,
                            "label": long_window.label,
                            "arm": arm_id,
                            **long_choice,
                        }
                    )
                if arm_id == global_predictive_arm:
                    for label in config["recordings"][recording]["labels"]:
                        window = sorted(
                            [candidate for candidate in evaluation if candidate.label == label],
                            key=lambda candidate: candidate.start,
                        )[0]
                        representative_panels.append(
                            representative_prediction(
                                meta,
                                window,
                                spec,
                                representation,
                                selected,
                                config,
                            )
                        )

        heldout = pd.concat(heldout_tables, ignore_index=True)
        spectra = pd.concat(spectrum_tables, ignore_index=True)
        long_metrics = pd.concat(long_tables, ignore_index=True)
        write_csv(stage_dir / "heldout_metrics.csv", heldout)
        write_csv(stage_dir / "heldout_eigenvalues.csv", spectra)
        write_csv(stage_dir / "long_block_metrics.csv", long_metrics)
        write_csv(stage_dir / "long_block_selected_configurations.csv", pd.DataFrame(long_selected))
        plot_heldout_skill(heldout, stage_dir / "08_heldout_forecast_skill")
        plot_representative_predictions(
            representative_panels,
            stage_dir / "09_representative_predictions",
        )
        plot_empirical_spectra(
            spectra[spectra["sensitivity"] == "primary"],
            stage_dir / "10_heldout_eigenplanes",
        )
        plot_long_block_comparison(
            heldout,
            long_metrics,
            stage_dir / "11_long_block_diagnostic",
        )
        plot_diagonal_gain(heldout, stage_dir / "12_multivariate_gain")
        plot_smoothing_trim_sensitivity(
            heldout,
            stage_dir / "13_smoothing_trim_sensitivity",
        )

        primary = heldout[
            (heldout["sensitivity"] == "primary") & (heldout["horizon_role"] == "gate")
        ]
        near_one = primary[(primary["actual_horizon_seconds"] - 1.0).abs() < 0.2]
        cell_summary = (
            near_one.groupby(["recording", "label", "arm"], as_index=False)
            .agg(
                windows=("window_id", "nunique"),
                median_skill_persistence=("skill_persistence", "median"),
                median_skill_diagonal=("skill_diagonal", "median"),
                finite_fraction=("finite_forecast", "mean"),
                explosive_fraction=("explosive_forecast", "mean"),
            )
        )
        write_csv(stage_dir / "heldout_cell_summary.csv", cell_summary)
        positive_by_arm = (
            cell_summary.assign(positive=cell_summary["median_skill_persistence"] > 0)
            .groupby("arm")["positive"]
            .sum()
            .to_dict()
        )
        preliminary = {
            "global_predictive_arm": global_predictive_arm,
            "global_tracking_arm": global_tracking_arm,
            "positive_cells_by_arm_before_bootstrap_ci": positive_by_arm,
            "maximum_positive_cells": int(max(positive_by_arm.values())),
            "all_primary_forecasts_finite": bool(primary["finite_forecast"].all()),
            "primary_explosive_fraction": float(primary["explosive_forecast"].mean()),
            "bootstrap_confidence_bounds_pending": True,
            "simulation_gate_2_pass": bool(simulation_gate["gate_2_pass"]),
            "diagnostic_override_after_failed_simulation_gate": diagnostic_override,
            "downstream_interpretation": (
                config.get("execution", {}).get("failed_gate_interpretation")
                if diagnostic_override
                else "confirmatory_gate_sequence"
            ),
        }
        write_json(stage_dir / "forecast_preliminary_checks.json", preliminary)
        complete_stage(context, "forecast", preliminary)
        return context
    except Exception as error:
        fail_stage(context, "forecast", f"{type(error).__name__}: {error}")
        (context.root / "forecast_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def run_stability_stage(config_path: str | Path | None = None) -> RunContext:
    """Run moving-block uncertainty, fixed-reference stability, and subset checks."""
    config = load_config() if config_path is None else load_config(config_path)
    context = active_run()
    if stage_state(context, "forecast") != "completed":
        raise RuntimeError("Forecasting must complete before stability analysis")
    start_stage(context, "stability")
    try:
        metadata = empirical_metadata(config)
        _, windows = _load_window_table(context)
        forecast_dir = context.stage_dir("forecast")
        preprocessing_dir = context.stage_dir("preprocessing")
        stage_dir = context.stage_dir("stability")
        selection = json.loads((forecast_dir / "selection_manifest.json").read_text())
        predictive_arm = str(selection["global_predictive_arm"])
        tracking_arm = str(selection["global_tracking_arm"])
        if predictive_arm != tracking_arm:
            raise RuntimeError(
                "This initial screen requires one development-selected arm for both prediction and tracking"
            )
        arm_id = predictive_arm
        spec = ArmSpec.from_config(arm_id, config["arms"][arm_id])

        reference_rows: list[dict[str, Any]] = []
        original_tables: list[pd.DataFrame] = []
        predictive_bootstrap_tables: list[pd.DataFrame] = []
        tracking_bootstrap_tables: list[pd.DataFrame] = []
        between_tables: list[pd.DataFrame] = []
        tracking_summary_rows: list[dict[str, Any]] = []
        subset_fit_tables: list[pd.DataFrame] = []
        subset_membership_tables: list[pd.DataFrame] = []
        subset_agreement_tables: list[pd.DataFrame] = []
        subset_audits: list[dict[str, Any]] = []
        projector_archive: dict[str, np.ndarray] = {}

        for recording_index, (recording, meta) in enumerate(metadata.items()):
            development = [
                window
                for window in windows
                if window.recording == recording
                and window.kind == "deployment"
                and window.split == "development"
            ]
            evaluation = [
                window
                for window in windows
                if window.recording == recording
                and window.kind == "deployment"
                and window.split == "evaluation"
            ]
            predictive_selected = selection["selected_predictive"][recording][arm_id]
            tracking_selected = selection["selected_tracking"][recording][arm_id]
            if predictive_selected is None or tracking_selected is None:
                raise RuntimeError(f"No eligible selected {arm_id} configuration for {recording}")
            representation = FrozenRepresentation.load(
                preprocessing_dir / "models" / f"{recording}__{arm_id}.npz"
            )
            tracking_rank = int(tracking_selected["rank"])
            development_scores = [
                load_scores(meta, window, spec, representation, rank=tracking_rank)
                for window in development
            ]
            reference_model, reference_subspace = fit_development_reference(
                development_scores,
                rank=tracking_rank,
                lag=int(tracking_selected["lag"]),
                eta=float(tracking_selected["eta"]),
                target_dimension=int(config["model"]["tracking_subspace_dimension"]),
                config=config,
            )
            projector_archive[f"{recording}__development_reference"] = (
                reference_subspace.projector
            )
            for group_index, group in enumerate(reference_subspace.groups):
                for eigen_index, eigenvalue in zip(
                    group.indices, group.eigenvalues, strict=True
                ):
                    reference_rows.append(
                        {
                            "recording": recording,
                            "arm": arm_id,
                            "rank": tracking_rank,
                            "lag": int(tracking_selected["lag"]),
                            "eta": float(tracking_selected["eta"]),
                            "group_index": group_index,
                            "group_dimension": group.dimension,
                            "group_energy": group.energy,
                            "operator_eigen_index": eigen_index,
                            "eigenvalue_real": eigenvalue.real,
                            "eigenvalue_imag": eigenvalue.imag,
                            "reference_captured_energy": reference_subspace.captured_energy,
                            **reference_model.diagnostics,
                        }
                    )
            np.savez_compressed(
                stage_dir / f"reference__{recording}.npz",
                operator=reference_model.operator,
                eigenvalues=reference_model.eigenvalues,
                basis=reference_subspace.basis,
                projector=reference_subspace.projector,
            )

            original, original_subspaces, tracking_scores = evaluate_tracking_windows(
                meta,
                evaluation,
                spec,
                representation,
                tracking_selected,
                reference_subspace,
                config,
            )
            original_tables.append(original)
            for window_id, subspace in original_subspaces.items():
                projector_archive[f"{recording}__{window_id}"] = subspace.projector

            predictive_bootstrap_tables.append(
                bootstrap_predictive_windows(
                    evaluation,
                    tracking_scores,
                    predictive_selected,
                    spec,
                    config,
                    recording_index,
                )
            )
            tracking_bootstrap = bootstrap_tracking_windows(
                evaluation,
                tracking_scores,
                tracking_selected,
                reference_subspace,
                original_subspaces,
                config,
                recording_index,
            )
            tracking_bootstrap_tables.append(tracking_bootstrap)
            between, resolution = tracking_resolution(
                original_subspaces,
                evaluation,
                tracking_bootstrap,
            )
            between_tables.append(between)
            resolution["original_matched_windows"] = len(original_subspaces)
            resolution["expected_original_windows"] = len(evaluation)
            resolution["tracking_resolution_eligible"] = bool(
                len(original_subspaces) == len(evaluation)
                and resolution["between_distance_count"] == 6
                and float(tracking_bootstrap["match_success"].mean())
                >= float(config["gates"]["finite_fit_fraction"])
            )
            tracking_summary_rows.append({"recording": recording, "arm": arm_id, **resolution})

            subset_fits, subset_membership, subset_result = (
                deterministic_neuron_subset_classes(
                    meta,
                    development,
                    evaluation,
                    spec,
                    representation,
                    tracking_selected,
                    config,
                    recording_index,
                )
            )
            subset_fit_tables.append(subset_fits)
            subset_membership_tables.append(subset_membership)
            subset_agreement_tables.append(subset_result["agreement"])
            subset_audits.append({"recording": recording, **subset_result["audit"]})

        reference_table = pd.DataFrame(reference_rows)
        original_table = pd.concat(original_tables, ignore_index=True)
        predictive_bootstrap = pd.concat(predictive_bootstrap_tables, ignore_index=True)
        tracking_bootstrap = pd.concat(tracking_bootstrap_tables, ignore_index=True)
        between_table = pd.concat(between_tables, ignore_index=True)
        tracking_summary = pd.DataFrame(tracking_summary_rows)
        subset_fits = pd.concat(subset_fit_tables, ignore_index=True)
        subset_membership = pd.concat(subset_membership_tables, ignore_index=True)
        subset_agreement = pd.concat(subset_agreement_tables, ignore_index=True)
        _, predictive_cell_summary = synchronized_cell_intervals(
            predictive_bootstrap,
            confidence_level=float(config["inference"]["confidence_level"]),
        )
        synchronized_stability = (
            tracking_bootstrap[tracking_bootstrap["match_success"]]
            .groupby(["recording", "label", "repetition"], as_index=False)
            .agg(
                windows=("window_id", "nunique"),
                median_reference_similarity=("reference_similarity", "median"),
                median_within_window_distance=("within_window_distance", "median"),
            )
        )
        stability_cell_summary = (
            synchronized_stability.groupby(["recording", "label"], as_index=False)
            .agg(
                repetitions=("repetition", "nunique"),
                windows_per_repetition_min=("windows", "min"),
                median_reference_similarity=("median_reference_similarity", "median"),
                median_within_window_distance=("median_within_window_distance", "median"),
            )
        )
        match_fraction = (
            tracking_bootstrap.groupby(["recording", "label"], as_index=False)["match_success"]
            .mean()
            .rename(columns={"match_success": "bootstrap_match_fraction"})
        )
        stability_cell_summary = stability_cell_summary.merge(
            match_fraction,
            on=["recording", "label"],
            how="outer",
        )
        stability_cell_summary["gate_eligible"] = (
            (stability_cell_summary["windows_per_repetition_min"] == 3)
            & (
                stability_cell_summary["bootstrap_match_fraction"]
                >= float(config["gates"]["finite_fit_fraction"])
            )
        )

        write_csv(stage_dir / "reference_eigengroups.csv", reference_table)
        write_csv(stage_dir / "original_tracking_fits.csv", original_table)
        write_csv(stage_dir / "predictive_bootstrap_metrics.csv", predictive_bootstrap)
        write_csv(stage_dir / "predictive_bootstrap_cell_summary.csv", predictive_cell_summary)
        write_csv(stage_dir / "tracking_bootstrap_metrics.csv", tracking_bootstrap)
        write_csv(stage_dir / "tracking_stability_cell_summary.csv", stability_cell_summary)
        write_csv(stage_dir / "between_window_projector_distances.csv", between_table)
        write_csv(stage_dir / "tracking_resolution_summary.csv", tracking_summary)
        write_csv(stage_dir / "neuron_subset_fits.csv", subset_fits)
        write_csv(stage_dir / "neuron_subset_membership.csv", subset_membership)
        write_csv(stage_dir / "neuron_subset_class_agreement.csv", subset_agreement)
        write_csv(stage_dir / "neuron_subset_audit.csv", pd.DataFrame(subset_audits))
        np.savez_compressed(stage_dir / "matched_projectors.npz", **projector_archive)

        failure_table = pd.concat(
            [
                original_table.loc[
                    ~original_table["match_success"],
                    ["recording", "label", "window_id", "failure"],
                ].assign(source="original"),
                tracking_bootstrap.loc[
                    ~tracking_bootstrap["match_success"],
                    ["recording", "label", "window_id", "repetition", "failure"],
                ].assign(source="bootstrap"),
            ],
            ignore_index=True,
        )
        write_csv(stage_dir / "matching_failures.csv", failure_table)

        plot_bootstrap_forecast_intervals(
            predictive_cell_summary,
            stage_dir / "14_bootstrap_forecast_intervals",
        )
        plot_subspace_stability(
            original_table,
            tracking_bootstrap,
            stage_dir / "15_subspace_stability",
        )
        plot_tracking_resolution(
            tracking_summary,
            stage_dir / "16_tracking_resolution",
        )
        plot_neuron_subset_classes(
            subset_agreement,
            stage_dir / "17_neuron_subset_classes",
        )

        near_one = predictive_cell_summary[
            predictive_cell_summary["horizon_role"] == "near_one_second"
        ]
        near_two = predictive_cell_summary[
            predictive_cell_summary["horizon_role"] == "near_two_seconds"
        ]
        positive_ci_cells = int(np.sum(near_one["skill_persistence_ci_lower"] > 0))
        stable_cells = int(
            np.sum(
                stability_cell_summary["gate_eligible"]
                & (
                    stability_cell_summary["median_reference_similarity"]
                    >= float(config["gates"]["empirical_subspace_overlap"])
                )
            )
        )
        summary = {
            "selected_arm": arm_id,
            "predictive_bootstrap_rows": int(len(predictive_bootstrap)),
            "tracking_bootstrap_rows": int(len(tracking_bootstrap)),
            "all_bootstrap_pairs_within_blocks": bool(
                (predictive_bootstrap["cross_block_pairs"] == 0).all()
                and (tracking_bootstrap["cross_block_pairs"] == 0).all()
            ),
            "positive_near_one_second_ci_cells": positive_ci_cells,
            "near_two_second_ci_median_positive_cells": int(
                np.sum(near_two["bootstrap_median_skill_persistence"] > 0)
            ),
            "gate_3_bootstrap_component_pass": bool(
                positive_ci_cells >= int(config["gates"]["empirical_required_cells"])
            ),
            "subspace_threshold_cells": stable_cells,
            "subset_classes_agree_all_cells": bool(
                subset_agreement["subset_class_agreement"].all()
            ),
            "gate_4_pass": bool(
                stable_cells >= int(config["gates"]["empirical_required_cells"])
                and subset_agreement["subset_class_agreement"].all()
            ),
            "gate_5_pass": bool(
                tracking_summary["tracking_resolution_eligible"].all()
                and (
                    tracking_summary["tracking_resolution_ratio"]
                    < float(config["gates"]["tracking_resolution_ratio"])
                ).all()
            ),
            "matching_failure_count": int(len(failure_table)),
        }
        write_json(stage_dir / "stability_gate_components.json", summary)
        complete_stage(context, "stability", summary)
        return context
    except Exception as error:
        fail_stage(context, "stability", f"{type(error).__name__}: {error}")
        (context.root / "stability_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def run_nulls_stage(config_path: str | Path | None = None) -> RunContext:
    """Run within-bout neuronwise circular shifts and stationary empirical nulls."""
    config = load_config() if config_path is None else load_config(config_path)
    context = active_run()
    if stage_state(context, "stability") != "completed":
        raise RuntimeError("Stability analysis must complete before negative controls")
    start_stage(context, "nulls")
    try:
        metadata = empirical_metadata(config)
        _, windows = _load_window_table(context)
        forecast_dir = context.stage_dir("forecast")
        stability_dir = context.stage_dir("stability")
        preprocessing_dir = context.stage_dir("preprocessing")
        stage_dir = context.stage_dir("nulls")
        selection = json.loads((forecast_dir / "selection_manifest.json").read_text())
        arm_id = str(selection["global_predictive_arm"])
        if arm_id != str(selection["global_tracking_arm"]):
            raise RuntimeError("Prediction and tracking must use the same frozen arm in this screen")
        spec = ArmSpec.from_config(arm_id, config["arms"][arm_id])
        null_tables: list[pd.DataFrame] = []
        for recording_index, (recording, meta) in enumerate(metadata.items()):
            development = [
                window
                for window in windows
                if window.recording == recording
                and window.kind == "deployment"
                and window.split == "development"
            ]
            evaluation = [
                window
                for window in windows
                if window.recording == recording
                and window.kind == "deployment"
                and window.split == "evaluation"
            ]
            predictive_selected = selection["selected_predictive"][recording][arm_id]
            tracking_selected = selection["selected_tracking"][recording][arm_id]
            if predictive_selected is None or tracking_selected is None:
                raise RuntimeError(f"Missing selected configuration for {recording}:{arm_id}")
            representation = FrozenRepresentation.load(
                preprocessing_dir / "models" / f"{recording}__{arm_id}.npz"
            )
            tracking_rank = int(tracking_selected["rank"])
            development_scores = [
                load_scores(meta, window, spec, representation, rank=tracking_rank)
                for window in development
            ]
            _, reference = fit_development_reference(
                development_scores,
                rank=tracking_rank,
                lag=int(tracking_selected["lag"]),
                eta=float(tracking_selected["eta"]),
                target_dimension=int(config["model"]["tracking_subspace_dimension"]),
                config=config,
            )
            null_tables.append(
                evaluate_negative_controls(
                    meta,
                    evaluation,
                    spec,
                    representation,
                    predictive_selected,
                    tracking_selected,
                    reference,
                    config,
                    recording_index,
                )
            )
        null_metrics = pd.concat(null_tables, ignore_index=True)
        heldout = pd.read_csv(forecast_dir / "heldout_metrics.csv")
        synchronized, null_summary = summarize_negative_controls(
            null_metrics,
            heldout,
            percentile=float(config["gates"]["null_percentile"]),
        )
        original_tracking = pd.read_csv(stability_dir / "original_tracking_fits.csv")
        write_csv(stage_dir / "negative_control_metrics.csv", null_metrics)
        write_csv(stage_dir / "negative_control_synchronized_cells.csv", synchronized)
        write_csv(stage_dir / "negative_control_cell_summary.csv", null_summary)
        failures = null_metrics.loc[
            ~null_metrics["tracking_match_success"],
            [
                "null_kind",
                "recording",
                "label",
                "window_id",
                "repetition",
                "tracking_failure",
            ],
        ].drop_duplicates()
        write_csv(stage_dir / "negative_control_matching_failures.csv", failures)
        plot_null_multivariate_gain(
            synchronized,
            null_summary,
            stage_dir / "18_null_multivariate_gain",
        )
        plot_null_subspace_similarity(
            synchronized,
            original_tracking,
            stage_dir / "19_null_subspace_similarity",
        )

        circular_near_one = null_summary[
            (null_summary["null_kind"] == "circular_shift")
            & (null_summary["horizon_role"] == "near_one_second")
        ]
        qualifier_cells = int(np.sum(circular_near_one["coordinated_mode_qualifier_cell"]))
        summary = {
            "selected_arm": arm_id,
            "null_metric_rows": int(len(null_metrics)),
            "circular_shift_repetitions_per_window": int(
                config["resampling"]["circular_shift_repetitions"]
            ),
            "stationary_repetitions_per_window": int(
                config["resampling"]["stationary_null_repetitions"]
            ),
            "all_circular_sources_inside_constant_label_bouts": bool(
                null_metrics["source_indices_inside_bout"].all()
            ),
            "tracking_match_fraction": float(null_metrics["tracking_match_success"].mean()),
            "coordinated_mode_qualifier_cells": qualifier_cells,
            "coordinated_mode_qualifier_any_cell": bool(qualifier_cells > 0),
            "coordinated_mode_qualifier_all_cells": bool(
                qualifier_cells == len(circular_near_one)
            ),
            "matching_failure_count": int(len(failures)),
        }
        write_json(stage_dir / "negative_control_summary.json", summary)
        complete_stage(context, "nulls", summary)
        return context
    except Exception as error:
        fail_stage(context, "nulls", f"{type(error).__name__}: {error}")
        (context.root / "nulls_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def run_decision_stage(config_path: str | Path | None = None) -> RunContext:
    """Apply the frozen feasibility gates and save a machine-auditable decision."""
    config = load_config() if config_path is None else load_config(config_path)
    context = active_run()
    if stage_state(context, "nulls") != "completed":
        raise RuntimeError("Negative controls must complete before the gate decision")
    start_stage(context, "decision")
    try:
        data_dir = context.stage_dir("data")
        simulation_dir = context.stage_dir("simulation")
        forecast_dir = context.stage_dir("forecast")
        stability_dir = context.stage_dir("stability")
        nulls_dir = context.stage_dir("nulls")
        stage_dir = context.stage_dir("decision")
        geometry = json.loads((data_dir / "geometry_checks.json").read_text())
        simulation = json.loads((simulation_dir / "simulation_gate.json").read_text())
        stability = json.loads(
            (stability_dir / "stability_gate_components.json").read_text()
        )
        nulls = json.loads((nulls_dir / "negative_control_summary.json").read_text())
        heldout = pd.read_csv(forecast_dir / "heldout_metrics.csv")
        primary = heldout[
            (heldout["sensitivity"] == "primary") & (heldout["horizon_role"] == "gate")
        ]
        finite_fraction = float(
            np.mean(
                primary["finite_operator"]
                & primary["finite_eigenvalues"]
                & primary["finite_forecast"]
            )
        )
        explosive_fraction = float(primary["explosive_forecast"].mean())
        gate_1 = bool(
            geometry["overlapping_deployment_pairs"] == 0
            and geometry["label_or_boundary_violations"] == 0
            and finite_fraction >= float(config["gates"]["finite_fit_fraction"])
            and stability["all_bootstrap_pairs_within_blocks"]
            and nulls["all_circular_sources_inside_constant_label_bouts"]
        )
        gate_2 = bool(simulation["gate_2_pass"])
        gate_3 = bool(
            stability["gate_3_bootstrap_component_pass"]
            and stability["near_two_second_ci_median_positive_cells"]
            >= int(config["gates"]["empirical_required_cells"])
            and finite_fraction >= float(config["gates"]["finite_fit_fraction"])
            and explosive_fraction <= float(config["gates"]["maximum_explosive_fraction"])
        )
        gate_4 = bool(stability["gate_4_pass"])
        gate_5 = bool(stability["gate_5_pass"])
        gates = pd.DataFrame(
            [
                {
                    "gate": "Gate 1 — geometry/numerics",
                    "threshold": "zero illegal joins; >=95% finite",
                    "observed": (
                        f"illegal geometry=0; finite={100*finite_fraction:.1f}%; "
                        "bootstrap joins=0"
                    ),
                    "pass": gate_1,
                },
                {
                    "gate": "Gate 2 — known-system recovery",
                    "threshold": "class>=0.80; Ssub>=0.75; eig err<=0.25; skill>0",
                    "observed": (
                        f"class={simulation['classification_accuracy']:.3f}; "
                        f"Ssub={simulation['median_subspace_overlap']:.3f}; "
                        f"eig err={simulation['median_eigenvalue_relative_error']:.3f}"
                    ),
                    "pass": gate_2,
                },
                {
                    "gate": "Gate 3 — empirical prediction",
                    "threshold": "lower CI>0 in >=3/4 cells; stable near 2 s",
                    "observed": (
                        f"CI-positive cells={stability['positive_near_one_second_ci_cells']}/4; "
                        f"2-s positive={stability['near_two_second_ci_median_positive_cells']}/4"
                    ),
                    "pass": gate_3,
                },
                {
                    "gate": "Gate 4 — reproducibility",
                    "threshold": "median Ssub>=0.80 in >=3/4; subset classes agree",
                    "observed": (
                        f"stable cells={stability['subspace_threshold_cells']}/4; "
                        f"subset agreement={stability['subset_classes_agree_all_cells']}"
                    ),
                    "pass": gate_4,
                },
                {
                    "gate": "Gate 5 — tracking resolution",
                    "threshold": "Rtrack<0.5 in both recordings",
                    "observed": "see tracking_resolution_summary.csv",
                    "pass": gate_5,
                },
            ]
        )
        all_gates = bool(gates["pass"].all())
        coordinated_qualifier = bool(nulls["coordinated_mode_qualifier_all_cells"])
        if all_gates and coordinated_qualifier:
            decision = "PASS"
            reason = "All five frozen feasibility gates and the coordinated-mode qualifier passed."
        elif not gate_2:
            decision = "FAIL"
            reason = (
                "The frozen rule declares failure when sparse/calcium-like known-system "
                "recovery fails, regardless of empirical prediction."
            )
        elif not gate_3 or not gate_4 or not gate_5:
            decision = "FAIL"
            reason = "Prediction or subspace estimation is not sufficiently reproducible/resolvable."
        else:
            decision = "CONDITIONAL"
            reason = "Core gates passed, but diagonal/null evidence limits the interpretation."
        decision_record = {
            "decision": decision,
            "reason": reason,
            "gate_1_pass": gate_1,
            "gate_2_pass": gate_2,
            "gate_3_pass": gate_3,
            "gate_4_pass": gate_4,
            "gate_5_pass": gate_5,
            "all_five_gates_pass": all_gates,
            "coordinated_mode_qualifier_pass": coordinated_qualifier,
            "coordinated_mode_qualifier_cells": nulls[
                "coordinated_mode_qualifier_cells"
            ],
            "downstream_work_authorized": False,
            "review_pause_required": True,
        }
        write_csv(stage_dir / "gate_table.csv", gates)
        write_json(stage_dir / "final_decision.json", decision_record)
        plot_gate_summary(gates, decision, stage_dir / "20_gate_summary")
        complete_stage(context, "decision", decision_record)
        return context
    except Exception as error:
        fail_stage(context, "decision", f"{type(error).__name__}: {error}")
        (context.root / "decision_failure_traceback.txt").write_text(traceback.format_exc())
        raise


def run_stage(stage: str, config_path: str | Path | None = None) -> RunContext:
    if stage == "data":
        return run_data_stage(config_path)
    if stage == "preprocessing":
        return run_preprocessing_stage(config_path)
    if stage == "precedent":
        return run_precedent_stage(config_path)
    if stage == "simulation":
        return run_simulation_stage(config_path)
    if stage == "forecast":
        return run_forecast_stage(config_path)
    if stage == "stability":
        return run_stability_stage(config_path)
    if stage == "nulls":
        return run_nulls_stage(config_path)
    if stage == "decision":
        return run_decision_stage(config_path)
    raise NotImplementedError(f"Stage {stage!r} is not implemented yet")
