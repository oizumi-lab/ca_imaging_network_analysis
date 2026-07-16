"""Numbered, self-explanatory intermediate figures for the pilot report."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/dmd-matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch


COLORS = {
    "P": "#0072B2",
    "B4": "#56B4E9",
    "Sz": "#009E73",
    "Sc": "#E69F00",
    "F": "#CC79A7",
    "development": "#0072B2",
    "evaluation": "#D55E00",
    "diagnostic": "#6A3D9A",
}


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 180,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.titleweight": "bold",
            "legend.frameon": False,
        }
    )


def save_figure(fig: plt.Figure, base_path: str | Path) -> None:
    base_path = Path(base_path)
    base_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(base_path.with_suffix(".png"), bbox_inches="tight", facecolor="white")
    fig.savefig(base_path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_window_map(
    metadata: dict[str, object],
    windows: pd.DataFrame,
    fs_hz: float,
    output: str | Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(len(metadata), 1, figsize=(13, 2.8 * len(metadata)), squeeze=False)
    label_colors = {0.0: "#d9d9d9", 0.5: "#fdbf6f", 1.0: "#33a02c", 2.0: "#6a3d9a"}
    for axis, (name, meta) in zip(axes[:, 0], metadata.items(), strict=True):
        time = np.arange(meta.n_frames) / fs_hz / 60
        axis.step(time, meta.state, where="post", color="#444444", linewidth=0.7)
        for code in np.unique(meta.state):
            mask = np.isclose(meta.state, code)
            changes = np.diff(np.r_[False, mask, False].astype(int))
            for start, stop in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1), strict=True):
                axis.axvspan(time[start], (stop / fs_hz / 60), color=label_colors.get(float(code), "#bbbbbb"), alpha=0.22)
        subset = windows[(windows["recording"] == name) & (windows["kind"] == "deployment")]
        for row in subset.itertuples():
            axis.axvspan(
                row.start / fs_hz / 60,
                row.stop / fs_hz / 60,
                color=COLORS[row.split],
                alpha=0.45,
                ymin=0.05,
                ymax=0.28,
            )
        long_subset = windows[(windows["recording"] == name) & (windows["kind"] == "long")]
        for row in long_subset.itertuples():
            axis.axvspan(
                row.start / fs_hz / 60,
                row.stop / fs_hz / 60,
                color=COLORS["diagnostic"],
                alpha=0.18,
                ymin=0.73,
                ymax=0.96,
            )
        for stop in meta.segment_stops[:-1]:
            axis.axvline(stop / fs_hz / 60, color="black", linestyle="--", linewidth=1)
        axis.set(title=f"{name}: raw labels and predeclared windows", ylabel="state code")
        axis.set_xlim(0, meta.n_frames / fs_hz / 60)
    axes[-1, 0].set_xlabel("Time (min)")
    fig.legend(
        handles=[
            Patch(color=COLORS["development"], alpha=0.45, label="Development W=300"),
            Patch(color=COLORS["evaluation"], alpha=0.45, label="Untouched evaluation W=300"),
            Patch(color=COLORS["diagnostic"], alpha=0.25, label="W=1500 diagnostic"),
        ],
        ncol=3,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.945),
    )
    fig.suptitle(
        "Stage 1A — Window geometry before any DMD fit",
        y=0.995,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    save_figure(fig, output)


def plot_signal_diagnostics(
    statistics: pd.DataFrame,
    autocorrelation: pd.DataFrame,
    output: str | Path,
) -> None:
    apply_style()
    recordings = list(statistics["recording"].drop_duplicates())
    fig, axes = plt.subplots(2, len(recordings), figsize=(7 * len(recordings), 7), squeeze=False)
    for column, recording in enumerate(recordings):
        stats = statistics[statistics["recording"] == recording]
        axis = axes[0, column]
        axis.bar(stats["arm"], 100 * stats["zero_fraction"], color=[COLORS[a] for a in stats["arm"]])
        axis.set_ylim(0, 102)
        axis.set_ylabel("Exact zeros (%)")
        axis.set_title(f"{recording}: sparsity")
        for index, value in enumerate(stats["zero_fraction"]):
            axis.text(index, 100 * value + 1, f"{100*value:.1f}", ha="center", va="bottom", fontsize=8)

        axis = axes[1, column]
        acf = autocorrelation[autocorrelation["recording"] == recording]
        for arm, group in acf.groupby("arm", sort=False):
            axis.plot(group["lag_seconds"], group["autocorrelation"], marker="o", color=COLORS[arm], label=arm)
        axis.axhline(0, color="#777777", linewidth=0.6)
        axis.set(xlabel="Lag (s)", ylabel="Pooled within-window autocorrelation", title=f"{recording}: temporal memory")
        axis.legend(ncol=3)
    fig.suptitle("Stage 1B — Candidate signals: sparsity and autocorrelation", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_pca_diagnostics(pca_table: pd.DataFrame, output: str | Path) -> None:
    apply_style()
    recordings = list(pca_table["recording"].drop_duplicates())
    fig, axes = plt.subplots(1, len(recordings), figsize=(7 * len(recordings), 4.8), squeeze=False)
    for axis, recording in zip(axes[0], recordings, strict=True):
        subset = pca_table[pca_table["recording"] == recording]
        for arm, group in subset.groupby("arm", sort=False):
            group = group.sort_values("component")
            axis.plot(
                group["component"],
                100 * group["cumulative_explained_variance"],
                marker=".",
                color=COLORS[arm],
                label=arm,
            )
        axis.set(
            xlabel="Number of retained PCs",
            ylabel="Cumulative development variance (%)",
            title=recording,
            xlim=(1, int(subset["component"].max())),
        )
        axis.legend(ncol=3)
    fig.suptitle("Stage 1C — Frozen recording-level PCA capacity", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_all_neuron_heatmaps(
    panels: Iterable[dict[str, object]],
    recording: str,
    output: str | Path,
) -> None:
    """Plot every eligible neuron for every arm and both labels; no row subsampling."""
    apply_style()
    panels = list(panels)
    arms = list(dict.fromkeys(str(panel["arm"]) for panel in panels))
    labels = list(dict.fromkeys(str(panel["label"]) for panel in panels))
    fig, axes = plt.subplots(len(arms), len(labels), figsize=(7 * len(labels), 2.1 * len(arms)), squeeze=False)
    for row_index, arm in enumerate(arms):
        for column_index, label in enumerate(labels):
            panel = next(item for item in panels if item["arm"] == arm and item["label"] == label)
            values = np.asarray(panel["values"])
            ordering = np.argsort(np.mean(np.abs(values), axis=1))[::-1]
            ordered = values[ordering]
            finite = ordered[np.isfinite(ordered)]
            limit = float(np.quantile(np.abs(finite), 0.995)) if finite.size else 1.0
            if limit <= 0:
                limit = 1.0
            image = axes[row_index, column_index].imshow(
                np.clip(ordered, -limit, limit),
                aspect="auto",
                interpolation="nearest",
                cmap="coolwarm",
                vmin=-limit,
                vmax=limit,
                rasterized=True,
            )
            axes[row_index, column_index].set_title(
                f"{arm} · {label} · all {values.shape[0]:,} eligible neurons"
            )
            axes[row_index, column_index].set_ylabel("Neurons (activity ordered)")
            axes[row_index, column_index].set_xlabel("Bin" if arm == "B4" else "Frame")
            fig.colorbar(image, ax=axes[row_index, column_index], fraction=0.02, pad=0.01)
    fig.suptitle(
        f"Stage 1D — {recording}: train-scaled activity in representative evaluation windows",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_pc_traces(
    panels: Iterable[dict[str, object]],
    recording: str,
    fs_hz: float,
    output: str | Path,
) -> None:
    apply_style()
    panels = list(panels)
    fig, axes = plt.subplots(len(panels), 1, figsize=(13, 1.8 * len(panels)), squeeze=False)
    for axis, panel in zip(axes[:, 0], panels, strict=True):
        scores = np.asarray(panel["scores"])
        effective_fs = fs_hz / int(panel["bin_frames"])
        time = np.arange(scores.shape[1]) / effective_fs
        offsets = 4 * np.arange(min(4, scores.shape[0]))
        scaled_scores = scores[: len(offsets)] / np.maximum(np.std(scores[: len(offsets)], axis=1, keepdims=True), 1e-12)
        for index, offset in enumerate(offsets):
            axis.plot(time, scaled_scores[index] + offset, linewidth=0.8, label=f"PC{index+1}")
        axis.set(title=f"{panel['arm']} · {panel['label']}", ylabel="PC (offset)")
    axes[-1, 0].set_xlabel("Time within evaluation window (s)")
    fig.suptitle(
        f"Stage 1E — {recording}: first four frozen-basis PC traces",
        y=0.995,
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    save_figure(fig, output)


def plot_precedent_spectrum(
    eigenvalues: np.ndarray,
    diagnostics: pd.DataFrame,
    summary: dict[str, object],
    output: str | Path,
) -> None:
    apply_style()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    theta = np.linspace(0, 2 * np.pi, 400)
    axes[0].plot(np.cos(theta), np.sin(theta), color="#777777", linestyle="--", linewidth=0.8)
    code_mask = diagnostics["v1_code_filter"].to_numpy(bool)
    axes[0].scatter(eigenvalues.real[~code_mask], eigenvalues.imag[~code_mask], s=7, alpha=0.35, color="#999999", label=r"$|\lambda|\leq0.25$")
    axes[0].scatter(eigenvalues.real[code_mask], eigenvalues.imag[code_mask], s=10, alpha=0.65, color="#0072B2", label=r"v1.0: $|\lambda|>0.25$")
    axes[0].axhline(0, color="#bbbbbb", linewidth=0.5)
    axes[0].axvline(0, color="#bbbbbb", linewidth=0.5)
    axes[0].set(aspect="equal", xlabel=r"Re$(\lambda)$", ylabel=r"Im$(\lambda)$", title="Discrete eigenvalue plane")
    axes[0].legend(loc="lower left", fontsize=8)

    axes[1].hist(diagnostics["modulus"], bins=40, color="#56B4E9", edgecolor="white")
    axes[1].axvline(0.25, color="#D55E00", linestyle="--", label="v1.0 threshold")
    axes[1].axvline(1.0, color="black", linestyle=":", label="unit circle")
    axes[1].set(xlabel=r"$|\lambda|$", ylabel="Mode count", title="Eigenvalue modulus")
    axes[1].legend(fontsize=8)

    rotations = diagnostics.loc[diagnostics["interpretable_rotation"], "rotations_per_decade"].dropna()
    if rotations.empty:
        axes[2].text(0.5, 0.5, "No three-cycle eligible\nstable rotations", ha="center", va="center", transform=axes[2].transAxes)
    else:
        axes[2].hist(rotations, bins=25, color="#009E73", edgecolor="white")
    axes[2].set(
        xlabel="Rotations per tenfold attenuation",
        ylabel="Conjugate pairs (positive-imaginary member counted once)",
        title="Code-filtered rotation summary",
    )
    annotation = (
        f"q={summary['q']} · pairs={summary['pair_count']:,}\n"
        f"SVD vs Gram rel. error={summary['svd_vs_gram_relative_error']:.2e}\n"
        f"spectral radius={summary['spectral_radius']:.3f}"
    )
    axes[2].text(0.98, 0.98, annotation, ha="right", va="top", transform=axes[2].transAxes, fontsize=8)
    fig.suptitle("Stage 2A — Near-literal Pachitariu v1.0 smoke test on example_data", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_simulation_recovery(metrics: pd.DataFrame, output: str | Path) -> None:
    apply_style()
    observations = list(metrics["observation"].drop_duplicates())
    systems = list(metrics["system_class"].drop_duplicates())
    positions = np.arange(len(observations))
    width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    colors = {"real_relaxation": "#0072B2", "rotational": "#D55E00"}
    for system_index, system in enumerate(systems):
        subset = metrics[metrics["system_class"] == system]
        grouped = subset.groupby("observation", sort=False)
        accuracy = grouped["classification_correct"].mean().reindex(observations)
        axes[0, 0].bar(positions + (system_index - 0.5) * width, accuracy, width, color=colors[system], alpha=0.85, label=system)
        overlap = grouped["latent_embedding_overlap"].median().reindex(observations)
        axes[0, 1].bar(positions + (system_index - 0.5) * width, overlap, width, color=colors[system], alpha=0.85)
        error = grouped["median_eigenvalue_relative_error"].median().reindex(observations)
        axes[1, 0].bar(positions + (system_index - 0.5) * width, error, width, color=colors[system], alpha=0.85)
        skill = grouped["skill_persistence_near_one_second"].median().reindex(observations)
        axes[1, 1].bar(positions + (system_index - 0.5) * width, skill, width, color=colors[system], alpha=0.85)
    axes[0, 0].axhline(0.8, color="black", linestyle="--", linewidth=0.8, label="Gate threshold")
    axes[0, 0].set(ylabel="Correct fraction", title="Real vs rotational classification", ylim=(0, 1.05))
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].axhline(0.75, color="black", linestyle="--", linewidth=0.8)
    axes[0, 1].set(ylabel="Median overlap", title="Target latent-embedding subspace", ylim=(0, 1.05))
    axes[1, 0].axhline(0.25, color="black", linestyle="--", linewidth=0.8)
    axes[1, 0].set(ylabel="Median relative error", title="Matched lag-2 eigenvalues")
    axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=0.8)
    axes[1, 1].set(ylabel="Median skill over persistence", title="Held-out forecast near 1 s")
    for axis in axes.ravel():
        axis.set_xticks(positions, observations, rotation=20, ha="right")
    fig.suptitle("Stage 2B — Known-system recovery through observation transformations", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_development_selection(
    summaries: pd.DataFrame,
    selected: pd.DataFrame,
    output: str | Path,
) -> None:
    apply_style()
    recordings = list(summaries["recording"].drop_duplicates())
    fig, axes = plt.subplots(1, len(recordings), figsize=(7 * len(recordings), 5), squeeze=False)
    for axis, recording in zip(axes[0], recordings, strict=True):
        subset = summaries[summaries["recording"] == recording]
        for arm_index, (arm, group) in enumerate(subset.groupby("arm", sort=False)):
            best_by_rank = group.groupby("rank", as_index=False)["median_skill_one_second"].max()
            axis.plot(
                best_by_rank["rank"],
                best_by_rank["median_skill_one_second"],
                marker="o",
                color=COLORS[arm],
                label=arm,
            )
        chosen = selected[selected["recording"] == recording]
        for row in chosen.itertuples():
            axis.scatter(row.rank, row.median_skill_one_second, s=120, facecolors="none", edgecolors=COLORS[row.arm], linewidths=2)
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.axvline(8, color="#777777", linestyle=":", linewidth=0.8, label="tracking q minimum")
        axis.set(
            xlabel="Rank q (best lag/ridge shown)",
            ylabel="Median development skill over persistence near 1 s",
            title=recording,
        )
        axis.legend(ncol=3, fontsize=8)
    fig.suptitle("Stage 3A — Anonymous blocked development selection", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_heldout_skill(metrics: pd.DataFrame, output: str | Path) -> None:
    apply_style()
    primary = metrics[(metrics["sensitivity"] == "primary") & (metrics["horizon_role"] == "gate")].copy()
    cells = list(primary[["recording", "label"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), squeeze=False)
    rng = np.random.default_rng(0)
    for axis, (recording, label) in zip(axes.ravel(), cells, strict=True):
        subset = primary[(primary["recording"] == recording) & (primary["label"] == label)]
        horizons = sorted(subset["actual_horizon_seconds"].unique())
        arms = list(subset["arm"].drop_duplicates())
        width = 0.13
        for arm_index, arm in enumerate(arms):
            arm_data = subset[subset["arm"] == arm]
            positions = np.arange(len(horizons)) + (arm_index - (len(arms) - 1) / 2) * width
            medians = [
                arm_data[np.isclose(arm_data["actual_horizon_seconds"], horizon)]["skill_persistence"].median()
                for horizon in horizons
            ]
            axis.bar(positions, medians, width, color=COLORS[arm], alpha=0.75, label=arm)
            for position, horizon in zip(positions, horizons, strict=True):
                values = arm_data[np.isclose(arm_data["actual_horizon_seconds"], horizon)]["skill_persistence"].to_numpy()
                axis.scatter(
                    position + rng.uniform(-0.025, 0.025, size=values.size),
                    values,
                    s=16,
                    color=COLORS[arm],
                    edgecolor="white",
                    linewidth=0.3,
                    zorder=3,
                )
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(np.arange(len(horizons)), [f"{value:.2f}" for value in horizons])
        axis.set(
            xlabel="Actual horizon (s)",
            ylabel="Skill over persistence",
            title=f"{recording} · {label} · three untouched windows",
        )
        axis.legend(ncol=3, fontsize=8)
    fig.suptitle("Stage 3B — Held-out DMD forecast skill, window-level points retained", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_empirical_spectra(spectra: pd.DataFrame, output: str | Path) -> None:
    apply_style()
    recordings = list(spectra["recording"].drop_duplicates())
    arms = list(spectra["arm"].drop_duplicates())
    fig, axes = plt.subplots(len(recordings), len(arms), figsize=(3.1 * len(arms), 3.2 * len(recordings)), squeeze=False)
    theta = np.linspace(0, 2 * np.pi, 300)
    for row_index, recording in enumerate(recordings):
        for column_index, arm in enumerate(arms):
            axis = axes[row_index, column_index]
            subset = spectra[(spectra["recording"] == recording) & (spectra["arm"] == arm)]
            axis.plot(np.cos(theta), np.sin(theta), color="#bbbbbb", linestyle="--", linewidth=0.6)
            for label_index, (label, group) in enumerate(subset.groupby("label", sort=False)):
                axis.scatter(
                    group["real"],
                    group["imag"],
                    s=12,
                    alpha=0.55,
                    color=("#0072B2" if label_index == 0 else "#D55E00"),
                    label=label,
                )
            axis.set(xlim=(-1.05, 1.05), ylim=(-1.05, 1.05), aspect="equal", title=f"{recording}\n{arm}")
            if row_index == len(recordings) - 1:
                axis.set_xlabel(r"Re$(\lambda)$")
            if column_index == 0:
                axis.set_ylabel(r"Im$(\lambda)$")
            axis.legend(fontsize=7)
    fig.suptitle("Stage 3D — Selected-config spectra in untouched windows", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_representative_predictions(
    panels: Iterable[dict[str, object]],
    output: str | Path,
) -> None:
    apply_style()
    panels = list(panels)
    fig, axes = plt.subplots(len(panels), 3, figsize=(15, 2.6 * len(panels)), squeeze=False)
    for row_index, panel in enumerate(panels):
        truth = np.asarray(panel["truth"])
        prediction = np.asarray(panel["prediction"])
        time = np.asarray(panel["time"])
        for pc in range(min(3, truth.shape[0])):
            axis = axes[row_index, pc]
            axis.plot(time, truth[pc], color="black", linewidth=1.1, label="observed")
            axis.plot(time, prediction[pc], color="#D55E00", linewidth=1.0, label="DMD")
            axis.set_title(
                f"{panel['recording']} · {panel['label']} · {panel['arm']} · PC{pc+1}"
            )
            if row_index == len(panels) - 1:
                axis.set_xlabel("Target time in held-out block (s)")
            if pc == 0:
                axis.set_ylabel("PC score")
            axis.legend(fontsize=7)
    fig.suptitle("Stage 3C — Representative rolling-origin forecasts near 1 s", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_long_block_comparison(
    heldout: pd.DataFrame,
    long_metrics: pd.DataFrame,
    output: str | Path,
) -> None:
    apply_style()
    primary = heldout[(heldout["sensitivity"] == "primary") & (heldout["horizon_role"] == "gate")]
    short_near = primary[(primary["actual_horizon_seconds"] - 1.0).abs() < 0.2]
    short_summary = (
        short_near.groupby(["recording", "label", "arm"], as_index=False)["skill_persistence"]
        .median()
        .rename(columns={"skill_persistence": "short_skill"})
    )
    long_near = long_metrics[(long_metrics["actual_horizon_seconds"] - 1.0).abs() < 0.2]
    long_summary = (
        long_near.groupby(["recording", "label", "arm"], as_index=False)["skill_persistence"]
        .median()
        .rename(columns={"skill_persistence": "long_skill"})
    )
    merged = short_summary.merge(long_summary, on=["recording", "label", "arm"])
    fig, axis = plt.subplots(figsize=(8.2, 7.2))
    for arm, group in merged.groupby("arm", sort=False):
        axis.scatter(group["short_skill"], group["long_skill"], s=65, color=COLORS[arm], label=arm)
        for row in group.itertuples():
            recording = row.recording.replace("mouse", "m").replace("_sleep", "").replace("_ane", "")
            axis.annotate(
                f"{recording} {row.label}",
                (row.short_skill, row.long_skill),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=6.5,
            )
    limits = [min(merged[["short_skill", "long_skill"]].min()) - 0.05, max(merged[["short_skill", "long_skill"]].max()) + 0.05]
    axis.plot(limits, limits, color="#777777", linestyle="--", linewidth=0.8)
    axis.axhline(0, color="black", linewidth=0.6)
    axis.axvline(0, color="black", linewidth=0.6)
    axis.set(
        xlim=limits,
        ylim=limits,
        xlabel="W=300 held-out skill near 1 s",
        ylabel="W=1500 held-out skill near 1 s",
        title="Does longer temporal support rescue prediction?",
    )
    axis.legend(ncol=3)
    fig.suptitle("Stage 3E — Prespecified long-block diagnostic", fontsize=13, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, output)


def plot_diagonal_gain(metrics: pd.DataFrame, output: str | Path) -> None:
    """Show whether full DMD adds value beyond independent PC dynamics."""
    apply_style()
    primary = metrics[(metrics["sensitivity"] == "primary") & (metrics["horizon_role"] == "gate")]
    near_one = primary[(primary["actual_horizon_seconds"] - 1.0).abs() < 0.2]
    cells = list(near_one[["recording", "label"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    rng = np.random.default_rng(1)
    for axis, (recording, label) in zip(axes.ravel(), cells, strict=True):
        subset = near_one[(near_one["recording"] == recording) & (near_one["label"] == label)]
        arms = list(subset["arm"].drop_duplicates())
        medians = [subset[subset["arm"] == arm]["skill_diagonal"].median() for arm in arms]
        axis.bar(arms, medians, color=[COLORS[arm] for arm in arms], alpha=0.75)
        for arm_index, arm in enumerate(arms):
            values = subset[subset["arm"] == arm]["skill_diagonal"].to_numpy()
            axis.scatter(
                arm_index + rng.uniform(-0.06, 0.06, values.size),
                values,
                s=22,
                color=COLORS[arm],
                edgecolor="white",
                linewidth=0.4,
                zorder=3,
            )
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set(
            ylabel="Full-DMD skill over diagonal AR",
            title=f"{recording} · {label} · three untouched windows",
        )
    fig.suptitle(
        "Stage 3F — Multivariate-value check near 1 s",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_smoothing_trim_sensitivity(metrics: pd.DataFrame, output: str | Path) -> None:
    """Compare stored-smoothed results before and after the prespecified edge trim."""
    apply_style()
    subset = metrics[
        metrics["arm"].isin(["Sz", "Sc"])
        & (metrics["horizon_role"] == "gate")
        & ((metrics["actual_horizon_seconds"] - 1.0).abs() < 0.2)
    ]
    paired = subset.pivot_table(
        index=["recording", "label", "arm", "window_id"],
        columns="sensitivity",
        values="skill_persistence",
    ).reset_index()
    paired = paired.dropna(subset=["primary", "trimmed_seven_frame_edges"])
    fig, axis = plt.subplots(figsize=(7, 6.5))
    markers = {"Sz": "o", "Sc": "s"}
    for arm, group in paired.groupby("arm", sort=False):
        axis.scatter(
            group["primary"],
            group["trimmed_seven_frame_edges"],
            marker=markers[arm],
            s=45,
            color=COLORS[arm],
            alpha=0.75,
            label=arm,
        )
    lower = float(min(paired["primary"].min(), paired["trimmed_seven_frame_edges"].min()) - 0.05)
    upper = float(max(paired["primary"].max(), paired["trimmed_seven_frame_edges"].max()) + 0.05)
    axis.plot([lower, upper], [lower, upper], color="#777777", linestyle="--", linewidth=0.8)
    axis.axhline(0, color="black", linewidth=0.5)
    axis.axvline(0, color="black", linewidth=0.5)
    axis.set(
        xlim=(lower, upper),
        ylim=(lower, upper),
        xlabel="Primary skill over persistence near 1 s",
        ylabel="Skill after trimming seven frames at both edges",
        title="Each point is one untouched window",
    )
    axis.legend(title="Stored-smoothed arm")
    fig.suptitle(
        "Stage 3G — Stored-smoothing boundary-support sensitivity",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_bootstrap_forecast_intervals(summary: pd.DataFrame, output: str | Path) -> None:
    """Plot synchronized three-window bootstrap intervals for the selected arm."""
    apply_style()
    roles = ["near_one_second", "near_two_seconds"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), squeeze=False)
    for axis, role in zip(axes[0], roles, strict=True):
        subset = summary[summary["horizon_role"] == role].copy()
        labels = [
            f"{row.recording.replace('mouse', 'm').replace('_sleep', '').replace('_ane', '')}\n{row.label}"
            for row in subset.itertuples()
        ]
        positions = np.arange(len(subset))
        center = subset["bootstrap_median_skill_persistence"].to_numpy()
        lower = center - subset["skill_persistence_ci_lower"].to_numpy()
        upper = subset["skill_persistence_ci_upper"].to_numpy() - center
        axis.errorbar(
            positions,
            center,
            yerr=np.vstack([lower, upper]),
            fmt="o",
            color=COLORS["P"],
            capsize=4,
            linewidth=1.4,
            label="P: bootstrap median and 95% interval",
        )
        axis.axhline(0, color="black", linestyle="--", linewidth=0.8)
        axis.set_xticks(positions, labels)
        axis.set(
            ylabel="Skill over persistence",
            title=("Near 1 s" if role == "near_one_second" else "Near 2 s"),
        )
        axis.legend(fontsize=8)
    fig.suptitle(
        "Stage 4A — Moving-block bootstrap forecast uncertainty",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_subspace_stability(
    original: pd.DataFrame,
    bootstrap: pd.DataFrame,
    output: str | Path,
) -> None:
    """Show fixed-reference projector similarities for original and bootstrap fits."""
    apply_style()
    cells = list(original[["recording", "label"]].drop_duplicates().itertuples(index=False, name=None))
    labels = [
        f"{recording.replace('mouse', 'm').replace('_sleep', '').replace('_ane', '')}\n{label}"
        for recording, label in cells
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    original_values = [
        original[
            (original["recording"] == recording)
            & (original["label"] == label)
            & original["match_success"]
        ]["reference_similarity"].to_numpy()
        for recording, label in cells
    ]
    bootstrap_values = [
        bootstrap[
            (bootstrap["recording"] == recording)
            & (bootstrap["label"] == label)
            & bootstrap["match_success"]
        ]["reference_similarity"].to_numpy()
        for recording, label in cells
    ]
    axes[0].boxplot(original_values, tick_labels=labels, showfliers=True)
    axes[1].boxplot(bootstrap_values, tick_labels=labels, showfliers=False)
    for axis, title in zip(
        axes,
        ["Original evaluation fits", "100 moving-block fits per window"],
        strict=True,
    ):
        axis.axhline(0.8, color="#D55E00", linestyle="--", linewidth=1, label="Gate threshold")
        axis.set(ylim=(0, 1.02), ylabel=r"$S_{sub}$ to development reference", title=title)
        axis.legend(fontsize=8)
    fig.suptitle(
        "Stage 4B — Fixed-reference invariant-subspace reproducibility",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_tracking_resolution(summary: pd.DataFrame, output: str | Path) -> None:
    """Compare estimation-noise distance with ordinary same-label window change."""
    apply_style()
    labels = [
        value.replace("mouse", "m").replace("_sleep", " sleep").replace("_ane", " anesthesia")
        for value in summary["recording"]
    ]
    positions = np.arange(len(summary))
    width = 0.32
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    axes[0].bar(
        positions - width / 2,
        summary["median_within_window_bootstrap_distance"],
        width,
        color="#56B4E9",
        label="within-window bootstrap",
    )
    axes[0].bar(
        positions + width / 2,
        summary["median_between_window_same_label_distance"],
        width,
        color="#D55E00",
        label="between-window same label",
    )
    axes[0].set_xticks(positions, labels)
    axes[0].set(ylabel="Normalized projector distance", title="Distance components")
    axes[0].legend(fontsize=8)
    axes[1].bar(positions, summary["tracking_resolution_ratio"], color=COLORS["P"], width=0.55)
    axes[1].axhline(0.5, color="#D55E00", linestyle="--", linewidth=1, label="Gate threshold")
    axes[1].set_xticks(positions, labels)
    axes[1].set(ylabel=r"$R_{track}$", title="Estimation noise / ordinary change")
    axes[1].legend(fontsize=8)
    fig.suptitle(
        "Stage 4C — Is Grassmann tracking resolvable at W=300?",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_neuron_subset_classes(agreement: pd.DataFrame, output: str | Path) -> None:
    """Visualize the qualitative spectrum classification in equal-N subsets."""
    apply_style()
    subsets = ["subset_A", "subset_B"]
    values = np.asarray(
        [
            [1 if row[subset] == "resolved_rotation" else 0 for subset in subsets]
            for _, row in agreement.iterrows()
        ]
    )
    cell_labels = [
        f"{row.recording.replace('mouse', 'm').replace('_sleep', '').replace('_ane', '')} · {row.label}"
        for row in agreement.itertuples()
    ]
    fig, axis = plt.subplots(figsize=(6.2, 4.6))
    image = axis.imshow(values, cmap="RdYlBu_r", vmin=0, vmax=1, aspect="auto")
    for row in range(values.shape[0]):
        for column in range(values.shape[1]):
            axis.text(
                column,
                row,
                "resolved rotation" if values[row, column] else "no resolved rotation",
                ha="center",
                va="center",
                fontsize=8,
            )
    axis.set_xticks(np.arange(2), ["Subset A", "Subset B"])
    axis.set_yticks(np.arange(len(cell_labels)), cell_labels)
    axis.set_title("Majority class across three untouched windows")
    colorbar = fig.colorbar(image, ax=axis, fraction=0.04, pad=0.03, ticks=[0, 1])
    colorbar.ax.set_yticklabels(["no resolved rotation", "resolved rotation"])
    fig.suptitle(
        "Stage 4D — Two deterministic disjoint 2,000-neuron subsets",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_null_multivariate_gain(
    synchronized: pd.DataFrame,
    summary: pd.DataFrame,
    output: str | Path,
) -> None:
    """Compare observed full-over-diagonal skill with both negative controls."""
    apply_style()
    near_one = synchronized[synchronized["horizon_role"] == "near_one_second"]
    cells = list(near_one[["recording", "label"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8.2), squeeze=False)
    colors = {"circular_shift": "#56B4E9", "stationary_iid": "#999999"}
    for axis, (recording, label) in zip(axes.ravel(), cells, strict=True):
        cell = near_one[(near_one["recording"] == recording) & (near_one["label"] == label)]
        for kind, group in cell.groupby("null_kind", sort=False):
            axis.hist(
                group["median_skill_diagonal"],
                bins=18,
                alpha=0.55,
                color=colors[kind],
                density=True,
                label=kind.replace("_", " "),
            )
        comparator = summary[
            (summary["recording"] == recording)
            & (summary["label"] == label)
            & (summary["horizon_role"] == "near_one_second")
            & (summary["null_kind"] == "circular_shift")
        ].iloc[0]
        axis.axvline(
            comparator["observed_median_skill_diagonal"],
            color="#D55E00",
            linewidth=2,
            label="observed median",
        )
        axis.axvline(
            comparator["null_skill_diagonal_percentile"],
            color="black",
            linestyle="--",
            linewidth=1,
            label="circular 95th percentile",
        )
        axis.set(
            xlabel="Full-DMD skill over diagonal AR",
            ylabel="Density",
            title=f"{recording} · {label}",
        )
        axis.legend(fontsize=7)
    fig.suptitle(
        "Stage 5A — Is multivariate forecast gain larger than neuronwise nulls?",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_null_subspace_similarity(
    synchronized: pd.DataFrame,
    original: pd.DataFrame,
    output: str | Path,
) -> None:
    """Compare null and original similarity to the development reference."""
    apply_style()
    near_one = synchronized[synchronized["horizon_role"] == "near_one_second"]
    cells = list(near_one[["recording", "label"]].drop_duplicates().itertuples(index=False, name=None))
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), squeeze=False)
    for axis, (recording, label) in zip(axes.ravel(), cells, strict=True):
        cell = near_one[(near_one["recording"] == recording) & (near_one["label"] == label)]
        values = [
            cell[cell["null_kind"] == kind]["median_reference_similarity"].dropna().to_numpy()
            for kind in ("circular_shift", "stationary_iid")
        ]
        axis.boxplot(values, tick_labels=["circular shift", "stationary iid"], showfliers=False)
        observed = original[
            (original["recording"] == recording)
            & (original["label"] == label)
            & original["match_success"]
        ]["reference_similarity"].median()
        axis.axhline(observed, color="#D55E00", linewidth=2, label="observed median")
        axis.axhline(0.8, color="black", linestyle="--", linewidth=1, label="Gate 4 threshold")
        axis.set(
            ylim=(0, 1.02),
            ylabel=r"$S_{sub}$ to development reference",
            title=f"{recording} · {label}",
        )
        axis.legend(fontsize=7)
    fig.suptitle(
        "Stage 5B — Fixed-reference subspace similarity under negative controls",
        fontsize=13,
        fontweight="bold",
    )
    fig.tight_layout()
    save_figure(fig, output)


def plot_gate_summary(gates: pd.DataFrame, decision: str, output: str | Path) -> None:
    """Render the frozen feasibility gates without hiding failed components."""
    apply_style()
    figure, axis = plt.subplots(figsize=(11, 5.8))
    positions = np.arange(len(gates))[::-1]
    colors = ["#009E73" if passed else "#D55E00" for passed in gates["pass"]]
    axis.barh(positions, np.ones(len(gates)), color=colors, alpha=0.85)
    for position, (_, row) in zip(positions, gates.iterrows(), strict=True):
        axis.text(
            0.02,
            position,
            f"{row['gate']}: {'PASS' if row['pass'] else 'FAIL'}",
            va="center",
            ha="left",
            color="white",
            fontweight="bold",
        )
        axis.text(1.02, position, row["observed"], va="center", ha="left", fontsize=8)
    axis.set_yticks([])
    axis.set_xlim(0, 1.85)
    axis.set_xticks([])
    axis.set_title("Green = threshold met; orange = threshold not met")
    figure.suptitle(
        f"Stage 6 — Frozen gate decision: {decision}",
        fontsize=14,
        fontweight="bold",
    )
    figure.tight_layout()
    save_figure(figure, output)
