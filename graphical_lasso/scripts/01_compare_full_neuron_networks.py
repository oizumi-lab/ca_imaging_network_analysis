# %% [markdown]
# # 01 · Compare full-neuron matrices, strongest pairs, and networks
#
# Run `00_estimate_full_neuron_matrices.py` first. This script never refits the
# estimator. It compares Pearson correlation, graphical-lasso-implied marginal
# correlation, and graphical-lasso partial correlation on identical full-neuron
# node sets.

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[2]
GLASSO_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(GLASSO_ROOT / "src"))

from src.funcnet import network as net  # noqa: E402

from glasso_analysis.comparison import (  # noqa: E402
    edge_jaccard,
    exact_fixed_density,
    sampled_matrix_agreement,
    summarize_graph,
)
from glasso_analysis.config import COMPARISON_DIR, CONFIG, MATRIX_DIR  # noqa: E402

COMPARISON_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR = COMPARISON_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings and output partition

# %%
requested_recording = os.environ.get("GLASSO_RECORDING")
matrix_only = os.environ.get("GLASSO_MATRIX_ONLY", "0") == "1"
summary_figures_only = os.environ.get("GLASSO_SUMMARY_FIGURES_ONLY", "0") == "1"
recording_names = list(CONFIG["recordings"])
if requested_recording:
    if requested_recording not in recording_names:
        raise ValueError(f"unknown GLASSO_RECORDING={requested_recording!r}")
    recording_names = [requested_recording]
plot_recording_names = recording_names.copy()
if summary_figures_only:
    recording_names = []

EDGE_DENSITIES = [float(value) for value in CONFIG["edge_comparison_densities"]]
NETWORK_DENSITIES = [float(value) for value in CONFIG["network_densities"]]
N_LOUVAIN = int(CONFIG["louvain_runs"])
PATH_SOURCES = int(CONFIG["path_length_sources"])
SEED = int(CONFIG["random_seed"])

matrix_rows: list[dict] = []
overlap_rows: list[dict] = []
network_rows: list[dict] = []


# %% [markdown]
# ## Matrix agreement and exact high-weight edge overlap

# %%
for recording_index, recording_name in enumerate(recording_names):
    recording_dir = MATRIX_DIR / recording_name
    population = json.loads((recording_dir / "population.json").read_text(encoding="utf-8"))
    coordinates_um = np.load(recording_dir / "coordinates_um.npy")

    for state_index, state in enumerate(population["states"]):
        print(f"\n=== {recording_name}: {state} ===", flush=True)
        state_dir = recording_dir / state
        matrices = {
            "pearson": np.load(state_dir / "pearson.npy", mmap_mode="r"),
            "glasso_marginal": np.load(state_dir / "glasso_marginal.npy", mmap_mode="r"),
            "glasso_partial": np.load(state_dir / "glasso_partial.npy", mmap_mode="r"),
        }
        partial_native_density = float(
            np.count_nonzero(
                np.triu(np.abs(np.asarray(matrices["glasso_partial"])) > 1e-10, 1)
            )
            / (coordinates_um.shape[0] * (coordinates_um.shape[0] - 1) / 2)
        )

        for method in ("glasso_marginal", "glasso_partial"):
            agreement = sampled_matrix_agreement(
                matrices["pearson"],
                matrices[method],
                max_pairs=1_000_000,
                seed=SEED + 10 * recording_index + state_index,
            )
            matrix_rows.append(
                {
                    "recording": recording_name,
                    "state": state,
                    "reference": "pearson",
                    "method": method,
                    "partial_native_density": partial_native_density,
                    **agreement,
                }
            )

        selections: dict[tuple[str, float], object] = {}
        for density in EDGE_DENSITIES:
            for method, matrix in matrices.items():
                try:
                    selection = exact_fixed_density(
                        matrix,
                        density,
                        require_nonzero=(method == "glasso_partial"),
                    )
                except ValueError as error:
                    overlap_rows.append(
                        {
                            "recording": recording_name,
                            "state": state,
                            "density": density,
                            "comparison": f"pearson_vs_{method}",
                            "available": False,
                            "reason": str(error),
                            "jaccard": np.nan,
                        }
                    )
                    continue
                selections[(method, density)] = selection
                if method != "pearson":
                    overlap_rows.append(
                        {
                            "recording": recording_name,
                            "state": state,
                            "density": density,
                            "comparison": f"pearson_vs_{method}",
                            "available": True,
                            "reason": "",
                            "jaccard": edge_jaccard(
                                selections[("pearson", density)], selection
                            ),
                            "pearson_threshold": selections[("pearson", density)].threshold,
                            "method_threshold": selection.threshold,
                            "method_native_density": selection.native_density,
                            "method_positive_edge_fraction": float(
                                np.mean(selection.weights > 0)
                            ),
                        }
                    )

        # %% [markdown]
        # ## Full-network measures
        #
        # Every feasible graph is summarized with sparse graph arithmetic. The
        # Louvain call intentionally uses the existing repository implementation
        # so estimator choice is the only pipeline change.

        # %%
        for density in ([] if matrix_only else NETWORK_DENSITIES):
            for method, matrix in matrices.items():
                try:
                    selection = selections.get((method, density)) or exact_fixed_density(
                        matrix,
                        density,
                        require_nonzero=(method == "glasso_partial"),
                    )
                except ValueError as error:
                    network_rows.append(
                        {
                            "recording": recording_name,
                            "state": state,
                            "method": method,
                            "requested_density": density,
                            "available": False,
                            "reason": str(error),
                        }
                    )
                    continue

                summary = summarize_graph(
                    selection,
                    coordinates_um,
                    path_sources=PATH_SOURCES,
                    seed=SEED + 100 * recording_index + 10 * state_index,
                )
                dense_adjacency = selection.adjacency.toarray()
                modularity = net.repeat_louvain(
                    dense_adjacency,
                    gamma=1.0,
                    n_runs=N_LOUVAIN,
                    seed=SEED,
                    warm_start=True,
                )
                del dense_adjacency
                ci_path = (
                    COMPARISON_DIR
                    / f"partition_{recording_name}_{state}_{method}_K{density:.4f}.npy"
                )
                np.save(ci_path, modularity["ci_max"], allow_pickle=False)
                network_rows.append(
                    {
                        "recording": recording_name,
                        "state": state,
                        "method": method,
                        "available": True,
                        "reason": "",
                        **summary,
                        "modularity_Q": modularity["Q_max"],
                        "n_modules": modularity["n_modules_max"],
                        "louvain_q_mean": float(modularity["Q_all"].mean()),
                        "louvain_q_sd": float(modularity["Q_all"].std()),
                    }
                )
                print(
                    f"{method:17s} K={density:.3%}: "
                    f"Q={modularity['Q_max']:.4f}, "
                    f"C={summary['clustering']:.4f}, "
                    f"L={summary['path_length_giant']:.3f}, "
                    f"GC={summary['giant_component_fraction']:.3f}",
                    flush=True,
                )

        # %% [markdown]
        # ## Compact matrix views

        # %%
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.7))
        limits = {
            "pearson": (-0.5, 0.5),
            "glasso_marginal": (-0.5, 0.5),
            "glasso_partial": (-0.25, 0.25),
        }
        for axis, (method, matrix) in zip(axes, matrices.items()):
            vmin, vmax = limits[method]
            image = axis.imshow(
                matrix,
                cmap="RdBu_r",
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                rasterized=True,
            )
            axis.set_title(method.replace("_", " "))
            axis.set_xlabel("neuron")
            axis.set_ylabel("neuron")
            fig.colorbar(image, ax=axis, shrink=0.72)
        fig.suptitle(f"{recording_name}: {state} — all {coordinates_um.shape[0]:,} neurons")
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"matrices_{recording_name}_{state}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)
        del matrices, selections


# %% [markdown]
# ## Save exact numerical tables

# %%
matrix_table = pd.DataFrame(matrix_rows)
overlap_table = pd.DataFrame(overlap_rows)
network_table = pd.DataFrame(network_rows)

suffix = f"_{requested_recording}" if requested_recording else ""
matrix_path = COMPARISON_DIR / f"matrix_agreement{suffix}.csv"
overlap_path = COMPARISON_DIR / f"edge_overlap{suffix}.csv"
network_path = COMPARISON_DIR / f"network_measures{suffix}.csv"
if not matrix_table.empty:
    matrix_table.to_csv(matrix_path, index=False)
elif matrix_path.exists():
    matrix_table = pd.read_csv(matrix_path)
if not overlap_table.empty:
    overlap_table.to_csv(overlap_path, index=False)
elif overlap_path.exists():
    overlap_table = pd.read_csv(overlap_path)
if not network_table.empty:
    network_table.to_csv(network_path, index=False)
elif network_path.exists():
    network_table = pd.read_csv(network_path)

print("saved:")
print(matrix_path)
print(overlap_path)
if network_path.exists():
    print(network_path)


# %% [markdown]
# ## Summary figures from the tables

# %%
if not overlap_table.empty:
    available_overlap = overlap_table[overlap_table["available"]].copy()
    for recording_name in plot_recording_names:
        subset = available_overlap[available_overlap["recording"] == recording_name]
        if subset.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), sharey=True)
        states = CONFIG["recordings"][recording_name]["states"]
        for axis, state in zip(axes, states):
            state_rows = subset[subset["state"] == state]
            for comparison, group in state_rows.groupby("comparison"):
                group = group.sort_values("density")
                axis.plot(
                    group["density"] * 100,
                    group["jaccard"],
                    marker="o",
                    label=comparison.replace("pearson_vs_", ""),
                )
            axis.set_xscale("log")
            axis.set_xlabel("fixed density (%)")
            axis.set_title(state)
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("edge-set Jaccard vs Pearson")
        axes[-1].legend()
        fig.suptitle(f"{recording_name}: overlap of strongest full-neuron pairs")
        fig.tight_layout()
        fig.savefig(
            FIGURE_DIR / f"edge_overlap_{recording_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

if not network_table.empty:
    available_network = network_table[network_table["available"]].copy()
    measures = [
        ("modularity_Q", "modularity Q"),
        ("clustering", "clustering C"),
        ("path_length_giant", "path length L"),
        ("giant_component_fraction", "giant-component fraction"),
    ]
    for recording_name in plot_recording_names:
        subset = available_network[available_network["recording"] == recording_name]
        if subset.empty:
            continue
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        for axis, (column, label) in zip(axes.ravel(), measures):
            for (state, method), group in subset.groupby(["state", "method"]):
                group = group.sort_values("requested_density")
                axis.plot(
                    group["requested_density"] * 100,
                    group[column],
                    marker="o",
                    label=f"{state}: {method}",
                )
            axis.set_xscale("log")
            axis.set_xlabel("fixed density (%)")
            axis.set_ylabel(label)
            axis.grid(alpha=0.25)
        handles, labels = axes[0, 0].get_legend_handles_labels()
        fig.suptitle(f"{recording_name}: full-neuron network measures", y=0.995)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.96),
            ncol=3,
            fontsize=8,
        )
        fig.tight_layout(rect=(0, 0, 1, 0.88))
        fig.savefig(
            FIGURE_DIR / f"network_measures_{recording_name}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(fig)

print("comparison complete")
