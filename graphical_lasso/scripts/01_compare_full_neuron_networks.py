# %% [markdown]
# # 01 · Simple fixed-density graph comparison
#
# This script keeps two analyses separate:
#
# 1. **Common comparison at K=0.1%** — usual Pearson versus graphical-lasso
#    partial correlation, with exactly the same number of edges.
# 2. **Usual-method reference at K=5%** — Pearson only, matching the reference
#    density used in the paper-facing modularity workflow.
#
# Both analyses display the raw binary graph matrices and the same four network
# measures for awake versus NREM/anesthesia.

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
    exact_fixed_density_blockwise,
    summarize_graph,
)
from glasso_analysis.config import COMPARISON_DIR, CONFIG, MATRIX_DIR  # noqa: E402

FIGURE_DIR = COMPARISON_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## 1. Choose the two fixed densities
#
# `COMMON_DENSITY=0.001` means 0.1% of all possible neuron pairs. Every fitted
# graphical-lasso partial-correlation matrix supports this density without zero
# padding. `REFERENCE_DENSITY=0.05` is the usual Pearson K=5% reference. The
# graphical-lasso partial graph is deliberately not constructed at K=5% because
# it does not contain enough nonzero edges.
#
# Set `GLASSO_RECORDING=mouse01_sleep` or `mouse05_ane` if only one comparison
# should be displayed.

# %%
COMMON_DENSITY = 0.001
REFERENCE_DENSITY = 0.05

requested_recording = os.environ.get("GLASSO_RECORDING")
recording_names = list(CONFIG["recordings"])
if requested_recording:
    if requested_recording not in recording_names:
        raise ValueError(f"unknown GLASSO_RECORDING={requested_recording!r}")
    recording_names = [requested_recording]

GRAPH_SPECS = {
    "pearson_common": {
        "analysis": "common_K0.1%",
        "method": "pearson",
        "label": "Pearson (usual)",
        "matrix_file": "pearson.npy",
        "density": COMMON_DENSITY,
        "color": "#377eb8",
    },
    "graphical_lasso_common": {
        "analysis": "common_K0.1%",
        "method": "graphical_lasso_partial",
        "label": "Graphical lasso (partial)",
        "matrix_file": "glasso_partial.npy",
        "density": COMMON_DENSITY,
        "color": "#e41a1c",
    },
    "pearson_reference": {
        "analysis": "usual_reference_K5%",
        "method": "pearson",
        "label": "Pearson (usual reference)",
        "matrix_file": "pearson.npy",
        "density": REFERENCE_DENSITY,
        "color": "#4d4d4d",
    },
}
COMMON_GRAPH_KEYS = ("pearson_common", "graphical_lasso_common")
REFERENCE_GRAPH_KEY = "pearson_reference"

N_LOUVAIN = int(CONFIG["louvain_runs"])
PATH_SOURCES = int(CONFIG["path_length_sources"])
SEED = int(CONFIG["random_seed"])


def display_state(state: str) -> str:
    """Use concise labels in plots."""
    return {"awake": "Awake", "nrem": "NREM", "anesthesia": "Anesthesia"}[state]


def binned_edge_density(adjacency, n_bins: int = 300) -> np.ndarray:
    """Compress a large binary adjacency into neuron-order edge-density bins."""
    n_nodes = adjacency.shape[0]
    node_bins = np.minimum(np.arange(n_nodes) * n_bins // n_nodes, n_bins - 1)
    bin_sizes = np.bincount(node_bins, minlength=n_bins)
    coo = adjacency.tocoo()
    edge_counts = np.zeros((n_bins, n_bins), dtype=np.int64)
    np.add.at(edge_counts, (node_bins[coo.row], node_bins[coo.col]), 1)

    possible = np.outer(bin_sizes, bin_sizes)
    diagonal = np.diag_indices(n_bins)
    possible[diagonal] = bin_sizes * (bin_sizes - 1)
    return np.divide(
        edge_counts,
        possible,
        out=np.zeros_like(edge_counts, dtype=np.float64),
        where=possible > 0,
    )


# %% [markdown]
# ## 2. Threshold the matrices and calculate network measures
#
# For each state and method:
#
# 1. rank all neuron pairs by absolute coefficient;
# 2. retain exactly the graph specification's fixed density;
# 3. make an unweighted binary graph;
# 4. calculate modularity, clustering, path length, and connectedness.
#
# The same neurons are used throughout. Pearson and graphical lasso have the
# same edge count in the K=0.1% comparison. K=5% remains a separate reference.

# %%
all_measure_rows: list[dict[str, float | int | str]] = []

for recording_index, recording_name in enumerate(recording_names):
    recording_dir = MATRIX_DIR / recording_name
    population = json.loads(
        (recording_dir / "population.json").read_text(encoding="utf-8")
    )
    states = list(population["states"])
    coordinates_um = np.load(recording_dir / "coordinates_um.npy")
    n_nodes = coordinates_um.shape[0]

    graphs = {}
    recording_rows: list[dict[str, float | int | str]] = []

    for state_index, state in enumerate(states):
        state_dir = recording_dir / state

        for graph_key, settings in GRAPH_SPECS.items():
            matrix = np.load(
                state_dir / str(settings["matrix_file"]),
                mmap_mode="r",
            )
            density = float(settings["density"])
            selection = exact_fixed_density_blockwise(
                matrix,
                density,
                require_nonzero=(graph_key == "graphical_lasso_common"),
            )
            graphs[(state, graph_key)] = selection
            del matrix

            graph_summary = summarize_graph(
                selection,
                coordinates_um,
                path_sources=PATH_SOURCES,
                seed=SEED + 100 * recording_index + 10 * state_index,
            )

            # The existing repository's Louvain routine is used unchanged.
            dense_adjacency = selection.adjacency.astype(np.uint8).toarray()
            modularity = net.repeat_louvain(
                dense_adjacency,
                gamma=1.0,
                n_runs=N_LOUVAIN,
                seed=SEED,
                warm_start=True,
            )
            del dense_adjacency

            row = {
                "recording": recording_name,
                "state": state,
                "graph_key": graph_key,
                "analysis": str(settings["analysis"]),
                "method": str(settings["method"]),
                "method_label": str(settings["label"]),
                "fixed_density": density,
                "louvain_runs": N_LOUVAIN,
                **graph_summary,
                "modularity_Q": float(modularity["Q_max"]),
                "n_modules": int(modularity["n_modules_max"]),
            }
            recording_rows.append(row)
            all_measure_rows.append(row)

            print(
                f"{recording_name:13s} | {display_state(state):10s} | "
                f"K={density:.2%} | {settings['label']:27s} | "
                f"edges={selection.adjacency.nnz // 2:,} "
                f"threshold={selection.threshold:.5f} | "
                f"Q={modularity['Q_max']:.4f} "
                f"C={graph_summary['clustering']:.4f} "
                f"L={graph_summary['path_length_giant']:.3f} "
                f"GC={graph_summary['giant_component_fraction']:.3f}",
                flush=True,
            )

    recording_table = pd.DataFrame(recording_rows)

    # %% [markdown]
    # ## 3. K=0.1% common comparison: raw binary graph matrices
    #
    # Rows are estimation methods and columns are states. Neurons remain in
    # their original shared order; they are not sorted to make blocks look
    # stronger. Every black point is one retained undirected edge.

    # %%
    fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
    for method_index, graph_key in enumerate(COMMON_GRAPH_KEYS):
        settings = GRAPH_SPECS[graph_key]
        for state_index, state in enumerate(states):
            selection = graphs[(state, graph_key)]
            axis = axes[method_index, state_index]
            axis.spy(
                selection.adjacency,
                marker=".",
                markersize=0.22,
                color="black",
                rasterized=True,
            )
            axis.set_title(
                f"{display_state(state)} — {settings['label']}\n"
                f"{selection.adjacency.nnz // 2:,} edges; "
                f"threshold={selection.threshold:.4f}"
            )
            axis.set_xlabel("neuron (original order)")
            axis.set_ylabel("neuron (original order)")

    fig.suptitle(
        f"{recording_name}: raw binary graph matrices at "
        f"K={COMMON_DENSITY:.2%} ({n_nodes:,} neurons)",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    adjacency_path = (
        FIGURE_DIR
        / f"simple_adjacency_{recording_name}_K{COMMON_DENSITY:.4f}.png"
    )
    fig.savefig(adjacency_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # %% [markdown]
    # ## 4. K=0.1% common comparison: network measures
    #
    # Each panel compares awake with NREM or anesthesia. Blue is the usual
    # Pearson graph and red is the graphical-lasso partial-correlation graph.
    # `L` is measured within the largest connected component; `GC` is the
    # fraction of all neurons contained in that component.

    # %%
    measures = [
        ("modularity_Q", "Modularity Q"),
        ("clustering", "Clustering C"),
        ("path_length_giant", "Path length L (giant component)"),
        ("giant_component_fraction", "Giant-component fraction"),
    ]
    x = np.arange(len(states))
    bar_width = 0.36
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))

    for axis, (column, label) in zip(axes.ravel(), measures):
        for method_index, graph_key in enumerate(COMMON_GRAPH_KEYS):
            settings = GRAPH_SPECS[graph_key]
            values = [
                float(
                    recording_table.loc[
                        (recording_table["state"] == state)
                        & (recording_table["graph_key"] == graph_key),
                        column,
                    ].iloc[0]
                )
                for state in states
            ]
            offset = (method_index - 0.5) * bar_width
            bars = axis.bar(
                x + offset,
                values,
                width=bar_width,
                color=str(settings["color"]),
                label=str(settings["label"]),
            )
            axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)

        axis.set_xticks(x, [display_state(state) for state in states])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=2,
    )
    fig.suptitle(
        f"{recording_name}: Pearson vs graphical lasso at K={COMMON_DENSITY:.2%}",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    measures_path = (
        FIGURE_DIR
        / f"simple_network_measures_{recording_name}_K{COMMON_DENSITY:.4f}.png"
    )
    fig.savefig(measures_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # %% [markdown]
    # ## 5. K=5% usual Pearson reference: raw binary graph matrices
    #
    # These panels provide the paper-facing density reference. They are not a
    # Pearson-versus-graphical-lasso comparison because the fitted partial graph
    # has insufficient nonzero support at K=5%.

    # %%
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharex=True, sharey=True)
    reference_settings = GRAPH_SPECS[REFERENCE_GRAPH_KEY]
    reference_block_density = {
        state: binned_edge_density(graphs[(state, REFERENCE_GRAPH_KEY)].adjacency)
        for state in states
    }
    display_max = max(
        float(np.quantile(block_density, 0.995))
        for block_density in reference_block_density.values()
    )
    display_max = max(display_max, 1e-12)
    for axis, state in zip(axes, states):
        selection = graphs[(state, REFERENCE_GRAPH_KEY)]
        image = axis.imshow(
            reference_block_density[state],
            cmap="Greys",
            vmin=0,
            vmax=display_max,
            interpolation="nearest",
            origin="upper",
            rasterized=True,
        )
        axis.set_title(
            f"{display_state(state)} — Pearson (usual)\n"
            f"{selection.adjacency.nnz // 2:,} edges; "
            f"threshold={selection.threshold:.4f}"
        )
        axis.set_xlabel("neuron-order bin")
        axis.set_ylabel("neuron-order bin")
        colorbar = fig.colorbar(image, ax=axis, shrink=0.78)
        colorbar.set_label("within-block edge fraction")

    fig.suptitle(
        f"{recording_name}: Pearson reference graph at "
        f"K={REFERENCE_DENSITY:.0%} ({n_nodes:,} neurons)",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    reference_adjacency_path = (
        FIGURE_DIR
        / f"pearson_reference_adjacency_{recording_name}_K{REFERENCE_DENSITY:.4f}.png"
    )
    fig.savefig(reference_adjacency_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    # %% [markdown]
    # ## 6. K=5% usual Pearson reference: network measures

    # %%
    fig, axes = plt.subplots(2, 2, figsize=(11, 8.5))
    for axis, (column, label) in zip(axes.ravel(), measures):
        values = [
            float(
                recording_table.loc[
                    (recording_table["state"] == state)
                    & (recording_table["graph_key"] == REFERENCE_GRAPH_KEY),
                    column,
                ].iloc[0]
            )
            for state in states
        ]
        bars = axis.bar(
            x,
            values,
            width=0.55,
            color=str(reference_settings["color"]),
        )
        axis.bar_label(bars, fmt="%.3f", fontsize=8, padding=2)
        axis.set_xticks(x, [display_state(state) for state in states])
        axis.set_ylabel(label)
        axis.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"{recording_name}: Pearson reference measures at K={REFERENCE_DENSITY:.0%}",
        fontsize=15,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    reference_measures_path = (
        FIGURE_DIR
        / f"pearson_reference_measures_{recording_name}_K{REFERENCE_DENSITY:.4f}.png"
    )
    fig.savefig(reference_measures_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

    del graphs

# %% [markdown]
# ## 7. Exact numerical table for both analyses

# %%
measure_table = pd.DataFrame(all_measure_rows)
table_path = COMPARISON_DIR / "simple_fixed_density_measures.csv"
measure_table.to_csv(table_path, index=False)

contrast_rows: list[dict[str, float | int | str]] = []
for recording_name in recording_names:
    states = list(CONFIG["recordings"][recording_name]["states"])
    awake_state, unconscious_state = states
    for graph_key, settings in GRAPH_SPECS.items():
        graph_rows = measure_table[
            (measure_table["recording"] == recording_name)
            & (measure_table["graph_key"] == graph_key)
        ].set_index("state")
        awake = graph_rows.loc[awake_state]
        unconscious = graph_rows.loc[unconscious_state]
        contrast_rows.append(
            {
                "recording": recording_name,
                "awake_state": awake_state,
                "unconscious_state": unconscious_state,
                "analysis": str(settings["analysis"]),
                "method_label": str(settings["label"]),
                "fixed_density": float(settings["density"]),
                "delta_modularity_Q": float(
                    unconscious["modularity_Q"] - awake["modularity_Q"]
                ),
                "delta_clustering": float(
                    unconscious["clustering"] - awake["clustering"]
                ),
                "delta_path_length_giant": float(
                    unconscious["path_length_giant"]
                    - awake["path_length_giant"]
                ),
                "delta_giant_component_fraction": float(
                    unconscious["giant_component_fraction"]
                    - awake["giant_component_fraction"]
                ),
            }
        )

contrast_table = pd.DataFrame(contrast_rows)
contrast_path = COMPARISON_DIR / "simple_fixed_density_state_contrasts.csv"
contrast_table.to_csv(contrast_path, index=False)

columns_to_display = [
    "recording",
    "state",
    "analysis",
    "method_label",
    "fixed_density",
    "n_edges",
    "threshold",
    "modularity_Q",
    "clustering",
    "path_length_giant",
    "giant_component_fraction",
    "n_isolates",
]
print("\nExact comparison table:")
print(measure_table[columns_to_display].to_string(index=False))
print("\nUnconscious minus awake contrasts:")
print(contrast_table.to_string(index=False))
print("\nsaved ->", table_path)
print("saved ->", contrast_path)
