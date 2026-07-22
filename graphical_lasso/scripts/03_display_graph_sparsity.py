# %% [markdown]
# # 03 · Display graphical-lasso sparsity panels
#
# This visualization-only script loads the completed full-neuron checkpoints;
# it does not refit graphical lasso. For every state it displays four panels:
# the regularization path, the native partial-correlation support matrix, the
# degree survival curve, and the spatial degree map. Figures are both rendered
# inline and saved under `results/02_network_comparison/figures/`.

# %%
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

SCRIPT_PATH = Path(__file__).resolve()
GLASSO_ROOT = SCRIPT_PATH.parents[1]
sys.path.insert(0, str(GLASSO_ROOT / "src"))

from glasso_analysis.config import COMPARISON_DIR, CONFIG, MATRIX_DIR  # noqa: E402

FIGURE_DIR = COMPARISON_DIR / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Settings
#
# Set `GLASSO_RECORDING=mouse01_sleep` or `mouse05_ane` to display one
# recording. The support is read in row blocks so the 7,693-neuron matrix does
# not need an additional full-size Boolean copy in memory.

# %%
requested_recording = os.environ.get("GLASSO_RECORDING")
recording_names = list(CONFIG["recordings"])
if requested_recording:
    if requested_recording not in recording_names:
        raise ValueError(f"unknown GLASSO_RECORDING={requested_recording!r}")
    recording_names = [requested_recording]

SUPPORT_TOL = float(CONFIG["precision_support_tolerance"])
BLOCK_ROWS = 512


def load_sparse_support(
    matrix_path: Path,
    *,
    support_tol: float,
    block_rows: int,
) -> csr_matrix:
    """Load an off-diagonal matrix support without a dense Boolean copy."""
    matrix = np.load(matrix_path, mmap_mode="r")
    n_nodes = matrix.shape[0]
    row_chunks: list[np.ndarray] = []
    column_chunks: list[np.ndarray] = []

    for start in range(0, n_nodes, block_rows):
        stop = min(start + block_rows, n_nodes)
        local_rows, columns = np.nonzero(np.abs(matrix[start:stop]) > support_tol)
        rows = local_rows + start
        off_diagonal = rows != columns
        row_chunks.append(rows[off_diagonal].astype(np.int32, copy=False))
        column_chunks.append(columns[off_diagonal].astype(np.int32, copy=False))

    rows = np.concatenate(row_chunks)
    columns = np.concatenate(column_chunks)
    adjacency = csr_matrix(
        (np.ones(rows.size, dtype=np.uint8), (rows, columns)),
        shape=(n_nodes, n_nodes),
    )
    adjacency = adjacency.maximum(adjacency.T).tocsr()
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


def component_sorted_order(adjacency: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    """Order components by size and nodes within each component by degree."""
    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    component_ids = np.argsort(-sizes, kind="stable")
    component_rank = np.empty(n_components, dtype=np.int64)
    component_rank[component_ids] = np.arange(n_components)
    degree = np.asarray(adjacency.getnnz(axis=1), dtype=np.int64)
    order = np.lexsort((-degree, component_rank[labels]))
    return order, sizes


# %% [markdown]
# ## Build and display the sparsity panels
#
# The dashed 1% and 5% lines in the first column are the repository's current
# fixed-density reference points. The native support panels use every nonzero
# partial-correlation edge at the primary alpha; no zero padding is applied.

# %%
summary_rows: list[dict[str, float | int | str]] = []

for recording_name in recording_names:
    recording_dir = MATRIX_DIR / recording_name
    population = json.loads((recording_dir / "population.json").read_text())
    coordinates_um = np.load(recording_dir / "coordinates_um.npy")
    states = list(population["states"])
    primary_alpha = float(CONFIG["primary_alpha_by_recording"][recording_name])

    fig, axes = plt.subplots(
        len(states),
        4,
        figsize=(19, 4.7 * len(states)),
        squeeze=False,
    )

    for state_index, state in enumerate(states):
        state_dir = recording_dir / state
        diagnostics = json.loads(
            (state_dir / "alpha_diagnostics.json").read_text(encoding="utf-8")
        )
        primary = next(
            item
            for item in diagnostics
            if np.isclose(float(item["alpha"]), primary_alpha)
        )
        adjacency = load_sparse_support(
            state_dir / "glasso_partial.npy",
            support_tol=SUPPORT_TOL,
            block_rows=BLOCK_ROWS,
        )
        degree = np.asarray(adjacency.getnnz(axis=1), dtype=np.int64)
        n_nodes = adjacency.shape[0]
        n_edges = adjacency.nnz // 2
        n_possible = n_nodes * (n_nodes - 1) // 2
        native_density = n_edges / n_possible
        order, component_sizes = component_sorted_order(adjacency)
        sorted_support = adjacency[order][:, order]

        if n_edges != int(primary["n_edges"]):
            raise RuntimeError(
                f"{recording_name}/{state}: support has {n_edges:,} edges but "
                f"diagnostics report {int(primary['n_edges']):,}"
            )

        n_components = component_sizes.size
        largest_component = int(component_sizes.max())
        isolates = int(np.count_nonzero(degree == 0))
        summary_rows.append(
            {
                "recording": recording_name,
                "state": state,
                "n_nodes": n_nodes,
                "primary_alpha": primary_alpha,
                "native_edges": n_edges,
                "native_density": native_density,
                "mean_degree": float(degree.mean()),
                "median_degree": float(np.median(degree)),
                "max_degree": int(degree.max()),
                "isolates": isolates,
                "n_components": n_components,
                "largest_component_nodes": largest_component,
                "largest_component_fraction": largest_component / n_nodes,
            }
        )

        path_axis, support_axis, degree_axis, spatial_axis = axes[state_index]

        path_rows = sorted(diagnostics, key=lambda item: float(item["alpha"]))
        path_alpha = np.array([float(item["alpha"]) for item in path_rows])
        path_density = np.array(
            [float(item["native_density"]) * 100 for item in path_rows]
        )
        path_axis.plot(path_alpha, path_density, marker="o", color="#2864a5")
        path_axis.scatter(
            [primary_alpha],
            [native_density * 100],
            marker="s",
            s=65,
            color="#d1495b",
            zorder=3,
            label=f"primary alpha={primary_alpha:g}",
        )
        for reference_density in (1.0, 5.0):
            path_axis.axhline(
                reference_density,
                color="#777777",
                linestyle="--",
                linewidth=1,
            )
        path_axis.set_yscale("log")
        path_axis.invert_xaxis()
        path_axis.set_xlabel("graphical-lasso alpha (lower = denser)")
        path_axis.set_ylabel("native partial-edge density (%)")
        path_axis.set_title(f"{state}: regularization path")
        path_axis.grid(alpha=0.25, which="both")
        path_axis.legend(fontsize=8, loc="lower left")

        support_axis.spy(
            sorted_support,
            marker=".",
            markersize=0.35,
            color="#111111",
            rasterized=True,
        )
        support_axis.set_title(
            "component-sorted native support\n"
            f"{n_edges:,} edges; K={native_density:.4%}"
        )
        support_axis.set_xlabel("neuron (component-sorted)")
        support_axis.set_ylabel("neuron (component-sorted)")

        positive_degree = np.sort(degree[degree > 0])
        survival = (positive_degree.size - np.arange(positive_degree.size)) / n_nodes
        degree_axis.step(positive_degree, survival, where="post", color="#3a7d44")
        degree_axis.axvline(
            np.median(degree),
            color="#d1495b",
            linestyle="--",
            label=f"median={np.median(degree):g}",
        )
        degree_axis.set_xscale("log")
        degree_axis.set_yscale("log")
        degree_axis.set_xlabel("native support degree")
        degree_axis.set_ylabel("fraction with degree >= x")
        degree_axis.set_title(
            f"degree survival; mean={degree.mean():.2f}, max={degree.max():,}\n"
            f"isolates={isolates:,} ({isolates / n_nodes:.2%})"
        )
        degree_axis.grid(alpha=0.25, which="both")
        degree_axis.legend(fontsize=8)

        spatial_plot = spatial_axis.scatter(
            coordinates_um[:, 0],
            coordinates_um[:, 1],
            c=np.log1p(degree),
            s=5,
            cmap="viridis",
            linewidths=0,
            rasterized=True,
        )
        spatial_axis.set_aspect("equal")
        spatial_axis.invert_yaxis()
        spatial_axis.set_xlabel("x (micrometers)")
        spatial_axis.set_ylabel("y (micrometers)")
        spatial_axis.set_title(
            "spatial native degree\n"
            f"components={n_components:,}; largest={largest_component:,} "
            f"({largest_component / n_nodes:.2%})"
        )
        colorbar = fig.colorbar(spatial_plot, ax=spatial_axis, shrink=0.76)
        colorbar.set_label("log(1 + degree)")

        print(
            f"{recording_name}/{state}: alpha={primary_alpha:g}, "
            f"edges={n_edges:,}, density={native_density:.6%}, "
            f"mean degree={degree.mean():.2f}, isolates={isolates:,}, "
            f"components={n_components:,}",
            flush=True,
        )

        del adjacency, sorted_support

    fig.suptitle(
        f"{recording_name}: graphical-lasso native partial-network sparsity "
        f"({coordinates_um.shape[0]:,} neurons)",
        fontsize=16,
        y=0.995,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    figure_path = FIGURE_DIR / f"sparsity_{recording_name}.png"
    fig.savefig(figure_path, dpi=180, bbox_inches="tight")
    plt.show()
    plt.close(fig)

# %% [markdown]
# ## Save the exact values shown in the panels

# %%
sparsity_table = pd.DataFrame(summary_rows)
suffix = f"_{requested_recording}" if requested_recording else ""
table_path = COMPARISON_DIR / f"sparsity_summary{suffix}.csv"
sparsity_table.to_csv(table_path, index=False)
print("saved sparsity summary ->", table_path)
