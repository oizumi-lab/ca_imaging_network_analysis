"""Fixed-density edge selection and full-network summaries."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path


@dataclass(frozen=True)
class EdgeSelection:
    adjacency: csr_matrix
    rows: np.ndarray
    cols: np.ndarray
    weights: np.ndarray
    threshold: float
    requested_density: float
    realized_density: float
    native_density: float


def exact_fixed_density(
    matrix: np.ndarray,
    density: float,
    *,
    require_nonzero: bool = False,
    support_tol: float = 1e-10,
) -> EdgeSelection:
    """Select exactly the strongest absolute coefficients with stable tie rules."""
    values_matrix = np.asarray(matrix)
    if values_matrix.ndim != 2 or values_matrix.shape[0] != values_matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")
    n_nodes = values_matrix.shape[0]
    iu = np.triu_indices(n_nodes, 1)
    signed = np.asarray(values_matrix[iu], dtype=np.float64)
    absolute = np.abs(signed)
    n_possible = signed.size
    n_keep = max(1, min(int(np.floor(density * n_possible)), n_possible))
    native_count = int(np.count_nonzero(absolute > support_tol))
    if require_nonzero and native_count < n_keep:
        raise ValueError(
            f"requested {n_keep} edges but matrix has only {native_count} nonzero edges"
        )

    threshold = float(np.partition(absolute, n_possible - n_keep)[n_possible - n_keep])
    above = np.flatnonzero(absolute > threshold)
    boundary = np.flatnonzero(absolute == threshold)
    need = n_keep - above.size
    chosen = np.concatenate((above, boundary[:need]))
    if chosen.size != n_keep:
        raise RuntimeError("exact-density boundary selection failed")
    order = np.argsort(chosen)
    chosen = chosen[order]
    rows = iu[0][chosen]
    cols = iu[1][chosen]
    weights = signed[chosen]
    data = np.ones(2 * n_keep, dtype=np.float64)
    adjacency = csr_matrix(
        (data, (np.r_[rows, cols], np.r_[cols, rows])),
        shape=(n_nodes, n_nodes),
    )
    return EdgeSelection(
        adjacency=adjacency,
        rows=rows,
        cols=cols,
        weights=weights,
        threshold=threshold,
        requested_density=float(density),
        realized_density=n_keep / n_possible,
        native_density=native_count / n_possible,
    )


def _trim_top_candidates(
    rows: np.ndarray,
    cols: np.ndarray,
    weights: np.ndarray,
    *,
    n_keep: int,
    n_nodes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Keep exact strongest candidates, resolving ties by matrix order."""
    absolute = np.abs(weights)
    if weights.size > n_keep:
        threshold = np.partition(absolute, weights.size - n_keep)[
            weights.size - n_keep
        ]
        above = np.flatnonzero(absolute > threshold)
        boundary = np.flatnonzero(absolute == threshold)
        need = n_keep - above.size
        if need < boundary.size:
            boundary_keys = rows[boundary].astype(np.int64) * n_nodes + cols[boundary]
            boundary = boundary[np.argsort(boundary_keys, kind="stable")[:need]]
        chosen = np.concatenate((above, boundary))
        rows = rows[chosen]
        cols = cols[chosen]
        weights = weights[chosen]

    keys = rows.astype(np.int64) * n_nodes + cols
    order = np.argsort(keys, kind="stable")
    return rows[order], cols[order], weights[order]


def exact_fixed_density_blockwise(
    matrix: np.ndarray,
    density: float,
    *,
    require_nonzero: bool = False,
    support_tol: float = 1e-10,
    block_rows: int = 256,
) -> EdgeSelection:
    """Exact fixed-density selection without materializing all pair values."""
    values_matrix = np.asarray(matrix)
    if values_matrix.ndim != 2 or values_matrix.shape[0] != values_matrix.shape[1]:
        raise ValueError("matrix must be square")
    if not 0 < density <= 1:
        raise ValueError("density must be in (0, 1]")

    n_nodes = values_matrix.shape[0]
    n_possible = n_nodes * (n_nodes - 1) // 2
    n_keep = max(1, min(int(np.floor(density * n_possible)), n_possible))
    candidate_rows = np.empty(0, dtype=np.int32)
    candidate_cols = np.empty(0, dtype=np.int32)
    candidate_weights = np.empty(0, dtype=np.float64)
    native_count = 0

    for start in range(0, n_nodes - 1, block_rows):
        stop = min(start + block_rows, n_nodes - 1)
        row_ids = np.arange(start, stop, dtype=np.int32)
        counts = n_nodes - row_ids - 1
        block_rows_array = np.repeat(row_ids, counts)
        block_cols_array = np.concatenate(
            [np.arange(row + 1, n_nodes, dtype=np.int32) for row in row_ids]
        )
        block_weights = np.concatenate(
            [np.asarray(values_matrix[row, row + 1 :]) for row in row_ids]
        )
        native_count += int(np.count_nonzero(np.abs(block_weights) > support_tol))

        block_rows_array, block_cols_array, block_weights = _trim_top_candidates(
            block_rows_array,
            block_cols_array,
            block_weights,
            n_keep=n_keep,
            n_nodes=n_nodes,
        )
        candidate_rows = np.concatenate((candidate_rows, block_rows_array))
        candidate_cols = np.concatenate((candidate_cols, block_cols_array))
        candidate_weights = np.concatenate(
            (candidate_weights, block_weights.astype(np.float64, copy=False))
        )
        candidate_rows, candidate_cols, candidate_weights = _trim_top_candidates(
            candidate_rows,
            candidate_cols,
            candidate_weights,
            n_keep=n_keep,
            n_nodes=n_nodes,
        )

    if require_nonzero and native_count < n_keep:
        raise ValueError(
            f"requested {n_keep} edges but matrix has only {native_count} nonzero edges"
        )
    if candidate_weights.size != n_keep:
        raise RuntimeError("exact blockwise density selection failed")

    data = np.ones(2 * n_keep, dtype=np.float64)
    adjacency = csr_matrix(
        (
            data,
            (
                np.r_[candidate_rows, candidate_cols],
                np.r_[candidate_cols, candidate_rows],
            ),
        ),
        shape=(n_nodes, n_nodes),
    )
    return EdgeSelection(
        adjacency=adjacency,
        rows=candidate_rows,
        cols=candidate_cols,
        weights=candidate_weights,
        threshold=float(np.min(np.abs(candidate_weights))),
        requested_density=float(density),
        realized_density=n_keep / n_possible,
        native_density=native_count / n_possible,
    )


def edge_jaccard(first: EdgeSelection, second: EdgeSelection) -> float:
    """Jaccard overlap of two undirected edge selections."""
    n_nodes = first.adjacency.shape[0]
    if second.adjacency.shape != first.adjacency.shape:
        raise ValueError("edge selections must have the same node set")
    a = np.sort(first.rows.astype(np.int64) * n_nodes + first.cols)
    b = np.sort(second.rows.astype(np.int64) * n_nodes + second.cols)
    intersection = np.intersect1d(a, b, assume_unique=True).size
    union = a.size + b.size - intersection
    return float(intersection / union) if union else 1.0


def average_clustering_sparse(adjacency: csr_matrix) -> float:
    """Mean binary local clustering, assigning degree<2 nodes coefficient zero."""
    graph = adjacency.astype(np.float64).tocsr()
    degree = np.asarray(graph.sum(axis=1)).ravel()
    twice_triangles = np.asarray(graph.multiply(graph @ graph).sum(axis=1)).ravel()
    coefficient = np.zeros(graph.shape[0], dtype=np.float64)
    valid = degree >= 2
    coefficient[valid] = twice_triangles[valid] / (
        degree[valid] * (degree[valid] - 1)
    )
    return float(coefficient.mean())


def sampled_path_length(
    adjacency: csr_matrix,
    *,
    n_sources: int,
    seed: int,
) -> tuple[float, int, float]:
    """Path length within the largest component and its node fraction."""
    n_components, labels = connected_components(adjacency, directed=False)
    sizes = np.bincount(labels, minlength=n_components)
    largest = int(np.argmax(sizes))
    keep = np.flatnonzero(labels == largest)
    fraction = keep.size / adjacency.shape[0]
    if keep.size < 2:
        return float("nan"), int(keep.size), float(fraction)
    graph = adjacency[keep][:, keep]
    rng = np.random.default_rng(seed)
    source_count = min(int(n_sources), keep.size)
    sources = np.sort(rng.choice(keep.size, source_count, replace=False))
    distance = shortest_path(graph, directed=False, unweighted=True, indices=sources)
    row_index = np.arange(source_count)
    mask = np.isfinite(distance)
    mask[row_index, sources] = False
    return float(distance[mask].mean()), int(keep.size), float(fraction)


def summarize_graph(
    selection: EdgeSelection,
    coordinates_um: np.ndarray,
    *,
    path_sources: int,
    seed: int,
) -> dict[str, float | int]:
    """Deterministic graph summaries excluding stochastic community detection."""
    adjacency = selection.adjacency
    degree = np.asarray(adjacency.sum(axis=1)).ravel()
    path, giant_size, giant_fraction = sampled_path_length(
        adjacency, n_sources=path_sources, seed=seed
    )
    distance = np.sqrt(
        np.sum(
            (coordinates_um[selection.rows] - coordinates_um[selection.cols]) ** 2,
            axis=1,
        )
    )
    result = {
        "threshold": selection.threshold,
        "requested_density": selection.requested_density,
        "realized_density": selection.realized_density,
        "native_density": selection.native_density,
    }
    mean_degree = float(degree.mean())
    result.update(
        {
            "n_nodes": int(adjacency.shape[0]),
            "n_edges": int(adjacency.nnz // 2),
            "positive_edge_fraction": float(np.mean(selection.weights > 0)),
            "mean_degree": mean_degree,
            "degree_sd": float(degree.std()),
            "degree_cv": float(degree.std() / mean_degree) if mean_degree else float("nan"),
            "n_isolates": int(np.count_nonzero(degree == 0)),
            "giant_component_size": giant_size,
            "giant_component_fraction": giant_fraction,
            "clustering": average_clustering_sparse(adjacency),
            "path_length_giant": path,
            "edge_distance_mean_um": float(distance.mean()),
            "edge_distance_median_um": float(np.median(distance)),
        }
    )
    return result


def sampled_matrix_agreement(
    first: np.ndarray,
    second: np.ndarray,
    *,
    max_pairs: int,
    seed: int,
) -> dict[str, float | int]:
    """Signed/absolute Pearson and rank agreement on a reproducible pair sample."""
    from scipy.stats import pearsonr, spearmanr

    if first.shape != second.shape:
        raise ValueError("matrices must have matching shapes")
    iu = np.triu_indices(first.shape[0], 1)
    rng = np.random.default_rng(seed)
    if iu[0].size > max_pairs:
        take = np.sort(rng.choice(iu[0].size, max_pairs, replace=False))
        index = (iu[0][take], iu[1][take])
    else:
        index = iu
    a = np.asarray(first[index], dtype=np.float64)
    b = np.asarray(second[index], dtype=np.float64)
    supported = np.abs(b) > 1e-10
    sign_concordance = (
        float(np.mean(np.sign(a[supported]) == np.sign(b[supported])))
        if np.any(supported)
        else float("nan")
    )
    return {
        "n_pairs": int(a.size),
        "signed_pearson": float(pearsonr(a, b).statistic),
        "signed_spearman": float(spearmanr(a, b).statistic),
        "absolute_pearson": float(pearsonr(np.abs(a), np.abs(b)).statistic),
        "absolute_spearman": float(spearmanr(np.abs(a), np.abs(b)).statistic),
        "second_nonzero_fraction": float(np.mean(supported)),
        "sign_concordance_on_second_nonzero": sign_concordance,
    }
