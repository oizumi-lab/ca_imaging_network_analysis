"""Spatial coarse-graining of single-cell activity into mesoscale parcels.

Ports the mesoscale pipeline of ``oizumi-lab/mouse_network`` used for **Figure 7**
of Kiyooka & Oomoto et al. (2026): neurons are grouped into spatial parcels of
``nnei`` neighbours, their smoothed-spike time series are averaged within each
parcel, and a functional network is then estimated between parcels exactly as at
the single-cell scale (correlation -> density threshold -> Louvain modularity).

Reference MATLAB:
- ``kiyooka/RG_analysis/get_close_clustering.m``  -> :func:`close_clustering`
- ``kiyooka/fov_map/spatial_coarse_graining.m``   -> :func:`coarse_grain`

The parcellation ``get_close_clustering`` is a **deterministic greedy** grouping:
visit neurons in order of ``x + y`` (a diagonal sweep of the FOV); each unvisited
neuron seeds a new parcel that claims itself plus its nearest ``nnei - 1`` still-
unvisited neurons. Every parcel therefore holds exactly ``nnei`` neurons (the last
one may hold fewer). ``nnei = 1`` returns the identity partition (no
coarse-graining), so the whole Figure-7 sweep is one code path.
"""

from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def close_clustering(x, y, n_neighbors: int, D: np.ndarray | None = None) -> np.ndarray:
    """Greedy spatial parcels of ``n_neighbors`` neurons (port of ``get_close_clustering.m``).

    Parameters
    ----------
    x, y : ``(N,)`` neuron centroid coordinates (any consistent unit).
    n_neighbors : parcel size ``nnei``. ``<= 1`` returns ``arange(N)`` (identity).
    D : optional precomputed ``(N, N)`` Euclidean distance matrix; pass it to
        reuse across several ``n_neighbors`` for the same neurons.

    Returns
    -------
    ``(N,)`` int array of 0-based parcel labels (contiguous, in creation order).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    N = x.size
    if n_neighbors <= 1:
        return np.arange(N)

    if D is None:
        D = squareform(pdist(np.column_stack([x, y])))

    sweep = np.argsort(x + y, kind="stable")          # visit order: diagonal sweep
    idx = np.full(N, -1, dtype=int)
    visited = np.zeros(N, dtype=bool)
    count = 0
    for seed in sweep:
        if visited[seed]:
            continue
        order = np.argsort(D[seed], kind="stable")    # nearest first (self at 0)
        order = order[~visited[order]]                # drop visited, keep distance order
        take = order[:n_neighbors]                    # self + nearest unvisited
        idx[take] = count
        visited[take] = True
        count += 1
    return idx


def same_module_by_distance(coords, ci, edges=(500.0, 1000.0, 1500.0, 2000.0, 2500.0)):
    """Proportion of node pairs assigned to the same module, per distance bin.

    Reproduces the metric behind Kiyooka et al. **Fig. 5G/H** (single cell) and
    **Fig. 7G/H** (mesoscale) — MATLAB ``dist_and_mod.m`` / ``calc_dist_module_pairs.m``:
    bin every node pair by the Euclidean distance between the two nodes, then
    report the fraction of pairs in each bin whose two nodes share a module label.

    Parameters
    ----------
    coords : ``(N, 2)`` node coordinates in **micrometres** (to match the paper's
        500-µm bins). For parcels, use the mean coordinate of their neurons.
    ci : ``(N,)`` module labels.
    edges : upper edges of the distance bins (µm). The result has
        ``len(edges) + 1`` entries: ``[0, e0), [e0, e1), ..., [e_last, inf)``
        (default bins 0–500, 500–1000, 1000–1500, 1500–2000, 2000–2500, 2500+).

    Returns
    -------
    ``(len(edges)+1,)`` array of same-module proportions (NaN for empty bins).
    Flat across distance ⇒ spatially intermixed modules; decreasing ⇒ localized.
    """
    coords = np.asarray(coords, dtype=float)
    ci = np.asarray(ci)
    N = ci.shape[0]
    iu = np.triu_indices(N, 1)                       # upper-triangle pair order
    d = pdist(coords)                                # condensed, same order as iu
    same = (ci[iu[0]] == ci[iu[1]]).astype(float)
    b = np.digitize(d, np.asarray(edges, float))     # bin index 0 .. len(edges)
    out = np.full(len(edges) + 1, np.nan)
    for k in range(len(edges) + 1):
        m = b == k
        if m.any():
            out[k] = same[m].mean()
    return out


def module_localization_index(coords: np.ndarray, ci: np.ndarray) -> float:
    """Nearest-neighbour module agreement relative to chance.

    For every node, find its spatially nearest other node and measure how often
    the pair shares a module.  Divide that observed agreement by the chance
    probability ``sum((module_count / n_nodes) ** 2)``.  Values above one
    indicate that modules are more spatially localized than the label-frequency
    baseline.

    Parameters
    ----------
    coords
        ``(N, 2)`` node or parcel coordinates in any consistent spatial unit.
    ci
        ``(N,)`` module labels aligned with ``coords``.
    """
    coords = np.asarray(coords, dtype=float)
    ci = np.asarray(ci)
    if coords.ndim != 2 or coords.shape[1] != 2:
        raise ValueError("coords must have shape (n_nodes, 2)")
    if ci.ndim != 1 or ci.size != coords.shape[0]:
        raise ValueError("ci must contain one module label per coordinate")
    if ci.size < 2:
        raise ValueError("at least two nodes are required")

    distances = squareform(pdist(coords))
    np.fill_diagonal(distances, np.inf)
    nearest = np.argmin(distances, axis=1)
    _, counts = np.unique(ci, return_counts=True)
    chance = np.sum((counts / ci.size) ** 2)
    return float(np.mean(ci[nearest] == ci) / chance)


def coarse_grain(X: np.ndarray, x, y, idx: np.ndarray):
    """Average signals and centroids within each parcel (port of ``spatial_coarse_graining.m``).

    Parameters
    ----------
    X : ``(N, T)`` per-neuron time series (rows aligned with ``x``/``y``/``idx``).
    x, y : ``(N,)`` neuron centroids.
    idx : ``(N,)`` parcel labels from :func:`close_clustering`.

    Returns
    -------
    (res, x_parcel, y_parcel) : ``res`` is ``(K, T)`` mean parcel time series;
    ``x_parcel``/``y_parcel`` are the ``(K,)`` parcel centroids (mean of members).
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ids = np.unique(idx)
    res = np.empty((ids.size, X.shape[1]), dtype=float)
    xp = np.empty(ids.size, dtype=float)
    yp = np.empty(ids.size, dtype=float)
    for i, c in enumerate(ids):
        m = idx == c
        res[i] = X[m].mean(axis=0)
        xp[i] = x[m].mean()
        yp[i] = y[m].mean()
    return res, xp, yp
