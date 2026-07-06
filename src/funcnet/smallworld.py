"""Small-world analysis: clustering coefficient, characteristic path length,
small-world-ness, and Small-World Propensity (SWP).

Ported from the MATLAB code in ``oizumi-lab/mouse_network`` (``kiyooka/SWP/`` and
``kiyooka/networkComparison/sliding2/characteristic_path_length_w.m``), which in
turn implements:

    Muldoon, Bridgeford & Bassett (2016), "Small-World Propensity and Weighted
    Brain Networks", Sci. Rep. 6:22057.  https://arxiv.org/abs/1505.02194

Pipeline used to produce the talk figures (``sw_summary.m`` + the driver
``script_20251218_calc_small_world.m``):

    spike_smoothed (window) → corr → |r| → density_threshold(K=0.01, binary)
        → largest connected component → small_world_propensity(...)

Definitions
-----------
For a graph with clustering C and characteristic path length L, compared to a
ring **lattice** (reg) and a **randomized** null (rand) with the same weights:

    small-world-ness   SMN   = (C_net / C_rand) / (L_net / L_rand)       (Humphries & Gurney 2008)
    ΔC (delta_C)             = (C_reg − C_net) / (C_reg − C_rand)  ∈ [0, 1]
    ΔL (delta_L)             = (L_net − L_rand) / (L_reg − L_rand)  ∈ [0, 1]
    SWP                      = 1 − sqrt(ΔC² + ΔL²) / sqrt(2)

Higher SMN, higher 1/ΔC (more clustered than random), and higher ΔL (longer
paths than random) all indicate a more locally-clustered / less globally
integrated network — the paper's signature of unconsciousness.

Performance note
----------------
The cost is the all-pairs shortest-path (APSP) on the largest connected
component (thousands of nodes). ``characteristic_path_length`` accepts
``n_sources`` to estimate L from a random sample of source nodes — an unbiased
estimator that is far cheaper than the full APSP and is accurate to ~1% with a
few hundred sources.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path

from .network import correlation_matrix, density_threshold


# ----------------------------------------------------------------------------
# Clustering coefficient  (port of clustering_coef_matrix.m)
# ----------------------------------------------------------------------------
def _is_binary(W: np.ndarray) -> bool:
    return bool(np.array_equal(W, (W > 0)))


def clustering_coef(W: np.ndarray, method: str = "O") -> np.ndarray:
    """Per-node clustering coefficient.

    method : 'O' Onnela (default), 'Z' Zhang, 'B' Barrat, 'bin' binary.
    For a binary matrix, Onnela and 'bin' coincide; we use a fast sparse
    triangle count in that case.
    """
    W = np.asarray(W, dtype=float)
    n = W.shape[0]

    if method == "bin" or (method == "O" and _is_binary(W)):
        # C_i = 2 t_i / (k_i (k_i − 1)),  t_i from diag(A^3) via sparse ops
        A = csr_matrix((W > 0).astype(np.float64))
        k = np.asarray(A.sum(1)).ravel()
        tri = np.asarray(A.multiply(A @ A).sum(1)).ravel()  # = diag(A^3) = 2·triangles
        C = np.zeros(n)
        nz = k >= 2
        C[nz] = tri[nz] / (k[nz] * (k[nz] - 1))
        return C

    if method == "O":  # Onnela, weighted
        K = (W != 0).sum(1).astype(float)
        W2 = W / W.max() if W.max() > 0 else W
        cyc3 = np.diag(np.linalg.matrix_power(np.cbrt(W2), 3))
        K = K.copy()
        K[cyc3 == 0] = np.inf
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(cyc3 / (K * (K - 1)))

    if method == "Z":  # Zhang & Horvath, weighted
        W2 = W / W.max() if W.max() > 0 else W
        cyc3 = np.diag(np.linalg.matrix_power(W2, 3))
        s1 = W2.sum(1)
        denom = s1 ** 2 - (W2 ** 2).sum(1)
        with np.errstate(divide="ignore", invalid="ignore"):
            return np.nan_to_num(cyc3 / denom)

    if method == "B":  # Barrat, weighted
        A = (W > 0).astype(float)
        s = W.sum(1)
        deg = A.sum(1)
        C = np.zeros(n)
        for i in range(n):
            num = np.sum(((W[i, :, None] + W[i, None, :]) / 2) * A[i, :, None] * A * A[i, None, :])
            denom = s[i] * (deg[i] - 1)
            C[i] = num / denom if denom > 0 else 0.0
        return C

    raise ValueError(f"unknown clustering method {method!r}")


def avg_clustering(W: np.ndarray, method: str = "O") -> float:
    """Mean clustering coefficient over nodes (NaNs ignored)."""
    return float(np.nanmean(clustering_coef(W, method)))


# ----------------------------------------------------------------------------
# Characteristic path length  (port of characteristic_path_length_w.m)
# ----------------------------------------------------------------------------
def characteristic_path_length(
    W: np.ndarray,
    n_sources: int | None = None,
    rng: np.random.RandomState | None = None,
) -> float:
    """Average shortest-path length over all ordered node pairs.

    Edge weights are turned into distances as ``1/W`` (so stronger connections
    are shorter), matching ``characteristic_path_length_w.m``. For a binary
    graph this is the mean hop count. If the graph is disconnected the value is
    ``inf`` (as in the MATLAB code, which then sets ΔL = 1).

    n_sources : if given, estimate L from this many random source nodes
        (unbiased, much faster than the full all-pairs computation).
    """
    n = W.shape[0]
    binary = _is_binary(W)
    if binary:
        graph = csr_matrix((W > 0).astype(np.int8))
    else:
        with np.errstate(divide="ignore"):
            dist = np.where(W > 0, 1.0 / np.where(W > 0, W, 1.0), 0.0)
        graph = csr_matrix(dist)

    if n_sources is not None and n_sources < n:
        src = (rng or np.random).choice(n, n_sources, replace=False)
        sp = shortest_path(graph, method="D", unweighted=binary, directed=False, indices=src)
        per_source = np.empty(len(src))
        for r_i, s in enumerate(src):
            row = sp[r_i].copy()
            row[s] = 0.0  # distance to self
            per_source[r_i] = row.sum() / (n - 1)  # inf propagates if disconnected
        return float(np.mean(per_source))

    sp = shortest_path(graph, method="D", unweighted=binary, directed=False)
    off = ~np.eye(n, dtype=bool)
    return float(sp[off].sum() / (n * (n - 1)))


# ----------------------------------------------------------------------------
# Null networks  (ports of regular_matrix_generator.m and randomize_matrix.m)
# ----------------------------------------------------------------------------
def randomize_matrix(W: np.ndarray, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Randomize edges: shuffle the upper-triangular weights across all pairs.

    Preserves the number of nodes and the exact weight distribution (for binary
    input, an Erdős–Rényi graph with the same edge count). Port of
    ``randomize_matrix.m``.
    """
    rng = rng or np.random
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    vals = W[iu].copy()
    vals = vals[rng.permutation(vals.size)]
    R = np.zeros_like(W, dtype=float)
    R[iu] = vals
    return R + R.T


def regular_lattice(W: np.ndarray, r: int, rng: np.random.RandomState | None = None) -> np.ndarray:
    """Ring-lattice null with the network's weights, radius ``r``.

    Node ``i`` connects to ``i±1 … i±r`` (mod n); the strongest weights are
    placed on the innermost ring, matching ``regular_matrix_generator.m``. The
    last (partial) ring is filled at random nodes. For a binary input this is a
    standard ring lattice with the same edge count.
    """
    rng = rng or np.random
    n = W.shape[0]
    iu = np.triu_indices(n, 1)
    w_desc = np.sort(W[iu][W[iu] > 0])[::-1]  # edge weights, strongest first
    num_edges = w_desc.size
    M = np.zeros((n, n), dtype=float)

    placed = 0
    i_all = np.arange(n)
    for z in range(1, r + 1):
        if placed >= num_edges:
            break
        j = (i_all + z) % n
        remaining = num_edges - placed
        if remaining >= n:
            rows = i_all
        else:  # last partial ring: random subset of nodes
            rows = np.sort(rng.choice(n, remaining, replace=False))
        w = w_desc[placed:placed + len(rows)]
        M[rows, j[rows]] = w
        M[j[rows], rows] = w
        placed += len(rows)
    return M


# ----------------------------------------------------------------------------
# Small-World Propensity  (port of small_world_propensity.m)
# ----------------------------------------------------------------------------
@dataclass
class SWResult:
    """Outputs of :func:`small_world_propensity` for one network."""
    SWP: float           # small-world propensity ∈ [0, 1]
    sw_ness: float       # small-world-ness SMN = (C/Crand)/(L/Lrand)
    delta_C: float       # clustering deviation ΔC ∈ [0, 1]
    delta_L: float       # path-length deviation ΔL ∈ [0, 1]
    net_clus: float
    net_path: float
    reg_clus: float
    rand_clus: float
    reg_path: float
    rand_path: float
    n: int               # number of nodes analysed


def small_world_propensity(
    W: np.ndarray,
    method: str = "O",
    n_sources: int | None = None,
    rng: np.random.RandomState | None = None,
) -> SWResult:
    """Small-world propensity of an (assumed undirected) network.

    Builds a ring-lattice and a randomized null with the same weights, then
    measures how the network's clustering and path length sit between them.
    """
    rng = rng or np.random.RandomState()
    W = np.asarray(W, dtype=float)
    n = W.shape[0]

    # approximate lattice radius from the average degree
    numb_connections = int((W > 0).sum())         # = 2 · edges (symmetric)
    avg_rad_eff = int(np.ceil((numb_connections / n) / 2))

    W_reg = regular_lattice(W, avg_rad_eff, rng)
    W_rand = randomize_matrix(W, rng)

    net_path = characteristic_path_length(W, n_sources, rng)
    reg_path = characteristic_path_length(W_reg, n_sources, rng)
    rand_path = characteristic_path_length(W_rand, n_sources, rng)

    net_clus = avg_clustering(W, method)
    reg_clus = avg_clustering(W_reg, method)
    rand_clus = avg_clustering(W_rand, method)

    # ΔL: where the network's path length sits between random and lattice
    num = net_path - rand_path
    if num < 0:
        num = 0.0
    if not np.isfinite([net_path, reg_path, rand_path]).all():
        delta_L = 1.0
    else:
        delta_L = min(num / (reg_path - rand_path), 1.0)

    # ΔC: where the network's clustering sits between random and lattice
    num_c = reg_clus - net_clus
    if num_c < 0:
        num_c = 0.0
    if np.isnan([reg_clus, rand_clus, net_clus]).any():
        delta_C = 1.0
    else:
        delta_C = min(num_c / (reg_clus - rand_clus), 1.0)

    SWP = 1.0 - np.sqrt(delta_C ** 2 + delta_L ** 2) / np.sqrt(2)
    # small-world-ness is undefined when the random null has zero clustering, or a
    # non-finite path length (a sparse random graph with no triangles / a
    # disconnected null → rand_path = inf); guard it so SWP (which does not divide
    # by rand_clus) is still returned.
    if rand_clus > 0 and net_path > 0 and np.isfinite([net_path, rand_path]).all() and rand_path > 0:
        sw_ness = (net_clus / rand_clus) / (net_path / rand_path)
    else:
        sw_ness = float("nan")

    return SWResult(
        SWP=float(SWP), sw_ness=float(sw_ness),
        delta_C=float(delta_C), delta_L=float(delta_L),
        net_clus=net_clus, net_path=net_path,
        reg_clus=reg_clus, rand_clus=rand_clus,
        reg_path=reg_path, rand_path=rand_path, n=n,
    )


# ----------------------------------------------------------------------------
# Driver  (port of sw_summary.m): corr → |r| → density threshold → largest CC
# ----------------------------------------------------------------------------
def largest_component(adj: np.ndarray) -> np.ndarray:
    """Row/col indices of the largest connected component of ``adj``."""
    _, labels = connected_components(csr_matrix(adj), directed=False)
    biggest = np.argmax(np.bincount(labels))
    return np.flatnonzero(labels == biggest)


def sw_summary(
    corr: np.ndarray,
    density: float = 0.01,
    method: str = "O",
    n_sources: int | None = None,
    rng: np.random.RandomState | None = None,
) -> SWResult:
    """Full small-world summary from a correlation matrix (port of ``sw_summary.m``).

    Thresholds ``|corr|`` to a binary graph at the given density, restricts to
    the largest connected component, and returns the small-world result.
    """
    adj, _ = density_threshold(np.abs(corr), density, weighted=False)
    idx = largest_component(adj)
    return small_world_propensity(adj[np.ix_(idx, idx)], method=method,
                                  n_sources=n_sources, rng=rng)


def sw_from_activity(
    X: np.ndarray,
    density: float = 0.01,
    method: str = "O",
    n_sources: int | None = None,
    rng: np.random.RandomState | None = None,
) -> SWResult:
    """Convenience: activity matrix (N × T) → correlation → :func:`sw_summary`."""
    return sw_summary(correlation_matrix(X), density=density, method=method,
                      n_sources=n_sources, rng=rng)
