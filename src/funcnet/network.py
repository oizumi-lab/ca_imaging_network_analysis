"""Functional-network construction and modularity, ported from the MATLAB repo.

Reference: https://github.com/oizumi-lab/mouse_network_2P
(functions/densityBasedThresh.m, functions/modularity_analysis.m,
functions/repeat_modularity_analysis.m, scripts/get_maxQ.m,
scripts/perform_consensus_clustering.m) and the dataset's
``example_network_analysis.m``.

Pipeline (single-cell functional network):

    spike_smoothed (N x T)
        -> correlation_matrix      : Pearson corr between neurons, zero diagonal
        -> density_threshold(K)    : keep strongest K-fraction of edges (binary)
        -> louvain_modularity(g)   : Louvain community detection, modularity Q
        [-> repeat_louvain / consensus_partition for robust partitions]

The Louvain routine is BCT's ``community_louvain`` via the ``bctpy`` package,
which is the same algorithm the paper used (resolution parameter ``gamma`` has
identical meaning: gamma>1 -> smaller modules, gamma<1 -> larger modules).
"""

from __future__ import annotations

import numpy as np
import bct
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components


# ----------------------------------------------------------------------------
# 1. Functional connectivity
# ----------------------------------------------------------------------------
def correlation_matrix(X: np.ndarray, zero_diagonal: bool = True) -> np.ndarray:
    """Pearson correlation between neurons.

    Parameters
    ----------
    X : ``(N, T)`` activity matrix (rows = neurons). Equivalent to MATLAB
        ``corr(X')`` where ``X`` is ``N x T``.

    Notes
    -----
    Neurons with zero variance produce NaNs; these are set to 0 (no edge), as in
    the MATLAB code (``corrMat(isnan(corrMat)) = 0``).
    """
    C = np.corrcoef(X)
    C = np.nan_to_num(C, nan=0.0)
    if zero_diagonal:
        np.fill_diagonal(C, 0.0)
    return C


# ----------------------------------------------------------------------------
# 2. Density-based thresholding  (port of densityBasedThresh.m)
# ----------------------------------------------------------------------------
def density_threshold(
    C: np.ndarray,
    K: float,
    weighted: bool = False,
    negative: bool = False,
) -> tuple[np.ndarray, float]:
    """Keep the strongest ``K`` fraction of edges.

    Faithful port of ``densityBasedThresh.m``: rank the upper-triangular
    correlation values, keep the top ``floor(K * N(N-1)/2)`` of them, and return
    a symmetric, zero-diagonal adjacency matrix.

    Parameters
    ----------
    C : ``(N, N)`` correlation matrix.
    K : target connection density in (0, 1], e.g. 0.05 = 5% of possible edges.
    weighted : if False, return a binary 0/1 matrix; if True, keep correlation
        weights on surviving edges.
    negative : if False, rank by signed correlation (keep most-positive edges, as
        the example does); if True, rank by absolute correlation.

    Returns
    -------
    (adj, thresh) : adjacency matrix and the correlation threshold applied.
    """
    C = C.copy()
    N = C.shape[0]
    np.fill_diagonal(C, 0.0)

    iu = np.triu_indices(N, k=1)
    vals = np.abs(C[iu]) if negative else C[iu]

    n_possible = N * (N - 1) // 2
    m = int(np.floor(K * n_possible))
    m = max(1, min(m, n_possible))

    # m-th largest value is the threshold (descending sort, 0-based index m-1).
    thresh = np.partition(vals, n_possible - m)[n_possible - m]

    rank = np.abs(C) if negative else C
    mask_u = np.zeros((N, N), dtype=bool)
    mask_u[iu] = rank[iu] >= thresh
    adj = mask_u | mask_u.T  # symmetric, zero diagonal by construction

    if weighted:
        return C * adj, float(thresh)
    return adj.astype(np.float64), float(thresh)


# ----------------------------------------------------------------------------
# 3. Modularity / community detection
# ----------------------------------------------------------------------------
def giant_component_init(adj: np.ndarray) -> np.ndarray:
    """Warm-start community vector, port of ``Ci_first`` in ``modularity_analysis.m``.

    Every node **outside** the largest connected component (all isolated /
    degree-0 nodes and any small components) shares a single community; each node
    **inside** the giant component starts as its own singleton. Passed to Louvain
    as the initial partition ``ci``.

    Why this matters: after density thresholding, some neurons have degree 0.
    Louvain can never move a degree-0 node (``k_i = 0`` -> no move changes Q), so
    with BCT's default init (``ci = 1..N``, every node its own community) each
    isolated neuron is frozen as its own singleton *module*, massively inflating
    the module count. Collapsing them into one shared community — as the paper's
    code does — is what keeps the reported number of modules meaningful.

    For a connected graph this returns ``1..N`` (identical to BCT's default), so
    it only changes the result when the thresholded network is disconnected.
    """
    N = adj.shape[0]
    n_comp, labels = connected_components(csr_matrix(adj != 0), directed=False)
    if n_comp == 1:
        return np.arange(1, N + 1)
    largest = int(np.argmax(np.bincount(labels)))
    conn = labels == largest
    ci0 = np.ones(N, dtype=int)
    ci0[conn] = np.arange(2, int(conn.sum()) + 2)
    return ci0


def louvain_modularity(
    W: np.ndarray,
    gamma: float = 1.0,
    B: str = "modularity",
    ci0: np.ndarray | None = None,
    seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """One Louvain run (BCT ``community_louvain``).

    Returns ``(ci, Q)`` where ``ci`` is the 1-based community label per node and
    ``Q`` is the modularity. For correlation (signed) matrices, pass
    ``B='negative_asym'``.
    """
    ci, Q = bct.community_louvain(W, gamma=gamma, ci=ci0, B=B, seed=seed)
    return ci, float(Q)


def repeat_louvain(
    W: np.ndarray,
    gamma: float = 1.0,
    n_runs: int = 200,
    B: str = "modularity",
    seed: int = 12345,
    warm_start: bool = True,
) -> dict:
    """Run Louvain ``n_runs`` times and keep the max-Q partition.

    Mirrors ``repeat_modularity_analysis.m`` + ``get_maxQ.m``: Louvain is
    stochastic, so the published pipeline runs it many times (200) and reports
    the highest-Q partition. Returns a dict with the full Q distribution, the
    partition matrix, and the best (max-Q) partition.

    ``warm_start`` (default True) matches ``modularity_analysis.m``: the
    giant-component initial partition (see :func:`giant_component_init`) is
    computed once from ``W`` and reused as the init for every run, so isolated
    nodes don't each become their own module. The per-run randomness comes only
    from the node-visit order (``seed``), exactly as in the MATLAB code. Set it
    to False to reproduce BCT's default ``ci = 1..N`` init.
    """
    rng = np.random.RandomState(seed)
    trial_seeds = rng.randint(1, 1_000_000, size=n_runs)

    N = W.shape[0]
    ci0 = giant_component_init(W) if warm_start else None
    Q_all = np.empty(n_runs)
    ci_all = np.empty((N, n_runs), dtype=int)
    for i, s in enumerate(trial_seeds):
        ci, Q = louvain_modularity(W, gamma=gamma, B=B, ci0=ci0, seed=int(s))
        Q_all[i] = Q
        ci_all[:, i] = ci

    best = int(np.argmax(Q_all))
    return {
        "Q_all": Q_all,
        "ci_all": ci_all,
        "Q_max": float(Q_all[best]),
        "ci_max": ci_all[:, best],
        "n_modules_max": int(np.unique(ci_all[:, best]).size),
    }


def modularity_from_activity(
    X: np.ndarray,
    density: float,
    gamma: float = 1.0,
    n_runs: int = 200,
    negative: bool = True,
    B: str = "modularity",
    seed: int = 12345,
    warm_start: bool = True,
) -> dict:
    """Run the standard activity-to-max-Q modularity pipeline.

    This convenience workflow consolidates the sequence repeated throughout the
    later tutorials: neuron activity → Pearson correlation → fixed-density
    binary graph → repeated Louvain optimization.  The lower-level functions
    remain available when a tutorial needs to show or modify an intermediate.

    Parameters
    ----------
    X
        ``(n_nodes, n_frames)`` activity matrix.
    density
        Fraction of possible edges retained by :func:`density_threshold`.
    gamma, n_runs, B, seed, warm_start
        Passed to :func:`repeat_louvain`.
    negative
        If true (the paper's main pipeline), rank correlations by absolute
        magnitude.  If false, retain the strongest positive correlations.

    Returns
    -------
    dict
        The :func:`repeat_louvain` result plus ``correlation_threshold``.  The
        full correlation and adjacency matrices are intentionally not retained,
        which keeps batch analyses from holding unnecessary quadratic arrays.
    """
    corr = correlation_matrix(X)
    adj, threshold = density_threshold(corr, density, negative=negative)
    result = repeat_louvain(
        adj,
        gamma=gamma,
        n_runs=n_runs,
        B=B,
        seed=seed,
        warm_start=warm_start,
    )
    result["correlation_threshold"] = threshold
    return result


def consensus_partition(
    ci_runs: np.ndarray,
    tau: float = 0.0,
    reps: int = 10,
    seed: int | None = None,
) -> np.ndarray:
    """Consensus clustering across many partitions.

    Port of ``perform_consensus_clustering.m``: build an agreement
    (co-assignment) matrix from the columns of ``ci_runs`` (each column is one
    Louvain partition), then resolve a single stable partition with BCT's
    ``consensus_und``.
    """
    n_runs = ci_runs.shape[1]
    agreement = bct.agreement(ci_runs) / n_runs
    return bct.consensus_und(agreement, tau=tau, reps=reps, seed=seed)


# ----------------------------------------------------------------------------
# 4. Independent modularity (for validation against BCT's Q)
# ----------------------------------------------------------------------------
def modularity_value(W: np.ndarray, ci: np.ndarray, gamma: float = 1.0) -> float:
    """Newman-Girvan modularity of partition ``ci`` on (un)weighted undirected W.

    Q = (1/2m) * sum_ij [ A_ij - gamma * k_i k_j / 2m ] * delta(c_i, c_j)

    Used only to cross-check ``louvain_modularity``'s reported Q in the
    reproduction script; it should agree to ~1e-12.
    """
    k = W.sum(axis=1)
    two_m = W.sum()
    if two_m == 0:
        return 0.0
    B = W - gamma * np.outer(k, k) / two_m
    same = ci[:, None] == ci[None, :]
    return float((B * same).sum() / two_m)


def n_modules(ci: np.ndarray) -> int:
    return int(np.unique(ci).size)
