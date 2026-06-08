"""funcnet — functional-network analysis for the RIKEN v2.0 calcium-imaging dataset.

Hands-on toolkit for the neural-data-analysis course: loading the v2.0 dataset
and computing correlation-based functional networks and modularity across brain
states (wakefulness, sleep, anesthesia).

Typical use::

    from funcnet import dataio, network as net
    from funcnet.paths import FIG_DIR

    rec = dataio.load_recording("mouse07_ane")
    C = net.correlation_matrix(dataio.activity(rec, "anesthesia", nonzero_only=True))
    adj, _ = net.density_threshold(C, K=0.05)
    print(net.repeat_louvain(adj, gamma=1.0, n_runs=200)["Q_max"])
"""

from __future__ import annotations

from . import dataio, network, paths

__all__ = ["dataio", "network", "paths"]
