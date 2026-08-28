"""funcnet — functional-network analysis for the RIKEN version-3 dataset.

Tutorial toolkit for loading and preparing the version-3 dataset, visualizing
population activity, and computing functional-network measures across
wakefulness, sleep, and anesthesia.

Typical use (the tutorial notebooks locate the repository before importing)::

    import numpy as np

    from src.funcnet import dataio, network as net, timeseries as ts

    rec = dataio.load_recording("mouse07_ane")
    rows = dataio.select_neuron_rows(rec)
    windows = ts.frame_windows(dataio.state_frames(rec, "anesthesia"), 2900)
    C = net.correlation_matrix(rec.spike_smoothed[np.ix_(rows, windows[0])])
    adj, _ = net.density_threshold(C, K=0.05)
    print(net.repeat_louvain(adj, gamma=1.0, n_runs=200)["Q_max"])
"""

from __future__ import annotations

from importlib import import_module

from . import (
    dataio,
    network,
    paths,
    timeseries,
)

_LAZY_MODULES = {
    "coarsegrain",
    "physiology",
    "statistics",
    "visualization",
}


def __getattr__(name: str):
    """Load optional SciPy/Matplotlib-backed categories only when requested."""
    if name in _LAZY_MODULES:
        module = import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include lazily exposed modules in interactive discovery."""
    return sorted(set(globals()) | set(__all__))

__all__ = [
    "coarsegrain",
    "dataio",
    "network",
    "paths",
    "physiology",
    "statistics",
    "timeseries",
    "visualization",
]
