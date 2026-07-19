"""Loader for the RIKEN v2.0 two-photon calcium-imaging dataset.

Dataset: "Single-cell calcium imaging dataset of large-scale neuronal activity
across wakefulness, sleep, and anesthesia" (RIKEN neurodata 20260409-001, v2.0;
Kiyooka & Oomoto et al., Cell Reports 2026).

Why this module exists
----------------------
The published analysis code (https://github.com/oizumi-lab/mouse_network_2P) was
written for dataset **v1.0**. The current **v2.0** release renamed several
variables and reorganised the structure. This module hides those differences
behind one clean ``Recording`` object so the rest of the project never has to
think about MATLAB internals.

Two gotchas this module handles for you
--------------------------------------
1. The ``.mat`` files are **MATLAB v7.3 (HDF5)**. ``scipy.io.loadmat`` CANNOT read
   them (it only supports <= v7.2), despite what the dataset README says. We use
   ``pymatreader`` instead, which decodes v7.3, transposes arrays back to MATLAB
   orientation (so matrices come out ``N x T`` as documented), and turns structs
   into nested dicts.
2. ``frame.used_frame`` / ``frame.boundary_ind`` are stored as **MATLAB 1-based**
   frame indices. We convert them to **0-based** so they can index NumPy arrays
   directly.

v1 -> v2 variable mapping
-------------------------
    smoothed_spike   -> spike_smoothed
    spike            -> spike_deconv
    dFF              -> dFF              (unchanged)
    atlasID (+ st)   -> ROIs.atlas       (region-acronym strings)
    x_coord/y_coord  -> ROIs.Centroid    (N x 2, pixels; 1.465 um/px)
    used_frame{1/2}  -> frame.used_frame{1,1}=awake, {1,2}=NREM or anesthesia
    boundary_ind     -> frame.boundary_ind
    SleepState       -> state            (1 x T)
    (separate files) -> data_info        ('sleep' | 'ane')
    (n/a)            -> nonzero_ROI       (activity filter used in the paper)

``ROIs.atlas`` is a MATLAB *string-class* array stored through an MCOS payload.
``pymatreader`` does not decode that object, so this loader reads the compact
UTF-16 payload directly with ``h5py`` and exposes one row-aligned cortical-region
acronym per neuron through ``Recording.atlas``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import h5py
import numpy as np
from pymatreader import read_mat

from .paths import PROJECT_ROOT, RAW_DIR  # noqa: F401  (re-exported for convenience)

# ----------------------------------------------------------------------------
# Constants from the dataset metadata
# ----------------------------------------------------------------------------
FS_HZ = 7.65            # imaging sampling rate
PX_TO_UM = 1.465        # micrometres per pixel (3000 um / 2048 px ~= 1.465)
SMOOTH_FRAMES = 15      # Gaussian smoothing window used to build spike_smoothed

# State-code legend (values found in the ``state`` vector).
SLEEP_STATE_CODES = {0.0: "awake", 0.5: "quiet_awake", 1.0: "nrem", 2.0: "rem"}
ANE_STATE_CODES = {0.0: "awake", 1.0: "anesthesia"}


@dataclass
class Recording:
    """A single recording session with v2.0 variables exposed by clean names.

    All time-series matrices are ``(n_neurons, n_frames)`` and row-aligned:
    row ``i`` is the same neuron in ``dFF``, ``spike_deconv``, ``spike_smoothed``,
    ``centroid``, ``atlas`` and ``nonzero_ROI``.
    """

    name: str
    data_info: str                       # 'sleep' or 'ane'
    dFF: np.ndarray                       # (N, T) relative fluorescence change
    spike_deconv: np.ndarray             # (N, T) OASIS-deconvolved spikes
    spike_smoothed: np.ndarray           # (N, T) Gaussian-smoothed; used for networks
    state: np.ndarray                    # (T,) brain-state code per frame
    centroid: np.ndarray                 # (N, 2) neuron centroids in pixels (x, y)
    used_frame: dict[str, np.ndarray]    # state label -> 0-based frame indices
    boundary_ind: np.ndarray             # 0-based frame indices of acquisition breaks
    atlas: list[str] | None = None       # region acronym per neuron, or None
    nonzero_ROI: np.ndarray | None = None  # (N,) bool: in the paper's network analysis
    animal_info: dict = field(default_factory=dict)
    fs: float = FS_HZ
    px_to_um: float = PX_TO_UM

    # -- convenience --------------------------------------------------------
    @property
    def n_neurons(self) -> int:
        return self.spike_smoothed.shape[0]

    @property
    def n_frames(self) -> int:
        return self.spike_smoothed.shape[1]

    @property
    def state_labels(self) -> list[str]:
        """The two states compared in this recording, awake first."""
        second = "nrem" if self.data_info == "sleep" else "anesthesia"
        return ["awake", second]

    @property
    def centroid_um(self) -> np.ndarray:
        """Neuron centroids in micrometres."""
        return self.centroid * self.px_to_um

    def __repr__(self) -> str:  # short, informative
        dur_min = self.n_frames / self.fs / 60
        return (
            f"Recording({self.name!r}, {self.data_info}, "
            f"N={self.n_neurons} neurons, T={self.n_frames} frames "
            f"~{dur_min:.0f} min, states={self.state_labels})"
        )


# ----------------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------------
def _as_1d_int(x) -> np.ndarray:
    """MATLAB scalars/vectors -> flat int64 NumPy array."""
    return np.atleast_1d(np.asarray(x)).ravel().astype(np.int64)


def _to_zero_based(idx) -> np.ndarray:
    """Convert MATLAB 1-based frame indices to 0-based, with a sanity check."""
    arr = _as_1d_int(idx) - 1
    if arr.size and arr.min() < 0:
        warnings.warn(
            "Found frame index < 1 before conversion; data may already be 0-based.",
            stacklevel=2,
        )
    return arr


def _resolve_path(path_or_name: str | Path) -> Path:
    """Accept a full path, a file name, or a bare recording name."""
    p = Path(path_or_name)
    if p.exists():
        return p
    cand = RAW_DIR / p.name
    if cand.exists():
        return cand
    cand = RAW_DIR / (p.name if p.suffix == ".mat" else f"{p.name}.mat")
    if cand.exists():
        return cand
    raise FileNotFoundError(
        f"Could not find '{path_or_name}'. Looked in {RAW_DIR}. "
        f"Run `python scripts/download_data.py` first."
    )


def _decode_mcos_string_payload(
    payload: np.ndarray,
    expected_count: int,
) -> list[str]:
    """Decode the packed MATLAB-v2 string payload used by ``ROIs.atlas``.

    The dataset stores ``[1, 2, N, 1]``, followed by ``N`` UTF-16 code-unit
    lengths and then four little-endian UTF-16 units per uint64 word. Keeping
    this format-specific operation separate makes it straightforward to test
    without loading a multi-gigabyte recording.
    """
    values = np.asarray(payload, dtype=np.uint64).ravel()
    if expected_count <= 0:
        raise ValueError("expected_count must be positive")
    header = np.array([1, 2, expected_count, 1], dtype=np.uint64)
    if values.size < 4 + expected_count or not np.array_equal(values[:4], header):
        raise ValueError("unrecognized MATLAB string payload header")

    lengths = values[4 : 4 + expected_count].astype(np.int64)
    if np.any(lengths < 0):
        raise ValueError("MATLAB string payload contains a negative length")
    total_units = int(lengths.sum())
    expected_words = (total_units + 3) // 4
    packed = values[4 + expected_count :]
    if packed.size != expected_words:
        raise ValueError("MATLAB string payload size does not match its lengths")

    shifts = np.array([0, 16, 32, 48], dtype=np.uint64)
    code_units = ((packed[:, None] >> shifts) & 0xFFFF).astype("<u2").ravel()
    code_units = code_units[:total_units]

    labels: list[str] = []
    start = 0
    for length in lengths:
        stop = start + int(length)
        labels.append(code_units[start:stop].tobytes().decode("utf-16le"))
        start = stop
    return labels


def _decode_v2_mcos_atlas(path: Path, expected_count: int) -> list[str] | None:
    """Locate and decode the unique MCOS string payload for ``ROIs.atlas``."""
    with h5py.File(path, "r") as mat:
        if "ROIs/atlas" not in mat:
            return None
        atlas_dataset = mat["ROIs/atlas"]
        if atlas_dataset.attrs.get("MATLAB_class") != b"string":
            return None
        if "#subsystem#/MCOS" not in mat:
            raise ValueError("MATLAB string object has no MCOS subsystem payload")

        decoded_candidates: list[list[str]] = []
        for reference in np.asarray(mat["#subsystem#/MCOS"]).ravel():
            if not reference:
                continue
            candidate = mat[reference]
            if (
                not isinstance(candidate, h5py.Dataset)
                or candidate.dtype.kind != "u"
                or candidate.dtype.itemsize != 8
            ):
                continue
            try:
                labels = _decode_mcos_string_payload(candidate[...], expected_count)
            except ValueError:
                continue
            decoded_candidates.append(labels)

    if len(decoded_candidates) != 1:
        raise ValueError(
            "expected one row-aligned MCOS atlas payload, found "
            f"{len(decoded_candidates)}"
        )
    return decoded_candidates[0]


def load_recording(path_or_name: str | Path) -> Recording:
    """Load one ``.mat`` recording into a :class:`Recording`.

    Parameters
    ----------
    path_or_name
        A path to a ``.mat`` file, or a recording name such as
        ``"mouse01_sleep"`` / ``"example_data"`` (resolved under data/raw).
    """
    path = _resolve_path(path_or_name)

    # pymatreader warns about the MCOS string object. We decode that field
    # separately below, while retaining pymatreader for the numerical arrays.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        raw = read_mat(str(path))

    data_info = str(np.asarray(raw["data_info"]).ravel()[0]) if not isinstance(
        raw["data_info"], str
    ) else raw["data_info"]

    rois = raw.get("ROIs", {}) or {}
    frame = raw.get("frame", {}) or {}

    # Centroid: ensure (N, 2).
    centroid = np.asarray(rois.get("Centroid"))
    if centroid.ndim == 2 and centroid.shape[0] == 2 and centroid.shape[1] != 2:
        centroid = centroid.T  # guard against transposed delivery

    # used_frame: list of two index vectors (awake, second state), 1-based -> 0-based.
    uf_raw = frame.get("used_frame")
    if isinstance(uf_raw, (list, tuple)):
        uf_list = list(uf_raw)
    else:  # single vector edge-case
        uf_list = [uf_raw]
    second_label = "nrem" if data_info == "sleep" else "anesthesia"
    labels = ["awake", second_label]
    used_frame = {
        labels[i]: _to_zero_based(uf_list[i])
        for i in range(min(len(labels), len(uf_list)))
    }

    # Future pymatreader versions may decode the string array directly. Prefer
    # that result when available; otherwise use the v2 MCOS payload decoder.
    atlas_raw = rois.get("atlas")
    atlas = None
    if isinstance(atlas_raw, (list, np.ndarray)) and np.size(atlas_raw) and not (
        isinstance(atlas_raw, np.ndarray) and atlas_raw.dtype.kind in "uif"
    ):
        atlas = [str(a) for a in np.atleast_1d(atlas_raw).ravel()]
    if atlas is None:
        try:
            atlas = _decode_v2_mcos_atlas(path, centroid.shape[0])
        except (KeyError, OSError, ValueError) as exc:
            warnings.warn(
                f"Could not decode cortical-region labels from {path.name}: {exc}",
                UserWarning,
                stacklevel=2,
            )

    nz = raw.get("nonzero_ROI")
    nonzero_ROI = None if nz is None else np.atleast_1d(np.asarray(nz)).ravel().astype(bool)

    rec = Recording(
        name=path.stem,
        data_info=data_info,
        dFF=np.asarray(raw["dFF"]),
        spike_deconv=np.asarray(raw["spike_deconv"]),
        spike_smoothed=np.asarray(raw["spike_smoothed"]),
        state=np.atleast_1d(np.asarray(raw["state"])).ravel().astype(float),
        centroid=centroid,
        used_frame=used_frame,
        boundary_ind=_to_zero_based(frame.get("boundary_ind", [])),
        atlas=atlas,
        nonzero_ROI=nonzero_ROI,
        animal_info=dict(raw.get("animal_info", {}) or {}),
    )
    _validate(rec)
    return rec


def _validate(rec: Recording) -> None:
    """Assert the row/column alignment promised by the dataset README."""
    n, t = rec.spike_smoothed.shape
    for nm in ("dFF", "spike_deconv"):
        assert getattr(rec, nm).shape == (n, t), f"{nm} shape != spike_smoothed"
    assert rec.state.shape[0] == t, "state length != n_frames"
    assert rec.centroid.shape[0] == n, "centroid rows != n_neurons"
    if rec.atlas is not None:
        assert len(rec.atlas) == n, "atlas length != n_neurons"
    if rec.nonzero_ROI is not None:
        assert rec.nonzero_ROI.shape[0] == n, "nonzero_ROI length != n_neurons"
    for lab, idx in rec.used_frame.items():
        if idx.size:
            assert idx.min() >= 0 and idx.max() < t, f"used_frame[{lab}] out of range"


# ----------------------------------------------------------------------------
# Helpers used by the analysis scripts
# ----------------------------------------------------------------------------
def state_frames(rec: Recording, which: str) -> np.ndarray:
    """0-based frame indices for a state label ('awake', 'nrem', 'anesthesia')."""
    if which not in rec.used_frame:
        raise KeyError(f"{which!r} not in used_frame; available: {list(rec.used_frame)}")
    return rec.used_frame[which]


def state_codes(rec: Recording) -> dict[float, str]:
    """Return the numeric state-code legend appropriate for ``rec``.

    Code ``1`` means NREM in sleep recordings but anesthesia in anesthesia
    recordings.  Centralizing this dispatch prevents visualizations and future
    analyses from silently applying the wrong categorical meaning.
    """
    if rec.data_info == "sleep":
        return SLEEP_STATE_CODES
    if rec.data_info == "ane":
        return ANE_STATE_CODES
    raise ValueError(f"Unknown recording type: {rec.data_info!r}")


def select_neuron_rows(
    rec: Recording,
    max_neurons: int | None = None,
    seed: int = 0,
    active_only: bool = True,
) -> np.ndarray:
    """Select reproducible neuron-row indices for a recording analysis.

    Parameters
    ----------
    rec
        Recording whose row-aligned matrices will be indexed.
    max_neurons
        Optional maximum row count.  If a subsample is needed it is drawn
        without replacement, sorted back into recording order, and seeded.
    seed
        Seed for NumPy's legacy ``RandomState``.  The tutorials historically
        used ``RandomState(0)``; retaining it keeps existing scientific examples
        numerically reproducible across this refactor.
    active_only
        If true, start from ``nonzero_ROI`` when available.  Otherwise all rows
        are eligible.  A recording without that mask also falls back to all rows.
    """
    if active_only and rec.nonzero_ROI is not None:
        rows = np.flatnonzero(rec.nonzero_ROI)
    else:
        rows = np.arange(rec.n_neurons)

    if max_neurons is None:
        return rows
    if isinstance(max_neurons, (bool, np.bool_)) or not isinstance(
        max_neurons, (int, np.integer)
    ):
        raise TypeError("max_neurons must be an integer or None")
    if max_neurons <= 0:
        raise ValueError("max_neurons must be positive")
    if rows.size <= max_neurons:
        return rows

    rng = np.random.RandomState(seed)
    return np.sort(rng.choice(rows, int(max_neurons), replace=False))


def activity(
    rec: Recording,
    which: str,
    signal: str = "spike_smoothed",
    nonzero_only: bool = False,
) -> np.ndarray:
    """Extract a ``(N, n_frames_in_state)`` activity matrix for one state.

    Parameters
    ----------
    which : state label ('awake' / 'nrem' / 'anesthesia').
    signal : which time series ('spike_smoothed', 'spike_deconv', 'dFF').
    nonzero_only : if True and ``nonzero_ROI`` exists, keep only the neurons the
        paper included in its network estimation (Section 2.9 of the README).
    """
    X = getattr(rec, signal)[:, state_frames(rec, which)]
    if nonzero_only and rec.nonzero_ROI is not None:
        X = X[rec.nonzero_ROI]
    return X


def list_recordings(include_example: bool = False) -> list[str]:
    """Recording names available under data/raw (sorted)."""
    names = sorted(p.stem for p in RAW_DIR.glob("*.mat"))
    if not include_example:
        names = [n for n in names if n != "example_data"]
    return names
