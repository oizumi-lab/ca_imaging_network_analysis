# Dataset v2.0 format & loading (persistent reference)

RIKEN neurodata **20260409-001**, version **2.0** — "Single-cell calcium imaging
dataset of large-scale neuronal activity across wakefulness, sleep, and
anesthesia" (Kiyooka & Oomoto et al., Cell Reports 2026).

## Files (data/raw/)

11 `.mat` files (~11 GB total) + the small example:
- `example_data.mat` — 1,000-neuron **subsample** (~84 MB), for fast dev/testing.
- Sleep (awake vs NREM): `mouse01_sleep`, `mouse02_sleep`, `mouse03_sleep`,
  `mouse04_day1_sleep`, `mouse04_day2_sleep`, `mouse05_sleep`.
- Anesthesia (awake vs isoflurane): `mouse03_ane`, `mouse05_ane`, `mouse06_ane`,
  `mouse07_ane`.

Recording specs: ~4,000–10,000 neurons (after QC), 7.65 Hz, layer 2/3,
G-CaMP7.09, 3 mm × 3 mm FOV (2048 px → **1.465 µm/px**), ~30–100 min.

## CRITICAL: file format is MATLAB v7.3 (HDF5)

The `.mat` files are HDF5 (`MATLAB 7.3 MAT-file ... HDF5 schema`). **`scipy.io.loadmat`
cannot read them** even though the dataset README claims it can. Use
`pymatreader.read_mat` (handles v7.3, transposes arrays to MATLAB `N×T`
orientation, structs→dicts, cells→lists). All loading is centralised in
`scripts/lib/dataio.py` — use `dataio.load_recording(name)`.

## Variable mapping (v1 MATLAB repo → v2.0 dataset)

| Concept | v1 (`mouse_network_2P`) | v2.0 | dataio.Recording field |
|---|---|---|---|
| Smoothed spikes (for networks) | `smoothed_spike` | `spike_smoothed` | `.spike_smoothed` |
| Deconvolved spikes | `spike` | `spike_deconv` | `.spike_deconv` |
| ΔF/F | `dFF` | `dFF` | `.dFF` |
| Region acronym / neuron | `atlasID` + `st` | `ROIs.atlas` (string class) | `.atlas` (None — see below) |
| Coordinates | `x_coord`, `y_coord` | `ROIs.Centroid` (N×2 px) | `.centroid` / `.centroid_um` |
| State frames | `used_frame{1/2}` | `frame.used_frame{1,1}`/`{1,2}` | `.used_frame[label]` |
| Frame breaks | `boundary_ind` | `frame.boundary_ind` | `.boundary_ind` |
| Brain state per frame | `SleepState`/`sleep_state` | `state` (1×T) | `.state` |
| Experiment type | (file name) | `data_info` ('sleep'/'ane') | `.data_info` |
| Activity filter | (n/a) | `nonzero_ROI` (N×1) | `.nonzero_ROI` (bool) |
| Animal metadata | (n/a) | `animal_info.{age,sex}` | `.animal_info` |

State codes — **sleep**: 0 awake, 0.5 quiet-awake, 1 NREM, 2 REM.
**ane**: 0 awake, 1 anesthesia.

## Gotchas (handled by dataio.py — keep them in mind)

1. **1-based → 0-based**: `frame.used_frame` and `frame.boundary_ind` are MATLAB
   1-based indices; the loader subtracts 1. `used_frame[awake]` etc. are ready to
   index NumPy arrays directly.
2. **Row alignment**: row *i* is the same neuron across `dFF`, `spike_deconv`,
   `spike_smoothed`, `centroid`, `nonzero_ROI`. `dataio` asserts this.
3. **`ROIs.atlas` = None**: it is a MATLAB *string-class* array stored as an
   opaque MCOS object inside `#subsystem#`; neither pymatreader nor h5py decode
   it. Region labels are therefore unavailable in Python for now. **Not needed**
   for single-cell modularity. If region/mesoscale analysis needs them later,
   options are: re-export from MATLAB as `cellstr`/char, or implement an MCOS
   string decoder.
4. **`nonzero_ROI`** marks neurons active within the paper's 1500-frame analysis
   windows. Apply it (`activity(..., nonzero_only=True)`) when reproducing the
   paper; the dataset otherwise keeps all QC-passed neurons.

## Reference modularity pipeline (paper / `example_network_analysis.m`)

`spike_smoothed → corr() → zero diag → density_threshold(K=0.05) →
community_louvain(gamma=1) → Q`. Full study: Louvain ×200, take max-Q, consensus
clustering, across densities 0.008–0.3 and gamma {0.5,1,1.5,2}. Ported in
`scripts/lib/network.py`.
