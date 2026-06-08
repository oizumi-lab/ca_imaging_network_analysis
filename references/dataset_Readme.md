# Two-photon calcium imaging dataset of cortical neuronal populations across wakefulness, sleep, and anesthesia

Large-scale two-photon calcium imaging dataset of ~4,000–10,000 neurons per recording in mouse cortex across wakefulness, sleep, and isoflurane anesthesia.  

## Dataset Metadata

**Version:** v2.0 (analysis dataset)  
**Release date:** 2026-03  
**License:** Creative Commons Attribution 4.0 International (CC-BY 4.0)  

---

## 1. Overview

This dataset contains deconvolved calcium activity time series recorded using two-photon microscopy and processed for functional network analysis across brain states (wakefulness/sleep or wakefulness/anesthesia using isoflurane).

The present release (v2.0) contains the processed neural activity data used in the associated publication, **including data from all mice used in the study**.  
In contrast, the previous release (v1.0) contained data from only a single animal and was intended as a preliminary subset of the dataset.

Only ROIs that passed quality-control criteria and were included in the analyses are provided in this release.

A partial subset of the dataset was previously released on Zenodo.  
The present version supersedes that preliminary release and contains the complete dataset used in the published analyses.

This dataset corresponds to the published study:

Kiyooka and Oomoto et al.,  
*Single-cell resolution functional networks during unconsciousness are segregated into spatially intermixed modules.*  
Cell Reports (2026).  
https://doi.org/10.1016/j.celrep.2025.116902  

A minimal MATLAB example for loading and accessing the dataset is provided in Section 2.

---

### Experimental Pipeline

1. Two-photon calcium imaging  
2. ROI detection  
3. Extraction of fluorescence traces (ΔF/F)  
4. Deconvolution (OASIS)  
5. Gaussian smoothing  
6. Functional network analysis  

The shared dataset includes **ΔF/F traces (step 3), deconvolved spike estimates (step 4), and smoothed spike estimates (step 5)** used for network analysis, along with anatomical and brain-state annotations.
 

All analyses in the associated publication were performed using `spike_smoothed` unless otherwise stated.  

For reproducibility of the published network analyses, the dataset also includes an **optional activity-based neuron filter (`nonzero_ROI`)**, which identifies neurons that were included in the network estimation procedure.  
This variable is specific to the analysis window used in the study and is **not required for general reuse of the dataset**, but can be applied when reproducing the analyses reported in the paper (see Section 2.9).

---

### Typical Dataset Scale

Typical recordings contain:

- Number of animals: 5 in the sleep dataset and 4 in the anesthesia dataset
- Number of neurons per recording (after quality control): ~4,000–10,000  
- Imaging depth: Layer 2/3 of the cortex (~100–150 µm depth)  
- Recording duration: ~30–100 minutes  
- Sampling rate: 7.65 Hz  

---

### Animal & Sensor Information

- Mouse strain: C57BL/6JJmsSlc (B6)  
- Calcium indicator: G-CaMP7.09  
- All animals were recorded using the same strain and sensor.

---

### Example Use Cases

This dataset can be used for:

- analysis of single-cell functional networks across brain states
- investigation of neural population dynamics during sleep or anesthesia
- spatial organization of functional modules in the cortex
- benchmarking methods for neural network inference from calcium imaging data

---

## 2. Data Structure

### File Format

Data are stored as MATLAB `.mat` files.

Each file contains a structured dataset including neural activity matrices, ROI information, and metadata used in the associated analyses.  

The dataset can be easily used in Python via `scipy.io.loadmat`, which loads MATLAB `.mat` files as NumPy arrays.

---

### Loading Example (MATLAB)

```matlab
data = load('mouse01_sleep.mat');
```

All variables described in this document will appear in the loaded structure.

Example:

```matlab
spike = data.spike_smoothed;
state = data.state;
```

---

### Loading Example (Python)

Python users can load .mat files using scipy.io.

```python
from scipy.io import loadmat

data = loadmat('mouse01_sleep.mat')

spike = data['spike_smoothed']
state = data['state']
```

Note that MATLAB structures may appear as nested dictionaries or arrays depending on the MATLAB file format.

---

### 2.1 Recording Conditions

#### `data_info`

Indicates recording condition:

- `'sleep'` : natural sleep experiment  
- `'ane'` : anesthesia experiment  

---
### 2.2 Time-Series Data

All time-series matrices have shape:

```
N_neurons x T_timepoints
```

#### `dFF`

- Raw fluorescence trace (ΔF/F)  
- Unit: relative fluorescence change  
- Shape: `N x T`  

#### `spike_deconv`

- Deconvolved spike estimate  
- Method: OASIS  
- Shape: `N x T`  

#### `spike_smoothed`

- Gaussian-filtered version of `spike_deconv`  
- Used for all network analyses  
- Shape: `N x T`  

Processing details:

Deconvolved spike traces were smoothed using a Gaussian filter with a temporal window length of **1.96 s (15 frames at 7.65 Hz)**.

---

### 2.3 Brain State Vector

#### `state`

Shape:

```
1 x T
```

If `data_info == 'sleep'`:

- `0` : awake  
- `0.5` : quiet awake  
- `1` : non-REM sleep  
- `2` : REM sleep  

If `data_info == 'ane'`:

- `0` : awake  
- `1` : anesthesia  

---

### 2.4 ROI Information

#### `ROIs.atlas`

Shape:

```
N_neurons x 1
```

- Brain region acronym (e.g., `'VISa2/3'`)  
- Atlas registration performed manually  

---

### 2.5 ROI Centroid Information

#### `ROIs.Centroid`

Shape:

```
N_neurons x 2
```

- Column 1: X coordinate (pixels)  
- Column 2: Y coordinate (pixels)  

Imaging specifications:

- Field of view: 3 mm x 3 mm  
- Resolution: 2048 x 2048 pixels  
- Pixel size: 1.465 µm/pixel  
- Sampling rate: 7.65 Hz  

---

### 2.6 Frame Discontinuity Information

#### `frame.boundary_ind`

Vector indicating frame indices where acquisition is discontinuous.

Recordings exceeding ~9500 frames were segmented due to microscope constraints.

Example:

```
[9500, 19000]
```

Users should treat segments separately for analyses requiring temporal continuity.

---

### 2.7 Frames Used for State-Specific Analyses

#### `frame.used_frame`

Cell array specifying frame indices used for state-specific analyses.

Shape:

```
1 x 2 cell
```

State assignment depends on `data_info`.

If `data_info == 'sleep'`:

- `{1,1}` : Awake frames  
- `{1,2}` : NREM sleep frames  

If `data_info == 'ane'`:

- `{1,1}` : Awake frames  
- `{1,2}` : Anesthesia frames  

Each cell contains a vector of frame indices:

```
1 x N_frames double
```

Example:

```
frame.used_frame{1,1} → 1 x 7000 double (awake frames)
frame.used_frame{1,2} → 1 x 8375 double (NREM frames)
```

Frames were selected based on the persistence of the brain state.

Only frames belonging to periods in which the same state persisted for **≥39 s (300 frames)** were included in the analysis.  
This criterion was applied to avoid ambiguity in state classification.

The vectors in `frame.used_frame` contain all frame indices belonging to these valid state segments.

These indices can be used to extract state-specific activity from the time-series matrices (e.g., `spike_deconv` or `spike_smoothed`).  

Note: Use curly braces `{}` to access frame indices stored in the cell array.
For example:

```
spike_deconv(:, frame.used_frame{1,1})
```
---

### 2.8 Animal Metadata

#### `animal_info`

Structure:

animal_info.age : age in weeks  
animal_info.sex : sex (M/F)  

---

### 2.9 Neuron Activity Filter Used in Network Analysis (optional)

#### `nonzero_ROI`

Shape:

```
N_neurons x 1
```

Binary flag indicating whether each neuron satisfied an additional activity criterion used in the network analysis.

- `1` : neuron included in the network analysis  
- `0` : neuron excluded due to lack of activity within analysis windows  

In the network analyses reported in the associated publication, functional connectivity was estimated using time windows of **1500 frames** (~196s).  
Neurons that showed **no detected activity within a given analysis window** can bias or destabilize network estimation. Therefore, neurons that were silent within analysis windows were excluded from the network analysis.

The `nonzero_ROI` flag indicates neurons that passed this additional activity-based criterion.

Importantly, this criterion depends on the **analysis window length used in the study** and is therefore specific to the network analysis performed in the associated publication.  
For this reason, the dataset itself contains **all neurons that passed the primary quality-control criteria**, regardless of whether they were excluded by this activity filter.

Users interested in reproducing the network analyses reported in the paper may apply this filter when extracting activity traces.

Example:

```
spike_deconv(nonzero_ROI==1, frame.used_frame{1,1});
```


This operation restricts the analysis to neurons that were included in the published network analyses.

---

## 3. Neuron Order Consistency (Critical)

Row alignment is strictly preserved across:

- `dFF`  
- `spike_deconv`  
- `spike_smoothed`  
- `ROIs.atlas`  
- `ROIs.Centroid`  
- `nonzero_ROI`  

Row `i` corresponds to the same neuron across all matrices.

Maintaining this alignment is essential for valid analysis.

---

## 4. Related Resources

### Dataset DOI (Zenodo)

https://doi.org/10.5281/zenodo.17667863  

### Network Analysis Code (GitHub)

https://github.com/oizumi-lab/mouse_network_2P  

This repository contains the full analysis pipeline used in the associated publication.

Note that the analysis code was originally developed for the earlier dataset structure (v1.0).  
The current dataset release (v2.0) has been reorganized and simplified to improve usability for general data reuse, and therefore the original code may require minor modifications to run directly with the v2.0 dataset.

Researchers interested in reproducing the exact analysis pipeline described in the paper may refer to the v1 dataset structure, while v2.0 is recommended for new analyses and general reuse.

---

## 5. Citation

If you use this dataset in your research, please cite both the associated publication and the dataset DOI.

Kiyooka, D., Oomoto, I., et al. (2026).  
*Single-cell resolution functional networks during unconsciousness are segregated into spatially intermixed modules.*  
Cell Reports.  
https://doi.org/10.1016/j.celrep.2025.116902  

The dataset is available via the following repositories:

Primary dataset DOI (Zenodo):  
https://doi.org/10.5281/zenodo.17667863  

Institutional mirror (RIKEN CBS Data Repository):  
[RIKEN DOI here]

Either dataset DOI may be cited.  

---

## 6. License

This dataset is distributed under the  
**Creative Commons Attribution 4.0 International (CC-BY 4.0)**.

Users are free to share and adapt the material for any purpose, including commercial use, provided appropriate credit is given.

---

## 7. Contact

For questions regarding the dataset, please contact:

Masanori Murayama  
RIKEN Center for Brain Science  
Email: masanori.murayama@riken.jp

Ikumi Oomoto  
RIKEN Center for Brain Science  
Email: ikumi.oomoto@riken.jp