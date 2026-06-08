# Sources

## Paper
- Kiyooka, D., Oomoto, I., et al. (2026). *Single-cell resolution functional
  networks during unconsciousness are segregated into spatially intermixed
  modules.* **Cell Reports.** https://doi.org/10.1016/j.celrep.2025.116902
- Preprint (bioRxiv, open): https://www.biorxiv.org/content/10.1101/2023.09.14.557838v2

## Dataset (v2.0, CC-BY 4.0)
- RIKEN neurodata portal: https://neurodata.riken.jp/id/20260409-001
  - DOI: 10.60178/cbs.20260409-001
- Zenodo mirror: https://doi.org/10.5281/zenodo.17667863
- Local copies of the dataset's own docs:
  - `dataset_Readme.md` — official data dictionary (variable definitions)
  - `dataset_Figure_guide.md` — maps manuscript figures → data files → code
  - `example_network_analysis.m` — the official minimal MATLAB example we port

## Reference code (MATLAB)
- https://github.com/oizumi-lab/mouse_network_2P — full analysis pipeline used in
  the paper. Written for dataset **v1.0**; see `.claude/rules/dataset-v2-format.md`
  for the v1→v2 differences our Python port handles.
- Brain Connectivity Toolbox (the `community_louvain` source): https://sites.google.com/site/bctnet/

## Related
- Dataset video overview: https://youtu.be/Rx8KDJF-d28
