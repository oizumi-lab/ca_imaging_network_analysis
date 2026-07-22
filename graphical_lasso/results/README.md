# Generated results

This directory receives full-neuron matrix caches, CSV/JSON summaries, and
figures from `../scripts/`. Large generated artifacts are intentionally ignored
by Git; the interpreted findings are preserved in `../documents/`.

`02_network_comparison/figures/sparsity_<recording>.png` and
`02_network_comparison/sparsity_summary.csv` are produced by the interactive
graph-sparsity display script.

The simplified fixed-density comparison produces
`simple_adjacency_<recording>_K0.0010.png`,
`simple_network_measures_<recording>_K0.0010.png`, and
`simple_fixed_density_measures.csv`. The Pearson-only K=5% reference adds
`pearson_reference_adjacency_<recording>_K0.0500.png`,
`pearson_reference_measures_<recording>_K0.0500.png`, and the combined contrast
table `simple_fixed_density_state_contrasts.csv`.
