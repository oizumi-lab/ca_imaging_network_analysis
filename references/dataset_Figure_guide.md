# FIGURE GUIDE

## Guide to Figures: Data, Analysis, and Code

This document describes the relationship between the figures in the manuscript and the corresponding data, analysis methods, and code resources.

## Overview

- All source data are available in this repository.
- Analysis scripts are available in the GitHub repository:  
  https://github.com/oizumi-lab/mouse_network_2P

For each figure, the following information is provided:

- Data files
- Analysis methods
- Corresponding code (if applicable)

## Important note on code usage

The analysis scripts available on GitHub were originally developed using the dataset deposited in Zenodo (version 1).

As a result, these scripts may not be directly compatible with the processed data files provided in this repository (e.g., `mouse01_sleep.mat`), due to differences in variable names and data structure.

Users who wish to run the GitHub scripts with the current dataset may need to modify variable names or adapt the data structure accordingly.

## Figure 1

**Description**  
Schematic illustration of the concept of modularity and two types of neurons.

**Data**  
Not applicable

**Analysis**  
Not applicable

**Code**  
Not applicable

## Figure 2

**Description**  
Experimental overview of wide-field two-photon calcium imaging.

**Data**  
`mouse05_sleep.mat`

**Analysis**  
Not applicable

**Code**  
Not applicable

## Figure 3

**Description**  
Estimation of functional networks and modularity.

**Data**  

`mouse01_sleep.mat`  
`mouse02_sleep.mat`  
`mouse03_sleep.mat`  
`mouse04_day1_sleep.mat`  
`mouse04_day2_sleep.mat`  
`mouse05_sleep.mat`  
`mouse03_ane.mat`  
`mouse05_ane.mat`  
`mouse06_ane.mat`  
`mouse07_ane.mat`

**Analysis**  

- Calculation of correlation matrices  
- Calculation of modularity  

**Code on GitHub**  

- fig3_ab_network_est_schema.m  
- fig3_cdef_compare_modularity.m

## Figure 4

**Description**  
Contribution of individual neurons to modularity.

**Data**  
Same as Figure 3 (derived from the results of Figure 3).

**Analysis**  

- Calculation of each neuron's contribution to modularity  

**Code on GitHub**  

- fig4_ab_Qi_hist.m  
- fig4_cdef_Qi_degree.m  

## Figure 5

**Description**  
Spatial distribution of functional network modules at single-cell resolution.

**Data**  
Same as Figure 3 (derived from the results of Figure 3).

**Analysis**  

- Calculation of spatial distributions  

**Code on GitHub**  

- fig5_abc_module_dist.m  
- fig5_def_module_composition_corr.m  
- fig5_gh_and_fig7_gh_dist_and_same_module_pair.m  

## Figure 6

**Description**  
Temporal stability of modules.

**Data**  
Same as Figure 3 (derived from the results of Figure 3).

**Analysis**  

- Calculation of module stability  

**Code on GitHub**  

- fig6_module_stability.m  

## Figure 7

**Description**  
Modularity and spatial distribution of modules across different spatial scales (coarse-graining).

**Data**  
Same as Figure 3 (derived from the results of Figure 3).

**Analysis**  

- Coarse-graining of neural activity  
- Calculation of modularity and module structure  

**Code on GitHub**  

- fig7_bcde_Q_diff_lv_of_coarse.m  
- fig7_f_coarse_module_dist_example.m  
- fig7_i_dist_corr_null_plot.m  
- fig7_j_scatter_density.m  
- fig7_kl_corr_dist_and_corr.m  