%% ==========================================================
% EXAMPLE NETWORK ANALYSIS
% correlation -> density threshold -> modularity detection -> cortical
% spatial module map
% ===================================================================

clear
clc

fprintf('\n=====================================\n')
fprintf(' Example Network Analysis\n')
fprintf('=====================================\n\n')

%% ----------------------------------------------------------
% TOOLBOX CHECK
%% ----------------------------------------------------------

fprintf('Checking Brain Connectivity Toolbox...\n')

if exist('community_louvain','file') ~= 2

    fprintf('\nBrain Connectivity Toolbox was NOT found.\n\n')

    fprintf('Download it from:\n')
    fprintf('https://sites.google.com/site/bctnet/\n\n')

    fprintf('Then add it to MATLAB path using:\n')
    fprintf('addpath(genpath(''brain-connectivity-toolbox''))\n\n')

    error('community_louvain.m is required.')

end

fprintf('Brain Connectivity Toolbox detected.\n\n')

%% ----------------------------------------------------------
% LOAD DATA
%% ----------------------------------------------------------

fprintf('Loading example_data.mat...\n')

data = load('example_data.mat','spike_smoothed','ROIs');

rng(1)   % Fix random seed

spike  = data.spike_smoothed;
coords = data.ROIs.Centroid;

[N,T] = size(spike);

fprintf('Neurons: %d  Frames: %d\n\n',N,T)

%% ----------------------------------------------------------
% CORRELATION MATRIX
%% ----------------------------------------------------------

fprintf('Computing correlation matrix...\n')

corrMat = corr(spike');

corrMat(1:N+1:end) = 0;

%% ----------------------------------------------------------
% DENSITY THRESHOLD
%% ----------------------------------------------------------

K = 0.05;
fprintf('Density threshold K = %.2f\n',K)

option.weighted = 0;
option.negative = 0;

[binaryMat,~] = densityBasedThresh(corrMat,K,option);

%% ----------------------------------------------------------
% MODULARITY
%% ----------------------------------------------------------

gamma = 1;

[Ci,Q] = community_louvain(binaryMat,gamma);

fprintf('Modularity Q = %.3f\n\n',Q)

%% ----------------------------------------------------------
% MODULE SORTED ADJACENCY
%% ----------------------------------------------------------

[~,sort_idx] = sort(Ci);

sorted_adj = binaryMat(sort_idx,sort_idx);

figure
imagesc(sorted_adj)

axis square
colormap(flipud(gray))

title(sprintf('Module-sorted adjacency matrix (Q = %.3f)',Q))

%% ----------------------------------------------------------
% CORTICAL SPATIAL MODULE MAP
%% ----------------------------------------------------------

nModules = numel(unique(Ci));

figure

scatter(coords(:,1),coords(:,2),35,Ci,'filled')

axis equal
axis ij

xlabel('X')
ylabel('Y')

title(sprintf('Cortical spatial module map (N = %d modules)', nModules))

colormap(lines)

fprintf('\nExample analysis finished successfully.\n\n')