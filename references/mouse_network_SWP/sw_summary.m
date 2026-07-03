function [SWP, sw_ness, delta_C, delta_L] = sw_summary(corrMat, density)
    
    binaryMat = densityBasedThresh(abs(corrMat), density);

    [bins, binsizes] = conncomp(graph(binaryMat));
    binsizesRes = max(binsizes);
    maxID = find(binsizes == max(binsizes));
    connectedID = find(bins == maxID)';

    [SWP, sw_ness, delta_C, delta_L, each_eff, C_coef, reg_path, rand_path, net_path, rand_clus] = small_world_propensity(binaryMat(connectedID, connectedID));