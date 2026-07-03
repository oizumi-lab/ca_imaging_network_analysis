function [f, each_eff] = characteristic_path_length_w(corrMat)
  %f:平均経路長 each_eff:各ノードのefficiency
  corrMat(corrMat < 0) = 0;
  corrMat = ones(size(corrMat)) ./ corrMat;
  dis = distances(graph(corrMat));
  N = length(dis);
  each_eff = ones(N, N) ./ dis;
  each_eff(find(each_eff == Inf)) = 0;
  each_eff = sum(each_eff) / (N-1);
  each_eff = each_eff';
  f = sum(sum(dis)) / (N * (N-1));
end
