sleep_or_ane = 'sleep';

if strcmpi(sleep_or_ane, 'sleep')
  mouse_list = {'1', '2', '3', '4', '5', 'Average'};
  special_case = 1;
elseif strcmpi(sleep_or_ane, 'ane')
  mouse_list = {'1', '2', '3', '4', 'Average'};
  special_case = 0;
end

file_name = append('/Users/daiki/Desktop/figure/20240424_rebuttal_mat/small_world_', sleep_or_ane, '.mat');
load(file_name);

measure_all  = {delta_C, delta_L, sw_ness, SWP};
measure_list = {'Clustering', 'Path length', 'Small-world-ness', 'Small-world propensity'};
ylim_list = {[1 3.5], [0 0.035], [14 40], [0.45 0.8]};


x_a6 = get_x_sleep(SWP, 1, special_case);
x_n6 = get_x_sleep(SWP, 2, special_case);

for measure_i = 1% :length(measure_all)
  aw_all = cell2mat(measure_all{measure_i}{1}');
  nr_all = cell2mat(measure_all{measure_i}{2}');
  if strcmpi(measure_list{measure_i}, 'Clustering')
    aw_all = 1 ./ aw_all;
    nr_all = 1 ./ nr_all;
  end

  [aw_mean, nr_mean] = plot_state_comparison(sleep_or_ane, aw_all, nr_all, x_a6, x_n6);
  xticklabels(mouse_list);
  ylabel(measure_list{measure_i});
  ylim(ylim_list{measure_i});
  saved_file_name = ['/Users/daiki/Desktop/figure_mouse_network/', sleep_or_ane, '_', measure_list{measure_i}, '.svg'];
  box off;
  saveas(gcf, saved_file_name);
  close all;
end