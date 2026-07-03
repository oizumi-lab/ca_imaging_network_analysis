kernel_size = 15;
density = 0.01;
if strcmpi(sleep_or_ane, 'sleep')
    fr = 1500;
    num_of_mice = 6;
elseif strcmpi(sleep_or_ane, 'ane')
    fr = 2900;
    num_of_mice = 4;
end

for mouse_num = 1:num_of_mice
    load_sleep_or_ane_data;
    for s_i = 1:length(used_frame)
        sleep_frame = used_frame{s_i};
        if length(sleep_frame) >= fr
            fr_size = floor(length(sleep_frame) / fr);
            for j = 1:fr_size
                tic;
                ind = 1+fr*(j-1) : fr*j;                
                dataMat = smoothMat(:, sleep_frame(ind));
                corrMat = corrcoef(dataMat');
                corrMat(isnan(corrMat)) = 0;
                corrMat = corrMat - diag(diag(corrMat));
                N = size(corrMat, 1);
                [SWP{s_i}{mouse_num}(j,1), sw_ness{s_i}{mouse_num}(j,1), delta_C{s_i}{mouse_num}(j,1), delta_L{s_i}{mouse_num}(j,1)] = sw_summary(corrMat, density);
                time = toc;
                fprintf('Mouse %d, state %d, frame %d/%d done. Time: %.2f sec\n', mouse_num, s_i, j, fr_size, time);
            end
        end
    end
end

file_name = append('/mnt/NAS/user_data/daiki-kiyooka/figure/20240424_rebuttal_mat/small_world_', sleep_or_ane, '.mat');

save(file_name, 'SWP', 'sw_ness', 'delta_C', 'delta_L');