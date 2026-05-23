%% RTDTtest_rd.m
%% RT（距离-时间）谱 + DT（多普勒-时间）谱批量演示
%% 遍历 DATA/ 下所有场景，各取第30个 .bin，生成 RT 图和 DT 图
clc; close all;

%% =====================================================
%% 路径配置
%% =====================================================
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '3D雷达信号数据读取参考代码'));
DATA_DIR = fullfile(script_dir, 'DATA');
FILE_IDX = 30;

%% =====================================================
%% 场景定义
%% =====================================================
scenarios = {
    'bend_0.8m',    'bend_3m', ...
    'diedao_0.8m',  'qianshuai_3m', ...
    'jumpup_0.8m',  'jumpup_3m', ...
    'run_0.8m',     'run_3m', ...
    'sit_0.8m',     'sit_3m', ...
    'stand_0.8m',   'stand_3m', ...
    'swing_0.8m',   'swing_3m', ...
    'walk',         'white'
};
scenarios_cn = {
    '弯腰0.8m','弯腰3m', ...
    '跌倒0.8m','前摔3m', ...
    '跳跃0.8m','跳跃3m', ...
    '跑步0.8m','跑步3m', ...
    '静坐0.8m','静坐3m', ...
    '站立0.8m','站立3m', ...
    '摆臂0.8m','摆臂3m', ...
    '走路','空房间'
};

n_scene = length(scenarios);
n_cols  = 4;
n_rows  = ceil(n_scene / n_cols);

% ----- 预分配存储 -----
rt_maps  = cell(n_scene, 1);   % 每个场景的 RT 谱 [n_range, n_frames]
dt_maps  = cell(n_scene, 1);   % 每个场景的 DT 谱 [n_chirps, n_frames]
all_para = cell(n_scene, 1);   % 雷达参数
valid    = false(n_scene, 1);  % 该场景是否有效

%% =====================================================
%% 主循环：遍历每个场景，生成 RT 和 DT 谱
%% =====================================================
for s = 1:n_scene
    scene_path = fullfile(DATA_DIR, scenarios{s});

    % ----- 找 LogFile -----
    logfiles = dir(fullfile(scene_path, '*_LogFile.txt'));
    if isempty(logfiles)
        fprintf('[%2d] %-16s 无 LogFile\n', s, scenarios{s});
        continue;
    end
    logfile = fullfile(scene_path, logfiles(1).name);
    para = readPara(logfile);

    % ----- 找第 FILE_IDX 个 bin -----
    bins = dir(fullfile(scene_path, '*.bin'));
    bin_names = {};
    for k = 1:length(bins)
        [~, name, ~] = fileparts(bins(k).name);
        if ~isempty(regexp(name, '^\d+$', 'once'))
            bin_names{end+1} = bins(k).name;
        end
    end
    bin_names = sort_nat(bin_names);
    if length(bin_names) < FILE_IDX
        fprintf('[%2d] %-16s bin 数量不足\n', s, scenarios{s});
        continue;
    end
    bin_file = fullfile(scene_path, bin_names{FILE_IDX});
    fprintf('[%2d] %-16s 处理中...\n', s, scenarios{s});

    % ----- 读取原始数据 -----
    [adcData, para] = readRawData(bin_file, para);
    n_range  = para.ADCSamples;
    n_chirps = para.ChirpNum;
    n_frames = size(adcData, 4);

    % ----- 距离分辨率 & 目标距离bin -----
    if contains(scenarios{s}, '3m')
        target_dist = 3.0;
    else
        target_dist = 0.8;
    end
    target_bin = round(target_dist / para.dr);

    range_win   = hanning(n_range);
    doppler_win = hanning(n_chirps);

    % ===== 第一阶段：最强距离bin搜索（和 micdopplertest_rd 一致）=====
    energy_acc = zeros(n_range, 1);
    for f = 1:n_frames
        frame = squeeze(adcData(:, :, :, f));
        sig = squeeze(frame(:, 1, :));            % RX0 [n_range, n_chirps]
        sig = sig - mean(sig, 2);                  % MTI 静态杂波抑制
        range_fft = fft(sig .* range_win, [], 1);
        energy_acc = energy_acc + mean(abs(range_fft).^2, 2);
    end
    search_range = max(1, target_bin-6) : min(n_range, target_bin+6);
    [~, idx] = max(energy_acc(search_range));
    best_bin = search_range(idx);
    fprintf('    strongest bin = %d\n', best_bin);

    % ===== 第二阶段：逐帧生成 RT 列 和 DT 列 =====
    rt_map = zeros(n_range, n_frames);    % [距离bin, 帧]
    dt_map = zeros(n_chirps, n_frames);   % [多普勒bin, 帧]
    bin_span = -3:3;

    for f = 1:n_frames
        frame = squeeze(adcData(:, :, :, f));
        sig = squeeze(frame(:, 1, :));            % RX0
        sig = sig - mean(sig, 2);                  % MTI

        % --- 距离 FFT ---
        range_fft = fft(sig .* range_win, [], 1);  % [n_range, n_chirps]

        % --- RT 列：取距离剖面（所有距离bin的平均功率） ---
        rt_map(:, f) = mean(abs(range_fft).^2, 2);

        % --- 多bin融合提取慢时间 ---
        bins_sel = best_bin + bin_span;
        bins_sel = bins_sel(bins_sel >= 1 & bins_sel <= n_range);
        slow_sig = sum(range_fft(bins_sel, :), 1).';  % [n_chirps, 1]

        % 轻量尖峰抑制
        amp = abs(slow_sig);
        th = median(amp) * 5;
        slow_sig(amp > th) = 0;

        % --- DT 列：多普勒 FFT ---
        doppler_fft_data = fft(slow_sig .* doppler_win);
        doppler_fft_data = fftshift(doppler_fft_data);  % 零速居中
        dt_map(:, f) = abs(doppler_fft_data).^2;
    end

    % 转 dB
    rt_map = 10 * log10(rt_map + 1e-10);
    dt_map = 10 * log10(dt_map + 1e-10);

    rt_maps{s} = rt_map;
    dt_maps{s} = dt_map;
    all_para{s} = para;
    valid(s) = true;
end

%% =====================================================
%% 图1：RT 谱（距离-时间）— 4×4 子图
%% =====================================================
figure('Name', 'RT 距离-时间谱图', 'Position', [30, 30, 1600, 900]);

for s = 1:n_scene
    if ~valid(s), continue; end
    para = all_para{s};
    rt_map = rt_maps{s};
    n_frames = size(rt_map, 2);

    range_axis = (0:n_range-1) * para.dr;
    time_axis  = (0:n_frames-1) * para.Frameinter;

    subplot(n_rows, n_cols, s);
    imagesc(time_axis, range_axis, rt_map);
    set(gca, 'YDir', 'normal');
    xlabel('时间 (s)');
    ylabel('距离 (m)');
    title(scenarios_cn{s}, 'FontSize', 9);
    colormap(gca, 'jet');
    xlim([time_axis(1), time_axis(end)]);
    ylim([0, 5]);  % 只显示 0~5m
end
sgtitle('RT 距离-时间谱图 — 所有场景', 'FontSize', 14, 'FontWeight', 'bold');

%% =====================================================
%% 图2：DT 谱（多普勒-时间）— 4×4 子图
%% =====================================================
figure('Name', 'DT 多普勒-时间谱图', 'Position', [60, 60, 1600, 900]);

for s = 1:n_scene
    if ~valid(s), continue; end
    para = all_para{s};
    dt_map = dt_maps{s};
    n_frames = size(dt_map, 2);

    v_max  = para.lambda / (4 * para.Chirptime * para.GroupNum);
    vel_axis = linspace(-v_max, v_max, n_chirps);
    time_axis = (0:n_frames-1) * para.Frameinter;

    subplot(n_rows, n_cols, s);
    imagesc(time_axis, vel_axis, dt_map);
    set(gca, 'YDir', 'normal');
    xlabel('时间 (s)');
    ylabel('速度 (m/s)');
    title(scenarios_cn{s}, 'FontSize', 9);
    colormap(gca, 'jet');
    xlim([time_axis(1), time_axis(end)]);
    ylim([-v_max, v_max]);
end
sgtitle('DT 多普勒-时间谱图 — 所有场景', 'FontSize', 14, 'FontWeight', 'bold');

fprintf('\n全部完成！\n');

%% =====================================================
%% 辅助函数：自然排序
%% =====================================================
function sorted = sort_nat(cell_arr)
    nums = zeros(length(cell_arr), 1);
    for k = 1:length(cell_arr)
        [~, name, ~] = fileparts(cell_arr{k});
        nums(k) = str2double(name);
    end
    [~, idx] = sort(nums);
    sorted = cell_arr(idx);
end
