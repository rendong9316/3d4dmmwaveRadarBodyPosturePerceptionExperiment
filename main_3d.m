%% 3D雷达数据处理主脚本 — 基础任务完整流水线
%% DCA1000 + IWR6843 (60GHz)
%% 读取 → Range-FFT → Doppler-FFT → 零速通道置零 → CFAR → 角度FFT → 3D点云
clear; clc; close all;

% 自动获取脚本所在目录，所有路径均以此为基准（无需手动修改）
script_dir = fileparts(mfilename('fullpath'));
BASE = fullfile(script_dir, '3D');
addpath(fullfile(script_dir, '3D雷达信号数据读取参考代码'));

% ===== 1. 解析雷达参数 =====
logfile = fullfile(BASE, 'stand_0.8m', '1_LogFile.txt');
para = readPara(logfile);
v_max = para.lambda / 4 / para.Chirptime / para.GroupNum;

fprintf('========== 雷达参数 ==========\n');
fprintf('载频:            %.1f GHz\n', para.f0/1e9);
fprintf('ADC采样数:       %d\n', para.ADCSamples);
fprintf('采样率:          %.0f kHz\n', para.Fs/1e3);
fprintf('调频斜率:        %.2f MHz/us\n', para.FrequencySlope/1e12);
fprintf('每帧Chirp数:     %d (GroupNum=%d × ChirpNum=%d)\n', ...
        para.GroupNum * para.ChirpNum, para.GroupNum, para.ChirpNum);
fprintf('帧数:            %d\n', para.FrameNum);
fprintf('带宽:            %.1f MHz\n', para.BandWidth/1e6);
fprintf('距离分辨率:      %.2f cm\n', para.dr*100);
fprintf('最大距离:        %.1f m\n', para.Fs*3e8/2/para.FrequencySlope);
fprintf('最大速度:        %.2f m/s\n', v_max);
fprintf('==============================\n');

% ===== 2. 场景列表 =====
scenarios = {'white', 'stand_0.8m', 'swing_0.8m', 'jumpup_0.8m'};
scenarios_cn = {
    '噪声（无目标）',
    '静止站立 @0.8m',
    '摆动 @0.8m',
    '跳跃 @0.8m'
};

% ===== 3. CFAR参数 =====
guard_r = 4;   guard_d = 4;      % 保护单元
train_r = 8;   train_d = 8;      % 训练单元
thresh_factor = 12.0;            % 阈值因子（线性，≈10.8dB）
n_angle_fft = 128;               % 角度FFT补零点数

% 预分配存储cell
all_rd_mid       = cell(4, 1);   % 中间帧RD谱(dB)
all_det_mid      = cell(4, 1);   % 中间帧检测掩码
all_points_frames = cell(4, 1);  % 所有帧的点云（cell数组）

% ===== 4. 逐场景处理 =====
for s = 1:4
    binfile = fullfile(BASE, scenarios{s}, '1.bin');
    fprintf('\n===== [场景 %d/4] %s =====\n', s, scenarios_cn{s});
    fprintf('  读取 %s ...\n', binfile);

    [adcData, para] = readRawData(binfile, para);
    n_frames = size(adcData, 4);
    fprintf('  数据维度: [%d %d %d %d]\n', size(adcData));

    % 预分配当前场景
    points_cell = cell(n_frames, 1);
    det_mask_mid = [];
    rd_db_mid = [];

    t_start = tic;

    for f_idx = 1:n_frames
        % --- 步骤1: 取单帧 ---
        frame = squeeze(adcData(:, :, :, f_idx));  % [256, 4, 245]

        % --- 步骤2: 距离维FFT ---
        range_data = rangeFFT(frame);  % [256, 4, 245] complex

        % --- 步骤3: 多普勒维FFT ---
        rd_data = dopplerFFT(range_data);  % [256, 4, 245] complex (fftshifted)

        % --- 步骤4: 静态杂波抑制（零速通道置零）---
        % 在Doppler FFT之后，将零多普勒bin置零，去除静止目标
        rd_data = staticClutterSuppression(rd_data);

        % --- 步骤5: 非相干合并RX → 功率谱 ---
        % 4通道功率求和（非相干积累），用于CFAR检测
        rd_power = squeeze(sum(abs(rd_data).^2, 2));  % [256, 245] 线性功率
        rd_db = 10 * log10(rd_power + 1e-10);          % dB

        % --- 步骤6: 二维CA-CFAR检测 ---
        detections = cfar2D(rd_power, guard_r, guard_d, train_r, train_d, thresh_factor);

        % --- 步骤7: 角度FFT + 点云生成 ---
        pts = generatePointCloud(detections, rd_data, para, n_angle_fft);

        % 存储
        points_cell{f_idx} = pts;

        % 保留中间帧的RD谱和检测结果用于可视化
        if f_idx == round(n_frames / 2)
            rd_db_mid = rd_db;
            det_mask_mid = detections;
        end

        % 进度显示（每10帧或首末帧）
        if mod(f_idx, 10) == 0 || f_idx == 1 || f_idx == n_frames
            n_pts = size(pts, 1);
            fprintf('    帧 %3d/%d: 检测 %d 个点\n', f_idx, n_frames, n_pts);
        end
    end

    elapsed = toc(t_start);
    fprintf('  完成，耗时 %.1f 秒\n', elapsed);

    all_rd_mid{s} = rd_db_mid;
    all_det_mid{s} = det_mask_mid;
    all_points_frames{s} = points_cell;
end

% ===== 5. 可视化 =====
fprintf('\n===== 生成可视化图表 =====\n');
visualizePointCloud(all_rd_mid, all_points_frames, all_det_mid, scenarios_cn, para, v_max);

% ===== 6. 汇总统计 =====
fprintf('\n========== 汇总统计 ==========\n');
for s = 1:4
    pts_cell = all_points_frames{s};
    total_pts = 0;
    for f = 1:length(pts_cell)
        if ~isempty(pts_cell{f})
            total_pts = total_pts + size(pts_cell{f}, 1);
        end
    end
    avg_pts = total_pts / length(pts_cell);
    fprintf('%-20s  平均 %.1f 点/帧\n', [scenarios_cn{s} ':'], avg_pts);
end
fprintf('==============================\n');
fprintf('\n流水线完成！\n');
