%% 3D雷达数据处理主脚本 — 基础任务完整流水线
%% DCA1000 + IWR6843 (60GHz)
%% 读取 → Range-FFT → Doppler-FFT → 零速通道置零 → CFAR → 角度FFT → 3D点云
clc; close all;

% 自动获取脚本所在目录，所有路径均以此为基准（无需手动修改）
script_dir = fileparts(mfilename('fullpath'));%获取脚本所在目录
BASE = fullfile(script_dir, '3D');
addpath(fullfile(script_dir, '3D雷达信号数据读取参考代码'));%添加搜索路径
addpath(script_dir);

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

%逐个场景遍历
for s = 1:1
    s=4;
    binfile = fullfile(BASE, scenarios{s}, '1.bin');
    fprintf('\n===== [场景 %d/4] %s =====\n', s, scenarios_cn{s});
    fprintf('  读取 %s ...\n', binfile);

    [adcData, para] = readRawData(binfile, para);
    n_frames = size(adcData, 4);
    fprintf('  数据维度: [%d %d %d %d]\n', size(adcData));

    static_clutter = mean(adcData, 4);
    adcData_clean = adcData - repmat(static_clutter, [1, 1, 1, size(adcData,4)]);

    %% ================================================================
    %%  第1阶段：全部计算（Range FFT → Doppler FFT → CFAR）
    %%  注意：此阶段不画任何图，避免figure渲染破坏复数矩阵内存
    %% ================================================================

    % 取第1帧，4个RX通道 [256, 4, 245]
    frame_data = squeeze(adcData_clean(:, :, :, 1));

    % ----- Range FFT：沿第1维，256个时域采样 → 256个距离bin -----
    range_win = hanning(256);
    range_win_3d = range_win .* ones(1, 4, 245);
    range_fft_data = fft(frame_data .* range_win_3d, [], 1);

    % ----- Doppler FFT：沿第3维，245个Chirp相位变化 → 速度 -----
    doppler_win = hanning(245);
    doppler_win_3d = ones(256, 4, 1) .* reshape(doppler_win, 1, 1, 245);
    rd_complex = fft(range_fft_data .* doppler_win_3d, [], 3);
    rd_complex = fftshift(rd_complex, 3);     % 零速居中

    % ----- 4通道功率求和 → dB -----
    rd_power = squeeze(sum(abs(rd_complex).^2, 2));  % [256, 245] 线性功率
    rd_map = 10 * log10(rd_power + eps);              % dB

    % 坐标轴
    range_axis = (0:255) * para.dr;
    doppler_axis = linspace(-v_max, v_max, 245);

    % ----- 二维 CFAR 检测 -----
    Tr = 2;  Gr = 1;   % Range方向 训练/保护单元
    Td = 2;  Gd = 1;   % Doppler方向 训练/保护单元
    offset_dB = 8;     % 阈值上调7dB
    cfar_result = rd_cfar_2d(rd_map, Tr, Td, Gr, Gd, offset_dB);

    [det_r, det_d] = find(cfar_result);
    n_det = length(det_r);
    fprintf('  CFAR检测到 %d 个目标点\n', n_det);

    %% ================================================================
    %%  第2阶段：画图
    %% ================================================================

    % ----- 图1：RD谱 -----
    figure;
    imagesc(doppler_axis, range_axis, rd_map);
    set(gca, 'YDir', 'normal');
    caxis([50, 130]);
    xlabel('速度 (m/s)');
    ylabel('距离 (m)');
    title('Range-Doppler 图');
    colorbar;
    colormap jet;

    % ----- 图2：RD谱 + CFAR检测叠加 -----
    figure;
    imagesc(doppler_axis, range_axis, rd_map);
    set(gca, 'YDir', 'normal');
    caxis([50, 130]);
    hold on;
    plot(doppler_axis(det_d), range_axis(det_r), 'rx', 'MarkerSize', 4, 'LineWidth', 1);
    hold off;
    xlabel('速度 (m/s)');
    ylabel('距离 (m)');
    title('RD谱 + CFAR目标检测（红色叉 = 检测点）');
    colorbar;
    colormap jet;


    % ----- 图3：单cfar检测 -----
    figure;
    imagesc(doppler_axis, range_axis, cfar_result);
    set(gca, 'YDir', 'normal');
    xlabel('速度 (m/s)');
    ylabel('距离 (m)');
    title('2D CFAR 检测结果');
    colormap gray;

end
