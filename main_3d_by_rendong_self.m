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
    s=2;
    binfile = fullfile(BASE, scenarios{s}, '1.bin');
    fprintf('\n===== [场景 %d/4] %s =====\n', s, scenarios_cn{s});
    fprintf('  读取 %s ...\n', binfile);

    [adcData, para] = readRawData(binfile, para);
    n_frames = size(adcData, 4);
    fprintf('  数据维度: [%d %d %d %d]\n', size(adcData));

    static_clutter = mean(adcData, 4);
    adcData_clean = adcData - repmat(static_clutter, [1, 1, 1, size(adcData,4)]);

    %% ================================================================
    %%  第1阶段：全部计算（Range FFT → Doppler FFT → CFAR → 角度FFT → 点云）
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

    % ----- 二维 CA-CFAR 目标检测 -----
    guard_r = 4;   guard_d = 4;
    train_r = 8;   train_d = 8;
    thresh_factor = 12.0;

    kernel_r = 2*train_r + 2*guard_r + 1;
    kernel_d = 2*train_d + 2*guard_d + 1;
    kernel = ones(kernel_r, kernel_d);
    kernel(train_r+1 : train_r+2*guard_r+1, train_d+1 : train_d+2*guard_d+1) = 0;
    n_train = sum(kernel(:));

    noise_sum = conv2(rd_power, kernel, 'same');
    noise_mean = noise_sum / n_train;
    threshold = noise_mean * thresh_factor;
    detections = rd_power > threshold;

    edge = train_r + guard_r;
    detections(1:edge, :) = false;
    detections(end-edge+1:end, :) = false;
    detections(:, 1:edge) = false;
    detections(:, end-edge+1:end) = false;

    [det_r, det_d] = find(detections);
    n_det = length(det_r);
    fprintf('  CFAR检测到 %d 个目标点\n', n_det);

    % ----- 角度维 FFT：对每个CFAR检测点提取方位角 -----
    n_angle_fft = 128;
    azimuth_all = zeros(n_det, 1);
    for i = 1:n_det
        rx_vec = double(reshape(rd_complex(det_r(i), :, det_d(i)), 4, 1));
        rx_padded = zeros(n_angle_fft, 1);
        rx_padded(1:4) = rx_vec;
        angle_spec = fftshift(abs(fft(rx_padded)));
        [~, peak] = max(angle_spec);
        sin_theta = (peak - n_angle_fft/2 - 1) / (n_angle_fft/2);
        azimuth_all(i) = asind(max(-1, min(1, sin_theta)));
    end

    % ----- RA谱：每个距离bin最强多普勒bin处做角度FFT -----
    ra_map = zeros(256, n_angle_fft);
    for r_bin = 1:256
        [~, d_peak] = max(rd_power(r_bin, :));
        rx_vec = double(reshape(rd_complex(r_bin, :, d_peak), 4, 1));
        rx_padded = zeros(n_angle_fft, 1);
        rx_padded(1:4) = rx_vec;
        ra_map(r_bin, :) = fftshift(abs(fft(rx_padded))).';
    end
    ra_map_db = 10 * log10(ra_map + eps);

    % 角度轴
    angle_axis = asind(linspace(-1, 1, n_angle_fft));

    % ----- 3D点云生成：极坐标 → 笛卡尔坐标 -----
    points = zeros(n_det, 3);
    for i = 1:n_det
        r_m = range_axis(det_r(i));
        v_ms = doppler_axis(det_d(i));
        az_deg = azimuth_all(i);
        points(i, :) = [r_m * sind(az_deg), r_m * cosd(az_deg), v_ms];
    end

    fprintf('  计算完成：%d个检测点 → %d个点云点\n', n_det, size(points,1));

    %% ================================================================
    %%  第2阶段：全部画图（此时不再访问任何复数矩阵，避免内存冲突）
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

    % ----- 图3：RA谱（距离-方位角）-----
    figure;
    imagesc(angle_axis, range_axis, ra_map_db);
    set(gca, 'YDir', 'normal');
    xlabel('方位角 (度)');
    ylabel('距离 (m)');
    title('距离-方位角 (RA) 谱');
    colorbar;
    colormap jet;

    % ----- 图4：3D点云散点图 -----
    figure;
    scatter(points(:,1), points(:,2), 20, points(:,3), 'filled');
    xlabel('X 方位向 (m)');
    ylabel('Y 距离向 (m)');
    title(sprintf('3D点云 — %s（第1帧，%d个点）', scenarios_cn{s}, n_det));
    colorbar;
    colormap jet;
    caxis([-v_max, v_max]);
    axis equal; grid on;
    xlim([-3, 3]);
    ylim([0, 5]);
end

