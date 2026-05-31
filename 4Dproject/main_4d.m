%% test_at_corrected.m - 正确的AT图绘制（基于192虚拟通道）
clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

sceneType = 'stand'; sceneIdx = 1;
scenario = sprintf('CCdata_%s_%04d', sceneType, sceneIdx);

nSamples = para.ADCSamples;      % 256
nLoops   = para.numLoops;        % 64
nChirps  = para.chirpsPerCycle;  % 12
nRX      = para.numRXPerDevice;  % 4 (每设备)
nDev     = para.numDevices;      % 4
nTX_per_dev = 3;                 % 每设备3个TX
nTX_total = nDev * nTX_per_dev;  % 12 TX
nRX_total = nDev * nRX;          % 16 RX

range_axis = (0:nSamples-1) * para.dr;
vmax = para.lambda / (4 * para.Chirptime * nChirps);
dopp_axis = linspace(-vmax, vmax, nLoops);

range_win = hanning(nSamples);
dopp_win  = hanning(nLoops);

r_min = round(1.0 / para.dr) + 1;
r_max = round(5.0 / para.dr);
zb = floor(nLoops/2) + 1;
non_dc = [1:zb-1, zb+1:nLoops];

%% 读取所有设备
devices = {'master', 'slave1', 'slave2', 'slave3'};
adcAll = cell(nDev,1);
fprintf('读取数据...\n');
for d = 1:nDev
    binFile = fullfile(dataRoot, scenario, sprintf('%s_0000_data.bin', devices{d}));
    fid = fopen(binFile, 'rb');
    rawData = fread(fid, 'int16');
    fclose(fid);
    rawData = rawData(1:2:end) + 1j*rawData(2:2:end);
    totalChirpsAll = length(rawData) / (nRX * nSamples);
    nFrames = floor(totalChirpsAll / (nChirps * nLoops));
    rawData = rawData(1 : nRX * nSamples * nFrames * nChirps * nLoops);
    rawData = reshape(rawData, nRX, nSamples, nChirps*nLoops, nFrames);
    rawData = reshape(rawData, nRX, nSamples, nLoops, nChirps, nFrames);
    adcAll{d} = rawData;
end
fprintf('读取完成! %d帧\n', nFrames);

% 计算时间轴（重要！）
time_axis = (0:nFrames-1) * para.Frameinter;
fprintf('时间轴: %.2f秒, 共%d帧\n', time_axis(end), nFrames);

%% 预处理：所有设备做Range+Doppler FFT
fprintf('预处理FFT...\n');
rd_all = cell(nDev,1);
for d = 1:nDev
    data = adcAll{d};
    rfft = fft(data .* range_win.', [], 2);
    rfft_mti = rfft - mean(rfft, 3);
    rd = fft(rfft_mti .* reshape(dopp_win, 1,1,nLoops,1), [], 3);
    rd = fftshift(rd, 3);
    rd_all{d} = rd;
end
fprintf('预处理完成!\n');

%% 关键：构建192虚拟通道的正确顺序
% 基于手册Figure 26-27

% 16个RX通道的位置（相对于各设备参考点）
% 每个AWR设备有4个RX，间距0.5 lambda
rx_positions_per_device = (0:nRX-1) * 0.5;  % [0, 0.5, 1.0, 1.5] lambda

% 4个设备之间的RX阵列偏移（从图26）
device_rx_offset = [0, 8, 16, 24];  % lambda单位

% 构建所有16个RX的绝对位置
rx_abs_positions = zeros(nRX_total, 1);
for dev = 1:nDev
    base = device_rx_offset(dev);
    for rx = 1:nRX
        idx = (dev-1)*nRX + rx;
        rx_abs_positions(idx) = base + rx_positions_per_device(rx);
    end
end

% 12个TX通道的位置（从图27）
% TX间距为2 lambda
tx_positions = 0:2:22;  % [0,2,4,6,8,10,12,14,16,18,20,22] lambda

% 构建192虚拟阵列位置（克罗内克积）
virtual_positions = zeros(nTX_total, nRX_total);
for tx = 1:nTX_total
    for rx = 1:nRX_total
        virtual_positions(tx, rx) = tx_positions(tx) + rx_abs_positions(rx);
    end
end

% 展平并排序（按位置从小到大，模拟均匀线阵）
virtual_positions_flat = virtual_positions(:);
[~, sort_idx] = sort(virtual_positions_flat);
nVirtual = nTX_total * nRX_total;  % 192

fprintf('虚拟通道数: %d\n', nVirtual);

%% 重要：根据您的chirp映射重新排列TX顺序
% 您提供的映射: chirp 0->Dev4 TX2, chirp1->Dev4 TX1, chirp2->Dev4 TX0
% chirp3->Dev3 TX2, chirp4->Dev3 TX1, chirp5->Dev3 TX0
% chirp6->Dev2 TX2, chirp7->Dev2 TX1, chirp8->Dev2 TX0
% chirp9->Dev1 TX2, chirp10->Dev1 TX1, chirp11->Dev1 TX0

% 注意：您的映射中Dev1是master，但chirp9-11对应Dev1
% 需要重新排列TX索引为物理顺序
tx_reorder = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];  % 根据映射调整
% 或者更明确地：
% chirp索引 -> 物理TX索引 (0-11)
chirp_to_tx = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];  % Dev4 TX2->TX11

%% 提取192通道数据
test_frame = 10;
fprintf('\n=== 单帧测试 ===\n');

% 提取该帧所有设备的RD数据
rd_master = rd_all{1}(:, :, :, :, test_frame);  % [4, 256, 64, 12] 注意：第4维是chirp
rd_slave1 = rd_all{2}(:, :, :, :, test_frame);  % [4, 256, 64, 12]
rd_slave2 = rd_all{3}(:, :, :, :, test_frame);  % [4, 256, 64, 12]
rd_slave3 = rd_all{4}(:, :, :, :, test_frame);  % [4, 256, 64, 12]

% 合并所有RX通道（16个）
rd_all_rx = cat(1, rd_master, rd_slave1, rd_slave2, rd_slave3);  % [16, 256, 64, 12]

% 重新排列chirp顺序以匹配物理TX顺序
% rd_all_rx: [16 RX, 256距离, 64多普勒, 12 chirp]
rd_all_rx_reordered = rd_all_rx(:, :, :, chirp_to_tx + 1);  % +1因为MATLAB索引从1开始

% 计算功率用于目标检测
pwr_rd = squeeze(mean(abs(rd_all_rx_reordered).^2, 1));  % [256, 64, 12]

% 进一步平均所有chirp找目标
pwr_rd_mean = mean(pwr_rd, 3);  % [256, 64]

% 找目标
[~, r_peak] = max(max(pwr_rd_mean(r_min:r_max, non_dc), [], 2));
r_peak = r_peak + r_min - 1;
[~, d_peak_local] = max(pwr_rd_mean(r_peak, non_dc));
d_peak = non_dc(d_peak_local);

fprintf('目标: 距离bin %d (%.2fm), 多普勒bin %d (%.2fm/s)\n', ...
    r_peak, range_axis(r_peak), d_peak, dopp_axis(d_peak));

% 提取该目标的所有192通道数据
% rd_all_rx_reordered: [16 RX, 256距离, 64多普勒, 12 TX物理顺序]
rd_target = squeeze(rd_all_rx_reordered(:, r_peak, d_peak, :));  % [16, 12]

% 展平并按虚拟阵列位置排序
virt_192_matrix = rd_target;  % [16 RX, 12 TX]
virt_192_flat = virt_192_matrix(:);  % [192, 1]
virt_192_sorted = virt_192_flat(sort_idx);

fprintf('192通道数据提取完成\n');

%% 角度FFT（使用正确的虚拟阵列）
n_fft = 512;
ang_spec = fftshift(fft(virt_192_sorted, n_fft));
ang_spec_pwr = abs(ang_spec).^2;
ang_axis = asind(linspace(-1, 1, n_fft));

% 计算理论角度分辨率
lambda = para.lambda;
d_ant = 0.5 * lambda;  % 虚拟天线间距（lambda/2）
array_length = nVirtual * d_ant;
angle_resolution = rad2deg(lambda / array_length);

figure('Name', '192通道角度谱');
plot(ang_axis, 10*log10(ang_spec_pwr), 'b-', 'LineWidth', 1.5);
xlabel('角度 (度)'); ylabel('功率 (dB)');
title(sprintf('192通道角度谱 - 理论分辨率: %.3f度', angle_resolution));
grid on; xlim([-90, 90]);
[~, max_idx] = max(ang_spec_pwr);
fprintf('峰值角度: %.1f度, 峰噪比: %.1f dB\n', ...
    ang_axis(max_idx), 10*log10(max(ang_spec_pwr)/mean(ang_spec_pwr)));

%% 逐帧生成AT图（使用正确的192通道）
fprintf('\n生成AT图...\n');
nAngles = 361;
ang_axis_at = linspace(-90, 90, nAngles);
at_map = zeros(nAngles, nFrames);

for f = 1:nFrames
    if mod(f, 20) == 0
        fprintf('处理帧 %d/%d\n', f, nFrames);
    end
    
    % 提取该帧所有设备数据
    rd_master = rd_all{1}(:, :, :, :, f);
    rd_slave1 = rd_all{2}(:, :, :, :, f);
    rd_slave2 = rd_all{3}(:, :, :, :, f);
    rd_slave3 = rd_all{4}(:, :, :, :, f);
    rd_all_rx = cat(1, rd_master, rd_slave1, rd_slave2, rd_slave3);
    
    % 重新排列chirp顺序
    rd_all_rx_reordered = rd_all_rx(:, :, :, chirp_to_tx + 1);
    
    % 计算功率找目标
    pwr_rd = squeeze(mean(abs(rd_all_rx_reordered).^2, 1));
    pwr_rd_mean = mean(pwr_rd, 3);
    
    [~, r_p] = max(max(pwr_rd_mean(r_min:r_max, non_dc), [], 2));
    r_p = r_p + r_min - 1;
    [~, d_p_local] = max(pwr_rd_mean(r_p, non_dc));
    d_p = non_dc(d_p_local);
    
    % 提取192通道
    rd_target_frame = squeeze(rd_all_rx_reordered(:, r_p, d_p, :));
    virt_192 = rd_target_frame(:);
    virt_192_sorted = virt_192(sort_idx);
    
    % 角度FFT
    ang_spec = fftshift(fft(virt_192_sorted, nAngles));
    at_map(:, f) = abs(ang_spec).^2;
end

at_map_dB = 10*log10(at_map + eps);

%% 可视化
figure('Color','w', 'Position', [100 100 1200 500]);

subplot(1,2,1);
imagesc(time_axis, ang_axis_at, at_map_dB);
set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('角度 (度)');
title(sprintf('AT图 (192虚拟通道) - %s', scenario));
colormap jet; colorbar;
% 自动调整颜色范围
clim([130,150]);

subplot(1,2,2);
plot(ang_axis_at, at_map_dB(:, end), 'b-', 'LineWidth', 1.5);
xlabel('角度 (度)'); ylabel('功率 (dB)');
title(sprintf('角度谱 - 最后一帧 (分辨率: %.3f度)', angle_resolution));
grid on; xlim([-90, 90]);
hold on;
[peaks, locs] = findpeaks(at_map_dB(:, end), 'MinPeakHeight', max(at_map_dB(:, end))-10);
if ~isempty(peaks)
    plot(ang_axis_at(locs), peaks, 'ro', 'MarkerSize', 8);
end

%% 验证虚拟阵列的相位响应
figure('Name', '虚拟阵列验证');
phase_data = angle(virt_192_sorted);
phase_unwrap = unwrap(phase_data);
plot(1:nVirtual, rad2deg(phase_unwrap), '.-', 'LineWidth', 1);
xlabel('虚拟天线序号'); ylabel('相位 (度)');
title('虚拟阵列相位响应 - 应为线性');
grid on;

fprintf('\n=== 完成 ===\n');
fprintf('虚拟通道数: %d\n', nVirtual);
fprintf('理论角度分辨率: %.3f度\n', angle_resolution);