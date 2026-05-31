%% AT.m — Azimuth-Time Map (方位角-时间图)
%% 基于手册 MMWCAS-RF-EVM Fig.26-27: 12TX×16RX = 192虚拟通道
%% 86个非重叠方位虚拟阵元 (λ/2间距), 理论分辨率 ~1.2°
%% 横轴时间, 纵轴方位角, 看人体左右转向/横向移动
clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

sceneType = 'walk';  sceneIdx = 1;
scenario = sprintf('CCdata_%s_%04d', sceneType, sceneIdx);
fprintf('场景: %s\n', scenario);

nSamples = para.ADCSamples;     % 256
nLoops   = para.numLoops;       % 64
nChirps  = para.chirpsPerCycle; % 12
nRX      = para.numRXPerDevice; % 4
nDev     = para.numDevices;     % 4
nTX_total = nDev * 3;           % 12
nRX_total = nDev * nRX;         % 16
nVirtual  = nTX_total * nRX_total;  % 192

range_win = hanning(nSamples);
dopp_win  = hanning(nLoops);
vmax = para.lambda / (4 * para.Chirptime * nChirps);
range_axis = (0:nSamples-1) * para.dr;
dopp_axis  = linspace(-vmax, vmax, nLoops);

r_min = round(0.5 / para.dr) + 1;
r_max = round(5.0 / para.dr);
zb = floor(nLoops/2) + 1;
non_dc = [1:zb-1, zb+1:nLoops];

% ===== 手册 Fig.26: 天线位置 (λ 为单位, λ@78.5GHz) =====
% ---- RX位置 (每设备4个, λ/2间距) ----
rx_per_device = (0:nRX-1) * 0.5;  % [0, 0.5, 1.0, 1.5] λ

% 设备间RX阵列偏移 (Fig.26: AWR #4=8λ, #1=0λ, #3=24λ, #2=16λ)
% 注: 设备编号与 mmWave Studio 一致: dev1=master, dev2=slave1, dev3=slave2, dev4=slave3
device_rx_base = [0, 16, 24, 8];  % Dev1, Dev2, Dev3, Dev4 的RX基地址 (λ)

rx_abs = zeros(nRX_total, 1);
for d = 1:nDev
    for rx = 1:nRX
        idx = (d-1)*nRX + rx;
        rx_abs(idx) = device_rx_base(d) + rx_per_device(rx);
    end
end

% ---- TX位置 (Fig.26: 12个TX, 2λ间距, 覆盖0~22λ) ----
tx_positions = (0:2:22)';  % 12个TX物理位置 (λ)

% ---- 构建192虚拟阵列 (TX_pos + RX_pos, 手册Fig.27) ----
virt_pos = zeros(nTX_total, nRX_total);
for tx = 1:nTX_total
    for rx = 1:nRX_total
        virt_pos(tx, rx) = tx_positions(tx) + rx_abs(rx);
    end
end
virt_flat = virt_pos(:);
[virt_sorted, sort_idx] = sort(virt_flat);
nUnique = length(unique(round(virt_sorted * 1000) / 1000));
angle_res = rad2deg(para.lambda / (max(virt_sorted) * para.lambda));  % 理论分辨率
fprintf('  虚拟通道: %d, 非重叠方位位置: %d, 理论分辨率: %.2f°\n', nVirtual, nUnique, angle_res);

% ===== TDM-MIMO chirp→物理TX映射 =====
% 手册Fig.26: chirp0→Dev4_TX2, chirp1→Dev4_TX1, ..., chirp11→Dev1_TX0
% 物理TX索引 (0-based): Dev4:TX2/1/0=11/10/9, Dev3:TX2/1/0=8/7/6,
%                        Dev2:TX2/1/0=5/4/3, Dev1:TX2/1/0=2/1/0
chirp_to_tx = [11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0];  % 0-based

% ===== 读取全部4设备 =====
fprintf('读取4设备...\n');
devices = {'master', 'slave1', 'slave2', 'slave3'};
adcAll = cell(nDev, 1);
for d = 1:nDev
    binFile = fullfile(dataRoot, scenario, sprintf('%s_0000_data.bin', devices{d}));
    fid = fopen(binFile, 'rb');
    rawData = fread(fid, 'int16');
    fclose(fid);
    rawData = rawData(1:2:end) + 1j*rawData(2:2:end);
    totalChirps = length(rawData) / (nRX * nSamples);
    nFrames = floor(totalChirps / (nChirps * nLoops));
    rawData = rawData(1 : nRX * nSamples * nFrames * nChirps * nLoops);
    rawData = reshape(rawData, nRX, nSamples, nChirps*nLoops, nFrames);
    rawData = reshape(rawData, nRX, nSamples, nLoops, nChirps, nFrames);
    adcAll{d} = rawData;
end
fprintf('  完成! %d帧\n', nFrames);
time_axis = (0:nFrames-1) * para.Frameinter;

% ===== 预处理: Range+Doppler FFT per device =====
fprintf('预处理FFT...\n');
rd_all = cell(nDev, 1);
for d = 1:nDev
    data = adcAll{d};  % [4, 256, 64, 12, nFrames]
    rfft = fft(data .* range_win.', [], 2);
    rfft_mti = rfft - mean(rfft, 3);
    rd = fft(rfft_mti .* reshape(dopp_win, 1,1,nLoops,1,1), [], 3);
    rd = fftshift(rd, 3);
    rd_all{d} = rd;  % [4, 256, 64, 12, nFrames]
end
fprintf('预处理完成!\n');

% ===== 逐帧生成AT图 =====
fprintf('生成 ATM...\n');
nAngles = 361;
ang_axis = linspace(-90, 90, nAngles);
at_map = zeros(nAngles, nFrames);

for f = 1:nFrames
    if mod(f, 20) == 0, fprintf('  帧 %d/%d\n', f, nFrames); end

    % 合并16 RX通道
    rd_frame = cat(1, rd_all{1}(:,:,:,:,f), rd_all{2}(:,:,:,:,f), ...
                     rd_all{3}(:,:,:,:,f), rd_all{4}(:,:,:,:,f));  % [16, 256, 64, 12]

    % 重排chirp为物理TX顺序
    rd_frame = rd_frame(:, :, :, chirp_to_tx + 1);  % [16, 256, 64, 12]

    % 找目标 (运动最强点)
    pwr_rd = squeeze(mean(abs(rd_frame).^2, 1));  % [256, 64, 12]
    pwr_mean = mean(pwr_rd, 3);  % [256, 64]
    [~, r_peak] = max(max(pwr_mean(r_min:r_max, non_dc), [], 2));
    r_peak = r_peak + r_min - 1;
    [~, d_local] = max(pwr_mean(r_peak, non_dc));
    d_peak = non_dc(d_local);

    % 提取192虚拟通道数据
    virt_data = squeeze(rd_frame(:, r_peak, d_peak, :));  % [16 RX, 12 TX]
    virt_192 = virt_data(:);  % [192, 1]
    virt_192 = virt_192(sort_idx);  % 按虚拟位置排序

    % 角度FFT
    ang_spec = fftshift(fft(virt_192, nAngles));
    at_map(:, f) = abs(ang_spec).^2;
end

% 背景归一化
at_bg = prctile(at_map, 10, 2);
at_map = 10*log10(at_map ./ (at_bg + 1e-6));

% ===== 画图 =====
figure('Color','w', 'Position', [100 100 900 500]);
imagesc(time_axis, ang_axis, at_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('方位角 (度)');
title(sprintf('ATM 方位角-时间谱 — %s (192通道, %.2f°分辨率)', scenario, angle_res));
colormap jet; colorbar; grid on;

c_lo = prctile(at_map(:), 5);
clim([c_lo, c_lo+20]);

outDir = fullfile(script_dir, '..', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end
print(gcf, fullfile(outDir, [scenario '_AT.png']), '-dpng', '-r150');
fprintf('ATM 完成! 数据范围: [%.1f, %.1f] dB\n', min(at_map(:)), max(at_map(:)));
