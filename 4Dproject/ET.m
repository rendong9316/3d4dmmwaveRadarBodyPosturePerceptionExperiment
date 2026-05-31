%% ET.m — Elevation-Time Map (俯仰角-时间图)
%% 基于手册 MMWCAS-RF-EVM: TX阵列含4个MRA俯仰阵元
%% RX全部位于同一俯仰面, 依靠TX端俯仰分集实现测高
%% 横轴时间, 纵轴俯仰角, 看蹲下/站起/抬手等竖直方向运动
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
nRX_total = nDev * nRX;         % 16

range_win = hanning(nSamples);
dopp_win  = hanning(nLoops);
vmax = para.lambda / (4 * para.Chirptime * nChirps);
range_axis = (0:nSamples-1) * para.dr;
dopp_axis  = linspace(-vmax, vmax, nLoops);

r_min = round(0.5 / para.dr) + 1;
r_max = round(5.0 / para.dr);
zb = floor(nLoops/2) + 1;
non_dc = [1:zb-1, zb+1:nLoops];

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

% ===== 手册: TX含4个MRA俯仰阵元 =====
% 每设备取TX0对应的chirp (1-based: Dev1=ch12, Dev2=ch9, Dev3=ch6, Dev4=ch3)
% 4个设备在PCB上呈2×2排列, 对应4个不同俯仰位置
tx0_chirps = [12, 9, 6, 3];  % Dev1~Dev4 TX0 的 chirp (1-based)

% 4个俯仰通道的物理位置 (MRA, 手册Table 6: B2/B4/B3)
elev_positions = [0, 0.5, 1.0, 1.5];  % λ 单位, MRA排布

n_ele = 256;
ele_axis = asind(linspace(-1, 1, n_ele));

% ===== 逐帧处理 ETM =====
fprintf('生成 ETM...\n');
et_map = zeros(n_ele, nFrames);

for f = 1:nFrames
    if mod(f, 20) == 0, fprintf('  帧 %d/%d\n', f, nFrames); end

    % 用Dev1 (master) 找目标位置
    frm1 = squeeze(adcAll{1}(:, :, :, :, f));
    rfft1 = fft(frm1 .* range_win.', [], 2);
    rd1 = rfft1 - mean(rfft1, 3);
    rd1 = fft(rd1 .* reshape(dopp_win, 1,1,nLoops,1), [], 3);
    rd1 = fftshift(rd1, 3);
    pwr1 = squeeze(mean(mean(abs(rd1).^2, 1), 4));
    [~, r_peak] = max(max(pwr1(r_min:r_max, non_dc), [], 2));
    r_peak = r_peak + r_min - 1;
    [~, d_local] = max(pwr1(r_peak, non_dc));
    d_peak = non_dc(d_local);

    % 4个俯仰通道: 每设备TX0, 所有loop取均值 (提高SNR)
    elev_channels = zeros(4, 1);
    for dev = 1:nDev
        frm = squeeze(adcAll{dev}(:, :, :, :, f));
        rfft = fft(frm .* range_win.', [], 2);
        % 取TX0 chirp, 目标range bin, 所有loop均值
        elev_channels(dev) = squeeze(mean(mean(rfft(:, r_peak, :, tx0_chirps(dev)), 3), 1));
    end

    % 俯仰FFT (4点→256点补零)
    elev_spec = fftshift(fft(elev_channels, n_ele));
    et_map(:, f) = abs(elev_spec).^2;
end

% 背景归一化
et_bg = prctile(et_map, 10, 2);
et_map = 10*log10(et_map ./ (et_bg + 1e-6));

% ===== 画图 =====
figure('Color','w', 'Position', [100 100 900 500]);
imagesc(time_axis, ele_axis, et_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('俯仰角 (度)');
title(sprintf('ETM 俯仰角-时间谱 — %s (4 MRA俯仰通道)', scenario));
colormap jet; colorbar; grid on;

c_lo = prctile(et_map(:), 5);
clim([c_lo, c_lo+20]);

outDir = fullfile(script_dir, '..', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end
print(gcf, fullfile(outDir, [scenario '_ET.png']), '-dpng', '-r150');
fprintf('ETM 完成! 数据范围: [%.1f, %.1f] dB\n', min(et_map(:)), max(et_map(:)));
