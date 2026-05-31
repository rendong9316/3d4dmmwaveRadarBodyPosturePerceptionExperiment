%% test_rt_only.m
%% 只生成RT图，慢慢调试
clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

sceneType = 'bend';  sceneIdx = 1;
scenario = sprintf('CCdata_%s_%04d', sceneType, sceneIdx);

nSamples = para.ADCSamples;      % 256
nLoops   = para.numLoops;        % 64
nChirps  = para.chirpsPerCycle;  % 12
nRX      = para.numRXPerDevice;  % 4

nChirpsPerFrame = nChirps * nLoops;  % 768

range_axis = (0:nSamples-1) * para.dr;
vmax = para.lambda / (4 * para.Chirptime * nChirps);
dopp_axis = linspace(-vmax, vmax, nLoops);

range_win = hanning(nSamples);
dopp_win  = hanning(nLoops);

r_min = round(0.5 / para.dr) + 1;
r_max = round(5.0 / para.dr);
zb = floor(nLoops/2) + 1;
non_dc = [1:zb-1, zb+1:nLoops];

%% 读取master数据
masterBin = fullfile(dataRoot, scenario, 'master_0000_data.bin');
fid = fopen(masterBin, 'rb');
rawData = fread(fid, 'int16');
fclose(fid);

rawData = rawData(1:2:end) + 1j*rawData(2:2:end);

totalChirps = length(rawData) / (nRX * nSamples);
nFrames = floor(totalChirps / nChirpsPerFrame);

rawData = rawData(1 : nRX * nSamples * nFrames * nChirpsPerFrame);
rawData = reshape(rawData, nRX, nSamples, nChirpsPerFrame, nFrames);
rawData = reshape(rawData, nRX, nSamples, nChirps, nLoops, nFrames);

fprintf('数据形状: [%d, %d, %d, %d, %d]\n', size(rawData));
fprintf('帧数: %d\n', nFrames);

time_axis = (0:nFrames-1) * para.Frameinter;

%% 初始化RT图
rt_map = zeros(nSamples, nFrames);

%% 逐帧处理
fprintf('\n处理中...\n');
for f = 1:nFrames
    frame = squeeze(rawData(:, :, :, :, f));  % [nRX, nSamples, nChirps, nLoops]
    
    % Range FFT
    rfft = fft(frame .* range_win.', [], 2);
    
    % MTI
    rd_mti = rfft - mean(rfft, 4);
    
    % Doppler FFT
    rd = fft(rd_mti .* reshape(dopp_win, 1,1,1,nLoops), [], 4);
    rd = fftshift(rd, 4);
    
    % 功率图：平均所有RX和chirp
    pwr_rd = squeeze(mean(sum(abs(rd).^2, 1), 3));  % [nRange, nLoops]
    
    % RTM：每个距离bin取非DC多普勒的最大值
    rt_map(:, f) = max(pwr_rd(:, non_dc), [], 2);
end

%% 转dB
rt_map_dB = 10*log10(rt_map + 1);

fprintf('\nRT图dB范围: [%.1f, %.1f]\n', min(rt_map_dB(:)), max(rt_map_dB(:)));

% 取2%和98%分位数
p2 = prctile(rt_map_dB(:), 2);
p98 = prctile(rt_map_dB(:), 98);
fprintf('2%%分位: %.1f, 98%%分位: %.1f\n', p2, p98);

%% 画图
figure('Color','w');
imagesc(time_axis, range_axis, rt_map_dB);
set(gca,'YDir','normal');
xlabel('时间 (s)');
ylabel('距离 (m)');
title(sprintf('RTM 距离-时间谱 - %s', scenario));
colormap jet;
colorbar;
ylim([0, 5]);
clim([100, 110]);

fprintf('\n=== 请反馈 ===\n');
fprintf('1. RT图dB范围是多少？\n');
fprintf('2. 是否能看到一条清晰的轨迹？\n');
fprintf('3. 背景是什么颜色？\n');
