%% DT.m
%% DT图 - 
clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

sceneType = 'bend';  sceneIdx = 1;
scenario = sprintf('CCdata_%s_%04d', sceneType, sceneIdx);
fprintf('场景: %s\n', scenario);

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

fprintf('帧数: %d\n', nFrames);

time_axis = (0:nFrames-1) * para.Frameinter;

%% DT图
dt_map = zeros(nLoops, nFrames);

fprintf('处理中...\n');
for f = 1:nFrames
    frame = squeeze(rawData(:, :, :, :, f));
    
    rfft = fft(frame .* range_win.', [], 2);
    rd_mti = rfft - mean(rfft, 4);
    rd = fft(rd_mti .* reshape(dopp_win, 1,1,1,nLoops), [], 4);
    rd = fftshift(rd, 4);
    
    pwr_rd = squeeze(mean(sum(abs(rd).^2, 1), 3));
    
    dt_map(:, f) = max(pwr_rd(r_min:r_max, :), [], 1);
end

dt_map_dB = 10*log10(dt_map + 1);

fprintf('DT图dB范围: [%.1f, %.1f]\n', min(dt_map_dB(:)), max(dt_map_dB(:)));

figure('Color','w');
imagesc(time_axis, dopp_axis, dt_map_dB);
set(gca,'YDir','normal');
xlabel('时间 (s)');
ylabel('速度 (m/s)');
title(sprintf('DTM - %s', scenario));
colormap jet;
colorbar;
clim([100, 110]);