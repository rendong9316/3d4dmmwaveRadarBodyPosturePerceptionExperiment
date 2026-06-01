%% plot_AT_complete_fixed3.m – 完整版（帧间关联 + 双阈值滞后 + 角度平滑）
clear; clc; close all;

%% ========= 用户配置 =========
jsonPath = "D:\downlowd_cloud\方向2-雷达数据demo\4D\CCconfig_json\CCconfig_json.mmwave.json";
dataFolder = "D:\downlowd_cloud\方向2-雷达数据demo\datasets_4Dradar\CCdata_bend_0001";
frameStart = 1;
frameEnd   = [];               % 空则自动使用全部帧
azimuthRange = [-60, 60];      % 方位角范围（度）
angleStep    = 0.5;            % 角度步长（度）

% 角度跟踪参数
Th_high = 8;          % 高阈值（dB），高于此值信任新角度
Th_low  = 4;          % 低阈值（dB），低于此值保持上一帧
maxAngChange = 15;    % 帧间最大允许角度变化（度），超过则保持上一帧
medFiltLen = 5;       % 角度轨迹中值滤波窗口长度（奇数）

% 调试选项
debugPlot = false;    % 是否绘制每帧的RD谱和角度谱（仅前几帧）
% ==================================

%% 1. 解析 JSON 配置文件
fprintf('读取配置文件...\n');
fid = fopen(jsonPath, 'r');
raw = fread(fid, inf, '*char')';
fclose(fid);
cfg = jsondecode(raw);

dev0 = cfg.mmWaveDevices(1);
rf = dev0.rfConfig;
prof = rf.rlProfiles(1).rlProfileCfg_t;
frameCfg = rf.rlFrameCfg_t;

% 基本参数
f0 = prof.startFreqConst_GHz * 1e9;
lambda = 3e8 / f0;
Fs = prof.digOutSampleRate * 1e3;
ADCSamples = prof.numAdcSamples;
idleTime = prof.idleTimeConst_usec * 1e-6;
rampEndTime = prof.rampEndTime_usec * 1e-6;
chirpTime = idleTime + rampEndTime;
freqSlope = prof.freqSlopeConst_MHz_usec * 1e12;

% 帧结构
chirpsPerCycle = frameCfg.chirpEndIdx - frameCfg.chirpStartIdx + 1;
numLoops = frameCfg.numLoops;
numFrames_json = frameCfg.numFrames;

% 级联设备信息
numDevices = length(cfg.mmWaveDevices);
numRXPerDevice = 4;
totalRX = numDevices * numRXPerDevice;

% 构建全局 TX -> chirp 映射
globalTxToChirpIdx = zeros(1, numDevices * 3);
maxTx = 0;
for d = 1:numDevices
    chirpCfgs = cfg.mmWaveDevices(d).rfConfig.rlChirps;
    for c = 1:length(chirpCfgs)
        txEn = hex2dec(chirpCfgs(c).rlChirpCfg_t.txEnable);
        if txEn > 0
            globalIdx = chirpCfgs(c).rlChirpCfg_t.chirpStartIdx + 1;
            txIdx = log2(txEn) + 1;
            globalTx = (d-1)*3 + txIdx;
            globalTxToChirpIdx(globalTx) = globalIdx;
            maxTx = max(maxTx, globalTx);
        end
    end
end
totalTX = maxTx;
globalTxToChirpIdx = globalTxToChirpIdx(1:totalTX);
fprintf('TX: %d, RX: %d\n', totalTX, totalRX);

bandwidth = freqSlope * ADCSamples / Fs;
dr = 3e8 / bandwidth / 2;
fprintf('距离分辨率: %.2f cm\n', dr*100);

%% 2. 读取原始数据（修正 IQ 解交织）
fprintf('读取原始数据...\n');
devNames = {'master', 'slave1', 'slave2', 'slave3'};
totalChirps = chirpsPerCycle * numLoops;

% 确定实际帧数
maxFrames = inf;
for d = 1:numDevices
    binFile = fullfile(dataFolder, sprintf('%s_0000_data.bin', devNames{d}));
    fid = fopen(binFile, 'rb');
    if fid == -1, error('无法打开: %s', binFile); end
    fseek(fid, 0, 'eof');
    fileBytes = ftell(fid);
    fclose(fid);
    samplesPerChirp = 2 * numRXPerDevice * ADCSamples;
    samplesPerFrame = samplesPerChirp * chirpsPerCycle * numLoops;
    framesInFile = floor(fileBytes / (samplesPerFrame * 2));
    maxFrames = min(maxFrames, framesInFile);
    fprintf('%s: %d 帧\n', devNames{d}, framesInFile);
end
if isempty(frameEnd)
    numFrames_total = min(maxFrames, numFrames_json);
else
    numFrames_total = min(frameEnd, maxFrames) - frameStart + 1;
end
fprintf('总可用帧数: %d\n', numFrames_total);

% 从第2帧开始有效处理（跳过第一帧），因此实际处理帧数少1
numFrames = numFrames_total - 1;
if numFrames < 1
    error('帧数不足，无法跳过第一帧');
end
fprintf('实际处理帧数（跳过首帧后）: %d\n', numFrames);

% 逐设备读取所有帧（包含第一帧，但后续处理时跳过）
adcDataAll = cell(numDevices, 1);
for d = 1:numDevices
    binFile = fullfile(dataFolder, sprintf('%s_0000_data.bin', devNames{d}));
    fid = fopen(binFile, 'rb');
    samplesPerChirp = 2 * numRXPerDevice * ADCSamples;
    samplesPerFrame = samplesPerChirp * chirpsPerCycle * numLoops;
    adcData = zeros(numRXPerDevice, ADCSamples, totalChirps, numFrames_total, 'single');
    for f = 1:numFrames_total
        globalFrameIdx = frameStart + f - 1;
        byteOffset = (globalFrameIdx - 1) * samplesPerFrame * 2;
        fseek(fid, byteOffset, 'bof');
        raw = fread(fid, samplesPerFrame, 'int16=>single');
        if length(raw) < samplesPerFrame
            warning('%s 帧 %d 数据不足', devNames{d}, globalFrameIdx);
            break;
        end
        % 关键修正：每两个 int16 组成一个复数 I/Q 样本
        complexRaw = complex(raw(1:2:end), raw(2:2:end));
        % 重组为 [RX, ADC] 格式（每个 chirp 独立）
        for chirp = 1:totalChirps
            start = (chirp-1) * numRXPerDevice * ADCSamples + 1;
            endIdx = start + numRXPerDevice * ADCSamples - 1;
            chunk = complexRaw(start:endIdx);
            adcData(:, :, chirp, f) = reshape(chunk, numRXPerDevice, ADCSamples);
        end
    end
    fclose(fid);
    adcDataAll{d} = adcData;
end

% 合并所有设备的 RX 通道
adcDataCombined = zeros(totalRX, ADCSamples, totalChirps, numFrames_total, 'single');
for d = 1:numDevices
    idx = (d-1)*numRXPerDevice + 1 : d*numRXPerDevice;
    adcDataCombined(idx, :, :, :) = adcDataAll{d};
end
fprintf('合并后数据尺寸: [%d, %d, %d, %d]\n', size(adcDataCombined));

%% 3. 构建真实虚拟阵列（请根据实际硬件替换以下坐标）
% 重要：以下坐标仅为示例，必须替换为您硬件的真实水平坐标（单位：米）
% 若坐标未知，可暂时使用均匀线阵近似（见注释），但角度会偏差。
rxPos_real = [0.0000, 0.0020, 0.0040, 0.0060, ...   % 芯片1 RX
              0.0160, 0.0180, 0.0200, 0.0220, ...   % 芯片2 RX
              0.0320, 0.0340, 0.0360, 0.0380, ...   % 芯片3 RX
              0.0480, 0.0500, 0.0520, 0.0540];      % 芯片4 RX
txPos_real = [-0.0040, 0, 0.0040, ...              % 芯片1 TX
               0.0120, 0.0160, 0.0200, ...         % 芯片2 TX
               0.0280, 0.0320, 0.0360, ...         % 芯片3 TX
               0.0440, 0.0480, 0.0520];            % 芯片4 TX
assert(length(rxPos_real) == totalRX, 'RX 坐标数量错误');
assert(length(txPos_real) == totalTX, 'TX 坐标数量错误');

% 构建虚拟阵元位置（TX+RX 卷积）
virtPos = zeros(totalTX * totalRX, 1);
idx = 1;
for tx = 1:totalTX
    for rx = 1:totalRX
        virtPos(idx) = txPos_real(tx) + rxPos_real(rx);
        idx = idx + 1;
    end
end
% 去重并排序
uniqPos = unique(virtPos);
% 以半波长间隔均匀插值，得到均匀虚拟线阵
minPos = uniqPos(1);
maxPos = uniqPos(end);
gridPos = (minPos : lambda/2 : maxPos)';
Nvirt = length(gridPos);
% 构建映射矩阵（最近邻插值）
mapMat = zeros(Nvirt, length(virtPos));
for i = 1:Nvirt
    [~, nearest] = min(abs(virtPos - gridPos(i)));
    mapMat(i, nearest) = 1;
end
fprintf('均匀虚拟阵元数: %d\n', Nvirt);
% 实际阵元间距（波长归一化）
d_lambda = (gridPos(2)-gridPos(1)) / lambda;

%% 4. 角度导向矢量
theta_deg = azimuthRange(1):angleStep:azimuthRange(2);
steer = exp(1j * 2*pi * d_lambda * (0:Nvirt-1)' * sind(theta_deg));

%% 5. 逐帧处理（从第2帧开始，使用帧间关联+滞后+角度变化限制）
anglePowerAll = zeros(length(theta_deg), numFrames);
azimuth_raw = zeros(numFrames, 1);
prevAng = 0;         % 上一帧角度，初始为0度
lowSNRcount = 0;

fprintf('开始逐帧处理（跳过第一帧，阈值高=%.1f dB，低=%.1f dB）...\n', Th_high, Th_low);
for f = 2:numFrames_total   % f 是原始帧索引，从2开始
    frame = adcDataCombined(:, :, :, f);
    
    % 重组为 [RX, ADC, TX, Doppler]
    txData = zeros(totalRX, ADCSamples, totalTX, numLoops, 'single');
    for tx = 1:totalTX
        chirpIdxInCycle = globalTxToChirpIdx(tx);
        for loop = 1:numLoops
            globalChirp = (loop-1)*chirpsPerCycle + chirpIdxInCycle;
            txData(:, :, tx, loop) = frame(:, :, globalChirp);
        end
    end
    
    % 距离 FFT
    rangeWin = hamming(ADCSamples, 'periodic').';
    rangeFFT = fft(bsxfun(@times, txData, rangeWin), [], 2);
    rangeFFT = rangeFFT(:, 1:ADCSamples/2, :, :);
    nRange = size(rangeFFT, 2);
    
    % 多普勒 FFT
    dopWin = hamming(numLoops, 'periodic').';
    dopWin4D = reshape(dopWin, 1,1,1,numLoops);
    dopFFT = fftshift(fft(bsxfun(@times, rangeFFT, dopWin4D), [], 4), 4);
    
    % 合并虚拟阵元
    virtOrig = reshape(permute(dopFFT, [1,3,2,4]), totalTX*totalRX, nRange, numLoops);
    [nOrig, nR, nD] = size(virtOrig);
    virtOrig2D = reshape(virtOrig, nOrig, nR * nD);
    virtUniform2D = mapMat * virtOrig2D;
    virtUniform = reshape(virtUniform2D, Nvirt, nR, nD);
    
    % 距离-多普勒谱（非相干累积）
    RDmap = squeeze(sum(abs(virtUniform).^2, 1));
    
    % 可选：抑制零多普勒附近的静态杂波（如果目标低速，建议注释）
    centerIdx = floor(numLoops/2) + 1;
    dopRange = 2;
    zeroIdx = max(1, centerIdx-dopRange) : min(numLoops, centerIdx+dopRange);
    RDmap(:, zeroIdx) = 0;
    
    % 噪声基底估计（最小30%像素平均）
    sortedRD = sort(RDmap(:));
    noiseFloor = mean(sortedRD(1:round(0.3*length(sortedRD))));
    noiseFloor_dB = 10*log10(noiseFloor + eps);
    [maxVal, maxIdx] = max(RDmap(:));
    maxVal_dB = 10*log10(maxVal + eps);
    peakSNR = maxVal_dB - noiseFloor_dB;
    
    % 获取候选角度
    [r, d] = ind2sub(size(RDmap), maxIdx);
    vec = squeeze(virtUniform(:, r, d));
    anglePower = abs(steer' * vec).^2;
    [~, maxAngIdx] = max(anglePower);
    candidateAng = theta_deg(maxAngIdx);
    
    % 双阈值滞后 + 角度变化限制
    if peakSNR >= Th_high
        % 高SNR：信任候选，但限制帧间变化
        angDiff = abs(candidateAng - prevAng);
        if angDiff <= maxAngChange
            currentAng = candidateAng;
        else
            currentAng = prevAng;
            fprintf('  帧 %d 角度变化过大(%.1f°)，保持 %.1f°\n', f, angDiff, prevAng);
        end
        lowSNRcount = 0;
    elseif peakSNR < Th_low
        % 低SNR：保持上一帧
        currentAng = prevAng;
        lowSNRcount = lowSNRcount + 1;
    else
        % 中间区域：连续低SNR超过2帧则强制更新
        lowSNRcount = lowSNRcount + 1;
        if lowSNRcount >= 3
            currentAng = candidateAng;
            lowSNRcount = 0;
            fprintf('  帧 %d 连续低SNR后强制更新角度为 %.1f°\n', f, currentAng);
        else
            currentAng = prevAng;
        end
    end
    
    % 存储结果（映射到处理帧序号 f-1）
    frameIdx = f - 1;   % 因为跳过了第一帧，有效帧从1开始
    anglePowerAll(:, frameIdx) = anglePower;
    azimuth_raw(frameIdx) = currentAng;
    prevAng = currentAng;
    
    if mod(f, 10) == 0
        fprintf('  帧 %d (原始索引) 完成, SNR=%.1f dB, 角度=%.1f°\n', f, peakSNR, currentAng);
    end
    
    % 调试绘图（前几帧）
    if debugPlot && f <= 5
        figure(100 + f);
        subplot(2,1,1);
        imagesc(10*log10(RDmap+eps)); colorbar; title(sprintf('帧%d RD谱', f));
        subplot(2,1,2);
        plot(theta_deg, 10*log10(anglePower+eps)); grid on;
        xlabel('角度(度)'); ylabel('功率(dB)'); title(sprintf('帧%d 角度谱', f));
        drawnow;
    end
end

% 中值滤波平滑角度轨迹
azimuth_smooth = medfilt1(azimuth_raw, medFiltLen);

%% 6. 绘图
heatmapDB = 20*log10(anglePowerAll + eps);
maxDB = max(heatmapDB(:));
minDB_display = maxDB - 40;   % 显示40dB动态范围

figure('Position', [100 100 1400 600]);
subplot(1,2,1);
imagesc(1:numFrames, theta_deg, heatmapDB);
axis xy; colormap(jet); colorbar;
xlabel('帧序号（有效帧，原始第2帧起）'); ylabel('方位角 (度)');
title(sprintf('角度-时间热图 (阈值高=%.1f,低=%.1f dB)', Th_high, Th_low));
caxis([minDB_display, maxDB]);
fprintf('%d\n',minDB_display);
fprintf('%d\n',maxDB);

subplot(1,2,2);
plot(1:numFrames, azimuth_raw, 'b-', 'LineWidth', 1, 'DisplayName', '原始角度'); hold on;
plot(1:numFrames, azimuth_smooth, 'r-', 'LineWidth', 1.5, 'DisplayName', '中值滤波后');
xlabel('帧序号（有效帧）'); ylabel('方位角 (度)');
title('主目标角度轨迹');
legend; grid on; ylim(azimuthRange);
xlim([1, numFrames]);

fprintf('AT 图生成完成！有效帧数: %d\n', numFrames);