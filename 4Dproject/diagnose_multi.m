%% diagnose_multi.m - Check multiple frames and frames for walking signal
clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

jsonPath = fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json');
para = read4DParam(jsonPath);
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

scenario = 'CCdata_walk_0001';
nRange = para.ADCSamples;
nLoops = para.numLoops;
nRX = para.numRXPerDevice;
nDev = para.numDevices;
chirpsPC = para.chirpsPerCycle;
totalTX = para.totalTX;
r_axis = (0:nRange-1)' * para.dr;
T_cycle = para.Chirptime * chirpsPC;
v_max = para.lambda / (4 * T_cycle);
v_res = 2*v_max/nLoops;
v_axis_correct = ((1:nLoops) - nLoops/2 - 1) * v_res;

fprintf('v_max=%.2f v_res=%.4f\n', v_max, v_res);

%% Test multiple frames - find which frames have signal at walking speed
testFrames = 5:5:70;
nTestFrames = length(testFrames);
bestSNR = zeros(nTestFrames, 2);  % [dB, range_m]

devNames = {'master', 'slave1', 'slave2', 'slave3'};
rWin = hanning(nRange);
dWin = hanning(nLoops);

fprintf('\n===== Scanning frames for walking signal =====\n');
fprintf('Frame | SNR@1m/s | bestR | SNR@0m/s \n');

rMin = round(1.0/para.dr);
rMax = round(5.0/para.dr);

for fi = 1:nTestFrames
    frameIdx = testFrames(fi);
    
    % Read all 4 devices
    rfft_devs = cell(nDev,1);
    for d = 1:nDev
        fpath = fullfile(dataRoot, scenario, [devNames{d} '_0000_data.bin']);
        adcData = read4DRawData(fpath, para, frameIdx:frameIdx);
        frm = squeeze(adcData(:, :, :, 1));
        rfft_devs{d} = fft(frm .* rWin.', [], 2);
    end
    
    % TDM + Doppler
    rdPwr = zeros(nRange, nLoops);
    for tx = 1:totalTX
        devId = para.txChirpMap(tx, 1);
        rd_tx = reshape(rfft_devs{devId}, nRX, nRange, chirpsPC, nLoops);
        rd_data = squeeze(rd_tx(:, :, tx, :));
        % WITH MTI
        rd_data = rd_data - mean(rd_data, 3);
        rd_data = fft(rd_data .* reshape(dWin, 1, 1, nLoops), [], 3);
        rd_data = fftshift(rd_data, 3);
        rdPwr = rdPwr + squeeze(mean(abs(rd_data).^2, 1));
    end
    rdPwr = rdPwr / totalTX;
    rdDB = 10*log10(rdPwr + eps);
    
    % Signal at walking speed (v=0.8-1.2 m/s)
    walkBins = find(abs(v_axis_correct) >= 0.6 & abs(v_axis_correct) <= 1.4);
    walkPwr = max(rdPwr(rMin:rMax, walkBins), [], 1);
    [bestWalkPwr, bestWalkIdx] = max(walkPwr(:));
    [~, bestCol] = ind2sub(size(max(rdPwr(rMin:rMax, walkBins), [], 1)), bestWalkIdx);
    bestWalkBin = walkBins(bestCol);
    bestWalkRange = rMin - 1 + find(rdPwr(rMin:rMax, bestWalkBin) == bestWalkPwr, 1);
    
    % Signal at zero velocity
    dcBin = nLoops/2 + 1;
    dcPwr = max(rdPwr(rMin:rMax, dcBin));
    
    fprintf(' %3d  |  %6.1f dB | R=%.1fm | %5.1f dB @DC=%d\n', ...
        frameIdx, 10*log10(bestWalkPwr), r_axis(bestWalkRange), ...
        10*log10(dcPwr), nLoops/2+1);
end

%% Now check: average over MULTIPLE frames to boost SNR
fprintf('\n===== Multi-frame integration test =====\n');

% Read frames 15-30, do frame averaging
frame_start = 15; frame_end = 30;
nFrames = frame_end - frame_start + 1;

% Pre-read frames
rdPwr_frames = zeros(nRange, nLoops, nFrames);
for fi = 1:nFrames
    frameIdx = frame_start + fi - 1;
    rfft_devs = cell(nDev,1);
    for d = 1:nDev
        fpath = fullfile(dataRoot, scenario, [devNames{d} '_0000_data.bin']);
        adcData = read4DRawData(fpath, para, frameIdx:frameIdx);
        frm = squeeze(adcData(:, :, :, 1));
        rfft_devs{d} = fft(frm .* rWin.', [], 2);
    end
    
    rdPwr = zeros(nRange, nLoops);
    for tx = 1:totalTX
        devId = para.txChirpMap(tx, 1);
        rd_tx = reshape(rfft_devs{devId}, nRX, nRange, chirpsPC, nLoops);
        rd_data = squeeze(rd_tx(:, :, tx, :));
        rd_data = rd_data - mean(rd_data, 3);  % MTI
        rd_data = fft(rd_data .* reshape(dWin, 1, 1, nLoops), [], 3);
        rd_data = fftshift(rd_data, 3);
        rdPwr = rdPwr + squeeze(mean(abs(rd_data).^2, 1));
    end
    rdPwr_frames(:, :, fi) = rdPwr / totalTX;
end

% Average power across frames (non-coherent)
rdPwr_avg = mean(rdPwr_frames, 3);
rdDB_avg = 10*log10(rdPwr_avg + eps);

% Check signal after averaging
fprintf('\nAfter %d-frame averaging (with MTI):\n', nFrames);
fprintf('Top velocity bins:\n');
avgPwr = mean(rdPwr_avg(rMin:rMax, :), 1);
[sortedPwr, sortIdx] = sort(avgPwr, 'descend');
for i = 1:15
    di = sortIdx(i);
    fprintf('  bin%2d: v=%+7.2f m/s pwr=%.1f dB\n', di, v_axis_correct(di), 10*log10(sortedPwr(i)));
end

fprintf('\nAt walking speed bins:\n');
walkBins = find(abs(v_axis_correct) >= 0.6 & abs(v_axis_correct) <= 1.4);
for bi = 1:length(walkBins)
    wb = walkBins(bi);
    [bestPwr, bestRi] = max(rdPwr_avg(rMin:rMax, wb));
    bestR = rMin - 1 + bestRi;
    if 10*log10(bestPwr) > 65
        fprintf('  v=%.2f m/s: best R=%.1fm %.1f dB\n', v_axis_correct(wb), r_axis(bestR), 10*log10(bestPwr));
    end
end

%% Save diagnosis figure
outDir = fullfile(script_dir, '..', 'figures');
if ~exist(outDir,'dir'), mkdir(outDir); end

figure('Color','w','Position',[50,50,900,600],'Visible','off');
imagesc(v_axis_correct, r_axis, rdDB_avg);
set(gca,'YDir','normal');
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title(sprintf('RD Spectrum - %d-frame avg with MTI (frames %d-%d)', nFrames, frame_start, frame_end));
colormap jet; colorbar; grid on;
xlim([-v_max, v_max]); ylim([0.5, 6]);
clim([prctile(rdDB_avg(:), 40), prctile(rdDB_avg(:), 99)]);
hold on;
plot([0.6,0.6], [0.5,6], 'w--', 'LineWidth',1);
plot([1.4,1.4], [0.5,6], 'w--', 'LineWidth',1);
plot([-0.6,-0.6], [0.5,6], 'w--', 'LineWidth',1);
plot([-1.4,-1.4], [0.5,6], 'w--', 'LineWidth',1);
hold off;
exportgraphics(gcf, fullfile(outDir, 'diagnose_RD_multiframe.png'), 'Resolution', 150);
close;

fprintf('\n===== Multi-frame diagnostic complete =====\n');
