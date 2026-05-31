%% diagnose_rd.m - 诊断RD谱和检测问题
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
% Correct v_axis after fftshift: DC at bin nLoops/2+1
v_axis_dc0 = ((0:nLoops-1) - nLoops/2) * (2*v_max/nLoops);
fprintf('T_cycle=%.1fus v_max=%.2fm/s vr=%.3fm/s\n', T_cycle*1e6, v_max, 2*v_max/nLoops);
fprintf('v_axis[33](DC)=%.3f v_axis[51](1m/s)=%.3f\n', v_axis_dc0(33), v_axis_dc0(51));

%% Read one frame
frameIdx = 20;
devNames = {'master', 'slave1', 'slave2', 'slave3'};
fprintf('\n===== Frame %d diagnostic =====\n', frameIdx);

rfft_devs = cell(nDev, 1);
for d = 1:nDev
    fpath = fullfile(dataRoot, scenario, [devNames{d} '_0000_data.bin']);
    adcData = read4DRawData(fpath, para, frameIdx:frameIdx);
    frm = squeeze(adcData(:, :, :, 1));
    rfft_devs{d} = fft(frm .* hanning(nRange).', [], 2);
end

%% TDM demux + Doppler FFT (no MTI)
dWin = hanning(nLoops);
rd_cube = cell(totalTX, 1);

for tx = 1:totalTX
    devId = para.txChirpMap(tx, 1);
    rd_tx = reshape(rfft_devs{devId}, nRX, nRange, chirpsPC, nLoops);
    rd_data = squeeze(rd_tx(:, :, tx, :));  % [4, 256, 64]
    rd_data = fft(rd_data .* reshape(dWin, 1, 1, nLoops), [], 3);
    rd_data = fftshift(rd_data, 3);
    rd_cube{tx} = rd_data;
end

%% Non-coherent RD
rdPwr = zeros(nRange, nLoops);
for tx = 1:totalTX
    rdPwr = rdPwr + squeeze(mean(abs(rd_cube{tx}).^2, 1));
end
rdPwr = rdPwr / totalTX;
rdDB = 10 * log10(rdPwr + eps);

%% Analysis 1: top velocity bin at each range
fprintf('\n--- Strongest velocity at each range ---\n');
rMin = round(1.0/para.dr);
rMax = round(5.0/para.dr);
for ri = rMin:round((rMax-rMin)/8):rMax
    [~, di] = max(rdPwr(ri, :));
    fprintf('  R=%.2fm: strongest bin=%d v=%.2f m/s pwr=%.1f dB\n', ...
        r_axis(ri), di, v_axis_dc0(di), rdDB(ri, di));
end

%% Analysis 2: energy by velocity bin
fprintf('\n--- Velocity bin energy (R=1-5m avg) ---\n');
avgPwr = mean(rdPwr(rMin:rMax, :), 1);
[sortedPwr, sortIdx] = sort(avgPwr, 'descend');
for i = 1:10
    di = sortIdx(i);
    fprintf('  bin%2d: v=%+7.2f m/s pwr=%.1f dB\n', di, v_axis_dc0(di), 10*log10(sortedPwr(i)));
end

%% Analysis 3: CFAR with different thresholds
fprintf('\n--- CFAR detection test ---\n');
gr = 4; gd = 2;
tr = 8; td = 4;
kr = 2*tr + 2*gr + 1;
kd = 2*td + 2*gd + 1;
kern = ones(kr, kd);
kern(tr+1:tr+2*gr+1, td+1:td+2*gd+1) = 0;
nTr = sum(kern(:));
noiseSum = conv2(rdPwr, kern, 'same');
noiseAvg = noiseSum / nTr;

for tf = [3, 5, 7, 10, 15]
    det = rdPwr > (noiseAvg * tf);
    e = tr + gr;
    det(1:e, :) = false; det(end-e+1:end, :) = false;
    det(:, 1:e) = false; det(:, end-e+1:end) = false;
    det(1:rMin-1, :) = false;
    det(rMax+1:end, :) = false;
    
    [detR, detD] = find(det);
    nDet = length(detR);
    velList = [];
    if nDet > 0, velList = v_axis_dc0(detD); end
    
    fprintf('  tf=%.0f: %d det, v=[%.2f,%.2f] m/s\n', tf, nDet, min(velList), max(velList));
end

%% Analysis 4: check signal at walking speed
targetV = 1.0;
[~, targetBin] = min(abs(v_axis_dc0 - targetV));
fprintf('\n--- Signal at v=%.1f m/s (bin %d) ---\n', targetV, targetBin);
slice = rdDB(:, targetBin);
[sliceSort, sliceIdx] = sort(slice(rMin:rMax), 'descend');
fprintf('  Top 5 ranges: ');
for i = 1:5
    ri = rMin + sliceIdx(i) - 1;
    fprintf('R=%.2fm(%.1fdB) ', r_axis(ri), sliceSort(i));
end
fprintf('\n');

r2m = round(2.0/para.dr);
fprintf('  R=2m: v=0 dB=%.1f, v=1m/s dB=%.1f, v=-1m/s dB=%.1f\n', ...
    rdDB(r2m, nLoops/2+1), rdDB(r2m, targetBin), rdDB(r2m, nLoops/2+1-round(1/0.055)));

%% Analysis 5: 16RX phase check
fprintf('\n--- Angle estimation diagnostic ---\n');
tx0_chirps = [3, 6, 9, 12];
rx16_az = zeros(16, 1);
for d_idx = 1:4
    rx16_az((d_idx-1)*nRX+1 : d_idx*nRX) = rd_cube{tx0_chirps(d_idx)}(:, r2m, targetBin);
end
amps = abs(rx16_az);
phases = angle(rx16_az);
fprintf('  R=2m v=1m/s 16RX amp: mean=%.2e min=%.2e max=%.2e\n', mean(amps), min(amps), max(amps));
fprintf('  Phase(deg): ');
fprintf('%.0f ', rad2deg(phases));
fprintf('\n');

N_PAD = 256;
az_spec = fftshift(abs(fft(rx16_az, N_PAD)));
[~, pk] = max(az_spec);
med = median(az_spec);
sinAz_axis = linspace(-1, 1, N_PAD);
fprintf('  Angle peak/median=%.2f peak at %.1f deg\n', az_spec(pk)/med, asind(sinAz_axis(pk)));

%% RD figure
outDir = fullfile(script_dir, '..', 'figures');
if ~exist(outDir,'dir'), mkdir(outDir); end

figure('Color','w','Position',[50,50,900,600], 'Visible','off');
imagesc(v_axis_dc0, r_axis, rdDB);
set(gca,'YDir','normal');
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title(sprintf('RD Spectrum (Frame %d, no MTI)', frameIdx));
colormap jet; colorbar; grid on;
xlim([-v_max, v_max]); ylim([0.5, 6]);
clim([prctile(rdDB(:), 30), prctile(rdDB(:), 99.5)]);
hold on;
plot([1,1], [0.5,6], 'w--', 'LineWidth',1);
plot([-1,-1], [0.5,6], 'w--', 'LineWidth',1);
plot([0,0], [0.5,6], 'k-', 'LineWidth',2);
hold off;
exportgraphics(gcf, fullfile(outDir, 'diagnose_RD.png'), 'Resolution', 150);
close;

fprintf('\n===== Diagnostic complete =====\n');
