%% Quick test: process one scene and verify
clear; close all; clc;
script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '4Dproject'));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

nSamples = para.ADCSamples;
nLoops   = para.numLoops;
nChirps  = para.chirpsPerCycle;
nRX      = para.numRXPerDevice;
nChirpsPerFrame = nChirps * nLoops;

range_win = hanning(nSamples);
dopp_win  = hanning(nLoops);
r_min = round(0.5 / para.dr) + 1;
r_max = round(5.0 / para.dr);
zb = floor(nLoops/2) + 1;
non_dc = [1:zb-1, zb+1:nLoops];

% Test walk_0001
scenario = 'CCdata_walk_0001';
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

fprintf('Scene: %s, Frames: %d, Shape: [%d %d %d %d %d]\n', scenario, nFrames, size(rawData));

dt_map = zeros(nLoops, nFrames);
for f = 1:nFrames
    frame = squeeze(rawData(:, :, :, :, f));
    rfft = fft(frame .* range_win.', [], 2);
    rd_mti = rfft - mean(rfft, 4);
    rd = fft(rd_mti .* reshape(dopp_win, 1,1,1,nLoops), [], 4);
    rd = fftshift(rd, 4);
    pwr_rd = squeeze(mean(sum(abs(rd).^2, 1), 3));
    dt_map(:, f) = max(pwr_rd(r_min:r_max, :), [], 1);
end

bg = prctile(dt_map, 10, 2);
dt_map_dB = 10*log10(dt_map ./ (bg + 1e-6));
dt_map_dB = max(min(dt_map_dB, 40), -30);

% resize test
IMG_SIZE = 64;
[nr, nc] = size(dt_map_dB);
[xo, yo] = meshgrid(linspace(1, nc, IMG_SIZE), linspace(1, nr, IMG_SIZE));
[xi, yi] = meshgrid(1:nc, 1:nr);
dt_resized = interp2(xi, yi, dt_map_dB, xo, yo, 'linear');

fprintf('DT map: [%d x %d] -> resized: [%d x %d]\n', nc, nr, size(dt_resized));
fprintf('DT dB range: [%.1f, %.1f]\n', min(dt_map_dB(:)), max(dt_map_dB(:)));

figure('Visible','off');
imagesc(dt_resized); colormap jet; colorbar; title('Resized DT Map (64x64)');
saveas(gcf, fullfile(script_dir, 'test_dt_sample.png'));
fprintf('Test image saved!\n');
disp('=== TEST PASSED ===');
