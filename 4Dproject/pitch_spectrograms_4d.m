%% pitch_spectrograms_4d.m
%% 4D雷达独有谱图：PT(俯仰-时间) RP(距离-俯仰) DP(多普勒-俯仰)
%% 利用4设备级联的俯仰维分辨能力（3D雷达做不到）
clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');
scenes = dir(dataRoot);
scenes = scenes([scenes.isdir] & ~ismember({scenes.name}, {'.', '..'}));
scenario = scenes(1).name;
fprintf('场景: %s\n', scenario);

devices = {'master', 'slave1', 'slave2', 'slave3'};
nDev = 4;
nRXperDev = 4;
nRXtotal = nDev * nRXperDev;  % 16
nRange = para.ADCSamples;      % 256
nLoops = para.numLoops;        % 64
chirpsPC = para.chirpsPerCycle; % 12

% ===== 读全部4设备 =====
fprintf('读取全部4设备 ...\n');
devAll = cell(nDev, 1);
for d = 1:nDev
    devBin = fullfile(dataRoot, scenario, sprintf('%s_0000_data.bin', devices{d}));
    devAll{d} = read4DRawData(devBin, para);  % [4, 256, 768, 79]
end
nFrames = size(devAll{1}, 4);
fprintf('  帧数: %d\n', nFrames);

% ===== 找目标距离bin（中间帧能量最强）=====
midF = round(nFrames/2);
range_win = hanning(nRange);
energy_frm = zeros(nRange, 1);
for d = 1:nDev
    frm = squeeze(devAll{d}(:, :, 1, midF));  % [4, 256]
    rfft = fft(frm .* range_win.', [], 2);
    energy_frm = energy_frm + squeeze(mean(abs(rfft).^2, 1));
end
[~, target_bin] = max(energy_frm(1:round(5/para.dr)));
fprintf('  目标距离bin: %d (%.1fm)\n', target_bin, target_bin*para.dr);

% ===== 俯仰角估计参数 =====
n_ele = 256;  % 俯仰FFT点数
n_azi = 256;  % 方位FFT点数
ele_axis = asind(linspace(-1, 1, n_ele));
azi_axis = asind(linspace(-1, 1, n_azi));

%% ===== PT谱：俯仰角-时间（遍历所有帧）=====
fprintf('生成 PT 谱 ...\n');
pt_step = 2;  % 每2帧取1帧
pt_frames = 1:pt_step:nFrames;
nPT = length(pt_frames);
pt_map = zeros(n_ele, nPT);

for fi = 1:nPT
    f = pt_frames(fi);
    % 16通道在目标距离bin处取复数
    rx16 = zeros(nRXtotal, 1);
    for d = 1:nDev
        frm = squeeze(devAll{d}(:, :, 1, f));  % [4, 256] 取第1个chirp
        rfft = fft(frm .* range_win.', [], 2);
        rx16((d-1)*nRXperDev+1 : d*nRXperDev) = rfft(:, target_bin);
    end
    % 2D FFT: 4设备(垂直) × 4RX(水平) → 俯仰×方位
    spec2d = fftshift(fft2(reshape(rx16, nRXperDev, nDev).', n_ele, n_azi));
    % 沿方位维求和 → 俯仰剖面
    pt_map(:, fi) = sum(abs(spec2d), 2);
end
pt_map = 10*log10(pt_map + eps);
pt_time = (pt_frames-1) * para.Frameinter;

%% ===== RP谱：距离-俯仰（取中间帧）=====
fprintf('生成 RP 谱 ...\n');
rp_map = zeros(n_ele, nRange);
% 取RX0通道的TDM解复用后数据做Range+Doppler处理
frm_master = squeeze(devAll{1}(:,:,:, midF));  % [4, 256, 768]
rfft_m = fft(frm_master .* range_win.', [], 2);  % [4, 256, 768]
rfft_m = reshape(rfft_m, nRXperDev, nRange, chirpsPC, nLoops);
rfft_m = rfft_m - mean(rfft_m, 4);  % MTI

for r = 1:nRange
    % 16通道在该距离bin处：取master + 3 slaves
    rx16 = zeros(nRXtotal, chirpsPC * nLoops);
    for d = 1:nDev
        frm = squeeze(devAll{d}(:, :, :, midF));
        rfft_d = fft(frm .* range_win.', [], 2);
        rfft_d = reshape(rfft_d, nRXperDev, nRange, chirpsPC, nLoops);
        rfft_d = rfft_d - mean(rfft_d, 4);
        rx16((d-1)*nRXperDev+1 : d*nRXperDev, :) = reshape(...
            squeeze(rfft_d(:, r, :, :)), nRXperDev, []);
    end
    % 沿chirp维取平均，得到稳定的16通道值
    rx16_r = mean(rx16, 2);
    spec2d = fftshift(fft2(reshape(rx16_r, nRXperDev, nDev).', n_ele, n_azi));
    rp_map(:, r) = sum(abs(spec2d), 2);
end
rp_map = 10*log10(rp_map + eps);
range_axis = (0:nRange-1) * para.dr;

%% ===== DP谱：多普勒-俯仰（取中间帧，目标bin）=====
fprintf('生成 DP 谱 ...\n');
dp_map = zeros(n_ele, nLoops);
dopp_win = hanning(nLoops);

for l = 1:nLoops
    rx16 = zeros(nRXtotal, 1);
    for d = 1:nDev
        frm = squeeze(devAll{d}(:, :, :, midF));
        rfft_d = fft(frm .* range_win.', [], 2);
        rfft_d = reshape(rfft_d, nRXperDev, nRange, chirpsPC, nLoops);
        rfft_d = rfft_d - mean(rfft_d, 4);
        % 该loop中所有chirp和RX的均值（在目标bin处）
        rx16((d-1)*nRXperDev+1 : d*nRXperDev) = ...
            squeeze(mean(rfft_d(:, target_bin, :, l), 3));
    end
    spec2d = fftshift(fft2(reshape(rx16, nRXperDev, nDev).', n_ele, n_azi));
    dp_map(:, l) = sum(abs(spec2d), 2) .* dopp_win(l)^2;
end
dp_map = 10*log10(dp_map + eps);
vmax = para.lambda / (4 * para.Chirptime * chirpsPC);
dopp_axis = linspace(-vmax, vmax, nLoops);

%% ===== 画图 =====
set_clim = @(m) clim([prctile(m(:),5), prctile(m(:),99.5)]);

% PT谱
figure('Color','w');
imagesc(pt_time, ele_axis, pt_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('俯仰角 (度)');
title(sprintf('4D PT 俯仰-时间谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(pt_map);

% RP谱
figure('Color','w');
imagesc(range_axis, ele_axis, rp_map); set(gca,'YDir','normal');
xlabel('距离 (m)'); ylabel('俯仰角 (度)');
title(sprintf('4D RP 距离-俯仰谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(rp_map);

% DP谱
figure('Color','w');
imagesc(dopp_axis, ele_axis, dp_map); set(gca,'YDir','normal');
xlabel('速度 (m/s)'); ylabel('俯仰角 (度)');
title(sprintf('4D DP 多普勒-俯仰谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(dp_map);

fprintf('PT/RP/DP 全部完成!\n');
