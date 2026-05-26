% 4D级联雷达数据处理 - 精简版
clear; close all; script_dir = fileparts(mfilename('fullpath')); addpath(genpath(script_dir));
% 读取雷达参数
para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
% 选择第一个场景
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');
scenes = dir(dataRoot); scenes = scenes([scenes.isdir] & ~ismember({scenes.name}, {'.', '..'}));
scenario = scenes(1).name;
% 读取Master原始数据 (第1帧)
adcMaster = read4DRawData(fullfile(dataRoot, scenario, 'master_0000_data.bin'), para);
[n_rx, n_range, ~, ~] = size(adcMaster);
frame = squeeze(adcMaster(:,:,:,1));
% Range FFT (汉宁窗)
range_fft = fft(frame .* hanning(n_range).', [], 2);
% TDM-MIMO解复用: 维度变为 [RX, Range, TX, Doppler]
rd = reshape(range_fft, n_rx, n_range, para.chirpsPerCycle, para.numLoops);
% 慢时间维去均值 (静态杂波抑制)
rd = rd - mean(rd, 4);
% Doppler FFT (汉宁窗 + fftshift)
rd = fft(rd .* reshape(hanning(para.numLoops),1,1,1,para.numLoops), [], 4);
rd = fftshift(rd, 4);
% 零速通道抑制 (3-bin)
zb = floor(para.numLoops/2) + 1;
rd(:,:,:,zb-1:zb+1) = 0;
% 非相干积累: RX求和, TX平均, 转dB
rd_pwr = squeeze(mean(sum(abs(rd).^2, 1), 3));
rd_map = 10*log10(rd_pwr + eps);
% 坐标轴
dr = para.dr;
vmax = para.lambda / (4 * para.Chirptime * para.chirpsPerCycle);
range_axis = (0:n_range-1) * dr;
doppler_axis = linspace(-vmax, vmax, para.numLoops);
% 仅显示0-80m，动态色标
max_r = min(80, n_range);
range_show = range_axis(1:max_r);
rd_show = rd_map(1:max_r, :);
rd_max_val = max(rd_show(:));
% 绘制RD谱
figure('Color','w'); imagesc(doppler_axis, range_show, rd_show); set(gca,'YDir','normal');
xlabel('速度 (m/s)'); ylabel('距离 (m)'); title(sprintf('4D Radar RD Map — %s', scenario));
colormap jet; colorbar; caxis([rd_max_val-35, rd_max_val]); grid on;
fprintf('场景 %s RD谱完成\n', scenario);

%% ===== DT / RT / AT 特征谱图 =====

% ----- 重读全部帧（仅 master，用于 RT/DT）-----
fprintf('处理全部帧用于 RT/DT ...\n');
adcFull = read4DRawData(fullfile(dataRoot, scenario, 'master_0000_data.bin'), para);
[n_rx, n_range, ~, n_frames] = size(adcFull);
range_win = hanning(n_range);
dopp_win  = hanning(para.numLoops);

rt_map = zeros(n_range, n_frames);      % RT: 距离 × 帧
dt_map = zeros(para.numLoops, n_frames); % DT: 速度 × 帧

% 找目标距离bin（能量最强的帧的中间几帧取平均）
mid_f = round(n_frames/2);
frame_mid = squeeze(adcFull(:, :, :, mid_f));
rfft_mid  = fft(frame_mid .* range_win.', [], 2);
rfft_mid  = mean(abs(rfft_mid).^2, [1, 3]);  % RX+Chirp平均
[~, target_bin] = max(rfft_mid(1:round(5/para.dr)));  % 0-5m内找最强

for f = 1:n_frames
    frame = squeeze(adcFull(:, :, :, f));
    % Range FFT
    rfft = fft(frame .* range_win.', [], 2);
    % RT列：距离剖面（RX+Chirp 平均功率）
    rt_map(:, f) = squeeze(mean(abs(rfft).^2, [1, 3]));

    % TDM解复用 + Doppler
    rd = reshape(rfft, n_rx, n_range, para.chirpsPerCycle, para.numLoops);
    rd = rd - mean(rd, 4);
    rd = fft(rd .* reshape(dopp_win, 1,1,1,para.numLoops), [], 4);
    rd = fftshift(rd, 4);
    rd(:, :, :, zb-1:zb+1) = 0;
    rd_pwr = squeeze(mean(sum(abs(rd).^2, 1), 3));  % [256, 64]
    % DT列：目标bin的多普勒剖面
    dt_map(:, f) = rd_pwr(target_bin, :);
end

rt_map = 10*log10(rt_map + eps);
dt_map = 10*log10(dt_map + eps);
time_axis = (0:n_frames-1) * para.Frameinter;

% ----- AT谱：需要全部4设备，只取每隔step帧加速 -----
fprintf('读取4设备用于AT ...\n');
n_angle = 256;
at_step = 4;  % 每4帧取1帧加速
at_frames = 1:at_step:n_frames;
n_at = length(at_frames);
at_map = zeros(n_angle, n_at);
devices = {'master', 'slave1', 'slave2', 'slave3'};
% 一次性读全部4设备（内存: 4×237MB≈950MB）
devAll = cell(4,1);
for d = 1:4
    devBin = fullfile(dataRoot, scenario, sprintf('%s_0000_data.bin', devices{d}));
    devAll{d} = read4DRawData(devBin, para);  % [4, 256, 768, 79]
end

for fi = 1:n_at
    f = at_frames(fi);
    % 4设备 × 4RX = 16通道，在目标距离bin处取复数
    rx_val = zeros(16, 1);
    for d = 1:4
        devFrame = squeeze(devAll{d}(:, :, 1, f));       % [4, 256]
        rfft_dev = fft(devFrame .* range_win.', [], 2);   % [4, 256]
        rx_val((d-1)*4+1 : d*4) = rfft_dev(:, target_bin);
    end
    at_spec = fftshift(abs(fft(rx_val, n_angle)));
    at_map(:, fi) = at_spec.^2;
end
at_map = 10*log10(at_map + eps);
angle_axis = asind(linspace(-1, 1, n_angle));
at_time = (at_frames-1) * para.Frameinter;

% ===== 画图 =====
% 自适应色标辅助函数：取数据的 [5%, 99.5%] 分位数作为色域
set_clim = @(m) clim([prctile(m(:),5), prctile(m(:),99.5)]);

% RT谱
figure('Color','w');
imagesc(time_axis, range_axis, rt_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('距离 (m)'); title(sprintf('4D RT 距离-时间谱 — %s', scenario));
colormap jet; colorbar; ylim([0, 5]); grid on; set_clim(rt_map);

% DT谱
figure('Color','w');
dopp_axis = linspace(-vmax, vmax, para.numLoops);
imagesc(time_axis, dopp_axis, dt_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('速度 (m/s)'); title(sprintf('4D DT 多普勒-时间谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(dt_map);

% AT谱
figure('Color','w');
imagesc(at_time, angle_axis, at_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('方位角 (度)'); title(sprintf('4D AT 方位角-时间谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(at_map);

fprintf('RT/DT/AT 全部完成!\n');

%% ===== 微多普勒谱（STFT时频分析，参照3D micdopplertest_rd.m）=====
fprintf('生成微多普勒谱 ...\n');

% 取master设备TX0的chirp序列（TDM周期中第12个chirp，即索引12）
tx_idx = para.chirpsPerCycle;  % master TX0 = chirp 12 (最后一个)

% 两遍扫描：①最强bin搜索 ②多bin融合提取慢时间
% 第一遍：累加能量找最强距离bin
energy_acc = zeros(n_range, 1);
for f = 1:n_frames
    frame = squeeze(adcFull(:, :, :, f));
    rfft = fft(frame .* range_win.', [], 2);
    rfft = reshape(rfft, n_rx, n_range, para.chirpsPerCycle, para.numLoops);
    % RX0, TX0 chirp, 所有loop的能量
    energy_acc = energy_acc + squeeze(mean(abs(rfft(1, :, tx_idx, :)).^2, 4));
end
[~, best_bin] = max(energy_acc(1:round(5/para.dr)));

% 第二遍：多bin融合(±3)提取慢时间，拼接所有帧
all_slow = [];
bin_span = -3:3;
for f = 1:n_frames
    frame = squeeze(adcFull(:, :, :, f));
    rfft = fft(frame .* range_win.', [], 2);
    rfft = reshape(rfft, n_rx, n_range, para.chirpsPerCycle, para.numLoops);
    rfft = rfft - mean(rfft, 4);  % MTI: 慢时间维去均值

    bins_sel = best_bin + bin_span;
    bins_sel = bins_sel(bins_sel >= 1 & bins_sel <= n_range);
    % 多bin + 多RX 求和融合
    slow = squeeze(sum(sum(rfft(:, bins_sel, tx_idx, :), 1), 2));  % [64, 1]
    % 尖峰抑制
    amp = abs(slow);
    slow(amp > median(amp)*5) = 0;
    all_slow = [all_slow; slow];
end

signal = detrend(all_slow(:));
fs_slow = 1 / (para.Chirptime * para.chirpsPerCycle);  % TX重复频率

% STFT
nperseg  = 128;
noverlap = 100;
nfft     = 512;
[S, F, T] = spectrogram(signal, hanning(nperseg), noverlap, nfft, fs_slow, 'centered');
S_db = 20 * log10(abs(S) + 1e-6);

% 频率→速度轴
vel_axis = F * para.lambda / 2;
vel_mask = abs(vel_axis) <= vmax;
S_db = S_db(vel_mask, :);
vel_axis = vel_axis(vel_mask);

% 画微多普勒谱
figure('Color','w');
imagesc(T, vel_axis, S_db); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('速度 (m/s)');
title(sprintf('4D 微多普勒谱 — %s (master TX0, bin=%d)', scenario, best_bin));
colormap jet; colorbar; set_clim(S_db); grid on;

fprintf('微多普勒完成!\n');