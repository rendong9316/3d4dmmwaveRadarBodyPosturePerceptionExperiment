%% main_4d.m
%% 4D级联雷达 四维时变谱图: RTM, DTM, ATM, ETM
%% 方法: 帧内FFT + 全距离累积 + 时域背景归一化 (不用STFT)
%% 从四张图可反推人体运动过程: 远近/快慢/左右/上下
clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');
scenes = dir(dataRoot);
scenes = scenes([scenes.isdir] & ~ismember({scenes.name}, {'.', '..'}));
sceneType = 'walk';  sceneIdx = 1;  % bend/boxing/walk/run/jump/...
targetScene = sprintf('CCdata_%s_%04d', sceneType, sceneIdx);
if exist(fullfile(dataRoot, targetScene), 'dir')
    scenario = targetScene;
else
    scenario = scenes(1).name;
end
fprintf('场景: %s\n', scenario);

nRange   = para.ADCSamples;       % 256
nLoops   = para.numLoops;         % 64
chirpsPC = para.chirpsPerCycle;   % 12
nRX = para.numRXPerDevice;        % 4
nDev = para.numDevices;           % 4

range_win = hanning(nRange);
dopp_win  = hanning(nLoops);
vmax = para.lambda / (4 * para.Chirptime * chirpsPC);
range_axis = (0:nRange-1) * para.dr;
dopp_axis  = linspace(-vmax, vmax, nLoops);
fprintf('  vmax=%.2f m/s, dr=%.2f cm\n', vmax, para.dr*100);

r_min = round(0.5 / para.dr) + 1;
r_max = round(5.0 / para.dr);
zb = floor(nLoops/2) + 1;
non_dc = [1:zb-1, zb+1:nLoops];  % 非DC bin索引

% ============ Phase 1: Master -> RTM + DTM ============
fprintf('Phase 1: RTM + DTM (Master, 全帧)...\n');
masterBin = fullfile(dataRoot, scenario, 'master_0000_data.bin');
adcMaster = read4DRawData(masterBin, para);
nFrames = size(adcMaster, 4);
time_axis = (0:nFrames-1) * para.Frameinter;

rt_map = zeros(nRange, nFrames);
dt_map = zeros(nLoops, nFrames);

for f = 1:nFrames
    frame = squeeze(adcMaster(:, :, :, f));

    % Range FFT + TDM解复用
    rfft = fft(frame .* range_win.', [], 2);
    rd = reshape(rfft, nRX, nRange, chirpsPC, nLoops);
    rd = rd - mean(rd, 4);  % MTI

    % Doppler FFT
    rd = fft(rd .* reshape(dopp_win, 1,1,1,nLoops), [], 4);
    rd = fftshift(rd, 4);
    rd_pwr = squeeze(mean(sum(abs(rd).^2, 1), 3));  % [256, 64]

    % RTM: 非DC Doppler能量沿距离的分布 (运动在哪里)
    rt_map(:, f) = sum(rd_pwr(:, non_dc), 2);

    % DTM: 所有距离bin的Doppler能量累加 (运动有多快)
    dt_map(:, f) = sum(rd_pwr(r_min:r_max, :), 1);
end

% ---- 时域背景归一化 (10分位数作为背景, 比值转dB) ----
rt_bg = prctile(rt_map, 10, 2);
rt_map = 10*log10(rt_map ./ (rt_bg + 1e-6));
dt_bg = prctile(dt_map, 10, 2);
dt_map = 10*log10(dt_map ./ (dt_bg + 1e-6));

% ============ Phase 2: 全部4设备 -> ATM + ETM ============
fprintf('Phase 2: ATM + ETM (全部4设备, 每2帧, 全距离累积)...\n');
devices = {'master', 'slave1', 'slave2', 'slave3'};
devAll = cell(nDev, 1);
for d = 1:nDev
    devBin = fullfile(dataRoot, scenario, sprintf('%s_0000_data.bin', devices{d}));
    devAll{d} = read4DRawData(devBin, para);
end

n_ele = 256;  n_azi = 256;
ele_axis = asind(linspace(-1, 1, n_ele));
azi_axis = asind(linspace(-1, 1, n_azi));

at_step = 2;
at_frames = 1:at_step:nFrames;
nAT = length(at_frames);
at_map = zeros(n_azi, nAT);
et_map = zeros(n_ele, nAT);

tx_chirp = para.chirpsPerCycle;  % Dev1 TX0

for fi = 1:nAT
    f = at_frames(fi);

    % 预计算每设备的MTI数据 (整帧一次)
    rd_mti = cell(nDev, 1);
    for d = 1:nDev
        frm = squeeze(devAll{d}(:, :, :, f));
        rfft_d = fft(frm .* range_win.', [], 2);
        rfft_d = reshape(rfft_d, nRX, nRange, chirpsPC, nLoops);
        rfft_d = rfft_d - mean(rfft_d, 4);
        rd_mti{d} = rfft_d;  % [4, 256, 12, 64]
    end

    % 跨所有ROI距离bin累积角度谱
    spec_acc = zeros(n_ele, n_azi);
    for r = r_min:r_max
        rx16 = zeros(para.totalRX, 1);
        for d = 1:nDev
            % 该距离bin, TX0 chirp, 所有loop的均值
            rx16((d-1)*nRX+1 : d*nRX) = ...
                squeeze(mean(rd_mti{d}(:, r, tx_chirp, :), 4));
        end
        spec2d = fftshift(fft2(reshape(rx16, nRX, nDev).', n_ele, n_azi));
        spec_acc = spec_acc + abs(spec2d).^2;
    end

    at_map(:, fi) = sum(spec_acc, 1);
    et_map(:, fi) = sum(spec_acc, 2);
end

% ---- 时域背景归一化 ----
at_bg = prctile(at_map, 10, 2);
at_map = 10*log10(at_map ./ (at_bg + 1e-6));
et_bg = prctile(et_map, 10, 2);
et_map = 10*log10(et_map ./ (et_bg + 1e-6));

at_time = (at_frames-1) * para.Frameinter;

% ============ 诊断 ============
fprintf('  RTM: [%.1f, %.1f] dB, std=%.1f\n', min(rt_map(:)), max(rt_map(:)), std(rt_map(:)));
fprintf('  DTM: [%.1f, %.1f] dB, std=%.1f\n', min(dt_map(:)), max(dt_map(:)), std(dt_map(:)));
fprintf('  ATM: [%.1f, %.1f] dB, std=%.1f\n', min(at_map(:)), max(at_map(:)), std(at_map(:)));
fprintf('  ETM: [%.1f, %.1f] dB, std=%.1f\n', min(et_map(:)), max(et_map(:)), std(et_map(:)));

% ============ 画图 ============
set_clim = @(m) clim([max(prctile(m(:),3), -15), min(prctile(m(:),97), 20)]);

figure('Color','w', 'Name','RTM');
imagesc(time_axis, range_axis, rt_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('距离 (m)');
title(sprintf('RTM 距离-时间谱 — %s', scenario));
colormap jet; colorbar; grid on;
ylim([r_min*para.dr, r_max*para.dr]); set_clim(rt_map);

figure('Color','w', 'Name','DTM');
imagesc(time_axis, dopp_axis, dt_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('速度 (m/s)');
title(sprintf('DTM 多普勒-时间谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(dt_map);

figure('Color','w', 'Name','ATM');
imagesc(at_time, azi_axis, at_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('方位角 (度)');
title(sprintf('ATM 方位角-时间谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(at_map);

figure('Color','w', 'Name','ETM');
imagesc(at_time, ele_axis, et_map); set(gca,'YDir','normal');
xlabel('时间 (s)'); ylabel('俯仰角 (度)');
title(sprintf('ETM 俯仰角-时间谱 — %s', scenario));
colormap jet; colorbar; grid on; set_clim(et_map);

fprintf('RTM / DTM / ATM / ETM 全部完成!\n');

% 导出
outDir = fullfile(script_dir, '..', 'figures');
if ~exist(outDir, 'dir'), mkdir(outDir); end
matPath = fullfile(outDir, [scenario '_4Dmaps.mat']);
save(matPath, 'rt_map', 'dt_map', 'at_map', 'et_map', ...
    'time_axis', 'at_time', 'range_axis', 'dopp_axis', ...
    'azi_axis', 'ele_axis', 'para', 'scenario');
fprintf('  数据: %s\n', matPath);

figs = findobj('Type', 'figure');
for i = 1:length(figs)
    fname = get(figs(i), 'Name');
    if isempty(fname), fname = sprintf('fig%d', i); end
    print(figs(i), fullfile(outDir, [scenario '_' fname '.png']), '-dpng', '-r150');
    fprintf('  保存: %s_%s.png\n', scenario, fname);
end
