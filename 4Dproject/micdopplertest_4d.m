%% micdopplertest_4d.m
%% 4D雷达微多普勒谱图批量演示 — 每类动作取1个样本
clear; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

jsonCfg = fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json');
para = read4DParam(jsonCfg);
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

% ===== 动作类别（取每类第一个样本）=====
actionTypes = {'bend','boxing','empty','falldown','jump',...
               'liedown','run','sit','stand','swing','walk'};
actionCN = {'弯腰','摆臂','空房间','跌倒','跳跃',...
            '躺下','跑步','静坐','站立','摆手','走路'};
nAct = length(actionTypes);

nCols = 4;
nRows = ceil(nAct / nCols);
figure('Name', '4D微多普勒谱图对比', 'Position', [20, 20, 1600, 900]);

% 预计算共用窗口
range_win = hanning(para.ADCSamples);
nLoops = para.numLoops;
chirpsPerCycle = para.chirpsPerCycle;
tx_idx = chirpsPerCycle;  % master TX0 = 最后一个chirp
vmax = para.lambda / (4 * para.Chirptime * chirpsPerCycle);
fs_slow = 1 / (para.Chirptime * chirpsPerCycle);

for a = 1:nAct
%for a = 1:2
    sceneName = sprintf('CCdata_%s_0001', actionTypes{a});
    scenePath = fullfile(dataRoot, sceneName);
    if ~exist(scenePath, 'dir')
        fprintf('[%2d] %s 目录不存在\n', a, sceneName); continue;
    end

    masterBin = fullfile(scenePath, 'master_0000_data.bin');
    fprintf('[%2d] %s 读取中...', a, actionCN{a});

    adc = read4DRawData(masterBin, para);  % [4, 256, 768, 79]
    [nRX, nRng, ~, nFrm] = size(adc);
    if nFrm < 10, fprintf('帧数不足\n'); continue; end

    % ---- 第一遍：最强bin搜索 ----
    energy = zeros(nRng, 1);
    for f = 1:nFrm
        rfft = fft(squeeze(adc(:,:,:,f)) .* range_win.', [], 2);
        rfft = reshape(rfft, nRX, nRng, chirpsPerCycle, nLoops);
        rfft = rfft - mean(rfft, 4);  % MTI
        energy = energy + squeeze(mean(abs(rfft(1,:,tx_idx,:)).^2, 4));
    end
    [~, best_bin] = max(energy(1:round(5/para.dr)));

    % ---- 第二遍：多bin融合 + STFT ----
    all_slow = [];
    for f = 1:nFrm
        rfft = fft(squeeze(adc(:,:,:,f)) .* range_win.', [], 2);
        rfft = reshape(rfft, nRX, nRng, chirpsPerCycle, nLoops);
        rfft = rfft - mean(rfft, 4);
        bins_sel = best_bin + (-3:3);
        bins_sel = bins_sel(bins_sel>=1 & bins_sel<=nRng);
        slow = squeeze(sum(sum(rfft(:, bins_sel, tx_idx, :), 1), 2));
        slow(abs(slow) > median(abs(slow))*5) = 0;
        all_slow = [all_slow; slow];
    end
    signal = detrend(all_slow(:));

    [S, F, T] = spectrogram(signal, hanning(128), 100, 512, fs_slow, 'centered');
    S_db = 20*log10(abs(S) + 1e-6);
    vel_axis = F * para.lambda / 2;
    mask = abs(vel_axis) <= vmax;
    S_db = S_db(mask, :); vel_axis = vel_axis(mask);

    % ---- 画子图 ----
    subplot(nRows, nCols, a);
    imagesc(T, vel_axis, S_db); set(gca,'YDir','normal');
    xlabel('时间 (s)'); ylabel('速度 (m/s)');
    title(sprintf('%s (bin=%d)', actionCN{a}, best_bin), 'FontSize', 9);
    colormap(gca,'jet');
    clim([80,110]);


    fprintf(' 完成\n');
end

sgtitle('4D级联雷达 — 十一类动作微多普勒谱图对比', 'FontSize', 14, 'FontWeight', 'bold');
fprintf('\n全部完成!\n');
