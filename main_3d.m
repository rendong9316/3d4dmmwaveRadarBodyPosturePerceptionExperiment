%% 3D雷达数据处理主脚本
%% DCA1000 + IWR6843 (60GHz) — Range-Doppler谱分析
clear; clc; close all;

BASE = 'D:\downlowd_cloud\方向2-雷达数据demo\3D';
addpath(fullfile(BASE, '..\3D雷达信号数据读取参考代码'));

% 1. 解析雷达参数
logfile = fullfile(BASE, 'stand_0.8m', '1_LogFile.txt');
para = readPara(logfile);
disp('========== 雷达参数 ==========');
fprintf('载频:            %.1f GHz\n', para.f0/1e9);
fprintf('ADC采样数:       %d\n', para.ADCSamples);
fprintf('采样率:          %.0f kHz\n', para.Fs/1e3);
fprintf('调频斜率:        %.2f MHz/us\n', para.FrequencySlope/1e12);
fprintf('每帧Chirp数:     %d\n', para.ChirpNum);
fprintf('帧数:            %d\n', para.FrameNum);
fprintf('带宽:            %.1f MHz\n', para.BandWidth/1e6);
fprintf('距离分辨率:      %.2f cm\n', para.dr*100);
fprintf('最大距离:        %.1f m\n', para.Fs*3e8/2/para.FrequencySlope);
v_max = para.lambda/4/(para.IDLEtime+para.ENDtime)/para.GroupNum;
fprintf('最大速度:        %.2f m/s\n', v_max);
disp('==============================');

% 2. 读取并处理所有场景
scenarios = {'white', 'stand_0.8m', 'swing_0.8m', 'jumpup_0.8m'};
labels_cn = {
    '噪声（无目标）',
    '静止站立 @0.8m',
    '摆动 @0.8m',
    '跳跃 @0.8m'
};

% 计算坐标轴
range_axis = (0:para.ADCSamples-1)' * para.dr;
vel_axis = linspace(-v_max, v_max, para.ChirpNum);

% 存储中间帧结果
all_rd_mid = cell(4, 1);
all_dop_evo = cell(4, 1);

for s = 1:4
    binfile = fullfile(BASE, scenarios{s}, '1.bin');
    fprintf('\n[处理] %s\n', scenarios{s});
    fprintf('  读取 %s ...\n', binfile);

    [adcData, para] = readRawData(binfile, para);
    fprintf('  数据维度: [%d %d %d %d]\n', size(adcData));

    n_frames = size(adcData, 4);

    % 计算每帧的RD图
    rd_stack = zeros(para.ADCSamples, para.ChirpNum, n_frames);
    for f_idx = 1:n_frames
        % 取一帧 [ADCSamples, numRX, ChirpNum]
        frame = squeeze(adcData(:, :, :, f_idx));

        % 对RX通道取平均
        data_1d = squeeze(mean(frame, 2));  % [ADCSamples, ChirpNum]

        % Range FFT + 加窗
        r_win = hanning(para.ADCSamples);
        range_fft = fft(data_1d .* r_win, [], 1);

        % Doppler FFT + 加窗
        d_win = hanning(para.ChirpNum);
        rd = fft(range_fft .* d_win.', [], 2);
        rd = fftshift(rd, 2);  % 零多普勒居中

        rd_stack(:, :, f_idx) = 20*log10(abs(rd) + 1e-10);
    end

    all_rd_mid{s} = rd_stack(:, :, round(n_frames/2));
    all_dop_evo{s} = rd_stack;
end

% 3. 四场景RD谱对比图
figure('Position', [100, 100, 1200, 900]);
for s = 1:4
    subplot(2, 2, s);
    imagesc(vel_axis, range_axis, all_rd_mid{s});
    set(gca, 'YDir', 'normal');
    caxis([-40, 40]);
    colormap('jet');
    colorbar;
    xlabel('速度 (m/s)');
    ylabel('距离 (m)');
    title(labels_cn{s});
    xlim([-3, 3]);
    ylim([0, 3]);
end
sgtitle('Range-Doppler 谱图对比（中间帧）');
saveas(gcf, fullfile(BASE, '..\RD_comparison.png'));

% 4. 跳跃场景 — 多普勒随时间演变
rd_jump = all_dop_evo{4};
n_frames_jump = size(rd_jump, 3);

% 取0.3~0.8m范围沿距离轴求和，看多普勒随时间变化
range_mask = range_axis >= 0.3 & range_axis <= 1.5;
dop_time = squeeze(mean(rd_jump(range_mask, :, :), 1));  % [Doppler, Frame]

figure('Position', [100, 100, 1200, 450]);

subplot(1, 2, 1);
imagesc(1:n_frames_jump, vel_axis, dop_time);
set(gca, 'YDir', 'normal');
caxis([-40, 40]);
colormap('jet');
colorbar;
xlabel('帧序号');
ylabel('速度 (m/s)');
title('跳跃场景 — 多普勒-时间演变');
ylim([-3, 3]);

% 5. 四场景平均多普勒谱对比
subplot(1, 2, 2);
hold on;
colors = {'k', 'b', 'g', 'r'};
for s = 1:4
    rd_s = all_dop_evo{s};
    range_mask = range_axis >= 0.3 & range_axis <= 1.5;
    dop_profile = squeeze(mean(rd_s(range_mask, :, :), [1, 3]));
    plot(vel_axis, dop_profile, 'Color', colors{s}, 'LineWidth', 1.5);
end
hold off;
xlabel('速度 (m/s)');
ylabel('功率 (dB)');
title('平均多普勒谱对比 (0.3~1.5m)');
legend(labels_cn, 'Location', 'best');
xlim([-3, 3]);
grid on;
saveas(gcf, fullfile(BASE, '..\Doppler_evolution.png'));

fprintf('\n===== 处理完成 =====\n');
