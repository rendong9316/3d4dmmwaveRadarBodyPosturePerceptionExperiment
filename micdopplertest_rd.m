%% 微多普勒谱图批量演示（简化增强版）
% 保留：
% 1. MTI
% 2. strongest bin 搜索
% 3. 多bin融合
% 4. 轻量异常值抑制
% 5. STFT

clc; % 清空命令行窗口
close all; % 关闭所有打开的图形窗口
%% =====================================================
%% 路径配置
%% =====================================================
script_dir = fileparts(mfilename('fullpath')); % 获取当前脚本所在目录的完整路径
addpath(fullfile(script_dir,'3D雷达信号数据读取参考代码')); % 添加数据读取函数的路径
DATA_DIR = fullfile(script_dir,'DATA'); % 设置数据存放的主目录
FILE_IDX = 30; % 选择每个场景中的第几个bin文件进行处理
%% =====================================================
%% 场景定义（英文名和中文显示名）
%% =====================================================
scenarios = {
    'bend_0.8m',    'bend_3m', ...
    'diedao_0.8m',  'qianshuai_3m', ...
    'jumpup_0.8m',  'jumpup_3m', ...
    'run_0.8m',     'run_3m', ...
    'sit_0.8m',     'sit_3m', ...
    'stand_0.8m',   'stand_3m', ...
    'swing_0.8m',   'swing_3m', ...
    'walk',         'white'
};
scenarios_cn = {
    '弯腰0.8m','弯腰3m', ...
    '跌倒0.8m','前摔3m', ...
    '跳跃0.8m','跳跃3m', ...
    '跑步0.8m','跑步3m', ...
    '静坐0.8m','静坐3m', ...
    '站立0.8m','站立3m', ...
    '摆臂0.8m','摆臂3m', ...
    '走路','空房间'
};
%% =====================================================
%% 绘图布局参数（一行4个子图，自动计算行数）
%% =====================================================
n_scene = length(scenarios); % 场景总数
n_cols = 4; % 每行显示4个子图
n_rows = ceil(n_scene / n_cols); % 计算需要的行数（向上取整）
figure('Name','微多普勒谱图（简化增强版）',...
       'Position',[30 30 1600 750]); % 创建大图窗，设置名称和位置大小
%% =====================================================
%% 主循环：遍历每个场景进行处理
%% =====================================================
for s = 1:n_scene
    %% =================================================
    %% 构建当前场景的完整路径
    %% =================================================
    scene_path = fullfile(DATA_DIR,scenarios{s});
    %% ===== 查找LogFile文件（包含雷达参数） =====
    logfiles = dir(fullfile(scene_path,'*_LogFile.txt')); % 搜索该场景下的LogFile文件
    if isempty(logfiles) % 如果没有找到LogFile，则跳过该场景
        fprintf('[%2d] %-16s 无LogFile\n',s,scenarios{s});
        continue;
    end
    logfile = fullfile(scene_path,logfiles(1).name); % 取第一个LogFile的完整路径
    para = readPara(logfile); % 从LogFile中读取雷达参数，存入结构体para
    %% =================================================
    %% 寻找该场景下的所有bin数据文件（纯数字命名的.bin文件）
    %% =================================================
    bins = dir(fullfile(scene_path,'*.bin')); % 列出所有.bin文件
    bin_names = {};
    for k = 1:length(bins)
        [~,name,~] = fileparts(bins(k).name); % 获取不带扩展名的文件名
        if ~isempty(regexp(name,'^\d+$','once')) % 如果文件名全是数字，则认为是数据文件
            bin_names{end+1} = bins(k).name;
        end
    end
    bin_names = sort_nat(bin_names); % 按文件名中的数字从小到大排序
    if length(bin_names) < FILE_IDX % 如果该场景的文件数量不足FILE_IDX，则跳过
        fprintf('[%2d] %-16s bin数量不足\n',s,scenarios{s});
        continue;
    end
    bin_file = fullfile(scene_path,bin_names{FILE_IDX}); % 选择第FILE_IDX个bin文件
    fprintf('[%2d] %-16s 处理中...\n',s,scenarios{s});
    %% =================================================
    %% 读取ADC原始数据
    %% =================================================
    [adcData,para] = readRawData(bin_file,para); % adcData: [距离门, 接收天线, 啁啾, 帧]
    n_range  = para.ADCSamples; % 距离门数量（每个啁啾的采样点数）
    n_frames = size(adcData,4); % 总帧数
    %% =================================================
    %% 计算最大不模糊速度 v_max
    %% =================================================
    v_max = para.lambda / (4 * para.Chirptime * para.GroupNum);
    %% =================================================
    %% 根据场景名称中的距离信息估算目标所在距离门索引（用于局部搜索）
    %% =================================================
    if contains(scenarios{s},'3m') % 如果场景名包含'3m'，则认为目标在3米处
        target_dist = 3.0;
    else
        target_dist = 0.8; % 否则认为在0.8米处
    end
    target_bin = round(target_dist / para.dr); % 距离门索引 = 距离 / 距离分辨率
    %% =================================================
    %% 距离FFT窗函数（汉宁窗，减少旁瓣）
    %% =================================================
    range_win = hanning(n_range);
    %% =================================================
    %% 第一阶段：搜索最强能量距离门（strongest bin）
    %% 方法：对所有帧的能量进行累加，然后在目标附近±6门内找峰值
    %% =================================================
    energy_acc = zeros(n_range,1); % 累加所有帧的能量
    for f = 1:n_frames
        frame = squeeze(adcData(:,:,:,f)); % 提取第f帧 [距离门, 天线, 啁啾]
        sig = squeeze(frame(:,1,:)); % 只使用RX0天线 [距离门, 啁啾]
        sig = sig - mean(sig,2); % MTI（动目标指示）：减去每个距离门的均值，抑制静止杂波
        range_fft = fft(sig .* range_win, [], 1); % 对每个啁啾做距离FFT，结果 [距离门, 啁啾]
        energy_acc = energy_acc + mean(abs(range_fft).^2, 2); % 累加该帧所有啁啾的平均能量
    end
    % 局部搜索范围：目标估计距离门 ±6，并限制在有效距离门内
    search_range = max(1, target_bin-6) : min(n_range, target_bin+6);
    [~, idx] = max(energy_acc(search_range)); % 在该范围内找峰值索引（局部）
    best_bin = search_range(idx); % 得到最终的最强能量距离门索引
    fprintf('    strongest bin = %d\n', best_bin);
    %% =================================================
    %% 第二阶段：提取慢时间信号（跨帧的相位历史）
    %% 方法：在最强距离门附近几个门内进行多bin融合，提高鲁棒性
    %% =================================================
    all_slow = []; % 用于拼接所有帧的慢时间信号
    bin_span = -3:3; % 融合范围：最强距离门 ±3
    for f = 1:n_frames
        frame = squeeze(adcData(:,:,:,f));
        sig = squeeze(frame(:,1,:)); % 取RX0
        sig = sig - mean(sig,2); % MTI
        range_fft = fft(sig .* range_win, [], 1); % 距离FFT
        %% =================================================
        %% 多bin融合：对最强门附近多个距离门的FFT结果求和
        %% 作用：当目标微动导致距离门间能量扩散时，仍能有效捕获信号
        %% =================================================
        bins_sel = best_bin + bin_span; % 选择的距离门索引
        bins_sel = bins_sel(bins_sel >= 1 & bins_sel <= n_range); % 剔除超出范围的门
        slow_sig = sum(range_fft(bins_sel,:), 1); % 沿距离门维度求和，结果 [1, 啁啾]
        slow_sig = slow_sig(:); % 转为列向量，每个元素对应一个啁啾
        %% =================================================
        %% 轻量级尖峰抑制：消除大幅噪声（如突然的干扰）
        %% 方法：幅度超过中位数5倍的置零
        %% =================================================
        amp = abs(slow_sig);
        th = median(amp) * 5; % 阈值：中位数的5倍
        slow_sig(amp > th) = 0;
        %% =================================================
        %% 拼接：将每一帧的慢时间信号追加到总序列中
        %% =================================================
        all_slow = [all_slow; slow_sig];
    end
    %% =================================================
    %% 长慢时间序列（总啁啾序列）
    %% =================================================
    signal = all_slow(:); % 确保为列向量
    signal = detrend(signal); % 去除线性趋势，消除低频漂移
    %% =================================================
    %% STFT（短时傅里叶变换）生成时频谱图
    %% =================================================
    nperseg  = 256;   % 每个窗口长度（啁啾数）
    noverlap = 220;   % 重叠点数（产生时间平滑）
    nfft     = 512;   % FFT点数（频率分辨率更高）
    fs_chirp = 1 / para.Chirptime; % 慢时间采样率 = 啁啾重复频率
    [S, F, T] = spectrogram( ...
        signal, ...
        hann(nperseg), ... % 汉宁窗，减少频谱泄漏
        noverlap, ...
        nfft, ...
        fs_chirp, ...
        'centered'); % 'centered'使频率轴关于0对称（速度可正可负）
    %% =================================================
    %% 转换为dB功率谱
    %% =================================================
    S_db = 20 * log10(abs(S) + 1e-6); % 加小量防止log(0)
    %% =================================================
    %% 将频率轴转换为速度轴（v = (f * lambda)/2）
    %% =================================================
    vel_axis = F * para.lambda / 2;
    vel_mask = abs(vel_axis) <= v_max; % 只显示不超过最大不模糊速度的范围
    vel_axis = vel_axis(vel_mask);
    S_db = S_db(vel_mask, :);
    %% =================================================
    %% 绘制当前场景的微多普勒谱图
    %% =================================================
    subplot(n_rows, n_cols, s); % 定位到第s个子图
    imagesc(T, vel_axis, S_db); % 显示谱图，T为时间轴，vel_axis为速度轴
    axis xy; % 使y轴方向从下到上（速度正方向向上）
    xlabel('时间 (s)');
    ylabel('速度 (m/s)');
    title(scenarios_cn{s}, 'FontSize', 9);
    colormap(gca, 'jet'); % 使用jet颜色映射
    caxis([60 100]); % 固定颜色范围60~100 dB，便于不同场景间对比
    ylim([-v_max, v_max]); % 速度范围限制在±v_max
    xlim([T(1), T(end)]); % 时间范围与STFT输出一致
end
%% =====================================================
%% 添加大标题
%% =====================================================
sgtitle('人体动作微多普勒谱图（简化增强版）',...
    'FontSize',14,...
    'FontWeight','bold');
fprintf('\n全部完成！\n');
%% =====================================================
%% 自然排序函数：将类似 "1.bin", "2.bin", "10.bin" 的文件名按数值排序
%% =====================================================
function sorted = sort_nat(cell_arr)
nums = zeros(length(cell_arr), 1); % 预分配数值数组
for k = 1:length(cell_arr)
    [~, name, ~] = fileparts(cell_arr{k}); % 提取文件名（不含扩展名）
    nums(k) = str2double(name); % 将文件名转为数值
end
[~, idx] = sort(nums); % 获取排序后的索引
sorted = cell_arr(idx); % 按索引重排原cell数组
end