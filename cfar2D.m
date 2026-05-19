function detections = cfar2D(rd_power, guard_r, guard_d, train_r, train_d, thresh_factor)
%CFAR2D 二维单元平均恒虚警率检测（CA-CFAR）
%
%   在Range-Doppler功率谱上逐像素滑动窗口，对该窗口的
%   训练单元取平均作为噪声功率估计；保护单元隔离目标能量
%   以避免目标自掩蔽；若当前检测单元功率 > 噪声×阈值因子，
%   则判为目标。
%
%   输入:
%     rd_power      — RD功率谱 [n_range, n_doppler] 线性值（非dB）
%     guard_r, guard_d  — 保护单元半宽 [距离, 多普勒] 默认 4
%     train_r, train_d  — 训练单元半宽 [距离, 多普勒] 默认 8
%     thresh_factor     — 阈值因子（线性），默认 5.0 (≈7dB)
%
%   输出:
%     detections — 二值检测掩码，与 rd_power 同尺寸
%
%   窗口结构示意:
%     ┌─────────────┐
%     │ T T T T T │  T = 训练单元（计算噪声均值）
%     │ T G G G T │  G = 保护单元（隔离目标）
%     │ T G C G T │  C = 当前检测单元 CUT
%     │ T G G G T │
%     │ T T T T T │
%     └─────────────┘
%
%   示例:
%     rd_power = squeeze(mean(abs(rd_data).^2, 2));
%     det = cfar2D(rd_power, 4, 4, 8, 8, 5.0);

    if nargin < 3
        guard_r = 4;  guard_d = 4;
    end
    if nargin < 5
        train_r = 8;  train_d = 8;
    end
    if nargin < 6
        thresh_factor = 5.0;
    end

    % 构建CFAR卷积核
    kernel_r = 2*train_r + 2*guard_r + 1;
    kernel_d = 2*train_d + 2*guard_d + 1;
    kernel = ones(kernel_r, kernel_d);

    % 将保护单元 + CUT 区域置零（不参与噪声估计）
    r_start = train_r + 1;
    r_end   = train_r + 2*guard_r + 1;
    d_start = train_d + 1;
    d_end   = train_d + 2*guard_d + 1;
    kernel(r_start:r_end, d_start:d_end) = 0;

    n_train = sum(kernel(:));

    % 卷积求和 → 每个像素的邻域训练单元噪声总功率
    noise_sum = conv2(rd_power, kernel, 'same');

    % 噪声平均功率
    noise_mean = noise_sum / n_train;

    % 自适应阈值
    threshold = noise_mean * thresh_factor;

    % 检测：当前功率 > 局部阈值
    detections = rd_power > threshold;

    % 排除边缘区域（卷积核未完全覆盖，噪声估计不可靠）
    edge_r = train_r + guard_r;
    edge_d = train_d + guard_d;
    detections(1:edge_r, :) = false;
    detections(end-edge_r+1:end, :) = false;
    detections(:, 1:edge_d) = false;
    detections(:, end-edge_d+1:end) = false;

end
