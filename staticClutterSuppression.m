function rd_data = staticClutterSuppression(rd_data, method)
%STATICCLUTTERSUPPRESSION 静态杂波抑制 — 去除多普勒域零速分量
%
%   静态杂波的来源：
%     1. 接收机直流偏置、FFT周期延拓等器件与算法因素
%     2. 墙面、地面、家具等静止目标的强回波
%   这些杂波集中在多普勒域的零速通道（Doppler FFT后的中心bin），
%   会严重压制动态目标的信号，干扰后续CFAR检测。
%
%   本函数实现PPT介绍的三种抑制方法，本质都是构造高通滤波器
%   滤除多普勒为零的直流分量。
%
%   输入:
%     rd_data — Range-Doppler复数数据 [n_range, n_rx, n_doppler]
%               必须是已经过 dopplerFFT + fftshift 的数据
%     method  — 可选，指定抑制方法：
%               'zeroVel'   (默认) 零速通道置零法
%               'MTI'       动目标显示
%               'phaseMean' 相位均值相消法
%
%   输出:
%     rd_data — 杂波抑制后的RD数据，同尺寸 complex
%
%   三种方法详解：
%
%   【方法1: 零速通道置零法 (zeroVel)】
%     原理: 静态目标的多普勒频率 = 0，在fftshift后的RD谱中
%           对应中心的零速通道。直接将该bin置零即可。
%     优点: 最简单直接，计算量最小
%     步骤: rd_data(:, :, zero_bin) = 0
%
%   【方法2: 动目标显示 MTI】
%     原理: 沿Chirp维做一阶差分 y[n] = x[n] - x[n-1]
%           静态目标各Chirp回波相同 → 差分后为零
%     优点: 经典方法，对慢速运动目标保留更好
%     注意: 此方法在Doppler FFT之前使用（本函数仍可处理RD域数据
%           但对RD域效果等价于零速通道附近衰减）
%
%   【方法3: 相位均值相消法 (phaseMean)】
%     原理: 沿Chirp维求复数均值（幅度+相位），再从每个Chirp减去
%           该均值即静态杂波的最佳估计
%     优点: 对静止杂波的估计最精确
%
%   示例:
%     rd_data = dopplerFFT(range_data);
%     rd_data = staticClutterSuppression(rd_data);

    if nargin < 2
        method = 'zeroVel';
    end

    [n_range, n_rx, n_doppler] = size(rd_data);

    switch lower(method)
        case 'zerovel'
            % ===== 零速通道置零法 =====
            % fftshift后，零多普勒位于中心bin
            % n_doppler为奇数时: zero_bin = (n_doppler+1)/2
            % n_doppler为偶数时: zero_bin = n_doppler/2 + 1
            zero_center = floor(n_doppler/2) + 1;
            % 由于有限观测时长+加窗的频谱泄露，静态杂波扩散到
            % 相邻bin，需置零5个bin（速度范围约±0.08 m/s）
            n_zero = 5;
            half_w = floor(n_zero / 2);
            zero_range = (zero_center - half_w) : (zero_center + half_w);
            rd_data(:, :, zero_range) = 0;

        case 'mti'
            % ===== 动目标显示 MTI（单延迟对消器）=====
            % 沿第3维（多普勒维）做一阶差分
            % 由于输入已经是RD域数据，这里沿多普勒维做高通滤波等价
            rd_data(:, :, 2:end) = rd_data(:, :, 2:end) - rd_data(:, :, 1:end-1);
            % 第一列无法差分，置零
            rd_data(:, :, 1) = 0;

        case 'phasemean'
            % ===== 相位均值相消法 =====
            % 沿多普勒维取复数均值，从每个bin减去该均值
            mean_doppler = mean(rd_data, 3);
            rd_data = rd_data - mean_doppler;

        otherwise
            error('未知方法: %s。可选: zeroVel, MTI, phaseMean', method);
    end

end
