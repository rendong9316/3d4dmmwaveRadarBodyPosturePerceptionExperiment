function [azimuth_deg, spectrum] = angleFFT(rd_complex, range_idx, doppler_idx, n_fft)
%ANGLEFFT 角度FFT — 利用RX阵列相位差估计方位角
%
%   在Range-Doppler谱的某个检测点上，提取4个RX通道的复数值，
%   这些复数的相位差反映了目标回波到达不同天线的波程差。
%   对此复数序列做FFT → 角度谱 → 峰值对应方位角。
%
%   原理（均匀线阵 ULA）：
%     相邻RX间距 d = λ/2
%     波程差 Δ = d × sin(θ) → 相位差 Δφ = 2π × d × sin(θ) / λ
%     4个RX形成一个空间采样序列，FFT可分辨不同sin(θ)方向
%
%   输入:
%     rd_complex  — RD复矩阵 [ADCSamples, numRX, ChirpNum]
%     range_idx   — 检测点的距离bin索引
%     doppler_idx — 检测点的多普勒bin索引
%     n_fft       — 补零后的FFT点数，默认128（增大提高角分辨率）
%
%   输出:
%     azimuth_deg — 方位角（度），-90° ~ +90°，0°为正前方
%     spectrum    — [n_fft × 1] 角度谱幅度
%
%   示例:
%     [az, spec] = angleFFT(rd_complex, 50, 120, 128);

    if nargin < 4
        n_fft = 128;
    end

    % 提取该检测点在4个RX上的复数值 [numRX, 1]
    rx_vector = squeeze(rd_complex(range_idx, :, doppler_idx));

    % Zero-pad FFT
    spectrum = fftshift(abs(fft(rx_vector, n_fft)));

    % 峰值位置 → sin(θ)
    [~, peak_idx] = max(spectrum);

    % FFT bin 到 sin(θ) 的映射（fftshift后的1-indexed索引）
    %   中心 (n_fft/2 + 1) = sinθ=0
    %   最右 (n_fft)     = sinθ≈+1
    %   最左 (1)         = sinθ≈-1
    sin_theta = (peak_idx - n_fft/2 - 1) / (n_fft/2);

    % 限幅，防止数值误差导致asind出界
    sin_theta = max(-1, min(1, sin_theta));

    % sin(θ) → 方位角（度）
    azimuth_deg = asind(sin_theta);

end
