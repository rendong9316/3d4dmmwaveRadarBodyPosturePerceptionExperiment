function rd_data = dopplerFFT(range_data)
%DOPPLERFFT 多普勒维FFT（沿慢时间Chirp维）
%
%   对距离FFT后的数据，沿Chirp维做FFT，将相位变化
%   转换为多普勒频率，从而得到目标的速度信息。
%   fftshift 将零多普勒移至频谱中心。
%
%   输入: range_data — 距离FFT结果 [ADCSamples, numRX, ChirpNum] complex
%   输出: rd_data — Range-Doppler谱 [ADCSamples, numRX, ChirpNum] complex
%         第3维已fftshift，中心对应零速
%
%   速度轴计算:
%     v_max = λ / (4 × Chirptime × GroupNum)
%     vel_axis = linspace(-v_max, v_max, ChirpNum)
%
%   示例:
%     rd_data = dopplerFFT(range_data);

    [n_range, n_rx, n_chirps] = size(range_data);

    % Hanning窗：沿多普勒维（第3维）
    win = hanning(n_chirps);
    win_3d = ones(n_range, n_rx, 1) .* reshape(win, 1, 1, n_chirps);

    % 多普勒维FFT
    rd_data = fft(range_data .* win_3d, [], 3);

    % 零多普勒居中
    rd_data = fftshift(rd_data, 3);

end
