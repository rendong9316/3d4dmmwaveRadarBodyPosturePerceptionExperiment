function range_data = rangeFFT(adcData)
%RANGEFFT 距离维FFT（沿快时间维）
%
%   对每个RX通道独立做距离维FFT，获得目标的距离谱。
%   加Hanning窗以抑制旁瓣泄露。
%
%   输入: adcData — 单帧数据 [ADCSamples, numRX, ChirpNum] complex
%         可以是杂波抑制后的数据
%   输出: range_data — 距离谱 [ADCSamples, numRX, ChirpNum] complex
%         第1维变为距离bin，每个bin对应一个距离门
%
%   距离轴计算:
%     range_axis(k) = k * dr,  k = 0, 1, ..., ADCSamples-1
%     其中 dr = c / (2 × BandWidth) 为距离分辨率
%
%   示例:
%     frame = staticClutterSuppression(frame);
%     range_data = rangeFFT(frame);

    [n_range, n_rx, n_chirps] = size(adcData);

    % Hanning窗：沿距离维（第1维）
    win = hanning(n_range);
    % 扩展为3D以便广播 [n_range × 1 × 1]
    win_3d = win .* ones(1, n_rx, n_chirps);

    % 距离维FFT
    range_data = fft(adcData .* win_3d, [], 1);

end
