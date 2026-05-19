function adcData = staticClutterSuppression(adcData)
%STATICCLUTTERSUPPRESSION 静态杂波抑制（MTI高通滤波）
%
%   通过沿慢时间维（Chirp维）减去均值，抑制静止目标的回波。
%   人体运动产生的多普勒频率不会被均值减法消除，而墙壁、地面等
%   静止目标的强回波被抑制，使运动目标的信号得以凸显。
%
%   原理：
%     S_suppressed(t) = S(t) - mean_t(S(t))
%     静止目标各Chirp回波相同 → 相减后趋于零
%     运动目标各Chirp回波有相位变化 → 信号保留
%
%   输入: adcData — 单帧原始数据 [ADCSamples, numRX, ChirpNum] complex
%   输出: adcData — 杂波抑制后数据，同尺寸 complex
%
%   示例:
%     frame = adcData(:, :, :, 1);
%     frame = staticClutterSuppression(frame);

    % 沿第3维（Chirp维）取均值，得到静态背景
    mean_chirp = mean(adcData, 3);

    % 减去静态背景
    adcData = adcData - mean_chirp;

end
