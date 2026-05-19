function points = generatePointCloud(detections, rd_complex, para, n_angle_fft)
%GENERATEPOINTCLOUD 从CFAR检测+角度估计生成3D点云
%
%   遍历RD谱上所有CFAR检测点，对每个点：
%     1. 距离bin → 径向距离 range
%     2. 多普勒bin → 径向速度 velocity
%     3. 4通道相位差 → 角度FFT → 方位角 azimuth
%     4. 极坐标 → 笛卡尔坐标: x = range×sin(az), y = range×cos(az)
%
%   输入:
%     detections  — CFAR检测掩码 [n_range, n_doppler] logical
%     rd_complex  — 复数RD数据 [n_range, numRX, n_doppler]
%     para        — 雷达参数结构体
%     n_angle_fft — 角度FFT点数，默认128
%
%   输出:
%     points — [N × 3] 矩阵，列 = [x, y, v]
%              x: 方位向坐标(m)，正值为右方
%              y: 距离向坐标(m)，正值为前方
%              v: 径向速度(m/s)，正值为远离

    if nargin < 4
        n_angle_fft = 128;
    end

    [n_range, ~, n_doppler] = size(rd_complex);

    % 坐标轴
    range_axis = (0:n_range-1)' * para.dr;
    v_max = para.lambda / 4 / para.Chirptime / para.GroupNum;
    vel_axis = linspace(-v_max, v_max, n_doppler);

    % 找到所有检测点的行列索引
    [r_idx, d_idx] = find(detections);
    n_det = length(r_idx);

    if n_det == 0
        points = [];
        return;
    end

    % 预分配
    points = zeros(n_det, 3);

    for i = 1:n_det
        ri = r_idx(i);
        di = d_idx(i);

        % 距离和速度
        range_m = range_axis(ri);
        vel_ms  = vel_axis(di);

        % 角度估计
        azimuth_deg = angleFFT(rd_complex, ri, di, n_angle_fft);

        % 极坐标 → 笛卡尔坐标
        x_m = range_m * sind(azimuth_deg);  % 方位向（左右）
        y_m = range_m * cosd(azimuth_deg);  % 距离向（前后）

        points(i, :) = [x_m, y_m, vel_ms];
    end

end
