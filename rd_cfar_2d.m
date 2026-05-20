function cfar_map = rd_cfar_2d(rd_map, Tr, Td, Gr, Gd, offset_dB)
% rd_map: [N_range x N_doppler] dB
% Tr,Td: Range/Doppler训练单元数
% Gr,Gd: Range/Doppler保护单元数
% offset_dB: 阈值超出背景的dB数

    [N_range, N_doppler] = size(rd_map);
    cfar_map = zeros(N_range, N_doppler);
    rd_power = 10.^(rd_map/10);   % 转换为线性功率

    for i = (Tr+Gr+1):(N_range-Tr-Gr)
        for j = (Td+Gd+1):(N_doppler-Td-Gd)
            % ----- Range 方向训练单元 -----
            range_idx = (i-Tr-Gr):(i+Tr+Gr);
            % 剔除保护单元（保留值 < i-Gr 或 > i+Gr 的元素）
            range_idx = range_idx( (range_idx < i-Gr) | (range_idx > i+Gr) );

            % ----- Doppler 方向训练单元 -----
            dop_idx = (j-Td-Gd):(j+Td+Gd);
            % 剔除保护单元（保留值 < j-Gd 或 > j+Gd 的元素）
            dop_idx = dop_idx( (dop_idx < j-Gd) | (dop_idx > j+Gd) );

            % 计算噪声平均水平（训练单元功率均值）
            noise_level = mean(mean(rd_power(range_idx, dop_idx)));

            % 计算检测阈值
            threshold = noise_level * 10^(offset_dB/10);

            % 被检测单元（CUT）功率
            CUT = rd_power(i, j);

            if CUT > threshold
                cfar_map(i, j) = 1;
            end
        end
    end
end