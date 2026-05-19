function visualizePointCloud(all_rd, all_points, all_detections, scenarios_cn, para, v_max)
%VISUALIZEPOINTCLOUD 可视化：RD谱+CFAR叠加、3D点云散点图、帧间点云数
%
%   生成三组图：
%     图1: 4场景 RD谱 + CFAR检测叠加（中间帧）
%     图2: 4场景 3D点云散点图（中间帧，颜色=速度）
%     图3: 4场景 点云数量随帧变化曲线
%
%   输入:
%     all_rd         — cell{4}，每个为 [n_range, n_doppler] 功率dB
%     all_points     — cell{4}，每个为 cell{n_frames}，cell内含 [N×3]
%     all_detections — cell{4}，每个为 cell{n_frames}，cell内含检测掩码
%     scenarios_cn   — cell{4}，场景中文名
%     para           — 雷达参数
%     v_max          — 最大速度 (m/s)

    n_range = para.ADCSamples;
    n_doppler = para.ChirpNum;
    range_axis = (0:n_range-1)' * para.dr;
    vel_axis = linspace(-v_max, v_max, n_doppler);

    % ===== 图1: RD谱 + CFAR检测叠加 =====
    figure('Name', 'Range-Doppler谱 + CFAR检测', 'Position', [50, 50, 1200, 700]);
    for s = 1:4
        subplot(2, 2, s);
        rd_db = all_rd{s};

        % 绘制RD谱背景
        imagesc(vel_axis, range_axis, rd_db);
        set(gca, 'YDir', 'normal');
        caxis([-40, 20]);
        colormap(gca, 'jet');
        colorbar;
        hold on;

        % 叠加CFAR检测点（红色叉）
        det = all_detections{s};
        [dr, dd] = find(det);
        if ~isempty(dr)
            plot(vel_axis(dd), range_axis(dr), 'rx', 'MarkerSize', 4, 'LineWidth', 1);
        end

        xlabel('速度 (m/s)');
        ylabel('距离 (m)');
        title(scenarios_cn{s});
        xlim([-v_max, v_max]);
        ylim([0, max(range_axis)]);
        hold off;
    end
    sgtitle('Range-Doppler谱 + CFAR目标检测（中间帧，红色叉=检测点）');

    % ===== 图2: 3D点云散点图 =====
    figure('Name', '3D点云', 'Position', [100, 50, 1200, 700]);
    for s = 1:4
        subplot(2, 2, s);
        pts_cell = all_points{s};
        if isempty(pts_cell)
            title([scenarios_cn{s} ' — 无点云']);
            continue;
        end
        mid = round(length(pts_cell) / 2);
        pts = pts_cell{mid};

        if isempty(pts)
            scatter([], []);
            title([scenarios_cn{s} ' — 中间帧无检测']);
            xlabel('X 方位向 (m)'); ylabel('Y 距离向 (m)');
            xlim([-3, 3]); ylim([0, 5]);
            axis equal; grid on;
            continue;
        end

        % 速度映射颜色
        scatter(pts(:,1), pts(:,2), 20, pts(:,3), 'filled');
        colormap(gca, 'jet');
        cbar = colorbar;
        cbar.Label.String = '速度 (m/s)';
        caxis([-v_max, v_max]);

        xlabel('X 方位向 (m)');
        ylabel('Y 距离向 (m)');
        title([scenarios_cn{s} ' — 中间帧点云']);
        xlim([-3, 3]);
        ylim([0, 5]);
        axis equal; grid on;
    end
    sgtitle('3D点云（中间帧，颜色=径向速度）');

    % ===== 图3: 点云数量随帧变化 =====
    figure('Name', '点云数量随帧变化', 'Position', [150, 100, 800, 700]);
    colors = {'k', 'b', 'g', 'r'};
    hold on;
    for s = 1:4
        pts_cell = all_points{s};
        n_frames = length(pts_cell);
        n_pts = zeros(n_frames, 1);
        for f = 1:n_frames
            if ~isempty(pts_cell{f})
                n_pts(f) = size(pts_cell{f}, 1);
            end
        end
        plot(1:n_frames, n_pts, 'Color', colors{s}, 'LineWidth', 1.5, ...
             'DisplayName', scenarios_cn{s});
    end
    hold off;
    xlabel('帧序号');
    ylabel('检测点数');
    title('每帧CFAR检测点数量');
    legend('Location', 'best');
    grid on;

end
