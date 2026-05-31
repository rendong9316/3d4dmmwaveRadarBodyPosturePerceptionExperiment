%% run_4Dpointcloud.m
%% 4D级联雷达点云生成 -- 复现 IEEE TVT 2025 论文方法
%% (重写版: 加入 SNR筛选、角度质量检测、空间约束)
%%
%% 参考: Wang et al., "Multi-Human Activity Recognition Based on
%%        Sequential 4D Point Clouds Using FMCW Radar"
%%        IEEE Trans. Veh. Technol., vol. 74, no. 10, Oct. 2025.
%%
%% 流程: Range-FFT → Doppler-FFT → CFAR+SNR筛选
%%       → 角度FFT(质量检测) → 极坐标→笛卡尔 → 空间过滤 → 3D点云
clear; close all;

%% ======================== 可调参数 ========================
scenario     = 'CCdata_walk_0001';  % 场景名
% 站立场景特殊参数
is_static_scenario = contains(scenario, 'jump') || contains(scenario, 'sit') ...
                     || contains(scenario, 'empty');
if is_static_scenario
    frame_start = 10; frame_end = 12;   % 静态场景帧数少
    USE_MTI = false;                     % 不抑制静态杂波
    SUPPRESS_ZERO_VEL = false;           % 保留零速
    SNR_MIN_DB = 12;                     % 静态场景需要更高SNR (无多普勒增益)
else
    frame_start = 15; frame_end = 30;    % 动态场景帧数多
    USE_MTI = true;                      % MTI抑制静态背景
    SUPPRESS_ZERO_VEL = true;            % 零速附近抑制
    SNR_MIN_DB = 8;
end

N_ANGLE_PAD  = 128;                % 角度FFT补零
ANGLE_PEAK_RATIO_MIN = 1.8;        % 角度峰均比最小阈值 (线性)
MAX_DET_PER_FRAME = 12;            % 每帧最大检测点数
Z_MIN = 0.1;  Z_MAX = 2.8;         % 高度约束 (m)
X_MAX = 2.5;                        % 方位约束 (m)
Y_MIN = 0.5;  Y_MAX = 6.0;         % 距离约束 (m)

%% ======================== 路径与参数 ========================
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);

jsonPath = fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json');
para = read4DParam(jsonPath);
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

fprintf('===== 4D雷达点云生成 =====\n');
fprintf('场景: %s | 帧: %d-%d | SNR_min: %ddB | 角度质量: %.1fx\n', ...
    scenario, frame_start, frame_end, SNR_MIN_DB, ANGLE_PEAK_RATIO_MIN);

%% ======================== 常量 ========================
nRange   = para.ADCSamples;
nLoops   = para.numLoops;
chirpsPC = para.chirpsPerCycle;
nRX      = para.numRXPerDevice;
nDev     = para.numDevices;
totalTX  = para.totalTX;
totalRX  = para.totalRX;

r_axis = (0:nRange-1)' * para.dr;
v_max  = para.lambda / (4 * para.Chirptime * chirpsPC);
v_axis = linspace(-v_max, v_max, nLoops);

fprintf('fc=%.0fGHz dr=%.1fcm vmax=%.1fm/s totalRX=%d\n', ...
    para.f0/1e9, para.dr*100, v_max, totalRX);
fprintf('MTI=%d | ZeroVelSupp=%d | SNR_min=%ddB\n', ...
    USE_MTI, SUPPRESS_ZERO_VEL, SNR_MIN_DB);

%% ======================== 读取数据 ========================
nFrames = frame_end - frame_start + 1;
devNames = {'master', 'slave1', 'slave2', 'slave3'};
adcAll = cell(nDev, 1);
for d = 1:nDev
    fpath = fullfile(dataRoot, scenario, [devNames{d} '_0000_data.bin']);
    fprintf('读取 %s (%d帧)...', devNames{d}, nFrames);
    adcAll{d} = read4DRawData(fpath, para, frame_start:frame_end);
    fprintf(' done\n');
end

%% ======================== 窗 ========================
rWin = hanning(nRange);
dWin = hanning(nLoops);

%% ======================== 虚拟阵列位置 ========================
% 4个设备的16根RX天线水平位置 (归一化到波长)
% 每个设备内4RX间距 λ/2, 设备间间距约 2λ
rx_pos_dev0 = (0:3) * 0.5;           % Dev0: 4 RX at [0, 0.5λ, λ, 1.5λ]
dev_spacing = 2.0;                    % 设备间距 (λ, 近似)
rx_positions = zeros(totalRX, 1);
for d = 1:nDev
    offset = (d-1) * (4*0.5 + dev_spacing);  % 每设备4RX + 间隙
    rx_positions((d-1)*nRX+1 : d*nRX) = rx_pos_dev0 + offset;
end
% 归一化到以阵列中心为0
rx_positions = rx_positions - mean(rx_positions);

% TX的chirp映射 (TX2=有俯仰偏移的发射天线)
% TX chirp映射 (1-based):
% TX0(无俯仰偏移, 用于方位估计): Dev4=3, Dev3=6, Dev2=9, Dev1=12
% TX2(有俯仰偏移, 用于俯仰估计): Dev4=1, Dev3=4, Dev2=7, Dev1=10
tx0_chirps = [3, 6, 9, 12];   % 每设备TX0
tx2_chirps = [1, 4, 7, 10];   % 每设备TX2

%% ======================== 逐帧处理 ========================
all_pts_cell  = {};
all_rd_cell   = {};
stats = struct('n_raw_det', [], 'n_angle_ok', [], 'n_spatial_ok', []);

fprintf('\n===== 逐帧处理 =====\n');

for fi = 1:nFrames
    realFrm = frame_start + fi - 1;

    % --- Step 1: Range-FFT ---
    rfft_devs = cell(nDev, 1);
    for d = 1:nDev
        frm = squeeze(adcAll{d}(:, :, :, fi));
        rfft_devs{d} = fft(frm .* rWin.', [], 2);
    end

    % --- Step 2: TDM + Doppler FFT ---
    rd_cube = cell(totalTX, 1);
    for tx = 1:totalTX
        devId = para.txChirpMap(tx, 1);
        rd_tx = reshape(rfft_devs{devId}, nRX, nRange, chirpsPC, nLoops);
        rd_data = squeeze(rd_tx(:, :, tx, :));
        if USE_MTI
            rd_data = rd_data - mean(rd_data, 3);  % MTI
        end
        rd_data = fft(rd_data .* reshape(dWin, 1, 1, nLoops), [], 3);
        rd_data = fftshift(rd_data, 3);
        rd_cube{tx} = rd_data;
    end

    % --- Step 3: 非相干积累 RD ---
    rdPwr = zeros(nRange, nLoops);
    for tx = 1:totalTX
        rdPwr = rdPwr + squeeze(mean(abs(rd_cube{tx}).^2, 1));
    end
    rdPwr = rdPwr / totalTX;
    rdDB = 10 * log10(rdPwr + eps);

    % --- Step 4: 2D CA-CFAR ---
    gr = 4;  gd = 2;
    tr = 8;  td = 4;
    tf_cfar = 10.0;  % CFAR阈值 (线性, ≈10dB)

    kr = 2*tr + 2*gr + 1;
    kd = 2*td + 2*gd + 1;
    kern = ones(kr, kd);
    kern(tr+1:tr+2*gr+1, td+1:td+2*gd+1) = 0;
    nTr = sum(kern(:));
    noiseSum = conv2(rdPwr, kern, 'same');
    noiseAvg = noiseSum / nTr;
    det = rdPwr > (noiseAvg * tf_cfar);

    e = tr + gr;
    det(1:e, :) = false;   det(end-e+1:end, :) = false;
    det(:, 1:e) = false;   det(:, end-e+1:end) = false;

    rMin = max(1, round(Y_MIN / para.dr));
    rMax = min(nRange, round(Y_MAX / para.dr));
    det(1:rMin-1, :) = false;
    det(rMax+1:end, :) = false;

    zb = floor(nLoops/2) + 1;
    if SUPPRESS_ZERO_VEL
        det(:, zb-1:zb+1) = false;
    end

    [detR, detD] = find(det);
    nRaw = length(detR);

    % --- Step 4b: SNR筛选 (相对局部噪声) ---
    if nRaw > 0
        snrLin = rdPwr(sub2ind([nRange, nLoops], detR, detD)) ./ ...
                 (noiseAvg(sub2ind([nRange, nLoops], detR, detD)) + eps);
        snrOK = snrLin > 10^(SNR_MIN_DB/10);

        % 取SNR最高的一批
        if sum(snrOK) > MAX_DET_PER_FRAME
            [~, si] = sort(snrLin(snrOK), 'descend');
            goodIdx = find(snrOK);
            snrOK(goodIdx(si(MAX_DET_PER_FRAME+1:end))) = false;
        end
        detR = detR(snrOK);
        detD = detD(snrOK);
    end
    nAfterSNR = length(detR);

    % --- Step 5: 角度估计 + 坐标变换 ---
    pts_raw = zeros(nAfterSNR, 5);   % [x,y,z,v,snr]
    angle_ok = false(nAfterSNR, 1);

    for i = 1:nAfterSNR
        ri = detR(i);
        di = detD(i);

        % ---- 5a: 提取16 RX复数值 (TX0, 无俯仰偏移, 用于方位) ----
        rx16_az = zeros(totalRX, 1);
        for d = 1:nDev
            rx16_az((d-1)*nRX+1 : d*nRX) = rd_cube{tx0_chirps(d)}(:, ri, di);
        end

        % ---- 5b: 1D FFT沿16RX → 方位角谱 ----
        az_spec = fftshift(abs(fft(rx16_az, N_ANGLE_PAD)));
        [pkVal_az, pkBin_az] = max(az_spec);
        medVal_az = median(az_spec);

        if pkVal_az < ANGLE_PEAK_RATIO_MIN * medVal_az
            continue;
        end

        sinAz = (pkBin_az - N_ANGLE_PAD/2 - 1) / (N_ANGLE_PAD/2);
        azDeg = asind(max(-1, min(1, sinAz)));

        % ---- 5c: 俯仰角 (TX2 vs TX0 相位差, 16通道平均) ----
        rx16_el_tx0 = zeros(totalRX, 1);
        rx16_el_tx2 = zeros(totalRX, 1);
        for d = 1:nDev
            rx16_el_tx0((d-1)*nRX+1 : d*nRX) = rd_cube{tx0_chirps(d)}(:, ri, di);
            rx16_el_tx2((d-1)*nRX+1 : d*nRX) = rd_cube{tx2_chirps(d)}(:, ri, di);
        end
        % TX2与TX0的相位差 (共轭相乘, 平均所有16通道)
        phase_diff = mean(angle(rx16_el_tx2 .* conj(rx16_el_tx0)));

        % h_TX ≈ 1λ (TX2垂直偏移, 归一化到波长)
        % Δφ = 2π·h·sin(φ) → sin(φ) = Δφ/(2π·h)
        h_tx_wavelength = 1.0;
        sinEl = phase_diff / (2 * pi * h_tx_wavelength);
        sinEl = max(-1, min(1, sinEl));
        elDeg = asind(sinEl);

        % ---- 5d: 极坐标 → 笛卡尔 (论文 Eq.5) ----
        rM  = r_axis(ri);
        vMS = v_axis(di);

        xM = rM * cosd(elDeg) * sind(azDeg);
        yM = rM * cosd(elDeg) * cosd(azDeg);
        zM = rM * sind(elDeg);

        % ---- 5e: 空间约束 ----
        if zM < Z_MIN || zM > Z_MAX || abs(xM) > X_MAX ...
           || yM < Y_MIN || yM > Y_MAX
            continue;
        end

        pts_raw(i, :) = [xM, yM, zM, vMS, 10*log10(pkVal_az/medVal_az)];
        angle_ok(i) = true;
    end

    % 只保留通过所有筛选的点
    pts = pts_raw(angle_ok, :);
    nFinal = size(pts, 1);

    all_pts_cell{fi} = pts;
    all_rd_cell{fi}  = rdDB;
    stats(fi).n_raw_det   = nRaw;
    stats(fi).n_angle_ok  = nAfterSNR;
    stats(fi).n_spatial_ok = nFinal;

    fprintf('Frame %2d: raw=%3d → SNR=%2d → angle+spatial=%2d pts\n', ...
        realFrm, nRaw, nAfterSNR, nFinal);
end

%% ======================== 汇总统计 ========================
allPts = [];
for f = 1:nFrames
    if ~isempty(all_pts_cell{f})
        p = all_pts_cell{f};
        allPts = [allPts; p(:,1:4), (frame_start+f-1)*ones(size(p,1),1)];
    end
end

fprintf('\n========================================\n');
fprintf('统计汇总:\n');
fprintf('  总帧数: %d\n', nFrames);
fprintf('  CFAR原始检测总数: %d\n', sum([stats.n_raw_det]));
fprintf('  SNR筛选后: %d\n', sum([stats.n_angle_ok]));
fprintf('  角度+空间筛选后: %d\n', sum([stats.n_spatial_ok]));
fprintf('  最终点云: %d 点\n', size(allPts, 1));
if size(allPts,1) > 0
    fprintf('  X范围: [%.2f, %.2f] m\n', min(allPts(:,1)), max(allPts(:,1)));
    fprintf('  Y范围: [%.2f, %.2f] m\n', min(allPts(:,2)), max(allPts(:,2)));
    fprintf('  Z范围: [%.2f, %.2f] m\n', min(allPts(:,3)), max(allPts(:,3)));
    fprintf('  V范围: [%.2f, %.2f] m/s\n', min(allPts(:,4)), max(allPts(:,4)));
end

if isempty(allPts)
    warning('无有效点云。请降低 SNR_MIN_DB 或 ANGLE_PEAK_RATIO_MIN。');
    return;
end

%% ======================== 可视化 ========================

% --- 图1: RD谱 + 最终检测点 ---
midF = round(nFrames / 2);
figure('Color','w','Position',[50,300,650,550]);
imagesc(v_axis, r_axis, all_rd_cell{midF});
set(gca,'YDir','normal');
hold on;
pMid = all_pts_cell{midF};
for i = 1:size(pMid,1)
    rV = sqrt(pMid(i,1)^2 + pMid(i,2)^2);
    [~,ri] = min(abs(r_axis - rV));
    [~,di] = min(abs(v_axis - pMid(i,4)));
    plot(v_axis(di), r_axis(ri), 'ro', 'MarkerSize',8,'LineWidth',1.5);
end
hold off;
xlabel('Velocity (m/s)'); ylabel('Range (m)');
title(sprintf('RD Map + Detections (Frame %d)', frame_start+midF-1));
colormap jet; colorbar;
cLim = [prctile(all_rd_cell{midF}(:),5), prctile(all_rd_cell{midF}(:),98)];
if ~isnan(cLim(1)), clim(cLim); end
xlim([-v_max, v_max]); ylim([Y_MIN, Y_MAX]); grid on;

% --- 图2: 3D点云 (色=速度, 大小=角度质量) ---
figure('Color','w','Position',[750,50,850,750]);
if size(allPts,1) > 1
    % 质量权重 (基于角度峰均比)
    quality = all_pts_cell{1}(:,5);  % 仅参考
    markerSize = 25 + 30 * (min(allPts(:,4)) + v_max) / (2*v_max);
    markerSize = max(10, min(40, markerSize));
end
scatter3(allPts(:,1), allPts(:,2), allPts(:,3), ...
    30, allPts(:,4), 'filled', 'MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('X Azimuth (m)','FontSize',12);
ylabel('Y Range (m)','FontSize',12);
zlabel('Z Height (m)','FontSize',12);
title(sprintf('4D Radar Point Cloud -- %s (%d frames, %d points)', ...
    strrep(scenario,'_','\_'), nFrames, size(allPts,1)), 'FontSize',14);
colormap jet; cb = colorbar; clim([-v_max, v_max]);
title(cb, 'Velocity (m/s)');
axis equal; grid on; box on; view(55, 25);
hold on;
plot3(0, 0, 0, 'k^','MarkerSize',14,'MarkerFaceColor','k');
text(0, 0.2, 0.3, 'Radar','FontSize',11,'Color','k','FontWeight','bold');

% 标注人体参考高度
plot3([-0.3,0.3], [2,2], [Z_MIN,Z_MIN], 'k--','LineWidth',0.5);
plot3([-0.3,0.3], [2,2], [Z_MAX,Z_MAX], 'k--','LineWidth',0.5);
text(0.4, 2, Z_MIN, sprintf('z=%.1fm', Z_MIN), 'FontSize',8);
text(0.4, 2, Z_MAX, sprintf('z=%.1fm', Z_MAX), 'FontSize',8);
hold off;
xlim([-X_MAX, X_MAX]); ylim([Y_MIN, Y_MAX]); zlim([Z_MIN-0.5, Z_MAX+0.5]);

% --- 图3: XY俯视图 (色=高度) ---
figure('Color','w','Position',[750,50,650,550]);
scatter(allPts(:,1), allPts(:,2), 35, allPts(:,3), 'filled', ...
    'MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('X Azimuth (m)'); ylabel('Y Range (m)');
title(sprintf('Top View (XY) -- %s', strrep(scenario,'_','\_')));
colormap jet; cb = colorbar; clim([Z_MIN, Z_MAX]);
title(cb, 'Height (m)');
axis equal; grid on; box on;
xlim([-X_MAX, X_MAX]); ylim([Y_MIN, Y_MAX]);
hold on; plot(0,0,'k^','MarkerSize',12,'MarkerFaceColor','k'); hold off;

% --- 图4: XZ侧视图 (色=速度) ---
figure('Color','w','Position',[750,50,650,550]);
scatter(allPts(:,1), allPts(:,3), 35, allPts(:,4), 'filled', ...
    'MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('X Azimuth (m)'); ylabel('Z Height (m)');
title(sprintf('Side View (XZ) -- %s', strrep(scenario,'_','\_')));
colormap jet; cb = colorbar; clim([-v_max, v_max]);
title(cb, 'Velocity (m/s)');
axis equal; grid on; box on;
xlim([-X_MAX, X_MAX]); ylim([Z_MIN-0.5, Z_MAX+0.5]);
hold on; plot(0, 1.0, 'k^','MarkerSize',12,'MarkerFaceColor','k'); hold off;

% --- 图5: YZ前视图 (色=速度) ---
figure('Color','w','Position',[750,50,650,550]);
scatter(allPts(:,2), allPts(:,3), 35, allPts(:,4), 'filled', ...
    'MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('Y Range (m)'); ylabel('Z Height (m)');
title(sprintf('Front View (YZ) -- %s', strrep(scenario,'_','\_')));
colormap jet; cb = colorbar; clim([-v_max, v_max]);
title(cb, 'Velocity (m/s)');
grid on; box on;
xlim([Y_MIN, Y_MAX]); ylim([Z_MIN-0.5, Z_MAX+0.5]);
hold on; plot(1, 0, 'k^','MarkerSize',12,'MarkerFaceColor','k'); hold off;

%% ======================== 保存 ========================
outDir = fullfile(script_dir, '..', 'figures');
if ~exist(outDir,'dir'), mkdir(outDir); end
figs = findobj('Type','figure');
fnames = {'1_RD_CFAR','2_3D_PointCloud','3_XY_TopView','4_XZ_SideView','5_YZ_FrontView'};
for i = 1:min(length(figs), 5)
    fn = sprintf('%s_%s.png', scenario, fnames{i});
    exportgraphics(figs(i), fullfile(outDir, fn), 'Resolution', 150);
    fprintf('保存: %s\n', fn);
end

fprintf('\n===== 完成 =====\n');
