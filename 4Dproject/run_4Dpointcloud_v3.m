%% run_4Dpointcloud.m  (v3.0 - DT.m/RT.m方法)
%% 4D级联雷达点云生成
%%
%% 核心改进:
%%   1. 仿DT.m/RT.m: master不分TDM构建RD功率图
%%   2. RT: 每距离bin非DC多普勒最大 -> 定位距离(~2.4m)
%%   3. DT: 全距离bin多普勒最大 -> 定位速度
%%   4. 功率dB阈值: >POWER_THRESH_DB 筛选 (参考clim=[100,110])
%%   5. 距离窗约束: RANGE_TARGET_M +- RANGE_WINDOW_M
%%   6. 只在目标区域做TDM分路+角度估计
clear; close all;

%% ======================== 参数 ========================
scenario     = 'CCdata_walk_0001';
frame_start = 35; frame_end = 55;
POWER_THRESH_DB = 95;          %% 功率阈值(dB)
RANGE_TARGET_M = 2.4;          %% 目标距离(m)
RANGE_WINDOW_M = 1.0;          %% 距离窗(+-m)
N_ANGLE_PAD = 256;
ANGLE_PEAK_RATIO_MIN = 1.1;      %% 角度峰均比 (大幅降低, TDM分路后SNR较低)
MAX_DET_PER_FRAME = 30;
Z_MIN = 0.3;  Z_MAX = 2.5;
X_MAX = 2.0;
Y_MIN = 0.5;  Y_MAX = 6.0;

%% ======================== 初始化 ========================
script_dir = fileparts(mfilename('fullpath'));
addpath(script_dir);
jsonPath = fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json');
para = read4DParam(jsonPath);
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

fprintf('===== 4D点云 v3.0 (DT.m/RT.m方法) =====\n');
fprintf('场景: %s | 帧: %d-%d | 功率阈值: %ddB | 目标距离: %.1f+-%.1fm\n', ...
    scenario, frame_start, frame_end, POWER_THRESH_DB, RANGE_TARGET_M, RANGE_WINDOW_M);

nRange = para.ADCSamples; nLoops = para.numLoops;
nChirps = para.chirpsPerCycle; nRX = para.numRXPerDevice;
nDev = para.numDevices; totalTX = para.totalTX; totalRX = para.totalRX;

r_axis = (0:nRange-1)' * para.dr;
v_max = para.lambda / (4 * para.Chirptime * nChirps);
v_axis = ((0:nLoops-1) - nLoops/2) * (2*v_max/nLoops);
rMin = max(1, round(0.5/para.dr)); rMax = min(nRange, round(5.0/para.dr));
zb = nLoops/2 + 1; non_dc = [1:zb-1, zb+1:nLoops];
nChirpsPerFrame = nChirps * nLoops;
rWin = hanning(nRange); dWin = hanning(nLoops);

fprintf('fc=%.0fGHz dr=%.1fcm vmax=%.1fm/s\n', para.f0/1e9, para.dr*100, v_max);

%% ======================== 读取数据 ========================
fprintf('读取master原始数据(DT.m方式)...');
masterBin = fullfile(dataRoot, scenario, 'master_0000_data.bin');
fid = fopen(masterBin, 'rb');
rawMaster = fread(fid, 'int16'); fclose(fid);
rawMaster = rawMaster(1:2:end) + 1j*rawMaster(2:2:end);

totalChirpsAll = length(rawMaster) / (nRX * nRange);
nFramesTotal = floor(totalChirpsAll / nChirpsPerFrame);
fprintf(' %d帧\n', nFramesTotal);

rawMaster = rawMaster(1 : nRX*nRange*nFramesTotal*nChirpsPerFrame);
rawMaster = reshape(rawMaster, nRX, nRange, nChirpsPerFrame, nFramesTotal);
rawMaster = reshape(rawMaster, nRX, nRange, nChirps, nLoops, nFramesTotal);

validFrames = frame_start:min(frame_end, nFramesTotal);
nFrames = length(validFrames);
if nFrames == 0, error('no valid frames'); end

fprintf('读取4设备数据(角度估计)...');
devNames = {'master','slave1','slave2','slave3'};
adcAngle = cell(nDev,1);
for d = 1:nDev
    fpath = fullfile(dataRoot, scenario, [devNames{d} '_0000_data.bin']);
    adcAngle{d} = read4DRawData(fpath, para, validFrames);
end
fprintf(' done\n');

tx0_chirps = [3,6,9,12]; tx1_chirps = [2,5,8,11]; tx2_chirps = [1,4,7,10];

%% ======================== 阶段1: RT/DT检测 ========================
all_pts_cell = {}; all_rd_cell = {};
stats = struct('n_raw_det',[],'n_power_ok',[],'n_angle_ok',[]);
rt_map_all = zeros(nRange, nFrames);
dt_map_all = zeros(nLoops, nFrames);

fprintf('\n===== 阶段1: RT/DT检测 =====\n');

for fi = 1:nFrames
    % DT.m/RT.m方式: master不分TDM
    frame = squeeze(rawMaster(:,:,:,:,validFrames(fi)));
    rfft = fft(frame .* rWin.', [], 2);
    rd_mti = rfft - mean(rfft, 4);
    rd = fft(rd_mti .* reshape(dWin,1,1,1,nLoops), [], 4);
    rd = fftshift(rd, 4);
    pwr_rd = squeeze(mean(sum(abs(rd).^2,1),3));
    pwr_rd_db = 10*log10(pwr_rd + 1);
    all_rd_cell{fi} = pwr_rd_db;
    rt_map_all(:,fi) = max(pwr_rd(:,non_dc), [], 2);
    dt_map_all(:,fi) = max(pwr_rd(rMin:rMax,:), [], 1);
end

%% RT分析
rt_map_db = 10*log10(rt_map_all + 1);
fprintf('\nRT图: dB=[%.1f,%.1f]\n', min(rt_map_db(:)), max(rt_map_db(:)));
fprintf('目标窗(%.1f+-%.1fm):\n', RANGE_TARGET_M, RANGE_WINDOW_M);
range_mask = (r_axis >= RANGE_TARGET_M-RANGE_WINDOW_M) & (r_axis <= RANGE_TARGET_M+RANGE_WINDOW_M);
for fi = 1:3:nFrames
    [~,bestR] = max(rt_map_all(rMin:rMax,fi)); bestR = bestR + rMin - 1;
    fprintf('  Frame %2d: R=%.2fm (%.1fdB)\n', validFrames(fi), r_axis(bestR), rt_map_db(bestR,fi));
end

%% DT分析
dt_map_db = 10*log10(dt_map_all + 1);
fprintf('\nDT图: dB=[%.1f,%.1f]\n', min(dt_map_db(:)), max(dt_map_db(:)));
for fi = 1:3:nFrames
    [~,bestD] = max(dt_map_all(:,fi));
    fprintf('  Frame %2d: 最强v=%.2fm/s (%.1fdB)\n', validFrames(fi), v_axis(bestD), dt_map_db(bestD,fi));
end

%% ======================== 阶段2: 功率阈值检测+角度估计 ========================
fprintf('\n===== 阶段2: 检测+角度估计 =====\n');

for fi = 1:nFrames
    realFrm = validFrames(fi);
    rdDB = all_rd_cell{fi};

    %% 功率阈值检测
    det_mask = rdDB > POWER_THRESH_DB;
    det_mask(~range_mask,:) = false;       %% 距离窗
    det_mask(:, zb-4:zb+4) = false;        %% 零速抑制
    det_mask(1:rMin-1,:) = false; det_mask(rMax+1:end,:) = false;
    det_mask(:,1:2)=false; det_mask(:,end-1:end)=false;
    [detR,detD] = find(det_mask);
    nRaw = length(detR);

    if nRaw > MAX_DET_PER_FRAME  %% Top-N
        pv = zeros(nRaw,1);
        for i=1:nRaw, pv(i)=rdDB(detR(i),detD(i)); end
        [~,si] = sort(pv,'descend');
        detR=detR(si(1:MAX_DET_PER_FRAME)); detD=detD(si(1:MAX_DET_PER_FRAME));
        nRaw = MAX_DET_PER_FRAME;
    end

    %% 角度估计
    pts_raw = zeros(nRaw,5); angle_ok = false(nRaw,1);
    nAngleFail = 0; nSpatialFail = 0;

    if nRaw > 0
        rfft_angle = cell(nDev,1);
        for d=1:nDev
            frm_angle = squeeze(adcAngle{d}(:,:,:,fi));
            rfft_angle{d} = fft(frm_angle .* rWin.', [], 2);
        end

        rd_cube = cell(totalTX,1);
        for tx=1:totalTX
            dt = para.txChirpMap(tx,1);
            rd_tx = reshape(rfft_angle{dt}, nRX, nRange, nChirps, nLoops);
            rd_d = squeeze(rd_tx(:,:,tx,:));
            %% 注意: master检测已做MTI, 角度估计不再重复MTI避免信号衰减
            rd_d = fft(rd_d.*reshape(dWin,1,1,nLoops),[],3);
            rd_cube{tx} = fftshift(rd_d,3);
        end

        for i=1:nRaw
            ri=detR(i); di=detD(i);

            rx32 = zeros(2*totalRX,1);
            for d=1:nDev
                i0=(d-1)*nRX*2+1;
                rx32(i0:i0+nRX-1)=rd_cube{tx0_chirps(d)}(:,ri,di);
                rx32(i0+nRX:i0+2*nRX-1)=rd_cube{tx1_chirps(d)}(:,ri,di);
            end

            az_spec=fftshift(abs(fft(rx32,N_ANGLE_PAD)));
            [pk,pi]=max(az_spec); medAz=median(az_spec);
            if pk<ANGLE_PEAK_RATIO_MIN*medAz
                nAngleFail = nAngleFail + 1; continue;
            end

            sA=(pi-N_ANGLE_PAD/2-1)/(N_ANGLE_PAD/2); sA=max(-1,min(1,sA));
            az=asind(sA);

            pd=zeros(totalRX,1); aw=zeros(totalRX,1);
            for d=1:nDev
                i0=(d-1)*nRX+1;
                t0=rd_cube{tx0_chirps(d)}(:,ri,di);
                t2=rd_cube{tx2_chirps(d)}(:,ri,di);
                pd(i0:i0+nRX-1)=angle(t2.*conj(t0));
                aw(i0:i0+nRX-1)=abs(t0);
            end
            aw=aw/(sum(aw)+eps); ph=sum(pd.*aw);
            sE=ph/(2*pi); sE=max(-1,min(1,sE));
            el=asind(sE);

            rM=r_axis(ri); vM=v_axis(di);
            xM=rM*cosd(el)*sind(az);
            yM=rM*cosd(el)*cosd(az);
            zM=rM*sind(el);

            if zM<Z_MIN||zM>Z_MAX||abs(xM)>X_MAX||yM<Y_MIN||yM>Y_MAX
                nSpatialFail = nSpatialFail + 1; continue; end

            pts_raw(i,:)=[xM,yM,zM,vM,10*log10(pk/median(az_spec))];
            angle_ok(i)=true;
        end
    end

    pts=pts_raw(angle_ok,:); nF=size(pts,1);
    all_pts_cell{fi}=pts;
    stats(fi).n_raw_det=nRaw; stats(fi).n_power_ok=nRaw; stats(fi).n_angle_ok=nF;

    if nF>0
        dbOk=rdDB(sub2ind([nRange,nLoops],detR(angle_ok),detD(angle_ok)));
        fprintf('Frame %2d: pwr=%d ang=%d(%.0fA/%.0fS fail) v=[%.1f,%.1f] z=[%.1f,%.1f] dB=[%.0f,%.0f]\n',...
            realFrm,nRaw,nF,nAngleFail,nSpatialFail,min(pts(:,4)),max(pts(:,4)),min(pts(:,3)),max(pts(:,3)),min(dbOk),max(dbOk));
    else
        fprintf('Frame %2d: pwr=%d ang=0(%.0fA/%.0fS fail)\n',realFrm,nRaw,nAngleFail,nSpatialFail);
    end
end

%% ======================== 汇总 ========================
allPts=[];
for f=1:nFrames
    if ~isempty(all_pts_cell{f})
        p=all_pts_cell{f};
        allPts=[allPts; p(:,1:4), validFrames(f)*ones(size(p,1),1)];
    end
end

fprintf('\n========================================\n');
fprintf('CFAR: %d -> 功率: %d -> 角度+空间: %d\n',...
    sum([stats.n_raw_det]),sum([stats.n_power_ok]),sum([stats.n_angle_ok]));
fprintf('点云: %d点\n',size(allPts,1));
if size(allPts,1)>0
    fprintf('X:[%.2f,%.2f] Y:[%.2f,%.2f] Z:[%.2f,%.2f] V:[%.2f,%.2f]\n',...
        min(allPts(:,1)),max(allPts(:,1)),min(allPts(:,2)),max(allPts(:,2)),...
        min(allPts(:,3)),max(allPts(:,3)),min(allPts(:,4)),max(allPts(:,4)));
end
if isempty(allPts), warning('no points'); return; end

%% ======================== 可视化 ========================
outDir = fullfile(script_dir, '..', 'figures');
if ~exist(outDir,'dir'), mkdir(outDir); end

midF = round(nFrames/2);
figure('Color','w','Position',[50,300,650,550],'Visible','off');
imagesc(v_axis, r_axis, all_rd_cell{midF});
set(gca,'YDir','normal'); hold on;
pMid = all_pts_cell{midF};
for i=1:size(pMid,1)
    rV=sqrt(pMid(i,1)^2+pMid(i,2)^2);
    [~,ri]=min(abs(r_axis-rV)); [~,di]=min(abs(v_axis-pMid(i,4)));
    plot(v_axis(di),r_axis(ri),'ro','MarkerSize',8,'LineWidth',1.5);
end
hold off; xlabel('Velocity (m/s)'); ylabel('Range (m)');
title(sprintf('RD + Detections (Frame %d)',validFrames(midF)));
colormap jet; colorbar; grid on;
cLim=[prctile(all_rd_cell{midF}(:),20),prctile(all_rd_cell{midF}(:),99)];
if ~isnan(cLim(1)), clim(cLim); end
xlim([-v_max,v_max]); ylim([Y_MIN,Y_MAX]);
% 标注功率阈值
hold on; plot([-v_max,v_max],[RANGE_TARGET_M,RANGE_TARGET_M],'w--','LineWidth',1); hold off;
exportgraphics(gcf,fullfile(outDir,[scenario '_1_RD_CFAR.png']),'Resolution',150);

figure('Color','w','Position',[750,50,850,750],'Visible','off');
scatter3(allPts(:,1),allPts(:,2),allPts(:,3),40,allPts(:,4),'filled','MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('X Azimuth (m)'); ylabel('Y Range (m)'); zlabel('Z Height (m)');
title(sprintf('4D Point Cloud - %s (%d frames, %d pts)',strrep(scenario,'_','\_'),nFrames,size(allPts,1)),'FontSize',14);
colormap jet; cb=colorbar; clim([-v_max,v_max]); title(cb,'V (m/s)');
axis equal; grid on; box on; view(55,25);
hold on; plot3(0,0,0,'k^','MarkerSize',14,'MarkerFaceColor','k');
text(0,0.2,0.3,'Radar','FontSize',11); hold off;
xlim([-X_MAX,X_MAX]); ylim([Y_MIN,Y_MAX]); zlim([0,3]);
exportgraphics(gcf,fullfile(outDir,[scenario '_2_3D_PointCloud.png']),'Resolution',150);

figure('Color','w','Position',[750,50,650,550],'Visible','off');
scatter(allPts(:,1),allPts(:,2),35,allPts(:,3),'filled','MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('X (m)'); ylabel('Y (m)'); title(sprintf('Top View - %s',strrep(scenario,'_','\_')));
colormap jet; cb=colorbar; clim([Z_MIN,Z_MAX]); title(cb,'Z (m)');
axis equal; grid on; xlim([-X_MAX,X_MAX]); ylim([Y_MIN,Y_MAX]);
hold on; plot(0,0,'k^','MarkerSize',12,'MarkerFaceColor','k'); hold off;
exportgraphics(gcf,fullfile(outDir,[scenario '_3_XY_TopView.png']),'Resolution',150);

figure('Color','w','Position',[750,50,650,550],'Visible','off');
scatter(allPts(:,1),allPts(:,3),35,allPts(:,4),'filled','MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('X (m)'); ylabel('Z (m)'); title(sprintf('Side View - %s',strrep(scenario,'_','\_')));
colormap jet; cb=colorbar; clim([-v_max,v_max]); title(cb,'V (m/s)');
axis equal; grid on; xlim([-X_MAX,X_MAX]); ylim([0,3]);
hold on; plot(0,1.0,'k^','MarkerSize',12,'MarkerFaceColor','k'); hold off;
exportgraphics(gcf,fullfile(outDir,[scenario '_4_XZ_SideView.png']),'Resolution',150);

figure('Color','w','Position',[750,50,650,550],'Visible','off');
scatter(allPts(:,2),allPts(:,3),35,allPts(:,4),'filled','MarkerEdgeColor',[.2 .2 .2],'LineWidth',0.3);
xlabel('Y (m)'); ylabel('Z (m)'); title(sprintf('Front View - %s',strrep(scenario,'_','\_')));
colormap jet; cb=colorbar; clim([-v_max,v_max]); title(cb,'V (m/s)');
grid on; xlim([Y_MIN,Y_MAX]); ylim([0,3]);
hold on; plot(1,0,'k^','MarkerSize',12,'MarkerFaceColor','k'); hold off;
exportgraphics(gcf,fullfile(outDir,[scenario '_5_YZ_FrontView.png']),'Resolution',150);

fprintf('\n===== 完成 =====\n');
close all;
