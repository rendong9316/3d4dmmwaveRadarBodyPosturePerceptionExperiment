%% read4DParam.m
%% 从 TI 级联雷达的 mmwave.json 配置文件中提取雷达工作参数
function para = read4DParam(jsonPath)
    fid = fopen(jsonPath, 'r');
    raw = fread(fid, inf, '*char')';
    fclose(fid);
    cfg = jsondecode(raw);

    % 取主设备（mmWaveDeviceId=0）的参数作为基准
    dev = cfg.mmWaveDevices(1);
    rf = dev.rfConfig;
    prof = rf.rlProfiles(1).rlProfileCfg_t;
    frame = rf.rlFrameCfg_t;

    % ---- 基本波形参数 ----
    para.f0 = prof.startFreqConst_GHz * 1e9;             % 载频 77 GHz
    para.lambda = 3e8 / para.f0;                         % 波长
    para.FrequencySlope = prof.freqSlopeConst_MHz_usec * 1e12;  % 调频斜率 Hz/s
    para.IDLEtime = prof.idleTimeConst_usec * 1e-6;      % 空闲时间
    para.ENDtime  = prof.rampEndTime_usec * 1e-6;        % 斜坡时间
    para.Chirptime = para.IDLEtime + para.ENDtime;       % 单 Chirp 时长
    para.ADCSamples = prof.numAdcSamples;                % 每次 Chirp 采样点数
    para.Fs = prof.digOutSampleRate * 1e3;               % 采样率 Hz

    % ---- 帧结构参数 ----
    para.numLoops  = frame.numLoops;                     % 每帧循环数
    para.numFrames = frame.numFrames;                    % 总帧数
    para.Frameinter = frame.framePeriodicity_msec * 1e-3;% 帧周期

    % ---- 多设备级联信息 ----
    para.numDevices = length(cfg.mmWaveDevices);         % 级联设备数 (4)
    para.numRXPerDevice = 4;                             % 每设备 RX 通道数
    para.totalRX = para.numDevices * para.numRXPerDevice; % 总 RX 通道 (16)

    % 统计总 TX 数和每设备活跃 TX
    txSet = [];  % 收集所有活跃的 TX 编号
    deviceInfo = cell(para.numDevices, 1);
    for d = 1:para.numDevices
        chirps = cfg.mmWaveDevices(d).rfConfig.rlChirps;
        activeTx = [];
        for c = 1:length(chirps)
            txEn = hex2dec(chirps(c).rlChirpCfg_t.txEnable);
            if txEn > 0
                activeTx = [activeTx, txEn];
            end
        end
        deviceInfo{d}.activeTxMask = unique(activeTx);
        txSet = [txSet, unique(activeTx)];
    end
    para.totalTX = length(unique(txSet));                % 总物理 TX 天线数 (9)
    para.chirpsPerCycle = frame.chirpEndIdx - frame.chirpStartIdx + 1;  % 每 TDM 周期 Chirp 数 (12)

    % ---- MIMO 虚拟通道数 ----
    para.virtualChannels = para.totalRX * para.totalTX;  % 192

    % ---- 派生参数 ----
    para.BandWidth = para.FrequencySlope * para.ADCSamples / para.Fs;
    para.dr = 3e8 / para.BandWidth / 2;                  % 距离分辨率
    para.df = 1 / para.Chirptime / para.numDevices;      % 多普勒采样间隔（每个设备发射间隔）
    para.dv = para.lambda * para.df / 2 / (para.numLoops * para.chirpsPerCycle);

    fprintf('  4D 级联雷达参数:\n');
    fprintf('    载频: %.0f GHz | 设备数: %d | TX: %d | RX: %d | 虚拟通道: %d\n', ...
            para.f0/1e9, para.numDevices, para.totalTX, para.totalRX, para.virtualChannels);
    fprintf('    ADC采样: %d | Chirp/周期: %d | Loops: %d | 帧数: %d\n', ...
            para.ADCSamples, para.chirpsPerCycle, para.numLoops, para.numFrames);
    fprintf('    带宽: %.1f MHz | 距离分辨率: %.2f cm\n', ...
            para.BandWidth/1e6, para.dr*100);
end
