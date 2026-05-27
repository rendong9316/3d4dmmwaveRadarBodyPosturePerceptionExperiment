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

    % ---- TDM-MIMO chirp->TX 映射表 ----
    % 扫描所有设备的 chirp 配置，统计物理 TX 总数并建立映射
    % txChirpMap(c, :) = [deviceID, txIndex]  (1-based chirp索引)
    % 含义：第 c 个 chirp 由 deviceID 号设备的 txIndex 号 TX 天线发射
    para.chirpsPerCycle = frame.chirpEndIdx - frame.chirpStartIdx + 1;  % 每 TDM 周期 Chirp 数 (12)
    para.txChirpMap = zeros(para.chirpsPerCycle, 2);
    totalTxCount = 0;
    for d = 1:para.numDevices
        chirps = cfg.mmWaveDevices(d).rfConfig.rlChirps;
        for c = 1:length(chirps)
            txEn = hex2dec(chirps(c).rlChirpCfg_t.txEnable);
            if txEn > 0
                totalTxCount = totalTxCount + 1;
                chirpIdx = chirps(c).rlChirpCfg_t.chirpStartIdx + 1;  % 0-based -> 1-based
                txIdx = log2(txEn) + 1;  % 0x1->1(TX0), 0x2->2(TX1), 0x4->3(TX2)
                para.txChirpMap(chirpIdx, :) = [d, txIdx];
            end
        end
    end
    para.totalTX = totalTxCount;  % 4设备 x 3TX = 12个物理发射天线

    % ---- MIMO 虚拟通道数 ----
    para.virtualChannels = para.totalRX * para.totalTX;  % 192 (16x12)

    % ---- 派生参数 ----
    para.BandWidth = para.FrequencySlope * para.ADCSamples / para.Fs;
    para.dr = 3e8 / para.BandWidth / 2;                  % 距离分辨率
    % TDM周期: 每个TX每 chirpsPerCycle 个chirp才发射一次
    para.df = 1 / (para.Chirptime * para.chirpsPerCycle); % 等效慢时间采样率 (每个TX的PRF)
    para.dv = para.lambda * para.df / 2 / para.numLoops;  % 速度分辨率

    fprintf('  4D 级联雷达参数:\n');
    fprintf('    载频: %.0f GHz | 设备数: %d | TX: %d | RX: %d | 虚拟通道: %d\n', ...
            para.f0/1e9, para.numDevices, para.totalTX, para.totalRX, para.virtualChannels);
    fprintf('    ADC采样: %d | Chirp/周期: %d | Loops: %d | 帧数: %d\n', ...
            para.ADCSamples, para.chirpsPerCycle, para.numLoops, para.numFrames);
    fprintf('    带宽: %.1f MHz | 距离分辨率: %.2f cm\n', ...
            para.BandWidth/1e6, para.dr*100);

    % 打印 TDM-MIMO 映射表
    fprintf('    TDM-MIMO chirp映射 (chirp -> 设备.TX):\n');
    for c = 1:para.chirpsPerCycle
        fprintf('      chirp%2d -> Dev%d TX%d\n', c-1, para.txChirpMap(c,1), para.txChirpMap(c,2)-1);
    end
end
