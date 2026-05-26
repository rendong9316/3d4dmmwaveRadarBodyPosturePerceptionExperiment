%% read4DRawData.m
%% 读取级联4D雷达原始ADC数据（单设备单文件版本）
%%
%% 输入:
%%   filename — _data.bin 文件的完整路径
%%   para     — read4DParam() 返回的参数结构体
%%   frames   — (可选) 要读取的帧号，默认根据文件大小自动计算
%%
%% 输出:
%%   adcData — 四维复数矩阵 [numRX, ADCSamples, totalChirps, nFrames]
%%             即: 通道数 × 采样点数 × 脉冲数 × 帧数
%%
%% 数据格式 (chInterleave=0, iqSwapSel=0):
%%   每 Chirp 的 4096 字节排列:
%%     [RX0_I×256, RX1_I×256, RX2_I×256, RX3_I×256,
%%      RX0_Q×256, RX1_Q×256, RX2_Q×256, RX3_Q×256]
%%   存储顺序: Chirp0→Chirp1→...→Chirp11, Loop0→Loop63, Frame0→Frame79
%%
function adcData = read4DRawData(filename, para, frames)
    totalChirps = para.chirpsPerCycle * para.numLoops;

    % 每 Chirp 的 int16 个数: 2(I/Q) × 4(RX) × 256(ADC)
    samplesPerChirp = 2 * para.numRXPerDevice * para.ADCSamples;
    samplesPerFrame = samplesPerChirp * para.chirpsPerCycle * para.numLoops;

    % 根据文件实际大小推算帧数（JSON 可能不准）
    fInfo = dir(filename);
    actualBytes = fInfo.bytes;
    maxFrames = floor(actualBytes / (samplesPerFrame * 2));  % int16=2字节
    if maxFrames < para.numFrames
        fprintf('  (JSON=%d帧, 文件实际=%d帧)\n', para.numFrames, maxFrames);
    end

    if nargin < 3
        frames = 1:maxFrames;
    else
        frames = frames(frames >= 1 & frames <= maxFrames);
    end
    nFrames = length(frames);
    if nFrames == 0
        error('没有可读取的帧');
    end

    % 预分配 [通道数, 采样点数, Chirp总数, 帧数]
    adcData = zeros(para.numRXPerDevice, para.ADCSamples, ...
                    totalChirps, nFrames, 'single');

    fid = fopen(filename, 'r');
    if fid < 0
        error('无法打开文件: %s', filename);
    end

    halfSamples = samplesPerChirp / 2;  % 1024 = 4(RX) × 256(ADC)

    for outF = 1:nFrames
        fIdx = frames(outF);
        byteOffset = (fIdx - 1) * samplesPerFrame * 2;
        fseek(fid, byteOffset, 'bof');

        raw = fread(fid, samplesPerFrame, 'int16=>single');
        if length(raw) < samplesPerFrame
            warning('帧 %d 数据不足', fIdx);
            break;
        end

        for chirp = 1:totalChirps
            offset = (chirp - 1) * samplesPerChirp + 1;
            chunk = raw(offset : offset + samplesPerChirp - 1);

            iPart = chunk(1:halfSamples);          % RX0_I..RX3_I
            qPart = chunk(halfSamples+1 : end);     % RX0_Q..RX3_Q

            iMat = reshape(iPart, para.ADCSamples, para.numRXPerDevice).';  % [4, 256]
            qMat = reshape(qPart, para.ADCSamples, para.numRXPerDevice).';  % [4, 256]
            adcData(:, :, chirp, outF) = complex(iMat, qMat);
        end
    end

    fclose(fid);
    fprintf('  读取完成: [%d %d %d %d]\n', size(adcData));
end
