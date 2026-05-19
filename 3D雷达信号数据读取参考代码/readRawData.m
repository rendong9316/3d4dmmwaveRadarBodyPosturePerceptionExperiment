function [retVal, para] = readRawData(filename, para)
%READRAWDATA 读取 DCA1000 采集板保存的原始 ADC 二进制数据
%
%   DCA1000 通过 LVDS 接口从 IWR6843 接收 ADC 采样数据，并以二进制
%   格式保存为 .bin 文件。本函数负责读取指定帧范围的数据，并重组为
%   [采样点 × 通道 × Chirp × 帧] 的四维复数矩阵。
%
%   输入:
%     filename — .bin 文件的完整路径（字符串）
%     para     — 结构体，由 readPara() 生成的雷达参数
%                可选的额外字段：
%                  para.FrameStart  — 起始帧号（默认 1）
%                  para.FrameLength — 读取帧数（默认读到最后一帧）
%
%   输出:
%     retVal — 四维复数矩阵 [ADCSamples, numRX×GroupNum, ChirpNum, FrameLength]
%              - 第 1 维: ADC 采样点（快时间 / 距离维）
%              - 第 2 维: 虚拟通道（4 RX × N TX，TDM-MIMO 下每 TX 对应一组）
%              - 第 3 维: Chirp 索引（慢时间 / 多普勒维）
%              - 第 4 维: 帧索引（超慢时间 / 时变维）
%     para   — 更新后的参数结构体（补全了 FrameStart / FrameLength 默认值）
%
%   DCA1000 数据存储格式:
%     - 每个 ADC 采样点为 16-bit 有符号整数（int16）
%     - I/Q 两路各存为两个独立的 int16（共 4 字节/采样点）
%     - 4 路 RX 数据以交错方式排列: [RX0_I, RX1_I, RX2_I, RX3_I,
%                                     RX0_Q, RX1_Q, RX2_Q, RX3_Q, ...]
%     - 先按采样点 (快时间), 再按 Chirp, 最后按帧的顺序排列
%
%   示例:
%     para = readPara("1_LogFile.txt");
%     [adcData, para] = readRawData("1.bin", para);

% ===== 1. 设置默认的帧范围 =====
% 如果用户没有指定起始帧号，默认从第 1 帧开始
if ~isfield(para, 'FrameStart')
    para.FrameStart = 1;
end

% 如果用户没有指定读取帧数，默认读到最后
% FrameLength = 总帧数 - 起始帧 + 1
if ~isfield(para, 'FrameLength')
    para.FrameLength = para.FrameNum - para.FrameStart + 1;
end

% ===== 2. 计算单帧数据量 =====
% FrameNum(变量复用): 单帧的 ADC 采样点总数（所有通道合计）
%   = ADCSamples × ChirpNum × GroupNum × numRX
%   注意：这里的 FrameNum 是临时局部变量，与 para.FrameNum（总帧数）不同
FrameNum = para.ADCSamples * para.ChirpNum * para.GroupNum * para.numRX;

% ===== 3. 打开文件并跳转到起始帧位置 =====
fid = fopen(filename, 'r');

% 计算字节偏移量:
%   2                → 每个 ADC 采样点 2 个 int16（I + Q）
%   FrameNum         → 单帧的总采样点数（所有通道、所有 Chirp）
%   (FrameStart - 1) → 跳过前面不需要的帧
%   2                → 每个 int16 占 2 字节
% 总偏移 = 2 × FrameNum × (FrameStart-1) × 2 字节
fseek(fid, (2 * FrameNum * (para.FrameStart - 1)) * 2, 'bof');

% ===== 4. 读取二进制数据 =====
% 读取格式: [4 行 × N 列] int16 矩阵
%   第 1 行 = RX0_I 序列（4 个 RX 的 I 路交错后，RX0 的部分）
%   第 2 行 = RX1_I 序列
%   第 3 行 = RX0_Q 序列
%   第 4 行 = RX1_Q 序列
%   （RX2/RX3 同理，按同样的去交错规律排列）
%
% 列数 = FrameNum × FrameLength / 2
%   因为每读取 4 个 int16（4 行）对应 2 个采样点的 I/Q 数据
adcData = fread(fid, [4, FrameNum * para.FrameLength / 2], 'int16');
fclose(fid);

% ===== 5. 合并 I/Q 两路为复数 =====
% 预分配复数矩阵 [2 行 × N 列]
%   第 1 行: 两个 RX（对应 adcData 的行 1 和行 3）
%   第 2 行: 另两个 RX（对应 adcData 的行 2 和行 4）
retVal = zeros(2, FrameNum * para.FrameLength / 2);

% I + jQ → 复数
%   行 1 = adcData 第 1 行 (I) + j × adcData 第 3 行 (Q) → 某对 RX 的复数信号
%   行 2 = adcData 第 2 行 (I) + j × adcData 第 4 行 (Q) → 另一对 RX 的复数信号
retVal(1, :) = adcData(1, :) + 1j * adcData(3, :);
retVal(2, :) = adcData(2, :) + 1j * adcData(4, :);

% ===== 6. 重组为四维矩阵 =====
% reshape 将一维复数序列折叠为:
%   [ADCSamples, numRX×GroupNum, ChirpNum, FrameLength]
%
% 维度含义:
%   第 1 维 (ADCSamples): 快时间采样，对应距离信息
%   第 2 维 (numRX×GroupNum): 通道维
%       numRX=4 根接收天线 × GroupNum=3 个发射天线（TDM 模式）= 12 个虚拟通道
%       排列顺序: RX0_TX0, RX1_TX0, ..., RX3_TX2 (共 12 个通道)
%   第 3 维 (ChirpNum): 慢时间维，对应多普勒信息
%       每个 Chirp 对应一组 TDM 循环（所有 TX 各发一次）
%   第 4 维 (FrameLength): 帧维，对应时间演变
%       FramePeriodicity_msec = 60ms 左右，80 帧约 5 秒的动作序列
%
% 'F' 参数表示按 Fortran/列优先顺序 reshape（MATLAB 默认）
retVal = reshape(retVal, para.ADCSamples, para.numRX * para.GroupNum, ...
                 para.ChirpNum, para.FrameLength);

end
