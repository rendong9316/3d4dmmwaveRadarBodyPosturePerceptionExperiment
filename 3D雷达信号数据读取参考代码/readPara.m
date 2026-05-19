function para = readPara(fpath)
%READPARA 从 DCA1000 采集板的 LogFile 中解析雷达工作参数
%
%   DCA1000 每次采集数据时，mmWave Studio 会在数据目录下生成一个
%   名为 1_LogFile.txt 的日志文件，其中记录了本次采集所用的全部
%   雷达参数配置。本函数负责从中提取关键参数并计算派生参数。
%
%   输入: fpath  — LogFile 文件的完整路径（字符串）
%   输出: para   — 结构体，包含所有雷达参数
%
%   示例: para = readPara("D:\data\1_LogFile.txt")
%
%   参数来源说明：
%     ProfileConfig 行：定义单个 Chirp 的波形参数
%     AdvancedFrameConfig 行：定义帧结构（一帧包含多少 Chirp、多少帧）
%
%   派生参数说明：
%     Chirptime  = IDLEtime + ENDtime               → 单个 Chirp 持续时间
%     BandWidth  = FrequencySlope × ADCSamples / Fs  → 有效带宽
%     dr         = c / (2 × BandWidth)               → 距离分辨率
%     df         = 1 / Chirptime / GroupNum          → 多普勒采样间隔
%     dv         = λ × df / 2 / ChirpNum             → 速度分辨率

fidin = fopen(fpath, 'r');
n = 0;

% ===== 第一遍扫描：定位 ProfileConfig 和 AdvancedFrameConfig 的行号 =====
while ~feof(fidin)
    tline = fgetl(fidin);  % 读取一行文本
    n = n + 1;

    if contains(tline, 'ProfileConfig')
        % 找到 ProfileConfig 行 — 包含 Chirp 级波形参数
        n1 = n;
    elseif contains(tline, 'AdvancedFrameConfig')
        % 找到 AdvancedFrameConfig 行 — 包含帧级结构参数
        n2 = n;
    end
end
fclose(fidin);

% ===== 第二遍扫描：读取对应行的参数值 =====
n = 0;
fidin = fopen(fpath, 'r');
while ~feof(fidin)
    tline = fgetl(fidin);
    n = n + 1;

    if n == n1
        % --- 解析 ProfileConfig 行 ---
        % 该行以逗号分隔，各字段位置固定，用逗号索引提取
        a = strfind(tline, ',');  % 找到所有逗号的位置

        % IDLEtime: 空闲时间（Chirp 之间 ADC 不工作的间隔）
        % 位于第 4 个逗号和第 5 个逗号之间，单位 μs → 转为秒（×1e-8，因为原始值×10）
        para.IDLEtime = str2double(tline(a(3)+1:a(4)-1)) * 1e-8;

        % STARTtime: ADC 开始采样的时间（相对于 Chirp 起始时刻）
        para.STARTtime = str2double(tline(a(4)+1:a(5)-1)) * 1e-8;

        % ENDtime: 斜坡结束时间（Chirp 频率上升阶段的持续时间）
        para.ENDtime = str2double(tline(a(5)+1:a(6)-1)) * 1e-8;

        % FrequencySlope: 调频斜率（MHz/μs → Hz/s）
        % 36210 是 IWR6843 (60GHz) 的转换系数（77GHz 器件用 48279）
        para.FrequencySlope = str2double(tline(a(8)+1:a(9)-1)) * 36210 / 1e6 * 1e12;

        % ADCSamples: 每个 Chirp 的 ADC 采样点数
        para.ADCSamples = str2double(tline(a(10)+1:a(11)-1));

        % Fs: ADC 采样率（ksps → Hz）
        para.Fs = str2double(tline(a(11)+1:a(12)-1)) * 1e3;

        % Chirptime: 单个 Chirp 的总时长 = 空闲时间 + 斜坡时间
        para.Chirptime = para.IDLEtime + para.ENDtime;
    end

    if n == n2
        % --- 解析 AdvancedFrameConfig 行 ---
        a = strfind(tline, ',');

        % FrameNum: 本次采集的总帧数（一帧 = 多组 Chirp 循环多次）
        para.FrameNum = str2double(tline(a(39)+1:a(40)-1));

        % GroupNum: 每个子帧包含的 Chirp 数
        % 在 TDM-MIMO 模式下，GroupNum = TX 数量（各 TX 轮流发射一轮为一个 Group）
        para.GroupNum = str2double(tline(a(5)+1:a(6)-1));

        % ChirpNum: 一帧中 Group 的循环次数
        % 一帧总 Chirp 数 = GroupNum × ChirpNum
        para.ChirpNum = str2double(tline(a(6)+1:a(7)-1));

        % Frameinter: 帧间隔时间（一帧结束到下一帧开始的时间）
        % 原始值单位 20ns，×0.5×1e-8 = ×1e-8 秒（系数 0.5 是原始公式）
        para.Frameinter = 0.5 * str2double(tline(a(7)+1:a(8)-1)) * 1e-8;
    end
end
fclose(fidin);

% ===== 固定参数（硬件相关）=====
para.numRX = 4;        % 接收天线数（IWR6843ISK 有 4 根 RX 天线）
para.f0 = 60e9;        % 载频 60 GHz（IWR6843 工作在 60-64 GHz 频段）
para.lambda = 3e8 / para.f0;  % 波长（光速 / 载频）
para.d = para.lambda / 2;     % 阵元间距（半波长，避免栅瓣）

% ===== 派生参数（由基本参数计算得出）=====
% 有效带宽: B = S × N / Fs
%   S: 调频斜率, N: ADC 采样点数, Fs: 采样率
para.BandWidth = para.FrequencySlope * para.ADCSamples / para.Fs;

% 距离分辨率: Δr = c / (2 × B)
%   带宽越大，距离分辨能力越强
para.dr = 3e8 / para.BandWidth / 2;

% 多普勒频率采样间隔: Δf = 1 / Chirptime / GroupNum
%   即一个 Chirp Group（一轮 TDM）周期的倒数
para.df = 1 / para.Chirptime / para.GroupNum;

% 速度分辨率（RD 谱中）: Δv = λ × Δf / (2 × ChirpNum)
%   取决于总观测时长（GroupNum × ChirpNum × Chirptime）
para.dv = para.lambda * para.df / 2 / para.ChirpNum;

end
