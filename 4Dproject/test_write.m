%% run_4Dpointcloud.m  (v3.1 final)
%% 4D点云 - TDM分路检测 + 距离窗约束 + 角度估计
clear; close all;

scenario = "CCdata_walk_0001";
frame_start = 35; frame_end = 55;
POWER_THRESH_DB = 95;
RANGE_TARGET_M = 2.4;
RANGE_WINDOW_M = 1.0;
N_ANGLE_PAD = 256;
ANGLE_PEAK_RATIO_MIN = 1.4;
MAX_DET_PER_FRAME = 30;
Z_MIN = 0.3; Z_MAX = 2.5;
X_MAX = 2.0;
Y_MIN = 0.5; Y_MAX = 6.0;

script_dir = fileparts(mfilename("fullpath"));
addpath(script_dir);
jsonPath = fullfile(script_dir, "..", "4D", "CCconfig_json", "CCconfig_json.mmwave.json");
para = read4DParam(jsonPath);
dataRoot = fullfile(script_dir, "..", "datasets_4Dradar");

fprintf("===== 4D v3.1 =====
");
nRange = para.ADCSamples; nLoops = para.numLoops;
nChirps = para.chirpsPerCycle; nRX = para.numRXPerDevice;
nDev = para.numDevices; totalTX = para.totalTX; totalRX = para.totalRX;
r_axis = (0:nRange-1)" * para.dr;
v_max = para.lambda / (4 * para.Chirptime * nChirps);
v_axis = ((0:nLoops-1) - nLoops/2) * (2*v_max/nLoops);
rMin = max(1, round(0.5/para.dr)); rMax = min(nRange, round(5.0/para.dr));
zb = nLoops/2 + 1;
range_mask = (r_axis >= RANGE_TARGET_M-RANGE_WINDOW_M) & (r_axis <= RANGE_TARGET_M+RANGE_WINDOW_M);
rWin = hanning(nRange); dWin = hanning(nLoops);
tx0 = [3,6,9,12]; tx1 = [2,5,8,11]; tx2 = [1,4,7,10];
devNames = {"master","slave1","slave2","slave3"};
fprintf("fc=%.0fGHz dr=%.1fcm vmax=%.1fm/s target=%.1f+-%.1fm
", para.f0/1e9, para.dr*100, v_max, RANGE_TARGET_M, RANGE_WINDOW_M);
fprintf("Writing this test...
");
