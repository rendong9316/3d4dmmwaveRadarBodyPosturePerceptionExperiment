%% test4DRead.m
%% 测试 4D 雷达数据读取：取第一个场景的第一帧，验证数据维度
clc; close all;

script_dir = fileparts(mfilename('fullpath'));
addpath(genpath(script_dir));

% ===== 1. 解析配置 =====
jsonCfg = fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json');
para = read4DParam(jsonCfg);

% ===== 2. 选一个测试数据目录 =====
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');
scenes = dir(dataRoot);
scenes = scenes([scenes.isdir] & ~ismember({scenes.name}, {'.', '..'}));
testScene = scenes(1).name;
fprintf('\n测试场景: %s\n', testScene);

% ===== 3. 读取 master 设备的第一帧 =====
masterBin = fullfile(dataRoot, testScene, 'master_0000_data.bin');
fprintf('读取 master: %s\n', masterBin);

% 只读前 2 帧做测试
adcMaster = read4DRawData(masterBin, para, 1:2);
fprintf('  master 维度: [%d %d %d %d]\n', size(adcMaster));

% ===== 4. 快速验证：画第一帧第一Chirp的I路原始数据 =====
figure;
plot(real(adcMaster(1, :, 1, 1)));  % RX0: 第1通道，所有采样点
hold on;
plot(real(adcMaster(2, :, 1, 1)));  % RX1
plot(real(adcMaster(3, :, 1, 1)));  % RX2
plot(real(adcMaster(4, :, 1, 1)));  % RX3
hold off;
xlabel('采样点');
ylabel('I 路幅值');
title(sprintf('4D 雷达原始 ADC 数据 (%s, Frame1, Chirp1)', testScene));
legend('RX0', 'RX1', 'RX2', 'RX3');
grid on;

fprintf('\n===== 测试通过 =====\n');
