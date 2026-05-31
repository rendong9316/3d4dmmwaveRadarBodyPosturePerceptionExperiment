%% generate_dataset.m — 批量生成DT图，制作训练/测试数据集
%% 输出: dt_dataset.mat (X_train, y_train, X_test, y_test, class_names)
clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '4Dproject'));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

%% 定义类别和数据集划分
class_names = {'bend','boxing','empty','falldown','jump', ...
               'liedown','run','sit','stand','swing','walk'};
nClasses = length(class_names);

all_scenes_per_class = [100, 100, 20, 100, 100, 100, 100, 100, 100, 100, 100];
base_nTrain = 80;
base_nTest  = 20;

rng(42);  % 固定随机种子

%% 计算每类实际训练/测试数
nTrain_per_class = zeros(nClasses, 1);
nTest_per_class  = zeros(nClasses, 1);
for cls = 1:nClasses
    nTotal = all_scenes_per_class(cls);
    if nTotal >= 100
        nTrain_per_class(cls) = base_nTrain;  % 80
        nTest_per_class(cls)  = base_nTest;   % 20
    else
        nTrain_per_class(cls) = floor(nTotal * 0.7);
        nTest_per_class(cls)  = nTotal - nTrain_per_class(cls);
    end
end
total_train = sum(nTrain_per_class);
total_test  = sum(nTest_per_class);

%% 雷达参数
nSamples = para.ADCSamples;      % 256
nLoops   = para.numLoops;        % 64
nChirps  = para.chirpsPerCycle;  % 12
nRX      = para.numRXPerDevice;  % 4
nChirpsPerFrame = nChirps * nLoops;

range_win = hanning(nSamples);
dopp_win  = hanning(nLoops);

r_min = round(0.5 / para.dr) + 1;
r_max = round(5.0 / para.dr);
zb = floor(nLoops/2) + 1;
non_dc = [1:zb-1, zb+1:nLoops];

%% 输出尺寸
IMG_SIZE = 64;

%% 预分配
X_train = zeros(IMG_SIZE, IMG_SIZE, total_train);
y_train = zeros(total_train, 1);
X_test  = zeros(IMG_SIZE, IMG_SIZE, total_test);
y_test  = zeros(total_test, 1);

train_idx = 1;
test_idx  = 1;

fprintf('========================================\n');
fprintf('批量生成DT数据集 (共计训练%d, 测试%d)\n', total_train, total_test);
fprintf('========================================\n\n');

%% 逐类别处理
for cls = 1:nClasses
    action = class_names{cls};
    nTotal = all_scenes_per_class(cls);
    nTr = nTrain_per_class(cls);
    nTe = nTest_per_class(cls);

    perm = randperm(nTotal);
    train_scenes = perm(1:nTr);
    test_scenes  = perm(nTr+1 : nTr+nTe);

    fprintf('[%s] %d场景, 训练:%d, 测试:%d\n', action, nTotal, nTr, nTe);

    %% --- 训练集 ---
    for s = 1:nTr
        sceneIdx = train_scenes(s);
        scenario = sprintf('CCdata_%s_%04d', action, sceneIdx);
        masterBin = fullfile(dataRoot, scenario, 'master_0000_data.bin');
        
        if ~exist(masterBin, 'file')
            fprintf('  x 训练 %s_%04d 缺失\n', action, sceneIdx);
            continue;
        end
        
        dt_resized = processOneScene(masterBin, nRX, nSamples, nChirps, nLoops, ...
                                      nChirpsPerFrame, range_win, dopp_win, ...
                                      r_min, r_max, non_dc, IMG_SIZE);
        
        if isempty(dt_resized)
            continue;
        end
        
        X_train(:, :, train_idx) = dt_resized;
        y_train(train_idx) = cls;
        train_idx = train_idx + 1;
    end
    
    %% --- 测试集 ---
    for s = 1:nTe
        sceneIdx = test_scenes(s);
        scenario = sprintf('CCdata_%s_%04d', action, sceneIdx);
        masterBin = fullfile(dataRoot, scenario, 'master_0000_data.bin');
        
        if ~exist(masterBin, 'file')
            fprintf('  x 测试 %s_%04d 缺失\n', action, sceneIdx);
            continue;
        end
        
        dt_resized = processOneScene(masterBin, nRX, nSamples, nChirps, nLoops, ...
                                      nChirpsPerFrame, range_win, dopp_win, ...
                                      r_min, r_max, non_dc, IMG_SIZE);
        
        if isempty(dt_resized)
            continue;
        end
        
        X_test(:, :, test_idx) = dt_resized;
        y_test(test_idx) = cls;
        test_idx = test_idx + 1;
    end
    
    fprintf('  ok %s: 训练=%d, 测试=%d\n', action, ...
            sum(y_train(1:train_idx-1)==cls), sum(y_test(1:test_idx-1)==cls));
end

%% 裁剪
X_train = X_train(:, :, 1:train_idx-1);
y_train = y_train(1:train_idx-1);
X_test  = X_test(:, :, 1:test_idx-1);
y_test  = y_test(1:test_idx-1);

fprintf('\n实际样本数: 训练=%d, 测试=%d\n', train_idx-1, test_idx-1);

%% 保存
outFile = fullfile(script_dir, 'dt_dataset.mat');
save(outFile, 'X_train', 'y_train', 'X_test', 'y_test', 'class_names', '-v7');
fprintf('数据集已保存: %s\n', outFile);

fprintf('\n各类别分布:\n');
for cls = 1:nClasses
    fprintf('  %-10s  训练:%d  测试:%d\n', class_names{cls}, ...
            sum(y_train == cls), sum(y_test == cls));
end

%% ===== 子函数 =====
function dt_resized = processOneScene(binFile, nRX, nSamples, nChirps, nLoops, ...
                                       nChirpsPerFrame, range_win, dopp_win, ...
                                       r_min, r_max, non_dc, IMG_SIZE)
    dt_resized = [];
    
    fid = fopen(binFile, 'rb');
    if fid == -1, return; end
    rawData = fread(fid, 'int16');
    fclose(fid);
    
    rawData = rawData(1:2:end) + 1j*rawData(2:2:end);
    
    totalChirps = length(rawData) / (nRX * nSamples);
    nFrames = floor(totalChirps / nChirpsPerFrame);
    
    if nFrames < 2, return; end
    
    rawData = rawData(1 : nRX * nSamples * nFrames * nChirpsPerFrame);
    rawData = reshape(rawData, nRX, nSamples, nChirpsPerFrame, nFrames);
    rawData = reshape(rawData, nRX, nSamples, nChirps, nLoops, nFrames);

    % Single precision for speed, per-frame FFT
    rawData = single(rawData);
    range_win_s = single(range_win);
    dopp_win_s  = single(dopp_win);

    dt_map = zeros(nLoops, nFrames, 'single');
    for f = 1:nFrames
        frame = squeeze(rawData(:, :, :, :, f));
        rfft = fft(frame .* range_win_s.', [], 2);
        rd_mti = rfft - mean(rfft, 4);
        rd = fft(rd_mti .* reshape(dopp_win_s, 1,1,1,nLoops), [], 4);
        rd = fftshift(rd, 4);
        pwr_rd = squeeze(mean(sum(abs(rd).^2, 1), 3));
        dt_map(:, f) = max(pwr_rd(r_min:r_max, :), [], 1);
    end

    bg = prctile(dt_map, 10, 2);
    dt_map_dB = 10*log10(dt_map ./ (bg + 1e-6));
    dt_map_dB = max(min(dt_map_dB, 40), -30);
    
    % 用interp2 resize (无需Image Processing Toolbox)
    [nr, nc] = size(dt_map_dB);
    [xo, yo] = meshgrid(linspace(1, nc, IMG_SIZE), linspace(1, nr, IMG_SIZE));
    [xi, yi] = meshgrid(1:nc, 1:nr);
    dt_resized = interp2(xi, yi, dt_map_dB, xo, yo, 'linear');
end
