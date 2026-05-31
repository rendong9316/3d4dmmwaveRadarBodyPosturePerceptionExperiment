%% gen_dt_images.m -- 批量生成DT图PNG，带进度条，支持断点续跑
clear; close all; clc;

script_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(script_dir, '..', '4Dproject'));

para = read4DParam(fullfile(script_dir, '..', '4D', 'CCconfig_json', 'CCconfig_json.mmwave.json'));
dataRoot = fullfile(script_dir, '..', 'datasets_4Dradar');

%% 类别与划分
class_names = {'bend','boxing','empty','falldown','jump', ...
               'liedown','run','sit','stand','swing','walk'};
scenes_per_class = [100, 100, 20, 100, 100, 100, 100, 100, 100, 100, 100];
nTrain = [80, 80, 14, 80, 80, 80, 80, 80, 80, 80, 80];
nTest  = [20, 20, 6,  20, 20, 20, 20, 20, 20, 20, 20];
rng(42);

%% 雷达参数
nSamples = para.ADCSamples; nLoops = para.numLoops;
nChirps = para.chirpsPerCycle; nRX = para.numRXPerDevice;
nChirpsPerFrame = nChirps * nLoops;

range_win = hanning(nSamples); dopp_win = hanning(nLoops);
r_min = round(0.5/para.dr)+1; r_max = round(5.0/para.dr);
IMG_SIZE = 64;
imgDir = fullfile(script_dir, 'dt_images');

% ===== 统计总任务量 =====
total = sum(nTrain) + sum(nTest);
fprintf('========================================\n');
fprintf('  DT图像数据集生成\n');
fprintf('  训练: %d | 测试: %d | 总计: %d 张\n', sum(nTrain), sum(nTest), total);
fprintf('========================================\n\n');

% ===== 检查 imresize 可用性 =====
has_imresize = (exist('imresize', 'file') == 2);
if ~has_imresize
    fprintf('[!] imresize 不可用，使用 interp2 替代\n');
end

total_count = 0; skip_count = 0; err_count = 0;
t_start = tic;

%% 预计算总任务数用于全局进度
total_tasks = sum(scenes_per_class);
processed_tasks = 0;

%% 逐类处理
for cls = 1:length(class_names)
    name = class_names{cls};
    nTotal = scenes_per_class(cls);
    nTr = nTrain(cls); nTe = nTest(cls);
    perm = randperm(nTotal);

    % 构建任务列表: [sceneIdx, is_train]
    tasks = [perm(1:nTr)', ones(nTr,1); perm(nTr+1:nTr+nTe)', zeros(nTe,1)];
    nTasks = size(tasks, 1);

    fprintf('\n[%d/%d] %s  (%d张)\n', cls, length(class_names), name, nTasks);
    t_cls = tic;

    for i = 1:nTasks
        sceneIdx = tasks(i,1);
        is_train = tasks(i,2);
        subfolder = 'train';
        if ~is_train, subfolder = 'test'; end

        outFile = fullfile(imgDir, subfolder, name, sprintf('%s_%04d.png', name, sceneIdx));
        
        % 确保目录存在
        [outDir, ~, ~] = fileparts(outFile);
        if ~exist(outDir, 'dir')
            mkdir(outDir);
        end
        
        % 断点续跑: 跳过已存在的图片
        if exist(outFile, 'file')
            skip_count = skip_count + 1;
            total_count = total_count + 1;
            processed_tasks = processed_tasks + 1;
            
            % 显示跳过信息
            overall_pct = 100 * processed_tasks / total_tasks;
            elapsed = toc(t_start);
            eta = elapsed / processed_tasks * (total_tasks - processed_tasks);
            fprintf('\r[全局 %5.1f%%] 跳过: %s/%s_%04d (已存在) | 耗时:%6.0fs | 剩余:%6.0fs', ...
                    overall_pct, subfolder, name, sceneIdx, elapsed, eta);
            continue;
        end

        % 读数据
        scenario = sprintf('CCdata_%s_%04d', name, sceneIdx);
        masterBin = fullfile(dataRoot, scenario, 'master_0000_data.bin');
        if ~exist(masterBin, 'file')
            err_count = err_count + 1;
            total_count = total_count + 1;
            processed_tasks = processed_tasks + 1;
            
            % 显示错误信息
            overall_pct = 100 * processed_tasks / total_tasks;
            elapsed = toc(t_start);
            fprintf('\r[全局 %5.1f%%] 错误: %s_%04d (文件不存在) | 耗时:%6.0fs', ...
                    overall_pct, name, sceneIdx, elapsed);
            continue;
        end

        fid = fopen(masterBin, 'rb');
        rawData = fread(fid, 'int16'); fclose(fid);
        rawData = rawData(1:2:end) + 1j*rawData(2:2:end);

        totalChirps = length(rawData) / (nRX * nSamples);
        nFrames = floor(totalChirps / nChirpsPerFrame);
        if nFrames < 2
            err_count = err_count + 1;
            total_count = total_count + 1;
            processed_tasks = processed_tasks + 1;
            
            overall_pct = 100 * processed_tasks / total_tasks;
            elapsed = toc(t_start);
            fprintf('\r[全局 %5.1f%%] 错误: %s_%04d (帧数不足) | 耗时:%6.0fs', ...
                    overall_pct, name, sceneIdx, elapsed);
            continue;
        end

        rawData = rawData(1 : nRX*nSamples*nFrames*nChirpsPerFrame);
        rawData = reshape(rawData, nRX, nSamples, nChirpsPerFrame, nFrames);
        rawData = reshape(rawData, nRX, nSamples, nChirps, nLoops, nFrames);

        % DT图计算
        dt_map = zeros(nLoops, nFrames);
        for f = 1:nFrames
            frame = squeeze(rawData(:, :, :, :, f));
            rfft = fft(frame .* range_win.', [], 2);
            rd_mti = rfft - mean(rfft, 4);
            rd = fft(rd_mti .* reshape(dopp_win, 1,1,1,nLoops), [], 4);
            rd = fftshift(rd, 4);
            pwr_rd = squeeze(mean(sum(abs(rd).^2, 1), 3));
            dt_map(:, f) = max(pwr_rd(r_min:r_max, :), [], 1);
        end

        % 背景归一化 + dB
        bg = prctile(dt_map, 10, 2);
        dt_db = 10*log10(dt_map ./ (bg + 1e-6));
        dt_db = max(min(dt_db, 40), -30);
        dt_norm = (dt_db + 30) / 70;  % 归一化到 [0, 1]

        % resize
        if has_imresize
            dt_resized = imresize(dt_norm, [IMG_SIZE, IMG_SIZE]);
        else
            [nr, nc] = size(dt_norm);
            [xo, yo] = meshgrid(linspace(1, nc, IMG_SIZE), linspace(1, nr, IMG_SIZE));
            [xi, yi] = meshgrid(1:nc, 1:nr);
            dt_resized = interp2(xi, yi, dt_norm, xo, yo, 'linear');
        end

        imwrite(dt_resized, outFile);
        total_count = total_count + 1;
        processed_tasks = processed_tasks + 1;

        % 实时显示进度（每张图片都更新）
        overall_pct = 100 * processed_tasks / total_tasks;
        elapsed = toc(t_start);
        eta = elapsed / processed_tasks * (total_tasks - processed_tasks);
        
        % 使用 \r 实现同一行刷新
        fprintf('\r[全局 %5.1f%%] 处理中: %s/%s_%04d | 已生成:%d | 跳过:%d | 错误:%d | 耗时:%6.0fs | 剩余:%6.0fs', ...
                overall_pct, subfolder, name, sceneIdx, ...
                total_count - skip_count - err_count, skip_count, err_count, ...
                elapsed, eta);
        
        % 每处理完一类，打印一个换行
        if i == nTasks
            fprintf('\n');
        end
    end
    fprintf('  类完成，耗时: %.1f 秒\n', toc(t_cls));
end

elapsed_total = toc(t_start);
fprintf('\n========================================\n');
fprintf('  完成!  总耗时: %.1f 分钟\n', elapsed_total/60);
fprintf('  生成: %d | 跳过: %d | 错误: %d\n', ...
        total_count - skip_count - err_count, skip_count, err_count);
fprintf('========================================\n');