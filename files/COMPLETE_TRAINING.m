%% FlowPic 加密流量分类 - 完整训练流程
%% 基于 Poliakov et al. (2025) 论文实现，数据集：MIRAGE-19

clear; clc; close all;

%% ── 第一部分：参数配置 ────────────────────────────────────────

fprintf('【第一部分】参数配置\n\n');

rng(42);

% 路径
DATASET_PATH = 'D:\Desktop\GitHub\flowpic\data\mirage';
CACHE_FILE   = 'D:\Desktop\GitHub\flowpic\data\flowpic_cache.mat';
% 注意：修改 FLOWPIC_SIZE 或归一化方式后，需手动删除缓存文件重新生成

% 数据子集（正式训练设 false）
USE_SUBSET  = false;
SUBSET_SIZE = 30000;

% FlowPic
FLOWPIC_SIZE = 32;
NUM_CHANNELS = 4;    % 上/下行包长 + 上/下行 IAT

% 训练超参数
MAX_EPOCHS    = 50;
BATCH_SIZE    = 128;
LEARNING_RATE = 0.0010;
L2_REG        = 0.0005;

% 数据增强（仅弱高斯噪声，不做翻转/平移等破坏时序语义的操作）
USE_AUGMENTATION = true;
AUG_NOISE_PROB   = 0.5;
AUG_NOISE_STD    = 0.05;

fprintf('  FlowPic : %dx%dx%d\n', FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS);
fprintf('  Epochs  : %d | Batch: %d | LR: %.4f | L2: %.4f\n', ...
        MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE, L2_REG);
fprintf('  增强    : %s (p=%.2f, std=%.3f)\n\n', ...
        string(USE_AUGMENTATION), AUG_NOISE_PROB, AUG_NOISE_STD);

%% ── 第二部分：数据加载 ────────────────────────────────────────

fprintf('【第二部分】数据加载\n\n');

if exist(CACHE_FILE, 'file')
    fprintf('发现缓存，直接加载...\n');
    tic;
    load(CACHE_FILE, 'X', 'Y', 'labels', 'class_names', 'num_samples', 'num_classes');
    fprintf('✓ 缓存加载完成（%.1f 秒）\n\n', toc);
else
    fprintf('未找到缓存，从原始 JSON 加载...\n');
    tic;
    try
        [flows, labels, class_names] = load_mirage_json(DATASET_PATH);
    catch ME
        fprintf('✗ 数据加载失败: %s\n', ME.message);
        return;
    end
    fprintf('✓ 数据加载完成（%.1f 秒）\n\n', toc);

    num_samples = length(flows);
    num_classes = length(class_names);

    %% ── 第三部分：子集采样（可选）────────────────────────────

    fprintf('【第三部分】数据采样\n\n');

    if USE_SUBSET && num_samples > SUBSET_SIZE
        fprintf('分层采样 %d 个样本...\n', SUBSET_SIZE);
        subset_idx = [];
        for c = 1:num_classes
            idx_c = find(labels == c);
            num_c = min(floor(SUBSET_SIZE / num_classes), length(idx_c));
            if num_c > 0
                subset_idx = [subset_idx; idx_c(randperm(length(idx_c), num_c))]; %#ok<AGROW>
            end
        end
        flows       = flows(subset_idx);
        labels      = labels(subset_idx);
        num_samples = length(flows);
        fprintf('✓ 子集采样完成：%d 个样本\n\n', num_samples);
    else
        fprintf('使用全部数据：%d 个样本\n\n', num_samples);
    end

    %% ── 第四部分：生成 FlowPic ───────────────────────────────

    fprintf('【第四部分】生成 FlowPic\n\n');
    fprintf('将 %d 条流转换为 %dx%dx%d FlowPic...\n进度: ', ...
            num_samples, FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS);

    tic;
    X = zeros(FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS, num_samples, 'single');
    progress_step = floor(num_samples / 20);

    for i = 1:num_samples
        X(:,:,:,i) = single(generate_flowpic(flows{i}.lengths, ...
                                              flows{i}.times, ...
                                              flows{i}.directions, ...
                                              FLOWPIC_SIZE));
        if mod(i, progress_step) == 0, fprintf('█'); end
    end
    fprintf(' 完成！（%.1f 秒）\n', toc);

    % log1p 归一化 + 逐样本逐通道 min-max 缩放到 [0, 1]
    fprintf('log1p 归一化 + 逐通道缩放到 [0,1]...\n');
    X = log1p(X);
    for i = 1:size(X, 4)
        for c = 1:NUM_CHANNELS
            ch = X(:,:,c,i);
            ch_max = max(ch(:));
            if ch_max > 0
                X(:,:,c,i) = ch / ch_max;
            end
        end
    end
    fprintf('✓ 归一化完成，数值范围: [%.4f, %.4f]\n\n', min(X(:)), max(X(:)));

    Y = categorical(labels);

    fprintf('保存缓存...\n');
    save(CACHE_FILE, 'X', 'Y', 'labels', 'class_names', 'num_samples', 'num_classes', '-v7.3');
    fprintf('✓ 缓存已保存：%s\n\n', CACHE_FILE);
end

%% ── 第五部分：数据集划分 ─────────────────────────────────────

fprintf('【第五部分】数据集划分（80/10/10 分层抽样）\n\n');

rng(42);

% 划出 10% 测试集
cv1         = cvpartition(Y, 'HoldOut', 0.1, 'Stratify', true);
X_train_val = X(:,:,:,training(cv1));
Y_train_val = Y(training(cv1));
X_test      = X(:,:,:,test(cv1));
Y_test      = Y(test(cv1));

% 从剩余 90% 中划出 1/9 作为验证集（≈整体 10%）
cv2     = cvpartition(Y_train_val, 'HoldOut', 1/9, 'Stratify', true);
X_train = X_train_val(:,:,:,training(cv2));
Y_train = Y_train_val(training(cv2));
X_val   = X_train_val(:,:,:,test(cv2));
Y_val   = Y_train_val(test(cv2));

fprintf('✓ 划分完成\n');
fprintf('  训练集: %d | 验证集: %d | 测试集: %d\n\n', ...
        length(Y_train), length(Y_val), length(Y_test));

% 数据增强（仅作用于训练集，增强后再送入训练）
if USE_AUGMENTATION
    fprintf('数据增强（弱高斯噪声）...\n');
    for i = 1:size(X_train, 4)
        if rand < AUG_NOISE_PROB
            X_train(:,:,:,i) = max(0, min(1, ...
                X_train(:,:,:,i) + randn(size(X_train(:,:,:,i))) * AUG_NOISE_STD));
        end
    end
    fprintf('✓ 数据增强完成\n\n');
end

%% ── 第六部分：构建模型 ───────────────────────────────────────

fprintf('【第六部分】构建 ResNet 模型\n\n');

input_size = [FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS];
fprintf('输入大小: %s | 类别数: %d\n', mat2str(input_size), num_classes);

try
    lgraph = create_flowpic_model(input_size, num_classes);
    fprintf('✓ 模型创建成功（共 %d 层）\n\n', length(lgraph.Layers));
catch ME
    fprintf('✗ 模型创建失败: %s\n', ME.message);
    return;
end

%% ── 第七部分：训练配置 ───────────────────────────────────────

fprintf('【第七部分】训练配置\n\n');

execution_env = 'cpu';
if canUseGPU(), execution_env = 'gpu'; end
fprintf('执行环境: %s\n', upper(execution_env));

% 不使用类别权重
class_weights = [];
fprintf('✓ 对照实验：不使用类别权重\n\n');

% 如需恢复类别权重，注释上面两行，取消注释下面四行：
% class_counts  = histcounts(double(Y_train), 1:num_classes+1);
% class_weights = 1 ./ (class_counts + 1);
% class_weights = class_weights / sum(class_weights) * num_classes;
% fprintf('✓ 类别权重计算完成\n\n');

options = trainingOptions('adam', ...
    'MaxEpochs',           MAX_EPOCHS, ...
    'MiniBatchSize',       BATCH_SIZE, ...
    'InitialLearnRate',    LEARNING_RATE, ...
    'LearnRateSchedule',   'piecewise', ...
    'LearnRateDropPeriod', 25, ...
    'LearnRateDropFactor', 0.1, ...
    'L2Regularization',    L2_REG, ...
    'GradientThreshold',   1.0, ...
    'ValidationPatience',  20, ...
    'ValidationData',      {X_val, Y_val}, ...
    'ValidationFrequency', max(1, floor(length(Y_train) / BATCH_SIZE)), ...
    'Shuffle',             'every-epoch', ...
    'Verbose',             true, ...
    'VerboseFrequency',    50, ...
    'Plots',               'training-progress', ...
    'ExecutionEnvironment', execution_env, ...
    'OutputNetwork',       'best-validation-loss');

%% ── 第八部分：训练 ───────────────────────────────────────────

fprintf('【第八部分】启动训练\n\n');

hyperparameters = struct(...
    'max_epochs',       MAX_EPOCHS, ...
    'batch_size',       BATCH_SIZE, ...
    'learning_rate',    LEARNING_RATE, ...
    'l2_reg',           L2_REG, ...
    'flowpic_size',     FLOWPIC_SIZE, ...
    'num_channels',     NUM_CHANNELS, ...
    'use_augmentation', USE_AUGMENTATION, ...
    'aug_noise_prob',   AUG_NOISE_PROB, ...
    'aug_noise_std',    AUG_NOISE_STD);

[net, results, exp_dir] = run_training_with_logging( ...
    X_train, Y_train, X_val, Y_val, X_test, Y_test, ...
    lgraph, options, class_names, hyperparameters, class_weights);

fprintf('\n✅ 训练完成！结果保存至：%s\n', exp_dir);