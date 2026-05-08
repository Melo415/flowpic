%% ============================================================
%% FlowPic 真实数据训练 - 完整流程
%% ============================================================
%% 
%% 本脚本演示如何使用真实网络流量数据（MIRAGE-19）
%% 从零开始构建FlowPic分类模型
%% 
%% 作者：基于 Poliakov et al. (2025) 论文实现
%% 日期：2026
%% 
%% ============================================================

clear; clc; close all;

fprintf('============================================================\n');
fprintf('    FlowPic真实数据训练 - 完整教程\n');
fprintf('============================================================\n\n');

%% ============================================================
%% 第一部分：环境配置和参数设置
%% ============================================================

fprintf('【第一部分】环境配置\n\n');

% 设置随机种子（保证结果可重复）
rng(42);

% 配置参数

% 1. 数据路径
DATASET_PATH = 'D:\Desktop\GitHub\flowpic\data\mirage';  % MIRAGE数据集路径

% 2. 数据量控制
USE_SUBSET = false;        % 是否使用数据子集
SUBSET_SIZE = 30000;       % 子集大小（建议：测试用1000-2000，正式训练用全部）

% 3. FlowPic配置
FLOWPIC_SIZE = 64;        % FlowPic大小（32x32）
NUM_CHANNELS = 4;         % 通道数（4通道：上下行长度+IAT）

% 4. 训练超参数
MAX_EPOCHS = 150;          % 训练轮数
BATCH_SIZE = 64;          % 批大小（GPU内存不足时改为32）
LEARNING_RATE = 0.0005;    % 初始学习率
L2_REG = 0.0001;          % L2正则化

% 4.1 数据增强配置（FlowPic是时序直方图，不做会破坏时间语义的增强）
USE_AUGMENTATION = true;   % 是否启用训练集增强
AUG_NOISE_PROB = 0.35;     % 弱高斯噪声触发概率
AUG_NOISE_STD = 0.02;      % 弱高斯噪声强度

% 5. 输出配置
SAVE_MODEL = true;        % 是否保存模型
SAVE_FIGURES = true;      % 是否保存图像
MODEL_NAME = 'flowpic_mirage_model';  % 模型名称

fprintf('配置参数:\n');
fprintf('  数据路径: %s\n', DATASET_PATH);
fprintf('  使用子集: %s (%d samples)\n', string(USE_SUBSET), SUBSET_SIZE);
fprintf('  FlowPic: %dx%dx%d\n', FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS);
fprintf('  训练: %d epochs, batch=%d, lr=%.4f\n', MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE);
fprintf('  增强: %s (仅弱高斯噪声, p=%.2f, std=%.3f)\n\n', ...
        string(USE_AUGMENTATION), AUG_NOISE_PROB, AUG_NOISE_STD);

%% ============================================================
%% 第二部分：数据加载
%% ============================================================

fprintf('【第二部分】加载真实数据\n\n');

% ── 缓存路径 ──────────────────────────────────────────
CACHE_FILE = 'D:\Desktop\GitHub\flowpic\data\flowpic_cache.mat';
%% 如果换了数据或修改了归一化方式，记得手动删掉缓存文件重新生成。

if exist(CACHE_FILE, 'file')
    fprintf('发现缓存文件，直接加载...\n');
    tic;
    load(CACHE_FILE, 'X', 'Y', 'labels', 'class_names', 'num_samples', 'num_classes');
    fprintf('✓ 缓存加载完成！用时: %.1f 秒\n\n', toc);
    % 跳过第二、三、四部分
else
    fprintf('未找到缓存，从原始JSON加载...\n');


fprintf('正在从 %s 加载MIRAGE数据...\n', DATASET_PATH);
fprintf('这可能需要几分钟，请耐心等待...\n\n');

tic;
try
    % 加载MIRAGE-19 JSON数据
    [flows, labels, class_names] = load_mirage_json(DATASET_PATH);
    load_time = toc;
    
    fprintf('✓ 数据加载成功！用时: %.1f 秒\n\n', load_time);
    
catch ME
    fprintf('✗ 数据加载失败: %s\n', ME.message);
    fprintf('\n请检查:\n');
    fprintf('  1. DATASET_PATH 是否正确\n');
    fprintf('  2. load_mirage_json.m 是否在当前目录\n');
    fprintf('  3. JSON文件格式是否正确\n\n');
    return;
end

num_samples = length(flows);
num_classes = length(class_names);

fprintf('数据集统计:\n');
fprintf('  总样本数: %d\n', num_samples);
fprintf('  类别数: %d\n', num_classes);
fprintf('  类别: %s\n', strjoin(class_names, ', '));

% 显示类别分布
fprintf('\n类别分布:\n');
for c = 1:num_classes
    count = sum(labels == c);
    fprintf('  %2d. %-20s: %5d (%.1f%%)\n', ...
            c, class_names{c}, count, 100*count/num_samples);
end

% 包统计
packet_counts = cellfun(@(x) length(x.lengths), flows);
fprintf('\n包统计:\n');
fprintf('  平均包数/流: %.1f\n', mean(packet_counts));
fprintf('  包数范围: [%d, %d]\n', min(packet_counts), max(packet_counts));
fprintf('  中位数: %.0f\n', median(packet_counts));

%% ============================================================
%% 第三部分：数据子集采样（可选）
%% ============================================================

if USE_SUBSET && num_samples > SUBSET_SIZE
    fprintf('\n【第三部分】采样数据子集\n\n');
    
    fprintf('从 %d 个样本中分层采样 %d 个...\n', num_samples, SUBSET_SIZE);
    
    % 分层采样（保持类别比例）
    subset_idx = [];
    for c = 1:num_classes
        idx_c = find(labels == c);
        num_c = min(floor(SUBSET_SIZE / num_classes), length(idx_c));
        
        if num_c > 0
            selected = idx_c(randperm(length(idx_c), num_c));
            subset_idx = [subset_idx; selected];
        end
    end
    
    % 应用子集
    flows = flows(subset_idx);
    labels = labels(subset_idx);
    num_samples = length(flows);
    
    fprintf('✓ 子集采样完成\n');
    fprintf('  最终样本数: %d\n', num_samples);
    
    % 显示子集类别分布
    fprintf('\n子集类别分布:\n');
    for c = 1:num_classes
        count = sum(labels == c);
        if count > 0
            fprintf('  %2d. %-20s: %5d (%.1f%%)\n', ...
                    c, class_names{c}, count, 100*count/num_samples);
        end
    end
else
    fprintf('\n【第三部分】使用全部数据\n\n');
    fprintf('样本数: %d\n', num_samples);
end

%% ============================================================
%% 第四部分：生成FlowPic特征
%% ============================================================

fprintf('\n【第四部分】生成FlowPic特征\n\n');

fprintf('将 %d 个网络流转换为 %dx%dx%d FlowPic...\n', ...
        num_samples, FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS);
fprintf('进度: ');

tic;
X = zeros(FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS, num_samples, 'single');

progress_step = floor(num_samples / 20);  % 5%进度间隔

for i = 1:num_samples
    % 生成FlowPic
    X(:,:,:,i) = single(generate_flowpic(flows{i}.lengths, ...
                                          flows{i}.times, ...
                                          flows{i}.directions, ...
                                          FLOWPIC_SIZE));
    
    % 显示进度
    if mod(i, progress_step) == 0
        fprintf('█');
    end
end

generation_time = toc;
fprintf(' 完成！\n');
fprintf('✓ FlowPic生成完成\n');
fprintf('  用时: %.1f 秒 (%.4f 秒/样本)\n', generation_time, generation_time/num_samples);
fprintf('  数据形状: %s\n', mat2str(size(X)));

% ── 关键改进：对FlowPic做 log1p 归一化 ──────────────────────────
% 问题：直方图计数值分布极度不均匀（大量0，少量大值）
%       直接喂给网络会导致梯度爆炸/消失
% 解决：log(1+x) 压缩计数值，让分布更均匀

fprintf('\n对FlowPic进行log1p归一化...\n');
X = log1p(X);

% 逐通道标准化到 [0, 1]（对每个样本独立归一化）
% 这样每张"图"的尺度一致，避免流量大的应用主导训练
fprintf('逐样本归一化到[0,1]...\n');
for i = 1:size(X, 4)
    for c = 1:NUM_CHANNELS
        channel = X(:,:,c,i);
        ch_max = max(channel(:));
        if ch_max > 0
            X(:,:,c,i) = channel / ch_max;
        end
    end
end

fprintf('✓ 归一化完成\n');
fprintf('  归一化后数值范围: [%.4f, %.4f]\n', min(X(:)), max(X(:)));

% 转换标签
Y = categorical(labels);

% ── 保存缓存 ──────────────────────────────────────────
fprintf('保存缓存文件（下次直接加载）...\n');
save(CACHE_FILE, 'X', 'Y', 'labels', 'class_names', 'num_samples', 'num_classes', '-v7.3');
fprintf('✓ 缓存已保存到: %s\n', CACHE_FILE);

end  % 对应第二部分开头的 else

% 检查数据质量
fprintf('\nFlowPic数据质量检查:\n');
fprintf('  数值范围: [%.2f, %.2f]\n', min(X(:)), max(X(:)));
fprintf('  平均值: %.2f\n', mean(X(:)));
fprintf('  非零元素: %.1f%%\n', 100 * sum(X(:)~=0) / numel(X));

%% ============================================================
%% 第五部分：数据集划分
%% ============================================================

fprintf('\n【第五部分】分层划分数据集\n\n');
fprintf('按照 80%%训练 + 10%%验证 + 10%%测试 划分（分层抽样，保证每类都有代表）...\n');

% 使用 Y（类别标签）进行分层抽样
rng(42);  % 固定随机种子，保证可复现

% 第一步：先划分 90% (train+val) 和 10% test
cv1 = cvpartition(Y, 'HoldOut', 0.1, 'Stratify', true);
idx_train_val = training(cv1);
idx_test = test(cv1);

X_train_val = X(:,:,:,idx_train_val);
Y_train_val = Y(idx_train_val);
X_test = X(:,:,:,idx_test);
Y_test = Y(idx_test);

% 第二步：从 train_val 中划 1/9 做 val
% 因为 90% 里的 1/9 正好是整体 10%
cv2 = cvpartition(Y_train_val, 'HoldOut', 1/9, 'Stratify', true);
idx_train = training(cv2);
idx_val = test(cv2);

X_train = X_train_val(:,:,:,idx_train);
Y_train = Y_train_val(idx_train);
X_val = X_train_val(:,:,:,idx_val);
Y_val = Y_train_val(idx_val);

% 打印划分结果
fprintf('✓ 分层划分完成\n');
fprintf(' 训练集: %d 样本 (%.1f%%)\n', length(Y_train), 100*length(Y_train)/num_samples);
fprintf(' 验证集: %d 样本 (%.1f%%)\n', length(Y_val),   100*length(Y_val)/num_samples);
fprintf(' 测试集: %d 样本 (%.1f%%)\n', length(Y_test),  100*length(Y_test)/num_samples);

% 检查每个集的类别分布
disp('训练集类别分布:'); tabulate(Y_train);
disp('验证集类别分布:'); tabulate(Y_val);
disp('测试集类别分布:'); tabulate(Y_test);

fprintf('【保守版数据增强】只对训练集，保留时序语义...\n');
if USE_AUGMENTATION
    for i = 1:size(X_train, 4)
        img = X_train(:,:,:,i);

        % 仅保留弱高斯噪声：
        % FlowPic 的横轴是时间，不能做 fliplr / circshift 这类会破坏时间语义的增强
        if rand < AUG_NOISE_PROB
            noise = randn(size(img)) * AUG_NOISE_STD;
            img = img + noise;
        end

        % 保证数值仍在 [0, 1]
        X_train(:,:,:,i) = max(0, min(1, img));
    end
    fprintf('✓ 已完成保守增强：仅弱高斯噪声，不做翻转 / 循环平移 / 随机通道置零\n');
else
    fprintf('✓ 已关闭训练集增强，直接使用原始 FlowPic 训练\n');
end



%% ============================================================
%% 第六部分：构建神经网络模型
%% ============================================================

fprintf('\n【第六部分】构建ResNet模型\n\n');

input_size = [FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS];

fprintf('创建 FlowPic ResNet 模型...\n');
fprintf('  输入大小: %s\n', mat2str(input_size));
fprintf('  输出类别: %d\n', num_classes);

try
    lgraph = create_flowpic_model(input_size, num_classes);
    fprintf('✓ 模型创建成功\n');
catch ME
    fprintf('✗ 模型创建失败: %s\n', ME.message);
    return;
end

% 模型统计
layers = lgraph.Layers;
fprintf('\n模型架构:\n');
fprintf('  总层数: %d\n', length(layers));
fprintf('  主要结构: ResNet with 3 residual blocks\n');
fprintf('  - Block 1: 64 filters\n');
fprintf('  - Block 2: 128 filters (stride=2)\n');
fprintf('  - Block 3: 256 filters (stride=2)\n');
fprintf('  - Classifier: 256 → %d classes\n', num_classes);

%% ============================================================
%% 第七部分：配置训练选项
%% ============================================================

fprintf('\n【第七部分】配置训练选项\n\n');

% 检测执行环境
if canUseGPU()
    execution_env = 'gpu';
    fprintf('  └─ Execution: GPU\n');
else
    execution_env = 'cpu';
    fprintf('  └─ Execution: CPU（建议使用GPU加速）\n');
end

% 计算类别权重
train_labels_num2 = double(Y_train);
class_counts = histcounts(train_labels_num2, 1:num_classes+1);
class_weights = 1 ./ (class_counts + 1);
class_weights = class_weights / sum(class_weights) * num_classes;
fprintf('类别权重计算完成\n');

options = trainingOptions('adam', ...
    'MaxEpochs',          MAX_EPOCHS, ...
    'MiniBatchSize',      BATCH_SIZE, ...
    'InitialLearnRate',   LEARNING_RATE, ...
    'LearnRateSchedule',  'none', ...
    'LearnRateDropFactor', 0.3, ...          
    'LearnRateDropPeriod', 50, ...           
    'L2Regularization',   L2_REG, ...
    'GradientThreshold',  1.0, ...           % 梯度裁剪，防爆炸
    'ValidationPatience', 12, ...             % 验证损失连续8次不降就停止
    'ValidationData',     {X_val, Y_val}, ...
    'ValidationFrequency', max(1, floor(length(Y_train)/BATCH_SIZE)), ...
    'Shuffle',            'every-epoch', ...
    'Verbose',            true, ...
    'VerboseFrequency',   5, ...
    'Plots',              'training-progress', ...
    'ExecutionEnvironment', execution_env, ...
    'OutputNetwork',      'best-validation-loss');  % 保存验证最优

%% ============================================================
%% 第八部分：训练模型（使用实验记录函数）
%% ============================================================
fprintf('\n【第八部分】启动带完整实验记录的训练...\n');

% 准备超参数结构体（传给记录函数）
hyperparameters = struct(...
    'max_epochs', MAX_EPOCHS, ...
    'batch_size', BATCH_SIZE, ...
    'learning_rate', LEARNING_RATE, ...
    'l2_reg', L2_REG, ...
    'flowpic_size', FLOWPIC_SIZE, ...
    'num_channels', NUM_CHANNELS, ...
    'use_augmentation', USE_AUGMENTATION, ...
    'aug_noise_prob', AUG_NOISE_PROB, ...
    'aug_noise_std', AUG_NOISE_STD);

% ====================== 核心调用（只保留这一段） ======================
[net, results, exp_dir] = run_training_with_logging( ...
    X_train, Y_train, X_val, Y_val, X_test, Y_test, ...
    lgraph, options, class_names, hyperparameters, class_weights);

fprintf('✅ 训练及所有实验记录完成！\n');
fprintf('   实验文件夹：%s\n', exp_dir);
fprintf('   推荐直接打开 experiment_summary.md 查看完整报告\n\n');