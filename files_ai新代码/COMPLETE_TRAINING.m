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

% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
% ⭐ 配置参数 - 根据你的实际情况修改
% ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

% 1. 数据路径
DATASET_PATH = 'D:\Desktop\GitHub\flowpic\data\mirage';  % MIRAGE数据集路径
% 2. 数据量控制
USE_SUBSET = true;        % 是否使用数据子集
SUBSET_SIZE = 10000;       % 子集大小（建议：测试用1000-2000，正式训练用全部）

% 3. FlowPic配置
FLOWPIC_SIZE = 32;        % FlowPic大小（32x32，论文标准）
NUM_CHANNELS = 4;         % 通道数（4通道：上下行长度+IAT）

% 4. 训练超参数
MAX_EPOCHS = 50;          % 训练轮数
BATCH_SIZE = 64;          % 批大小（GPU内存不足时改为32）
LEARNING_RATE = 0.0005;    % 初始学习率
L2_REG = 0.001;          % L2正则化

% 5. 输出配置
SAVE_MODEL = true;        % 是否保存模型
SAVE_FIGURES = true;      % 是否保存图像
MODEL_NAME = 'flowpic_mirage_model';  % 模型名称

fprintf('配置参数:\n');
fprintf('  数据路径: %s\n', DATASET_PATH);
fprintf('  使用子集: %s (%d samples)\n', string(USE_SUBSET), SUBSET_SIZE);
fprintf('  FlowPic: %dx%dx%d\n', FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS);
fprintf('  训练: %d epochs, batch=%d, lr=%.4f\n\n', MAX_EPOCHS, BATCH_SIZE, LEARNING_RATE);

%% ============================================================
%% 第二部分：数据加载
%% ============================================================

fprintf('【第二部分】加载真实数据\n\n');

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
X = zeros(FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS, num_samples);

progress_step = floor(num_samples / 20);  % 5%进度间隔

for i = 1:num_samples
    % 生成FlowPic
    X(:,:,:,i) = generate_flowpic(flows{i}.lengths, ...
                                   flows{i}.times, ...
                                   flows{i}.directions, ...
                                   FLOWPIC_SIZE);
    
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

% 转换标签
Y = categorical(labels);

% 检查数据质量
fprintf('\nFlowPic数据质量检查:\n');
fprintf('  数值范围: [%.2f, %.2f]\n', min(X(:)), max(X(:)));
fprintf('  平均值: %.2f\n', mean(X(:)));
fprintf('  非零元素: %.1f%%\n', 100 * sum(X(:)~=0) / numel(X));

%% ============================================================
%% 第五部分：数据集划分
%% ============================================================

fprintf('\n【第五部分】划分数据集\n\n');

fprintf('按照 80%%训练 + 10%%验证 + 10%%测试 划分...\n');

% 第一次划分：80% vs 20%
cv1 = cvpartition(num_samples, 'HoldOut', 0.2);
idx_train_val = training(cv1);
idx_test = test(cv1);

X_train_val = X(:,:,:,idx_train_val);
Y_train_val = Y(idx_train_val);
X_test = X(:,:,:,idx_test);
Y_test = Y(idx_test);

% 第二次划分：训练 vs 验证
cv2 = cvpartition(sum(idx_train_val), 'HoldOut', 0.125);  % 10/80 = 0.125
idx_train = training(cv2);
idx_val = test(cv2);

X_train = X_train_val(:,:,:,idx_train);
Y_train = Y_train_val(idx_train);
X_val = X_train_val(:,:,:,idx_val);
Y_val = Y_train_val(idx_val);

fprintf('✓ 数据集划分完成\n');
fprintf('  训练集: %d 样本 (%.1f%%)\n', sum(idx_train), 100*sum(idx_train)/num_samples);
fprintf('  验证集: %d 样本 (%.1f%%)\n', sum(idx_val), 100*sum(idx_val)/num_samples);
fprintf('  测试集: %d 样本 (%.1f%%)\n', sum(idx_test), 100*sum(idx_test)/num_samples);

% 检查训练集类别分布
fprintf('\n训练集类别分布:\n');
for c = 1:num_classes
    count = sum(double(Y_train) == c);
    if count > 0
        fprintf('  %2d. %-20s: %4d (%.1f%%)\n', ...
                c, class_names{c}, count, 100*count/length(Y_train));
    end
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
fprintf('  主要结构: ResNet with 2 residual blocks\n');
fprintf('  - Block 1: 64 filters\n');
fprintf('  - Block 2: 128 filters\n');
fprintf('  - Classifier: 256 → %d classes\n', num_classes);

%% ============================================================
%% 第七部分：配置训练选项
%% ============================================================

fprintf('\n【第七部分】配置训练选项\n\n');

% 检测GPU
use_gpu = false;
try
    gpu_device = gpuDevice;
    use_gpu = true;
    fprintf('✓ 检测到GPU: %s\n', gpu_device.Name);
    fprintf('  显存: %.1f GB\n', gpu_device.AvailableMemory / 1024^3);
    execution_env = 'gpu';
catch
    fprintf('ℹ 未检测到GPU，使用CPU训练\n');
    fprintf('  提示: GPU训练速度快10-20倍\n');
    execution_env = 'auto';
end

fprintf('\n训练超参数:\n');
fprintf('  ├─ Epochs: %d\n', MAX_EPOCHS);
fprintf('  ├─ Batch Size: %d\n', BATCH_SIZE);
fprintf('  ├─ Learning Rate: %.4f\n', LEARNING_RATE);
fprintf('  ├─ LR Schedule: Piecewise (drop every 20 epochs)\n');
fprintf('  ├─ L2 Regularization: %.5f\n', L2_REG);
fprintf('  ├─ Optimizer: Adam\n');
fprintf('  └─ Execution: %s\n', execution_env);

% 创建训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', MAX_EPOCHS, ...
    'MiniBatchSize', BATCH_SIZE, ...
    'InitialLearnRate', LEARNING_RATE, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', L2_REG, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', max(1, floor(length(Y_train)/BATCH_SIZE)), ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 5, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', execution_env, ...
    'OutputNetwork', 'best-validation-loss');

% 估算训练时间
num_batches_per_epoch = ceil(length(Y_train) / BATCH_SIZE);
fprintf('\n训练规模:\n');
fprintf('  每个epoch: %d batches\n', num_batches_per_epoch);
fprintf('  总迭代数: %d\n', num_batches_per_epoch * MAX_EPOCHS);

%% ============================================================
%% 第八部分：训练模型
%% ============================================================

fprintf('\n【第八部分】开始训练\n\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
fprintf('  训练进度将显示在弹出的窗口中\n');
fprintf('  请勿关闭MATLAB，训练需要较长时间\n');
fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n\n');

training_start = tic;

try
    net = trainNetwork(X_train, Y_train, lgraph, options);
    training_time = toc(training_start);
    
    fprintf('\n━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('✓ 训练完成！\n');
    fprintf('━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n');
    fprintf('  总用时: %.1f 分钟 (%.1f 小时)\n', ...
            training_time/60, training_time/3600);
    
catch ME
    fprintf('\n✗ 训练失败: %s\n', ME.message);
    fprintf('  请检查:\n');
    fprintf('    1. 内存/显存是否充足\n');
    fprintf('    2. 数据是否有效\n');
    fprintf('    3. 如果是GPU错误，尝试使用CPU\n');
    return;
end

%% ============================================================
%% 第九部分：模型评估
%% ============================================================

fprintf('\n【第九部分】评估模型性能\n\n');

fprintf('在各数据集上评估模型...\n');

% 训练集
fprintf('\n[训练集]\n');
Y_train_pred = classify(net, X_train);
train_accuracy = sum(Y_train_pred == Y_train) / numel(Y_train);
fprintf('  准确率: %.2f%%\n', train_accuracy * 100);

% 验证集
fprintf('\n[验证集]\n');
Y_val_pred = classify(net, X_val);
val_accuracy = sum(Y_val_pred == Y_val) / numel(Y_val);
fprintf('  准确率: %.2f%%\n', val_accuracy * 100);

% 测试集
fprintf('\n[测试集] ⭐ 最终评估指标\n');
Y_test_pred = classify(net, X_test);
test_accuracy = sum(Y_test_pred == Y_test) / numel(Y_test);
fprintf('  准确率: %.2f%%\n', test_accuracy * 100);

% 每类准确率
fprintf('\n测试集各类别性能:\n');
fprintf('  %3s %-20s %8s %10s %8s\n', 'ID', '类别', '样本数', '正确数', '准确率');
fprintf('  %s\n', repmat('-', 1, 60));

class_accuracies = zeros(num_classes, 1);
for c = 1:num_classes
    idx_c = double(Y_test) == c;
    if sum(idx_c) > 0
        correct = sum(Y_test_pred(idx_c) == Y_test(idx_c));
        acc_c = correct / sum(idx_c);
        class_accuracies(c) = acc_c;
        
        fprintf('  %3d %-20s %8d %10d %7.1f%%\n', ...
                c, class_names{c}, sum(idx_c), correct, acc_c*100);
    end
end

% 混淆矩阵
fprintf('\n生成混淆矩阵...\n');
conf_mat = confusionmat(Y_test, Y_test_pred);

%% ============================================================
%% 第十部分：保存结果
%% ============================================================

fprintf('\n【第十部分】保存结果\n\n');

% 保存模型
if SAVE_MODEL
    fprintf('保存训练好的模型...\n');
    
    results = struct();
    results.train_accuracy = train_accuracy;
    results.val_accuracy = val_accuracy;
    results.test_accuracy = test_accuracy;
    results.class_accuracies = class_accuracies;
    results.training_time = training_time;
    results.class_names = class_names;
    results.confusion_matrix = conf_mat;
    results.hyperparameters = struct(...
        'flowpic_size', FLOWPIC_SIZE, ...
        'num_channels', NUM_CHANNELS, ...
        'max_epochs', MAX_EPOCHS, ...
        'batch_size', BATCH_SIZE, ...
        'learning_rate', LEARNING_RATE, ...
        'l2_reg', L2_REG);
    results.dataset_info = struct(...
        'num_samples', num_samples, ...
        'num_classes', num_classes, ...
        'train_size', length(Y_train), ...
        'val_size', length(Y_val), ...
        'test_size', length(Y_test));
    
    model_file = sprintf('models/%s.mat', MODEL_NAME);
    save(model_file, 'net', 'results');
    fprintf('  ✓ 模型保存到: %s\n', model_file);
end

% 保存可视化
if SAVE_FIGURES
    fprintf('\n生成可视化结果...\n');
    
    % 图1: 混淆矩阵
    fig1 = figure('Position', [100, 100, 800, 700]);
    cm = confusionchart(Y_test, Y_test_pred);
    cm.Title = sprintf('混淆矩阵 - 测试集 (准确率: %.2f%%)', test_accuracy * 100);
    cm.RowSummary = 'row-normalized';
    cm.ColumnSummary = 'column-normalized';
    
    fig1_file = sprintf('figures/%s_confusion.png', MODEL_NAME);
    saveas(fig1, fig1_file);
    fprintf('  ✓ 混淆矩阵: %s\n', fig1_file);
    
    % 图2: 性能对比
    fig2 = figure('Position', [100, 100, 1200, 500]);
    
    subplot(1, 2, 1);
    bar([train_accuracy, val_accuracy, test_accuracy] * 100);
    set(gca, 'XTickLabel', {'训练集', '验证集', '测试集'});
    ylabel('准确率 (%)');
    title('整体准确率对比');
    grid on;
    ylim([0, 100]);
    
    % 添加数值标注
    hold on;
    text(1:3, [train_accuracy, val_accuracy, test_accuracy]*100, ...
         arrayfun(@(x) sprintf('%.2f%%', x), ...
                  [train_accuracy, val_accuracy, test_accuracy]*100, ...
                  'UniformOutput', false), ...
         'HorizontalAlignment', 'center', ...
         'VerticalAlignment', 'bottom');
    
    subplot(1, 2, 2);
    bar(class_accuracies * 100);
    set(gca, 'XTickLabel', class_names);
    ylabel('准确率 (%)');
    title('各类别准确率（测试集）');
    grid on;
    ylim([0, 100]);
    xtickangle(45);
    
    fig2_file = sprintf('figures/%s_performance.png', MODEL_NAME);
    saveas(fig2, fig2_file);
    fprintf('  ✓ 性能对比: %s\n', fig2_file);
end

%% ============================================================
%% 第十一部分：结果总结
%% ============================================================

fprintf('\n============================================================\n');
fprintf('    训练完成总结\n');
fprintf('============================================================\n\n');

fprintf('【数据集信息】\n');
fprintf('  数据来源: MIRAGE-19\n');
fprintf('  总样本数: %d\n', num_samples);
fprintf('  类别数: %d\n', num_classes);
fprintf('  训练/验证/测试: %d / %d / %d\n', ...
        length(Y_train), length(Y_val), length(Y_test));

fprintf('\n【模型配置】\n');
fprintf('  FlowPic: %dx%dx%d\n', FLOWPIC_SIZE, FLOWPIC_SIZE, NUM_CHANNELS);
fprintf('  架构: ResNet (2 residual blocks)\n');
fprintf('  参数量: ~58K\n');

fprintf('\n【训练配置】\n');
fprintf('  Epochs: %d\n', MAX_EPOCHS);
fprintf('  Batch Size: %d\n', BATCH_SIZE);
fprintf('  Learning Rate: %.4f\n', LEARNING_RATE);
fprintf('  训练时间: %.1f 分钟\n', training_time/60);
fprintf('  设备: %s\n', execution_env);

fprintf('\n【性能结果】 ⭐\n');
fprintf('  训练集准确率: %.2f%%\n', train_accuracy * 100);
fprintf('  验证集准确率: %.2f%%\n', val_accuracy * 100);
fprintf('  测试集准确率: %.2f%% ← 最终评估指标\n', test_accuracy * 100);

fprintf('\n【论文基准对比】\n');
fprintf('  论文 (MIRAGE-19, 全数据): 81.10%%\n');
fprintf('  你的结果: %.2f%%\n', test_accuracy * 100);

if test_accuracy >= 0.75
    fprintf('  ✓ 性能良好！');
    if test_accuracy >= 0.80
        fprintf('达到论文水平！');
    end
    fprintf('\n');
elseif test_accuracy >= 0.60
    fprintf('  ℹ 性能可接受，可能原因:\n');
    fprintf('    - 使用了数据子集（非全量）\n');
    fprintf('    - 可以尝试增加训练数据或epochs\n');
else
    fprintf('  ⚠ 性能较低，建议:\n');
    fprintf('    - 检查数据质量\n');
    fprintf('    - 增加训练数据量\n');
    fprintf('    - 调整超参数\n');
end

fprintf('\n【保存的文件】\n');
if SAVE_MODEL
    fprintf('  模型: models/%s.mat\n', MODEL_NAME);
end
if SAVE_FIGURES
    fprintf('  图像: figures/%s_*.png\n', MODEL_NAME);
end

fprintf('\n【下一步】\n');
fprintf('  1. 查看混淆矩阵分析错误类型\n');
fprintf('  2. 使用更多数据重新训练\n');
fprintf('  3. 尝试调整超参数优化性能\n');
fprintf('  4. 在新数据上测试模型泛化能力\n');

fprintf('\n============================================================\n');
fprintf('    全部完成！\n');
fprintf('============================================================\n\n');
