%% 使用MIRAGE数据集训练FlowPic模型

clear; clc;
fprintf('====================================================\n');
fprintf('   使用MIRAGE数据集训练\n');
fprintf('====================================================\n\n');

%% 配置
rng(42);

% 指定数据集路径
DATASET_PATH = 'data/mirage';  % 你的MIRAGE数据路径
USE_SUBSET = true;              % 是否使用子集（推荐先用子集测试）
SUBSET_SIZE = 5000;             % 子集大小（完整数据集可能有10万+）

%% [1/8] 加载MIRAGE数据
fprintf('[1/8] 加载MIRAGE数据集...\n');

try
    [flows, labels, class_names] = load_mirage_data(DATASET_PATH);
    fprintf('  ✓ 数据加载成功\n');
catch ME
    fprintf('  ✗ 数据加载失败: %s\n', ME.message);
    fprintf('\n请检查:\n');
    fprintf('  1. DATASET_PATH 是否正确\n');
    fprintf('  2. 数据集格式是否为CSV\n');
    fprintf('  3. CSV列名是否匹配（见 load_mirage_data.m）\n');
    return;
end

num_samples = length(flows);
num_classes = length(class_names);

% 使用子集（可选）
if USE_SUBSET && num_samples > SUBSET_SIZE
    fprintf('\n  使用数据子集进行训练\n');
    
    % 分层采样（保持类别比例）
    subset_idx = [];
    for c = 1:num_classes
        idx_c = find(labels == c);
        num_c = min(floor(SUBSET_SIZE / num_classes), length(idx_c));
        subset_idx = [subset_idx; idx_c(randperm(length(idx_c), num_c))];
    end
    
    flows = flows(subset_idx);
    labels = labels(subset_idx);
    num_samples = length(flows);
    
    fprintf('  ✓ 子集大小: %d\n', num_samples);
end

fprintf('  最终使用: %d 个样本, %d 个类别\n', num_samples, num_classes);

%% [2/8] 生成FlowPic特征
fprintf('\n[2/8] 生成FlowPic特征...\n');

flowpic_size = 32;
num_channels = 4;

fprintf('  配置: %dx%dx%d FlowPic\n', flowpic_size, flowpic_size, num_channels);
fprintf('  正在生成');

tic;
X = zeros(flowpic_size, flowpic_size, num_channels, num_samples);

for i = 1:num_samples
    if mod(i, 500) == 0
        fprintf('.');
    end
    
    X(:,:,:,i) = generate_flowpic(flows{i}.lengths, ...
                                   flows{i}.times, ...
                                   flows{i}.directions, ...
                                   flowpic_size);
end

generation_time = toc;
fprintf(' 完成！\n');
fprintf('  ✓ 生成用时: %.2f 秒 (%.4f 秒/样本)\n', ...
        generation_time, generation_time/num_samples);

Y = categorical(labels);

%% [3/8] 数据集划分
fprintf('\n[3/8] 划分数据集...\n');

cv1 = cvpartition(num_samples, 'HoldOut', 0.2);
idx_train_val = training(cv1);
idx_test = test(cv1);

X_train_val = X(:,:,:,idx_train_val);
Y_train_val = Y(idx_train_val);
X_test = X(:,:,:,idx_test);
Y_test = Y(idx_test);

cv2 = cvpartition(sum(idx_train_val), 'HoldOut', 0.125);
idx_train = training(cv2);
idx_val = test(cv2);

X_train = X_train_val(:,:,:,idx_train);
Y_train = Y_train_val(idx_train);
X_val = X_train_val(:,:,:,idx_val);
Y_val = Y_train_val(idx_val);

fprintf('  数据集划分:\n');
fprintf('    - 训练集: %d (%.1f%%)\n', sum(idx_train), 100*sum(idx_train)/num_samples);
fprintf('    - 验证集: %d (%.1f%%)\n', sum(idx_val), 100*sum(idx_val)/num_samples);
fprintf('    - 测试集: %d (%.1f%%)\n', sum(idx_test), 100*sum(idx_test)/num_samples);

%% [4/8] 创建模型
fprintf('\n[4/8] 创建模型...\n');

input_size = [flowpic_size, flowpic_size, num_channels];
lgraph = create_flowpic_model(input_size, num_classes);
fprintf('  ✓ 模型创建成功\n');

%% [5/8] 配置训练选项
fprintf('\n[5/8] 配置训练选项...\n');

use_gpu = false;
try
    gpu_device = gpuDevice;
    use_gpu = true;
    fprintf('  ✓ 检测到GPU: %s\n', gpu_device.Name);
    execution_env = 'gpu';
catch
    fprintf('  ℹ 未检测到GPU，使用CPU训练\n');
    execution_env = 'auto';
end

max_epochs = 50;
mini_batch_size = 64;
initial_learn_rate = 0.001;

fprintf('  训练超参数:\n');
fprintf('    - Epochs: %d\n', max_epochs);
fprintf('    - Batch Size: %d\n', mini_batch_size);
fprintf('    - Initial LR: %.4f\n', initial_learn_rate);
fprintf('    - Execution: %s\n', execution_env);

options = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_learn_rate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.0001, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', max(1, floor(length(Y_train)/mini_batch_size)), ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 10, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', execution_env, ...
    'OutputNetwork', 'best-validation-loss');

%% [6/8] 训练模型
fprintf('\n[6/8] 开始训练...\n');
fprintf('  ----------------------------------------\n\n');

training_start = tic;
net = trainNetwork(X_train, Y_train, lgraph, options);
training_time = toc(training_start);

fprintf('\n  ----------------------------------------\n');
fprintf('  ✓ 训练完成！总用时: %.2f 分钟\n', training_time/60);

%% [7/8] 评估模型
fprintf('\n[7/8] 评估模型性能...\n');

Y_train_pred = classify(net, X_train);
train_accuracy = sum(Y_train_pred == Y_train) / numel(Y_train);
fprintf('  [训练集] 准确率: %.2f%%\n', train_accuracy * 100);

Y_val_pred = classify(net, X_val);
val_accuracy = sum(Y_val_pred == Y_val) / numel(Y_val);
fprintf('  [验证集] 准确率: %.2f%%\n', val_accuracy * 100);

Y_test_pred = classify(net, X_test);
test_accuracy = sum(Y_test_pred == Y_test) / numel(Y_test);
fprintf('  [测试集] 准确率: %.2f%%\n', test_accuracy * 100);

fprintf('\n  测试集各类准确率:\n');
class_accuracies = zeros(num_classes, 1);
for c = 1:num_classes
    idx_c = double(Y_test) == c;
    if sum(idx_c) > 0
        acc_c = sum(Y_test_pred(idx_c) == Y_test(idx_c)) / sum(idx_c);
        class_accuracies(c) = acc_c;
        fprintf('    %d. %-15s: %.2f%% (%d/%d)\n', ...
                c, class_names{c}, acc_c*100, ...
                sum(Y_test_pred(idx_c) == Y_test(idx_c)), sum(idx_c));
    end
end

%% [8/8] 保存结果
fprintf('\n[8/8] 保存结果...\n');

results = struct();
results.train_accuracy = train_accuracy;
results.val_accuracy = val_accuracy;
results.test_accuracy = test_accuracy;
results.class_accuracies = class_accuracies;
results.training_time = training_time;
results.class_names = class_names;
results.confusion_matrix = confusionmat(Y_test, Y_test_pred);
results.dataset = 'MIRAGE';
results.num_samples = num_samples;

save('models/mirage_model.mat', 'net', 'results');
fprintf('  ✓ 模型保存到: models/mirage_model.mat\n');

% 可视化
fig = figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
cm = confusionchart(Y_test, Y_test_pred);
cm.Title = sprintf('混淆矩阵 (准确率: %.2f%%)', test_accuracy * 100);

subplot(1, 2, 2);
bar(class_accuracies * 100);
set(gca, 'XTickLabel', class_names);
ylabel('准确率 (%)');
title('各类别准确率');
xtickangle(45);
grid on;
ylim([0, 100]);

saveas(fig, 'figures/mirage_results.png');
fprintf('  ✓ 结果图保存到: figures/mirage_results.png\n');

%% 总结
fprintf('\n====================================================\n');
fprintf('   训练完成！\n');
fprintf('====================================================\n\n');

fprintf('【MIRAGE数据集结果】\n');
fprintf('  测试集准确率: %.2f%%\n', test_accuracy * 100);
fprintf('  训练时间: %.2f 分钟\n', training_time/60);
fprintf('  样本数: %d\n', num_samples);
fprintf('  类别数: %d\n', num_classes);

fprintf('\n【论文基准（参考）】\n');
fprintf('  MIRAGE-19: 81.10%%\n');
fprintf('  MIRAGE-22: 96.71%%\n');

fprintf('\n【文件保存位置】\n');
fprintf('  模型: models/mirage_model.mat\n');
fprintf('  结果图: figures/mirage_results.png\n\n');