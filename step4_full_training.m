%% 步骤4：完整训练流程
% 使用示例数据进行完整的模型训练、验证和测试

clear; clc;
fprintf('====================================================\n');
fprintf('   步骤4：完整训练流程\n');
fprintf('====================================================\n\n');

%% 全局设置
rng(42);  % 设置随机种子保证可重复性

%% [1/8] 加载数据
fprintf('[1/8] 加载数据...\n');

if ~exist('data/sample_flows.mat', 'file')
    error('错误：找不到数据文件。请先运行 step1_generate_sample_data.m');
end

load('data/sample_flows.mat', 'flows', 'labels', 'class_names');
num_samples = length(flows);
num_classes = length(class_names);

fprintf('  ✓ 已加载 %d 个样本\n', num_samples);
fprintf('  ✓ 类别数: %d\n', num_classes);
fprintf('  类别: %s\n', strjoin(class_names, ', '));

%% [2/8] 生成FlowPic特征
fprintf('\n[2/8] 生成FlowPic特征...\n');

flowpic_size = 32;
num_channels = 4;

fprintf('  配置: %dx%dx%d FlowPic\n', flowpic_size, flowpic_size, num_channels);
fprintf('  正在生成');

tic;
X = zeros(flowpic_size, flowpic_size, num_channels, num_samples);

for i = 1:num_samples
    if mod(i, 50) == 0
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

% 转换标签
Y = categorical(labels);

%% [3/8] 数据集划分
fprintf('\n[3/8] 划分数据集...\n');

% 80% 训练 + 验证, 20% 测试
cv1 = cvpartition(num_samples, 'HoldOut', 0.2);
idx_train_val = training(cv1);
idx_test = test(cv1);

X_train_val = X(:,:,:,idx_train_val);
Y_train_val = Y(idx_train_val);
X_test = X(:,:,:,idx_test);
Y_test = Y(idx_test);

% 从训练+验证中再分出验证集 (10% of total = 12.5% of train_val)
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

% 检查类别平衡
fprintf('\n  训练集类别分布:\n');
for c = 1:num_classes
    count = sum(double(Y_train) == c);
    fprintf('    %d. %-10s: %3d (%.1f%%)\n', ...
            c, class_names{c}, count, 100*count/length(Y_train));
end

%% [4/8] 创建模型
fprintf('\n[4/8] 创建模型...\n');

input_size = [flowpic_size, flowpic_size, num_channels];
lgraph = create_flowpic_model(input_size, num_classes);

fprintf('  ✓ 模型创建成功\n');

%% [5/8] 配置训练选项
fprintf('\n[5/8] 配置训练选项...\n');

% 检测GPU
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

% 训练超参数（与论文一致）
max_epochs = 50;
mini_batch_size = 64;
initial_learn_rate = 0.001;
l2_regularization = 0.0001;

fprintf('  训练超参数:\n');
fprintf('    - Epochs: %d\n', max_epochs);
fprintf('    - Batch Size: %d\n', mini_batch_size);
fprintf('    - Initial LR: %.4f\n', initial_learn_rate);
fprintf('    - L2 Regularization: %.4f\n', l2_regularization);
fprintf('    - Optimizer: Adam\n');
fprintf('    - LR Schedule: Piecewise\n');
fprintf('    - Execution: %s\n', execution_env);

% 创建训练选项
options = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_learn_rate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', l2_regularization, ...
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
fprintf('  预计时间: 根据硬件而定\n');
fprintf('  监控: 训练进度图已打开\n');
fprintf('  ----------------------------------------\n\n');

training_start = tic;

try
    net = trainNetwork(X_train, Y_train, lgraph, options);
    training_time = toc(training_start);
    
    fprintf('\n  ----------------------------------------\n');
    fprintf('  ✓ 训练完成！总用时: %.2f 分钟\n', training_time/60);
    
catch ME
    fprintf('\n  ✗ 训练失败：%s\n', ME.message);
    fprintf('  请检查:\n');
    fprintf('    1. 内存是否充足\n');
    fprintf('    2. 数据是否有效\n');
    fprintf('    3. GPU驱动是否正常（如使用GPU）\n');
    return;
end

%% [7/8] 评估模型
fprintf('\n[7/8] 评估模型性能...\n');

% 在训练集上评估
fprintf('  [训练集]\n');
Y_train_pred = classify(net, X_train);
train_accuracy = sum(Y_train_pred == Y_train) / numel(Y_train);
fprintf('    准确率: %.2f%%\n', train_accuracy * 100);

% 在验证集上评估
fprintf('  [验证集]\n');
Y_val_pred = classify(net, X_val);
val_accuracy = sum(Y_val_pred == Y_val) / numel(Y_val);
fprintf('    准确率: %.2f%%\n', val_accuracy * 100);

% 在测试集上评估
fprintf('  [测试集]\n');
Y_test_pred = classify(net, X_test);
test_accuracy = sum(Y_test_pred == Y_test) / numel(Y_test);
fprintf('    准确率: %.2f%%\n', test_accuracy * 100);

% 计算每类准确率
fprintf('\n  测试集各类准确率:\n');
class_accuracies = zeros(num_classes, 1);
for c = 1:num_classes
    idx_c = double(Y_test) == c;
    if sum(idx_c) > 0
        acc_c = sum(Y_test_pred(idx_c) == Y_test(idx_c)) / sum(idx_c);
        class_accuracies(c) = acc_c;
        fprintf('    %d. %-10s: %.2f%% (%d/%d)\n', ...
                c, class_names{c}, acc_c*100, ...
                sum(Y_test_pred(idx_c) == Y_test(idx_c)), sum(idx_c));
    end
end

%% [8/8] 可视化结果
fprintf('\n[8/8] 生成可视化结果...\n');

% 混淆矩阵
fig1 = figure('Name', '混淆矩阵', 'Position', [100, 100, 800, 700]);
cm = confusionchart(Y_test, Y_test_pred);
cm.Title = sprintf('混淆矩阵 - 测试集 (准确率: %.2f%%)', test_accuracy * 100);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';
saveas(fig1, 'figures/step4_confusion_matrix.png');
fprintf('  ✓ 混淆矩阵保存到: figures/step4_confusion_matrix.png\n');

% 准确率对比图
fig2 = figure('Name', '准确率对比', 'Position', [100, 100, 1000, 400]);

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
     'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom');

subplot(1, 2, 2);
bar(class_accuracies * 100);
set(gca, 'XTickLabel', class_names);
ylabel('准确率 (%)');
title('各类别准确率（测试集）');
grid on;
ylim([0, 100]);
xtickangle(45);

saveas(fig2, 'figures/step4_accuracy_comparison.png');
fprintf('  ✓ 准确率对比保存到: figures/step4_accuracy_comparison.png\n');

% 预测示例可视化
fig3 = figure('Name', '预测示例', 'Position', [100, 100, 1400, 900]);

num_examples = 9;
example_indices = randperm(length(Y_test), num_examples);

for idx = 1:num_examples
    i = example_indices(idx);
    
    subplot(3, 3, idx);
    
    % 显示FlowPic (Channel 1)
    imagesc(X_test(:,:,1,i));
    colorbar;
    colormap('hot');
    
    % 标题显示真实标签和预测
    true_label = char(Y_test(i));
    pred_label = char(Y_test_pred(i));
    
    if strcmp(true_label, pred_label)
        title_color = 'green';
        title_prefix = '✓';
    else
        title_color = 'red';
        title_prefix = '✗';
    end
    
    title(sprintf('%s 真实: %s\n预测: %s', title_prefix, true_label, pred_label), ...
          'Color', title_color, 'FontSize', 10);
    
    axis square;
end

sgtitle('预测示例 (Channel 1: 上行包长度)', 'FontSize', 14);
saveas(fig3, 'figures/step4_prediction_examples.png');
fprintf('  ✓ 预测示例保存到: figures/step4_prediction_examples.png\n');

%% 保存结果
fprintf('\n  保存模型和结果...\n');

results = struct();
results.train_accuracy = train_accuracy;
results.val_accuracy = val_accuracy;
results.test_accuracy = test_accuracy;
results.class_accuracies = class_accuracies;
results.training_time = training_time;
results.class_names = class_names;
results.confusion_matrix = confusionmat(Y_test, Y_test_pred);

save('models/trained_model.mat', 'net', 'results');
fprintf('  ✓ 模型保存到: models/trained_model.mat\n');

%% 与论文结果对比
fprintf('\n====================================================\n');
fprintf('   训练完成！\n');
fprintf('====================================================\n\n');

fprintf('【你的结果】\n');
fprintf('  测试集准确率: %.2f%%\n', test_accuracy * 100);
fprintf('  训练时间: %.2f 分钟\n', training_time/60);
fprintf('  样本数: %d\n', num_samples);

fprintf('\n【论文结果（参考）】\n');
fprintf('  MIRAGE-19 数据集:\n');
fprintf('    - 单通道 (32×32): 77.02%%\n');
fprintf('    - 双通道 (32×32): 80.21%%\n');
fprintf('    - 四通道 (32×32): 81.10%%\n');
fprintf('\n  MIRAGE-22 数据集:\n');
fprintf('    - 单通道 (32×32): 95.36%%\n');
fprintf('    - 双通道 (32×32): 95.96%%\n');
fprintf('    - 四通道 (32×32): 96.71%%\n');

fprintf('\n【说明】\n');
fprintf('  ⚠ 这是使用模拟数据的结果，与论文使用的真实数据集不同\n');
fprintf('  ⚠ 要获得与论文相近的结果，需要使用MIRAGE-19或MIRAGE-22数据集\n');
fprintf('  ✓ 如果准确率 > 70%%，说明模型工作正常\n');
fprintf('  ✓ 如果准确率 < 50%%，可能需要调试数据或模型\n');

fprintf('\n【下一步】\n');
fprintf('  1. 运行 step5_predict_new_data.m 测试预测功能\n');
fprintf('  2. 运行 step4_train_mirage.m 使用真实数据集（MIRAGE-19）重新训练\n');