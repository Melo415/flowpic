%% 使用MIRAGE数据集训练FlowPic模型
clear; clc;
fprintf('====================================================\n');
fprintf('   使用MIRAGE数据集训练FlowPic模型\n');
fprintf('====================================================\n\n');

%% 1. 配置参数（核心适配你的环境）
rng(42);  % 固定随机种子，结果可复现

% ⭐ 关键修改：改为你的MIRAGE数据集绝对路径
DATASET_PATH = 'D:\Desktop\GitHub\flowpic\data\mirage';  
USE_SUBSET = true;              % 先用子集测试，稳定后改false
SUBSET_SIZE = 1000;             % MIRAGE-19建议子集1000，完整用2000+
flowpic_size = 32;              % FlowPic图片尺寸（32x32）
num_channels = 4;               % FlowPic通道数（长度/时间/方向/速率）
max_epochs = 30;                % 训练轮数（GPU可设50，CPU设30）
mini_batch_size = 32;           % 批次大小（CPU建议32，GPU建议64）
initial_learn_rate = 0.001;     % 初始学习率

%% 2. 加载MIRAGE数据（适配JSON格式）
fprintf('[1/8] 加载MIRAGE数据集...\n');

try
    % 关键修改：调用你的JSON加载函数（替代原load_mirage_data）
    [flows, labels, class_names] = load_mirage_json(DATASET_PATH);
    fprintf('  ✓ 数据加载成功\n');
    fprintf('  - 总流数: %d, 类别数: %d\n', length(flows), length(class_names));
catch ME
    fprintf('  ✗ 数据加载失败: %s\n', ME.message);
    fprintf('\n请检查:\n');
    fprintf('  1. DATASET_PATH 是否正确（当前: %s）\n', DATASET_PATH);
    fprintf('  2. load_mirage_json.m 是否在当前路径\n');
    fprintf('  3. JSON文件是否为MIRAGE-19格式\n');
    return;
end

num_samples = length(flows);
num_classes = length(class_names);

% 使用子集（分层采样，保持类别比例）
if USE_SUBSET && num_samples > SUBSET_SIZE
    fprintf('\n  使用分层子集训练（%d个样本）\n', SUBSET_SIZE);
    
    subset_idx = [];
    samples_per_class = floor(SUBSET_SIZE / num_classes);
    
    for c = 1:num_classes
        idx_c = find(labels == c);
        if ~isempty(idx_c)
            % 每个类别取固定数量，不足则取全部
            num_take = min(samples_per_class, length(idx_c));
            subset_idx = [subset_idx, idx_c(randperm(length(idx_c), num_take))];
        end
    end
    
    % 补足到SUBSET_SIZE（避免样本数不足）
    if length(subset_idx) < SUBSET_SIZE
        remaining = randperm(num_samples, SUBSET_SIZE - length(subset_idx));
        subset_idx = unique([subset_idx, remaining]);
    end
    
    flows = flows(subset_idx);
    labels = labels(subset_idx);
    num_samples = length(flows);
    
    fprintf('  ✓ 最终子集大小: %d\n', num_samples);
end

fprintf('  训练使用: %d 样本, %d 类别\n', num_samples, num_classes);

%% 3. 生成FlowPic特征（核心：补充FlowPic生成逻辑）
fprintf('\n[2/8] 生成FlowPic特征 (%dx%dx%d)...\n', flowpic_size, flowpic_size, num_channels);
fprintf('  正在生成');

tic;
% 初始化FlowPic特征矩阵（4维：高x宽x通道x样本）
X = zeros(flowpic_size, flowpic_size, num_channels, num_samples);

for i = 1:num_samples
    if mod(i, 100) == 0
        fprintf('.');
    end
    
    % 提取当前流的核心数据
    lengths = flows{i}.lengths;
    times = flows{i}.times;
    directions = flows{i}.directions;
    
    % 生成FlowPic（调用核心函数）
    X(:,:,:,i) = generate_flowpic(lengths, times, directions, flowpic_size);
end

generation_time = toc;
fprintf(' 完成！\n');
fprintf('  ✓ 生成用时: %.2f 秒 (%.4f 秒/样本)\n', ...
        generation_time, generation_time/num_samples);

% 标签转为分类变量（适配MATLAB训练）
Y = categorical(labels);

%% 4. 数据集划分（训练/验证/测试）
fprintf('\n[3/8] 划分数据集（训练80%/验证10%/测试10%）...\n');

% 第一步：划分训练+验证（80%）和测试（20%）
cv1 = cvpartition(num_samples, 'HoldOut', 0.2);
idx_train_val = training(cv1);
idx_test = test(cv1);

% 第二步：从训练+验证集中划分训练（90%）和验证（10%）
cv2 = cvpartition(sum(idx_train_val), 'HoldOut', 0.125); % 0.125*0.8=0.1
idx_train = training(cv2);
idx_val = test(cv2);

% 整理数据
X_train = X(:,:,:,idx_train_val(idx_train));
Y_train = Y(idx_train_val(idx_train));
X_val = X(:,:,:,idx_train_val(idx_val));
Y_val = Y(idx_train_val(idx_val));
X_test = X(:,:,:,idx_test);
Y_test = Y(idx_test);

fprintf('  - 训练集: %d (%.1f%%)\n', length(Y_train), 100*length(Y_train)/num_samples);
fprintf('  - 验证集: %d (%.1f%%)\n', length(Y_val), 100*length(Y_val)/num_samples);
fprintf('  - 测试集: %d (%.1f%%)\n', length(Y_test), 100*length(Y_test)/num_samples);

%% 5. 创建FlowPic模型（CNN模型）
fprintf('\n[4/8] 创建FlowPic CNN模型...\n');

input_size = [flowpic_size, flowpic_size, num_channels];
lgraph = create_flowpic_model(input_size, num_classes);
fprintf('  ✓ FlowPic模型创建成功\n');

%% 6. 配置训练选项（适配CPU/GPU）
fprintf('\n[5/8] 配置训练选项...\n');

% 检测GPU
use_gpu = false;
execution_env = 'auto';
try
    gpu_device = gpuDevice;
    use_gpu = true;
    execution_env = 'gpu';
    fprintf('  ✓ 检测到GPU: %s (加速训练)\n', gpu_device.Name);
    mini_batch_size = 64; % GPU可用更大批次
catch
    fprintf('  ℹ 未检测到GPU，使用CPU训练（建议减小批次）\n');
    mini_batch_size = 16; % CPU适配小批次
end

% 训练选项配置
options = trainingOptions('adam', ...
    'MaxEpochs', max_epochs, ...
    'MiniBatchSize', mini_batch_size, ...
    'InitialLearnRate', initial_learn_rate, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 10, ... % 适配30轮训练
    'L2Regularization', 0.0001, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', max(1, floor(length(Y_train)/mini_batch_size)), ...
    'Shuffle', 'every-epoch', ...
    'Verbose', true, ...
    'VerboseFrequency', 5, ...
    'Plots', 'training-progress', ... % 显示训练曲线
    'ExecutionEnvironment', execution_env, ...
    'OutputNetwork', 'best-validation-loss', ... % 保存最优模型
    'GradientThreshold', 1); % 梯度裁剪，防止爆炸

%% 7. 训练模型
fprintf('\n[6/8] 开始训练模型...\n');
fprintf('  ----------------------------------------\n\n');

training_start = tic;
try
    net = trainNetwork(X_train, Y_train, lgraph, options);
catch ME
    fprintf('  ✗ 训练失败: %s\n', ME.message);
    fprintf('  建议：减小batch size或降低学习率\n');
    return;
end
training_time = toc(training_start);

fprintf('\n  ----------------------------------------\n');
fprintf('  ✓ 训练完成！总用时: %.2f 分钟\n', training_time/60);

%% 8. 评估模型性能
fprintf('\n[7/8] 评估模型性能...\n');

% 训练集准确率
Y_train_pred = classify(net, X_train);
train_accuracy = sum(Y_train_pred == Y_train) / numel(Y_train);
fprintf('  [训练集] 准确率: %.2f%%\n', train_accuracy * 100);

% 验证集准确率
Y_val_pred = classify(net, X_val);
val_accuracy = sum(Y_val_pred == Y_val) / numel(Y_val);
fprintf('  [验证集] 准确率: %.2f%%\n', val_accuracy * 100);

% 测试集准确率
Y_test_pred = classify(net, X_test);
test_accuracy = sum(Y_test_pred == Y_test) / numel(Y_test);
fprintf('  [测试集] 准确率: %.2f%%\n', test_accuracy * 100);

% 各类别准确率
fprintf('\n  测试集各类别准确率:\n');
class_accuracies = zeros(num_classes, 1);
for c = 1:num_classes
    idx_c = double(Y_test) == c;
    if sum(idx_c) > 0
        acc_c = sum(Y_test_pred(idx_c) == Y_test(idx_c)) / sum(idx_c);
        class_accuracies(c) = acc_c;
        fprintf('    %d. %-15s: %.2f%% (%d/%d)\n', ...
                c, class_names{c}, acc_c*100, ...
                sum(Y_test_pred(idx_c) == Y_test(idx_c)), sum(idx_c));
    else
        class_accuracies(c) = 0;
        fprintf('    %d. %-15s: 0.00%% (0/0)\n', c, class_names{c});
    end
end

%% 9. 保存结果和可视化
fprintf('\n[8/8] 保存结果和可视化...\n');

% 整理结果
results = struct();
results.train_accuracy = train_accuracy;
results.val_accuracy = val_accuracy;
results.test_accuracy = test_accuracy;
results.class_accuracies = class_accuracies;
results.training_time = training_time;
results.class_names = class_names;
results.confusion_matrix = confusionmat(Y_test, Y_test_pred);
results.dataset = 'MIRAGE-19';
results.num_samples = num_samples;
results.flowpic_size = flowpic_size;

% 保存模型和结果
save('models/mirage_flowpic_model.mat', 'net', 'results');
fprintf('  ✓ 模型保存到: models/mirage_flowpic_model.mat\n');

% 可视化结果（混淆矩阵+类别准确率）
fig = figure('Position', [100, 100, 1200, 500], 'Color', 'white');

% 子图1：混淆矩阵
subplot(1, 2, 1);
cm = confusionchart(Y_test, Y_test_pred);
cm.Title = sprintf('混淆矩阵 (测试集准确率: %.2f%%)', test_accuracy * 100);
cm.RowSummary = 'row-normalized';
cm.ColumnSummary = 'column-normalized';

% 子图2：各类别准确率
subplot(1, 2, 2);
bar(1:num_classes, class_accuracies * 100, 'FaceColor', [0.2, 0.6, 0.8]);
set(gca, 'XTick', 1:num_classes, 'XTickLabel', class_names);
ylabel('准确率 (%)', 'FontSize', 10);
title('各类别准确率', 'FontSize', 12);
xtickangle(45);
grid on;
ylim([0, 100]);
xlabel('应用类别', 'FontSize', 10);

% 保存图片
saveas(fig, 'figures/mirage_flowpic_results.png');
fprintf('  ✓ 结果图保存到: figures/mirage_flowpic_results.png\n');

%% 总结
fprintf('\n====================================================\n');
fprintf('   FlowPic模型训练完成！\n');
fprintf('====================================================\n\n');

fprintf('【核心结果】\n');
fprintf('  测试集准确率: %.2f%%\n', test_accuracy * 100);
fprintf('  训练时间: %.2f 分钟\n', training_time/60);
fprintf('  样本数: %d (类别数: %d)\n', num_samples, num_classes);
fprintf('  FlowPic尺寸: %dx%dx%d\n', flowpic_size, flowpic_size, num_channels);

fprintf('\n【参考基准】\n');
fprintf('  MIRAGE-19 FlowPic论文基准: 81.10%%\n');
fprintf('  若准确率偏低，可尝试：\n');
fprintf('  1. 增大训练样本数（关闭USE_SUBSET）\n');
fprintf('  2. 增加训练轮数（max_epochs=50）\n');
fprintf('  3. 调整FlowPic尺寸（64x64）\n');

fprintf('\n【文件位置】\n');
fprintf('  模型文件: models/mirage_flowpic_model.mat\n');
fprintf('  结果图片: figures/mirage_flowpic_results.png\n\n');

% ===================== 补充依赖函数 =====================
% 函数1：generate_flowpic - 生成FlowPic特征图
function flowpic = generate_flowpic(lengths, times, directions, pic_size)
    % 输入：
    %   lengths: 包长度数组（一维）
    %   times: 时间戳数组（一维）
    %   directions: 方向数组（±1，一维）
    %   pic_size: FlowPic尺寸（如32）
    % 输出：
    %   flowpic: pic_size x pic_size x 4 的FlowPic特征图
    
    % 数据清洗
    lengths = lengths(~isnan(lengths));
    times = times(~isnan(times));
    directions = directions(~isnan(directions));
    
    % 统一长度（取前pic_size^2个包，不足补0）
    max_packets = pic_size * pic_size;
    if length(lengths) > max_packets
        lengths = lengths(1:max_packets);
        times = times(1:max_packets);
        directions = directions(1:max_packets);
    else
        pad_len = max_packets - length(lengths);
        lengths = [lengths, zeros(1, pad_len)];
        times = [times, times(end)+zeros(1, pad_len)];
        directions = [directions, zeros(1, pad_len)];
    end
    
    % 归一化
    lengths_norm = (lengths - min(lengths)) / (max(lengths) - min(lengths) + eps);
    times_norm = (times - min(times)) / (max(times) - min(times) + eps);
    directions_norm = (directions + 1) / 2; % 把±1转为0-1
    rates_norm = lengths_norm ./ (times_norm + eps); % 速率特征
    rates_norm = (rates_norm - min(rates_norm)) / (max(rates_norm) - min(rates_norm) + eps);
    
    % 重塑为pic_size x pic_size
    ch1 = reshape(lengths_norm, pic_size, pic_size); % 通道1：包长度
    ch2 = reshape(times_norm, pic_size, pic_size);   % 通道2：时间
    ch3 = reshape(directions_norm, pic_size, pic_size); % 通道3：方向
    ch4 = reshape(rates_norm, pic_size, pic_size);   % 通道4：速率
    
    % 组合为4通道FlowPic
    flowpic = cat(3, ch1, ch2, ch3, ch4);
end

% 函数2：create_flowpic_model - 创建FlowPic CNN模型
function lgraph = create_flowpic_model(input_size, num_classes)
    % 输入：
    %   input_size: 输入尺寸 [H, W, C]
    %   num_classes: 类别数
    % 输出：
    %   lgraph: CNN模型图
    
    % 构建FlowPic经典CNN结构
    layers = [
        % 输入层
        imageInputLayer(input_size, 'Name', 'input')
        
        % 卷积块1
        convolution2dLayer(3, 32, 'Padding', 'same', 'Name', 'conv1')
        batchNormalizationLayer('Name', 'bn1')
        reluLayer('Name', 'relu1')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool1')
        
        % 卷积块2
        convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv2')
        batchNormalizationLayer('Name', 'bn2')
        reluLayer('Name', 'relu2')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool2')
        
        % 卷积块3
        convolution2dLayer(3, 128, 'Padding', 'same', 'Name', 'conv3')
        batchNormalizationLayer('Name', 'bn3')
        reluLayer('Name', 'relu3')
        maxPooling2dLayer(2, 'Stride', 2, 'Name', 'pool3')
        
        % 全连接层
        fullyConnectedLayer(256, 'Name', 'fc1')
        reluLayer('Name', 'relu4')
        dropoutLayer(0.5, 'Name', 'dropout1')
        
        % 输出层
        fullyConnectedLayer(num_classes, 'Name', 'fc_out')
        softmaxLayer('Name', 'softmax')
        classificationLayer('Name', 'classOutput')];
    
    % 转为层图（方便后续修改）
    lgraph = layerGraph(layers);
end