%% FlowPic 多通道加密流量分类 - 训练示例
% 基于论文: Multichannel Histograms for Flow Classification (NOMS 2025)

clear; clc;

%% 1. 数据准备
fprintf('正在准备数据...\n');

% 这里需要你的实际数据
% 假设你有一个cell array，每个cell是一个flow的数据
% flows{i}.lengths - 数据包长度
% flows{i}.times - 时间戳
% flows{i}.directions - 方向 (1或-1)
% flows{i}.label - 类别标签

% 示例：生成模拟数据
num_samples = 1000;
num_classes = 10;
flows = cell(num_samples, 1);
labels = zeros(num_samples, 1);

for i = 1:num_samples
    % 随机生成流数据（实际使用时替换为真实数据）
    num_packets = randi([50, 500]);
    flows{i}.lengths = randi([40, 1460], num_packets, 1);
    flows{i}.times = sort(rand(num_packets, 1) * 15);  % 0-15秒
    flows{i}.directions = randi([0, 1], num_packets, 1) * 2 - 1;  % 1或-1
    labels(i) = randi([1, num_classes]);
end

%% 2. 生成Flowpic特征
fprintf('正在生成flowpic特征...\n');

flowpic_size = 32;  % 32x32直方图
num_channels = 4;   % 四通道

X = zeros(flowpic_size, flowpic_size, num_channels, num_samples);

for i = 1:num_samples
    if mod(i, 100) == 0
        fprintf('处理进度: %d/%d\n', i, num_samples);
    end
    
    X(:,:,:,i) = generate_flowpic(flows{i}.lengths, ...
                                   flows{i}.times, ...
                                   flows{i}.directions, ...
                                   flowpic_size);
end

% 转换标签为categorical
Y = categorical(labels);

%% 3. 数据集划分
fprintf('划分数据集...\n');

% 80% 训练, 10% 验证, 10% 测试
cv = cvpartition(num_samples, 'HoldOut', 0.2);
idx_train_val = training(cv);
idx_test = test(cv);

X_train_val = X(:,:,:,idx_train_val);
Y_train_val = Y(idx_train_val);
X_test = X(:,:,:,idx_test);
Y_test = Y(idx_test);

% 再划分训练和验证集
cv2 = cvpartition(sum(idx_train_val), 'HoldOut', 0.125);  % 10/80 = 0.125
idx_train = training(cv2);
idx_val = test(cv2);

X_train = X_train_val(:,:,:,idx_train);
Y_train = Y_train_val(idx_train);
X_val = X_train_val(:,:,:,idx_val);
Y_val = Y_train_val(idx_val);

fprintf('训练集: %d, 验证集: %d, 测试集: %d\n', ...
        sum(idx_train), sum(idx_val), sum(idx_test));

%% 4. 创建模型
fprintf('创建模型...\n');

input_size = [flowpic_size, flowpic_size, num_channels];
lgraph = create_flowpic_model(input_size, num_classes);

% 可视化网络结构
figure;
plot(lgraph);
title('FlowPic ResNet模型结构');

%% 5. 训练选项
fprintf('设置训练参数...\n');

options = trainingOptions('adam', ...
    'MaxEpochs', 50, ...
    'MiniBatchSize', 64, ...
    'InitialLearnRate', 0.001, ...
    'LearnRateSchedule', 'piecewise', ...
    'LearnRateDropFactor', 0.1, ...
    'LearnRateDropPeriod', 20, ...
    'L2Regularization', 0.0001, ...
    'ValidationData', {X_val, Y_val}, ...
    'ValidationFrequency', 30, ...
    'Verbose', true, ...
    'VerboseFrequency', 30, ...
    'Plots', 'training-progress', ...
    'ExecutionEnvironment', 'auto');  % 'gpu'如果有GPU

%% 6. 训练模型
fprintf('开始训练...\n');
tic;
net = trainNetwork(X_train, Y_train, lgraph, options);
training_time = toc;
fprintf('训练完成，用时: %.2f 秒\n', training_time);

%% 7. 测试评估
fprintf('评估模型...\n');

% 在测试集上预测
Y_pred = classify(net, X_test);

% 计算准确率
accuracy = sum(Y_pred == Y_test) / numel(Y_test);
fprintf('测试集准确率: %.2f%%\n', accuracy * 100);

% 混淆矩阵
figure;
confusionchart(Y_test, Y_pred);
title(sprintf('混淆矩阵 (准确率: %.2f%%)', accuracy * 100));

%% 8. 保存模型
save('flowpic_model.mat', 'net', 'accuracy', 'training_time');
fprintf('模型已保存到 flowpic_model.mat\n');

%% 9. 与论文结果对比
fprintf('\n=== 结果对比 ===\n');
fprintf('你的结果: %.2f%%\n', accuracy * 100);
fprintf('论文结果 (MIRAGE-19): 81.10%%\n');
fprintf('论文结果 (MIRAGE-22): 96.71%%\n');
fprintf('注意：这是模拟数据，实际数据集结果会不同\n');
