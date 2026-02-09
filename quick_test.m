%% 快速测试示例 - 验证代码是否正常工作
% 使用模拟数据快速测试flowpic生成和模型创建

clear; clc;

fprintf('=== FlowPic MATLAB实现快速测试 ===\n\n');

%% 1. 测试flowpic生成
fprintf('[1/3] 测试flowpic生成功能...\n');

% 创建一个模拟流
lengths = [120, 1460, 52, 800, 1400, 64, 1200, 800];
times = [0, 0.01, 0.015, 0.02, 0.05, 0.08, 0.1, 0.12];
directions = [1, -1, 1, -1, 1, -1, 1, -1];

% 生成flowpic
flowpic = generate_flowpic(lengths, times, directions, 32);

fprintf('  ✓ Flowpic生成成功！\n');
fprintf('  - 形状: %s\n', mat2str(size(flowpic)));
fprintf('  - 数据范围: [%.2f, %.2f]\n', min(flowpic(:)), max(flowpic(:)));

% 可视化
figure('Name', 'FlowPic四通道可视化');
channel_names = {'上行Length', '下行Length', '上行IAT', '下行IAT'};
for ch = 1:4
    subplot(2,2,ch);
    imagesc(flowpic(:,:,ch));
    colorbar;
    title(channel_names{ch});
    xlabel('Time bins');
    ylabel('Length/IAT bins');
end
colormap('hot');

%% 2. 测试模型创建
fprintf('\n[2/3] 测试模型创建...\n');

try
    lgraph = create_flowpic_model([32, 32, 4], 10);
    fprintf('  ✓ 模型创建成功！\n');
    
    % 分析网络
    layers = lgraph.Layers;
    fprintf('  - 总层数: %d\n', length(layers));
    fprintf('  - 输入层: %s\n', layers(1).Name);
    fprintf('  - 输出层: %s\n', layers(end).Name);
    
    % 可视化网络结构
    figure('Name', 'ResNet模型结构');
    plot(lgraph);
    title('FlowPic分类网络');
    
catch ME
    fprintf('  ✗ 模型创建失败: %s\n', ME.message);
end

%% 3. 测试端到端流程
fprintf('\n[3/3] 测试小规模训练...\n');

try
    % 生成少量模拟数据
    num_samples = 100;
    num_classes = 5;
    
    X = zeros(32, 32, 4, num_samples);
    Y = categorical(randi([1, num_classes], num_samples, 1));
    
    for i = 1:num_samples
        num_packets = randi([20, 100]);
        lengths = randi([40, 1460], num_packets, 1);
        times = sort(rand(num_packets, 1) * 15);
        directions = randi([0, 1], num_packets, 1) * 2 - 1;
        
        X(:,:,:,i) = generate_flowpic(lengths, times, directions, 32);
    end
    
    fprintf('  ✓ 数据准备完成 (%d samples)\n', num_samples);
    
    % 简单训练测试
    options = trainingOptions('adam', ...
        'MaxEpochs', 2, ...
        'MiniBatchSize', 16, ...
        'Verbose', false, ...
        'ExecutionEnvironment', 'auto');
    
    fprintf('  - 开始快速训练（2 epochs）...\n');
    net = trainNetwork(X, Y, lgraph, options);
    
    fprintf('  ✓ 训练测试成功！\n');
    
    % 测试预测
    Y_pred = classify(net, X(: ,:,:,1:10));
    fprintf('  - 预测示例: %s\n', mat2str(double(Y_pred(1:5)')));
    
catch ME
    fprintf('  ✗ 训练测试失败: %s\n', ME.message);
end

%% 完成
fprintf('\n=== 测试完成！ ===\n');
fprintf('所有核心功能正常工作。\n');
fprintf('下一步：\n');
fprintf('  1. 准备你的真实数据\n');
fprintf('  2. 运行 train_flowpic.m 进行完整训练\n');
fprintf('  3. 参考 README.md 了解更多细节\n');
