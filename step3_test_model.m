%% 步骤3：测试模型创建
% 详细测试ResNet模型的创建和结构验证

clear; clc;
fprintf('====================================================\n');
fprintf('   步骤3：测试模型创建\n');
fprintf('====================================================\n\n');

%% 加载数据
fprintf('[1/6] 加载数据...\n');

if ~exist('data/sample_flows.mat', 'file')
    error('错误：找不到数据文件。请先运行 step1_generate_sample_data.m');
end

load('data/sample_flows.mat', 'class_names');
num_classes = length(class_names);
fprintf('  ✓ 类别数: %d\n', num_classes);

%% 创建模型
fprintf('\n[2/6] 创建FlowPic ResNet模型...\n');

input_size = [32, 32, 4];  % 32x32x4 FlowPic
fprintf('  输入大小: %s\n', mat2str(input_size));
fprintf('  输出类别: %d\n', num_classes);

try
    lgraph = create_flowpic_model(input_size, num_classes);
    fprintf('  ✓ 模型创建成功！\n');
catch ME
    fprintf('  ✗ 模型创建失败：%s\n', ME.message);
    fprintf('    请检查 create_flowpic_model.m 文件\n');
    return;
end

%% 分析模型结构
fprintf('\n[3/6] 分析模型结构...\n');

layers = lgraph.Layers;
fprintf('  层信息:\n');
fprintf('    - 总层数: %d\n', length(layers));
fprintf('    - 输入层: %s\n', layers(1).Name);
fprintf('    - 输出层: %s\n', layers(end).Name);

% 统计层类型
layer_types = cellfun(@class, num2cell(layers), 'UniformOutput', false);
unique_types = unique(layer_types);

fprintf('\n  层类型统计:\n');
for i = 1:length(unique_types)
    type_name = unique_types{i};
    count = sum(strcmp(layer_types, type_name));
    
    % 简化类名显示
    simple_name = strrep(type_name, 'nnet.cnn.layer.', '');
    simple_name = strrep(simple_name, 'nnet.layer.', '');
    fprintf('    - %-30s: %2d 层\n', simple_name, count);
end

% 统计参数量（理论计算，因为LayerGraph阶段权重未初始化）
fprintf('\n  参数统计:\n');

total_params = 0;
trainable_params = 0;

for i = 1:length(layers)
    layer = layers(i);
    
    % 卷积层参数
    if isa(layer, 'nnet.cnn.layer.Convolution2DLayer')
        filter_size = layer.FilterSize;
        num_filters = layer.NumFilters;
        num_channels = layer.NumChannels;
        
        % 权重: num_filters × (filter_h × filter_w × num_channels)
        weights = num_filters * (filter_size(1) * filter_size(2) * num_channels);
        % 偏置: num_filters
        bias = num_filters;
        
        layer_params = weights + bias;
        total_params = total_params + layer_params;
        trainable_params = trainable_params + layer_params;
    end
    
    % 全连接层参数
    if isa(layer, 'nnet.cnn.layer.FullyConnectedLayer')
        output_size = layer.OutputSize;
        
        % 根据层名推断输入大小
        if strcmp(layer.Name, 'fc1')
            % res2输出128通道，经过2×2 avgpool后是 128×2×2=512
            input_size_calc = 128 * 2 * 2;
        elseif strcmp(layer.Name, 'fc2')
            input_size_calc = 256;  % fc1的输出
        else
            input_size_calc = 512;  % 默认
        end
        
        weights = output_size * input_size_calc;
        bias = output_size;
        
        layer_params = weights + bias;
        total_params = total_params + layer_params;
        trainable_params = trainable_params + layer_params;
    end
    
    % 批归一化层参数
    if isa(layer, 'nnet.cnn.layer.BatchNormalizationLayer')
        num_ch = layer.NumChannels;
        total_params = total_params + num_ch * 4;  % scale, offset, mean, var
        trainable_params = trainable_params + num_ch * 2;  % 只有scale和offset可训练
    end
end

fprintf('    - 总参数量: %d (%.2f K)\n', total_params, total_params/1000);
fprintf('    - 可训练参数: %d (%.2f K)\n', trainable_params, trainable_params/1000);
fprintf('    ℹ 注：LayerGraph阶段参数未初始化，以上为理论计算值\n');

%% 可视化网络结构
fprintf('\n[4/6] 可视化网络结构...\n');

try
    fig1 = figure('Name', '网络结构图', 'Position', [100, 100, 1400, 800]);
    plot(lgraph);
    title('FlowPic ResNet模型结构', 'FontSize', 14);
    
    % 调整布局
    set(gca, 'FontSize', 8);
    
    saveas(fig1, 'figures/step3_model_structure.png');
    fprintf('  ✓ 已保存到: figures/step3_model_structure.png\n');
catch ME
    fprintf('  ⚠ 可视化失败（可能图太大）：%s\n', ME.message);
    fprintf('    可以手动运行 plot(lgraph) 查看\n');
end

%% 检查网络连接
fprintf('\n[5/6] 检查网络连接...\n');

try
    % 分析网络
    analyzeNetwork(lgraph);
    fprintf('  ✓ 网络结构有效\n');
    fprintf('  ℹ 网络分析器已打开，可查看详细信息\n');
catch ME
    fprintf('  ⚠ 网络分析器打开失败：%s\n', ME.message);
end

% 检查是否有连接问题
connections = lgraph.Connections;
fprintf('  连接数: %d\n', size(connections, 1));

% 查找未连接的层
all_layer_names = {layers.Name};
connected_sources = connections.Source;
connected_destinations = connections.Destination;

unconnected = [];
for i = 1:length(layers)
    layer_name = layers(i).Name;
    
    % 跳过输入层和输出层
    if contains(layer_name, 'input') || contains(layer_name, 'output')
        continue;
    end
    
    % 检查是否作为源或目标出现
    is_source = any(strcmp(connected_sources, layer_name));
    is_dest = any(strcmp(connected_destinations, layer_name));
    
    if ~is_source && ~is_dest
        unconnected = [unconnected; {layer_name}];
    end
end

if isempty(unconnected)
    fprintf('  ✓ 所有层都已正确连接\n');
else
    fprintf('  ✗ 发现未连接的层:\n');
    for i = 1:length(unconnected)
        fprintf('    - %s\n', unconnected{i});
    end
    fprintf('\n  ⚠ 网络结构有问题！\n');
    fprintf('  请查看 FIX_NETWORK_ERROR.md 了解如何修复\n');
    fprintf('  或运行 verify_fix 来验证修复\n\n');
    return;
end

%% 测试前向传播
fprintf('\n[6/6] 测试前向传播...\n');

% 创建随机输入（确保batch_size >= 类别数）
batch_size = max(8, num_classes);
test_input = rand(input_size(1), input_size(2), input_size(3), batch_size);
fprintf('  输入形状: %s\n', mat2str(size(test_input)));

% 转换网络为可执行形式
fprintf('  正在组装网络...\n');
try
    % 设置简单的训练选项
    options = trainingOptions('adam', ...
        'MaxEpochs', 1, ...
        'MiniBatchSize', batch_size, ...
        'Verbose', false);
    
    % 创建临时标签 - 确保包含所有类别
    % 先创建包含所有类别的序列，再用随机数填充剩余
    temp_labels_array = [1:num_classes, randi(num_classes, 1, batch_size - num_classes)];
    temp_labels_array = temp_labels_array(randperm(batch_size));  % 打乱顺序
    temp_labels = categorical(temp_labels_array');
    
    fprintf('  正在进行快速训练测试（1个batch）...\n');
    fprintf('  ℹ Batch包含所有%d个类别以避免标签不匹配\n', num_classes);
    tic;
    net = trainNetwork(test_input, temp_labels, lgraph, options);
    train_time = toc;
    
    fprintf('  ✓ 训练测试成功！用时: %.2f 秒\n', train_time);
    
    % 测试预测
    fprintf('  测试预测...\n');
    tic;
    predictions = classify(net, test_input);
    predict_time = toc;
    
    fprintf('  ✓ 预测成功！\n');
    fprintf('    - 输入: %d 个样本\n', batch_size);
    fprintf('    - 预测结果: %s\n', mat2str(double(predictions(1:min(5,batch_size))')));
    fprintf('    - 预测用时: %.4f 秒\n', predict_time);
    fprintf('    - 吞吐量: %.1f 样本/秒\n', batch_size/predict_time);
    
catch ME
    fprintf('  ✗ 前向传播测试失败：%s\n', ME.message);
    fprintf('    堆栈追踪:\n');
    for i = 1:min(3, length(ME.stack))
        fprintf('      %s (line %d)\n', ME.stack(i).name, ME.stack(i).line);
    end
    
    % 提供调试建议
    if contains(ME.message, 'output size')
        fprintf('\n  💡 提示：这通常是标签类别数不匹配导致的\n');
        fprintf('      请确认数据文件中的class_names正确\n');
    end
    return;
end

%% 性能预估
fprintf('\n[性能预估]\n');

samples_per_epoch = 400;  % 假设训练集400个样本
epochs = 50;
batch_size_actual = 64;

num_batches = ceil(samples_per_epoch / batch_size_actual);
time_per_batch = train_time;  % 使用测试得到的时间

estimated_time_per_epoch = num_batches * time_per_batch;
estimated_total_time = estimated_time_per_epoch * epochs;

fprintf('  假设配置:\n');
fprintf('    - 训练样本: %d\n', samples_per_epoch);
fprintf('    - Batch size: %d\n', batch_size_actual);
fprintf('    - Epochs: %d\n', epochs);
fprintf('  预估训练时间:\n');
fprintf('    - 每个epoch: %.1f 秒 (%.1f 分钟)\n', ...
        estimated_time_per_epoch, estimated_time_per_epoch/60);
fprintf('    - 总计: %.1f 秒 (%.1f 分钟, %.1f 小时)\n', ...
        estimated_total_time, estimated_total_time/60, estimated_total_time/3600);

if estimated_total_time > 3600
    fprintf('  ⚠ 预计训练时间较长，建议使用GPU加速\n');
end

%% 保存模型架构
fprintf('\n  保存模型架构...\n');
save('data/model_architecture.mat', 'lgraph', 'input_size', 'num_classes');
fprintf('  ✓ 已保存到: data/model_architecture.mat\n');

%% 总结
fprintf('\n====================================================\n');
fprintf('   步骤3完成！\n');
fprintf('====================================================\n');
fprintf('✓ 模型创建成功\n');
fprintf('✓ 网络结构验证通过\n');
fprintf('✓ 前向传播测试通过\n');
fprintf('✓ 总参数量: %d (%.2f K)\n', total_params, total_params/1000);
fprintf('✓ 可视化已保存到 figures/ 目录\n\n');

fprintf('下一步：运行 step4_full_training.m 进行完整训练\n\n');
