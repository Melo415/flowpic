function lgraph = create_flowpic_model(input_size, num_classes)
% 创建用于flowpic分类的ResNet风格CNN模型
% 输入:
%   input_size: [height, width, channels] 例如 [32, 32, 4]
%   num_classes: 分类类别数
% 输出:
%   lgraph: layerGraph对象

if nargin < 1
    input_size = [32, 32, 4];
end
if nargin < 2
    num_classes = 20;  % 默认20类
end

% 输入层
layers = [
    imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'none')
    
    % 初始卷积
    convolution2dLayer(3, 64, 'Padding', 'same', 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    dropoutLayer(0.2, 'Name', 'drop1')
];

lgraph = layerGraph(layers);

% 添加残差块1 (64 filters)
lgraph = addResidualBlock(lgraph, 'drop1', 64, 'res1');

% 添加残差块2 (128 filters, 带下采样)
lgraph = addResidualBlock(lgraph, 'res1_relu', 128, 'res2', true);

% 分类头
layers_head = [
    averagePooling2dLayer(2, 'Stride', 2, 'Name', 'avgpool')
    dropoutLayer(0.2, 'Name', 'drop_final')
    
    % Feed-forward classifier
    fullyConnectedLayer(256, 'Name', 'fc1')
    reluLayer('Name', 'relu_fc')
    fullyConnectedLayer(num_classes, 'Name', 'fc2')
    
    softmaxLayer('Name', 'softmax')
    classificationLayer('Name', 'output')
];

lgraph = addLayers(lgraph, layers_head);
lgraph = connectLayers(lgraph, 'res2_relu', 'avgpool');

end

function lgraph = addResidualBlock(lgraph, prevLayer, numFilters, blockName, downsample)
% 添加残差块
% downsample: 是否进行下采样 (stride=2)

if nargin < 5
    downsample = false;
end

if downsample
    stride = 2;
else
    stride = 1;
end

% 主路径
main_path = [
    convolution2dLayer(3, numFilters, 'Padding', 'same', ...
                      'Stride', stride, 'Name', [blockName '_conv1'])
    batchNormalizationLayer('Name', [blockName '_bn1'])
    reluLayer('Name', [blockName '_relu1'])
    
    convolution2dLayer(3, numFilters, 'Padding', 'same', ...
                      'Name', [blockName '_conv2'])
    batchNormalizationLayer('Name', [blockName '_bn2'])
    dropoutLayer(0.2, 'Name', [blockName '_drop'])
];

lgraph = addLayers(lgraph, main_path);
lgraph = connectLayers(lgraph, prevLayer, [blockName '_conv1']);

% 跳跃连接 (shortcut)
if downsample
    % 需要1x1卷积调整维度
    shortcut = [
        convolution2dLayer(1, numFilters, 'Stride', stride, ...
                          'Name', [blockName '_shortcut_conv'])
        batchNormalizationLayer('Name', [blockName '_shortcut_bn'])
    ];
    lgraph = addLayers(lgraph, shortcut);
    lgraph = connectLayers(lgraph, prevLayer, [blockName '_shortcut_conv']);
    shortcut_out = [blockName '_shortcut_bn'];
else
    shortcut_out = prevLayer;
end

% 添加元素相加层
add_layer = additionLayer(2, 'Name', [blockName '_add']);
lgraph = addLayers(lgraph, add_layer);

% 连接
lgraph = connectLayers(lgraph, [blockName '_drop'], [blockName '_add/in1']);
lgraph = connectLayers(lgraph, shortcut_out, [blockName '_add/in2']);

% ReLU激活（直接添加到add后面，作为连接的一部分）
relu_layer = reluLayer('Name', [blockName '_relu']);
lgraph = addLayers(lgraph, relu_layer);
lgraph = connectLayers(lgraph, [blockName '_add'], [blockName '_relu']);

end
