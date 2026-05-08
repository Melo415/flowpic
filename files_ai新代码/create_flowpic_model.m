function lgraph = create_flowpic_model(input_size, num_classes)
% CREATE_FLOWPIC_MODEL
% 创建用于 FlowPic 分类的改进版 ResNet CNN 模型
%
% 结构特点：
%   1. 3个残差块（64 → 128 → 256 filters）
%   2. Global Average Pooling 代替固定尺寸池化
%   3. Dropout 仅放在分类头
%   4. 最后一层名称固定为 output，便于训练脚本替换类别权重

if nargin < 1
    input_size = [32, 32, 4];
end
if nargin < 2
    num_classes = 18;
end

%% 初始层
layers = [
    imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'none')

    convolution2dLayer(7, 64, 'Padding', 3, 'Name', 'conv1')
    batchNormalizationLayer('Name', 'bn1')
    reluLayer('Name', 'relu1')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'maxpool1')  % 32 -> 16
];

lgraph = layerGraph(layers);

%% 残差块
lgraph = addResidualBlock(lgraph, 'maxpool1',  64, 'res1', false);
lgraph = addResidualBlock(lgraph, 'res1_out', 128, 'res2', true);
lgraph = addResidualBlock(lgraph, 'res2_out', 256, 'res3', true);

%% 分类头
layers_head = [
    globalAveragePooling2dLayer('Name', 'gap')
    fullyConnectedLayer(256, 'Name', 'fc1')
    batchNormalizationLayer('Name', 'bn_fc1')
    reluLayer('Name', 'relu_fc1')
    dropoutLayer(0.5, 'Name', 'drop2')
    fullyConnectedLayer(num_classes, 'Name', 'fc2')
    softmaxLayer('Name', 'softmax')
    % 这里先用均匀权重占位，训练脚本里会 replaceLayer 注入真实 class weights
    classificationLayer('Name', 'output', ...
        'Classes', categorical(1:num_classes)', ...
        'ClassWeights', ones(1, num_classes))
];

lgraph = addLayers(lgraph, layers_head);
lgraph = connectLayers(lgraph, 'res3_out', 'gap');

end

% -------------------------------------------------------------------------
function lgraph = addResidualBlock(lgraph, prevLayer, numFilters, blockName, downsample)
% 添加标准残差块（Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU）

if nargin < 5
    downsample = false;
end

stride = 1;
if downsample
    stride = 2;
end

% 主分支
main_path = [
    convolution2dLayer(3, numFilters, 'Padding', 'same', ...
                       'Stride', stride, 'Name', [blockName '_conv1'])
    batchNormalizationLayer('Name', [blockName '_bn1'])
    reluLayer('Name', [blockName '_relu1'])
    
    dropoutLayer(0.15, 'Name', [blockName '_drop'])

    convolution2dLayer(3, numFilters, 'Padding', 'same', ...
                       'Name', [blockName '_conv2'])
    batchNormalizationLayer('Name', [blockName '_bn2'])
    
];

lgraph = addLayers(lgraph, main_path);
lgraph = connectLayers(lgraph, prevLayer, [blockName '_conv1']);

% Shortcut 分支
if downsample
    shortcut = [
        convolution2dLayer(1, numFilters, 'Stride', stride, ...
                           'Name', [blockName '_sc_conv'])
        batchNormalizationLayer('Name', [blockName '_sc_bn'])
    ];
    lgraph = addLayers(lgraph, shortcut);
    lgraph = connectLayers(lgraph, prevLayer, [blockName '_sc_conv']);
    shortcut_out = [blockName '_sc_bn'];
else
    shortcut_out = prevLayer;
end

% Add + ReLU
add_layer  = additionLayer(2, 'Name', [blockName '_add']);
relu_layer = reluLayer('Name', [blockName '_out']);

lgraph = addLayers(lgraph, add_layer);
lgraph = addLayers(lgraph, relu_layer);

lgraph = connectLayers(lgraph, [blockName '_bn2'], [blockName '_add/in1']);
lgraph = connectLayers(lgraph, shortcut_out,        [blockName '_add/in2']);
lgraph = connectLayers(lgraph, [blockName '_add'],  [blockName '_out']);

end
