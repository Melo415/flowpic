function lgraph = create_flowpic_model(input_size, num_classes)
% CREATE_FLOWPIC_MODEL
% 按照 Poliakov et al. (2025) 论文 Figure 2 构建模型
%
% 架构：
%   stem: Conv 7x7 → BN → ReLU → MaxPool(stride=2)  [32→16]
%   残差块1: 64 filters，不下采样                      [16×16]
%   残差块2: 128 filters，stride=2 下采样              [8×8]
%   Average Pooling (4×4, stride=4)                    [2×2]  → 展平 512 维
%   分类头: FC(512→256) → ReLU → FC(256→classes)
%   Dropout(0.2) 放在两个残差块内 + 分类头前
%
% 注意：论文只用 2 个残差块。
%       论文用 Average Pooling(输出2×2)，不用 Global Average Pooling。
%       论文用 AdamW，MATLAB 无内置 AdamW，用 Adam + L2 等效。

if nargin < 1
    input_size = [32, 32, 4];
end
if nargin < 2
    num_classes = 18;
end

%% Stem（初始层）
% 32×32 → maxpool → 16×16
stem = [
    imageInputLayer(input_size, 'Name', 'input', 'Normalization', 'none')
    convolution2dLayer(7, 64, 'Padding', 3, 'Name', 'stem_conv')
    batchNormalizationLayer('Name', 'stem_bn')
    reluLayer('Name', 'stem_relu')
    maxPooling2dLayer(2, 'Stride', 2, 'Name', 'stem_pool')   % 32 → 16
];

lgraph = layerGraph(stem);

%% 两个残差块（对应论文 Figure 2）
% res1: 64 filters，不下采样  → 16×16×64
% res2: 128 filters，stride=2 → 8×8×128
lgraph = addResidualBlock(lgraph, 'stem_pool', 64,  'res1', false);
lgraph = addResidualBlock(lgraph, 'res1_out',  128, 'res2', true);

%% 分类头
% AveragePool(4×4) 把 8×8 → 2×2，展平得 128×4 = 512 维
% FC(512→256) → ReLU → FC(256→classes)
head = [
    averagePooling2dLayer(4, 'Stride', 4, 'Name', 'avg_pool')   % 8×8 → 2×2
    dropoutLayer(0.2, 'Name', 'head_drop')
    fullyConnectedLayer(512, 'Name', 'fc1')                      % 128ch × 2×2 = 512
    reluLayer('Name', 'fc1_relu')
    fullyConnectedLayer(256, 'Name', 'fc2')
    reluLayer('Name', 'fc2_relu')
    fullyConnectedLayer(num_classes, 'Name', 'fc_out')
    softmaxLayer('Name', 'softmax')
    % 占位 ClassWeights，训练脚本里 replaceLayer 注入真实权重
    classificationLayer('Name', 'output', ...
        'Classes',      categorical(1:num_classes)', ...
        'ClassWeights', ones(1, num_classes))
];

lgraph = addLayers(lgraph, head);
lgraph = connectLayers(lgraph, 'res2_out', 'avg_pool');   % 连接点：res2_out → avg_pool

end

% -------------------------------------------------------------------------
function lgraph = addResidualBlock(lgraph, prevLayer, numFilters, blockName, downsample)
% 标准残差块：Conv→BN→ReLU→Dropout→Conv→BN → (+shortcut) → ReLU
% Dropout(0.2) 在两个 Conv 之间，与论文 Figure 2 对应

if nargin < 5, downsample = false; end

stride = 1;
if downsample, stride = 2; end

% ── 主分支 ──────────────────────────────────────────────────
main = [
    convolution2dLayer(3, numFilters, 'Padding', 'same', ...
                       'Stride', stride, 'Name', [blockName '_conv1'])
    batchNormalizationLayer('Name', [blockName '_bn1'])
    reluLayer('Name',          [blockName '_relu1'])
    dropoutLayer(0.2, 'Name',  [blockName '_drop'])     % 论文 dropout=0.2
    convolution2dLayer(3, numFilters, 'Padding', 'same', ...
                       'Name', [blockName '_conv2'])
    batchNormalizationLayer('Name', [blockName '_bn2'])
];

lgraph = addLayers(lgraph, main);
lgraph = connectLayers(lgraph, prevLayer, [blockName '_conv1']);

% ── Shortcut 分支 ───────────────────────────────────────────
if downsample
    sc = [
        convolution2dLayer(1, numFilters, 'Stride', stride, ...
                           'Name', [blockName '_sc_conv'])
        batchNormalizationLayer('Name', [blockName '_sc_bn'])
    ];
    lgraph = addLayers(lgraph, sc);
    lgraph = connectLayers(lgraph, prevLayer, [blockName '_sc_conv']);
    sc_out = [blockName '_sc_bn'];
else
    sc_out = prevLayer;
end

% ── Add + ReLU ──────────────────────────────────────────────
lgraph = addLayers(lgraph, additionLayer(2, 'Name', [blockName '_add']));
lgraph = addLayers(lgraph, reluLayer('Name', [blockName '_out']));

lgraph = connectLayers(lgraph, [blockName '_bn2'],  [blockName '_add/in1']);
lgraph = connectLayers(lgraph, sc_out,              [blockName '_add/in2']);
lgraph = connectLayers(lgraph, [blockName '_add'],  [blockName '_out']);

end