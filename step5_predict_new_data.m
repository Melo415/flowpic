%% 步骤5：使用训练好的模型预测新数据
% 演示如何使用训练好的模型对新的网络流进行分类

clear; clc;
fprintf('====================================================\n');
fprintf('   步骤5：预测新数据\n');
fprintf('====================================================\n\n');

%% [1/5] 加载训练好的模型
fprintf('[1/5] 加载训练好的模型...\n');

if ~exist('models/trained_model.mat', 'file')
    error('错误：找不到训练好的模型。请先运行 step4_full_training.m');
end

load('models/trained_model.mat', 'net', 'results');
fprintf('  ✓ 模型加载成功\n');
fprintf('  模型性能（测试集）: %.2f%%\n', results.test_accuracy * 100);
fprintf('  类别: %s\n', strjoin(results.class_names, ', '));

%% [2/5] 创建新的测试数据
fprintf('\n[2/5] 生成新的测试流...\n');

% 生成3个新的模拟流
new_flows = cell(3, 1);

% 流1: 模拟Facebook（中等包长，均衡流量）
fprintf('  生成流1: 模拟Facebook流量\n');
new_flows{1}.lengths = round(normrnd(800, 400, 150, 1));
new_flows{1}.lengths = max(40, min(1460, new_flows{1}.lengths));
new_flows{1}.times = cumsum([0; exprnd(0.05, 149, 1)]);
new_flows{1}.directions = ones(150, 1);
new_flows{1}.directions(randperm(150, 60)) = -1;  % 60个下行
new_flows{1}.true_label = 'Facebook';

% 流2: 模拟Spotify（小包，音频流）
fprintf('  生成流2: 模拟Spotify流量\n');
new_flows{2}.lengths = round(normrnd(400, 200, 200, 1));
new_flows{2}.lengths = max(40, min(1460, new_flows{2}.lengths));
new_flows{2}.times = cumsum([0; exprnd(0.02, 199, 1)]);
new_flows{2}.directions = ones(200, 1);
new_flows{2}.directions(randperm(200, 100)) = -1;  % 均衡
new_flows{2}.true_label = 'Spotify';

% 流3: 模拟YouTube（大包，下行为主）
fprintf('  生成流3: 模拟YouTube流量\n');
new_flows{3}.lengths = round(normrnd(1000, 500, 180, 1));
new_flows{3}.lengths = max(40, min(1460, new_flows{3}.lengths));
new_flows{3}.times = cumsum([0; exprnd(0.04, 179, 1)]);
new_flows{3}.directions = ones(180, 1);
new_flows{3}.directions(randperm(180, 126)) = -1;  % 70% 下行
new_flows{3}.true_label = 'YouTube';

fprintf('  ✓ 生成了 %d 个新流\n', length(new_flows));

%% [3/5] 转换为FlowPic格式
fprintf('\n[3/5] 将新流转换为FlowPic...\n');

flowpic_size = 32;
num_new = length(new_flows);
X_new = zeros(flowpic_size, flowpic_size, 4, num_new);

for i = 1:num_new
    X_new(:,:,:,i) = generate_flowpic(new_flows{i}.lengths, ...
                                       new_flows{i}.times, ...
                                       new_flows{i}.directions, ...
                                       flowpic_size);
end

fprintf('  ✓ FlowPic生成完成\n');
fprintf('  形状: %s\n', mat2str(size(X_new)));

%% [4/5] 进行预测
fprintf('\n[4/5] 使用模型预测...\n');

% 预测
tic;
predictions = classify(net, X_new);
prediction_time = toc;

% 获取预测概率
[pred_scores, pred_labels] = predict(net, X_new);

fprintf('  ✓ 预测完成！用时: %.4f 秒\n', prediction_time);
fprintf('  平均每个: %.4f 秒\n', prediction_time/num_new);

% 显示预测结果
fprintf('\n  预测结果:\n');
fprintf('  %-6s %-15s %-15s %-10s %s\n', '流ID', '真实标签', '预测标签', '置信度', '状态');
fprintf('  %s\n', repmat('-', 1, 70));

for i = 1:num_new
    true_label = new_flows{i}.true_label;
    pred_label = char(predictions(i));
    confidence = max(pred_scores(i, :));
    
    if strcmp(true_label, pred_label)
        status = '✓ 正确';
    else
        status = '✗ 错误';
    end
    
    fprintf('  %-6d %-15s %-15s %.2f%%      %s\n', ...
            i, true_label, pred_label, confidence*100, status);
end

%% [5/5] 详细分析和可视化
fprintf('\n[5/5] 详细分析...\n');

fig = figure('Name', '预测分析', 'Position', [100, 100, 1400, 1000]);

for i = 1:num_new
    % 第一行：FlowPic可视化
    subplot(num_new, 5, (i-1)*5 + 1);
    imagesc(X_new(:,:,1,i));
    colorbar;
    colormap('hot');
    title(sprintf('流%d - Ch1 上行长度', i));
    
    subplot(num_new, 5, (i-1)*5 + 2);
    imagesc(X_new(:,:,2,i));
    colorbar;
    colormap('hot');
    title('Ch2 下行长度');
    
    subplot(num_new, 5, (i-1)*5 + 3);
    imagesc(X_new(:,:,3,i));
    colorbar;
    colormap('hot');
    title('Ch3 上行IAT');
    
    subplot(num_new, 5, (i-1)*5 + 4);
    imagesc(X_new(:,:,4,i));
    colorbar;
    colormap('hot');
    title('Ch4 下行IAT');
    
    % 预测概率分布
    subplot(num_new, 5, (i-1)*5 + 5);
    bar(pred_scores(i, :));
    set(gca, 'XTickLabel', results.class_names);
    ylabel('概率');
    title(sprintf('预测: %s (%.1f%%)', ...
                  char(predictions(i)), max(pred_scores(i,:))*100));
    xtickangle(45);
    grid on;
    ylim([0, 1]);
end

sgtitle('新流预测分析', 'FontSize', 14);
saveas(fig, 'figures/step5_prediction_analysis.png');
fprintf('  ✓ 分析图保存到: figures/step5_prediction_analysis.png\n');

%% 保存预测结果
fprintf('\n  保存预测结果...\n');

prediction_results = struct();
prediction_results.flows = new_flows;
prediction_results.predictions = predictions;
prediction_results.scores = pred_scores;
prediction_results.time = prediction_time;

save('results/prediction_results.mat', 'prediction_results');
fprintf('  ✓ 结果保存到: results/prediction_results.mat\n');

%% 实用函数示例
fprintf('\n====================================================\n');
fprintf('   预测完成！\n');
fprintf('====================================================\n\n');

fprintf('【如何使用训练好的模型】\n\n');

fprintf('1. 加载模型:\n');
fprintf('   load(''models/trained_model.mat'', ''net'');\n\n');

fprintf('2. 准备新数据:\n');
fprintf('   lengths = [120, 1460, 52, ...];  %% 包长度\n');
fprintf('   times = [0, 0.01, 0.02, ...];    %% 时间戳\n');
fprintf('   directions = [1, -1, 1, ...];    %% 方向\n\n');

fprintf('3. 生成FlowPic:\n');
fprintf('   flowpic = generate_flowpic(lengths, times, directions, 32);\n\n');

fprintf('4. 预测:\n');
fprintf('   prediction = classify(net, flowpic);\n');
fprintf('   [scores, ~] = predict(net, flowpic);\n\n');

fprintf('5. 查看结果:\n');
fprintf('   fprintf(''预测: %%s (置信度: %%.2f%%%%)\\n'', ...\n');
fprintf('           char(prediction), max(scores)*100);\n\n');

fprintf('【批量预测示例】\n\n');
fprintf('%% 假设你有多个流\n');
fprintf('num_flows = 100;\n');
fprintf('X_batch = zeros(32, 32, 4, num_flows);\n');
fprintf('for i = 1:num_flows\n');
fprintf('    X_batch(:,:,:,i) = generate_flowpic(...);\n');
fprintf('end\n');
fprintf('predictions = classify(net, X_batch);\n\n');

fprintf('完成！现在你可以使用模型预测任意网络流了。\n\n');
