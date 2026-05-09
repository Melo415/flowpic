function [net, results, exp_dir] = run_training_with_logging(X_train, Y_train, X_val, Y_val, X_test, Y_test, ...
    lgraph, options, class_names, hyperparameters, class_weights)
% RUN_TRAINING_WITH_LOGGING
% 带完整实验记录的训练函数

    fprintf('\n【实验记录系统启动】\n');

    %% ====================== 1. 创建本次实验文件夹 ======================
    results_root = "D:\Desktop\GitHub\flowpic\files\training_results";
    if ~exist(results_root, 'dir')
        mkdir(results_root);
    end

    exp_name = datestr(now, 'yyyy-mm-dd_HH-MM-SS');
    exp_dir = fullfile(results_root, ['FlowPic_ResNet_' exp_name]);
    mkdir(exp_dir);
    fprintf('本次实验记录文件夹：\n   %s\n\n', exp_dir);

    %% ====================== 2. 保存超参数 ======================
    exp_info = hyperparameters;
    exp_info.date = datestr(now);

    if isprop(options, 'ExecutionEnvironment')
        exp_info.execution_env = options.ExecutionEnvironment;
    else
        exp_info.execution_env = 'unknown';
    end

    exp_info.class_weights_used = ~isempty(class_weights);

    save(fullfile(exp_dir, 'experiment_info.mat'), 'exp_info');

    fid = fopen(fullfile(exp_dir, 'hyperparameters.txt'), 'w');
    if fid == -1
        error('无法创建 hyperparameters.txt，请检查输出目录权限。');
    end

    fprintf(fid, 'FlowPic ResNet 实验记录\n');
    fprintf(fid, '时间: %s\n\n', exp_info.date);
    fprintf(fid, '=== 核心超参数 ===\n');
    fprintf(fid, 'Epochs               : %d\n', exp_info.max_epochs);
    fprintf(fid, 'Batch Size           : %d\n', exp_info.batch_size);
    fprintf(fid, 'Learning Rate        : %.6f\n', exp_info.learning_rate);
    fprintf(fid, 'L2 Regularization    : %.6f\n', exp_info.l2_reg);
    fprintf(fid, 'FlowPic Size         : %dx%dx%d\n', exp_info.flowpic_size, exp_info.flowpic_size, exp_info.num_channels);
    fprintf(fid, 'Execution Env        : %s\n', exp_info.execution_env);
    fprintf(fid, 'Class Weights Used   : %s\n', string(exp_info.class_weights_used));

    if isfield(exp_info, 'dropout_fc1')
        fprintf(fid, 'Dropout FC1          : %.2f\n', exp_info.dropout_fc1);
    end
    if isfield(exp_info, 'dropout_fc2')
        fprintf(fid, 'Dropout FC2          : %.2f\n', exp_info.dropout_fc2);
    end
    if isfield(exp_info, 'learn_rate_drop_factor')
        fprintf(fid, 'LR Drop Factor       : %.3f\n', exp_info.learn_rate_drop_factor);
    end
    if isfield(exp_info, 'learn_rate_drop_period')
        fprintf(fid, 'LR Drop Period       : %d\n', exp_info.learn_rate_drop_period);
    end
    if isfield(exp_info, 'gradient_threshold')
        fprintf(fid, 'Gradient Threshold   : %.3f\n', exp_info.gradient_threshold);
    end
    if isfield(exp_info, 'validation_patience')
        fprintf(fid, 'Validation Patience  : %d\n', exp_info.validation_patience);
    end

    fclose(fid);
    fprintf('✓ 超参数已完整记录\n');

    %% ====================== 3. 保存训练集类别分布 ======================
    class_names = class_names(:);

    % 根据训练集标签重新统计类别数
    class_counts = histcounts(double(Y_train), 1:(numel(class_names) + 1));
    class_counts = class_counts(:);

    if nargin < 11 || isempty(class_weights)
        class_weights = ones(1, numel(class_names));
    end
    class_weights = reshape(class_weights, [], 1);

    fprintf('调试信息：\n');
    fprintf('  numel(class_names)   = %d\n', numel(class_names));
    fprintf('  numel(class_counts)  = %d\n', numel(class_counts));
    fprintf('  numel(class_weights) = %d\n', numel(class_weights));

    assert(numel(class_names) == numel(class_counts), ...
        'class_names 和 class_counts 长度不一致');
    assert(numel(class_names) == numel(class_weights), ...
        'class_names 和 class_weights 长度不一致');

    dist_table = table((1:numel(class_names))', class_names, ...
                       class_counts, class_weights, ...
                       'VariableNames', {'ClassID', 'ClassName', 'Count', 'Weight'});

    writetable(dist_table, fullfile(exp_dir, 'dataset_distribution.csv'));

    %% ====================== 4. 训练（使用传入的 options） ======================
    fprintf('开始训练并记录...\n');
    training_start = tic;

    % 替换输出层中的类别权重
    layer_names = string({lgraph.Layers.Name});
    if any(layer_names == "output")
        weighted_output = classificationLayer( ...
            'Name', 'output', ...
            'Classes', categorical(1:numel(class_names))', ...
            'ClassWeights', reshape(class_weights, 1, []));
        lgraph = replaceLayer(lgraph, 'output', weighted_output);
    else
        error(['未找到名为 "output" 的分类层，无法注入 ClassWeights。' newline ...
               '请检查 create_flowpic_model.m 中最后一层名称是否为 output。']);
    end

    % 尝试记录训练前已存在的 figure，便于后续寻找训练进度图
    figs_before = findall(groot, 'Type', 'Figure');

    [net, train_info] = trainNetwork(X_train, Y_train, lgraph, options);

    % 从训练历史里提取两个关键指标
    [~, loss_idx] = min(train_info.ValidationLoss);
    [~, acc_idx]  = max(train_info.ValidationAccuracy);
    
    best_loss_epoch    = loss_idx;
    best_loss_val      = train_info.ValidationLoss(loss_idx);
    acc_at_best_loss   = train_info.ValidationAccuracy(loss_idx);
    
    best_acc_epoch     = acc_idx;
    best_acc_val       = train_info.ValidationAccuracy(acc_idx);
    loss_at_best_acc   = train_info.ValidationLoss(acc_idx);
    
    fprintf('\n── 训练结果对比 ──────────────────────────────\n');
    fprintf('  Loss 最低：第 %2d 轮  Loss=%.4f  对应准确率=%.2f%%\n', ...
            best_loss_epoch, best_loss_val, acc_at_best_loss);
    fprintf('  准确率最高：第 %2d 轮  Acc=%.2f%%  对应Loss=%.4f\n', ...
            best_acc_epoch, best_acc_val, loss_at_best_acc);
    fprintf('  当前保存模型：Loss 最低（第 %d 轮）\n', best_loss_epoch);
    fprintf('──────────────────────────────────────────────\n\n');
    
    %% ── 绘制并保存训练曲线 ──────────────────────────────
    fprintf('正在保存训练曲线...\n');
    
    % 训练数据是逐 iteration 的，验证数据频率较低，需要对齐 x 轴
    n_iter     = length(train_info.TrainingLoss);
    n_val      = length(train_info.ValidationLoss);
    val_x      = linspace(1, n_iter, n_val);  % 把验证点映射到 iteration 轴
    
    fig = figure('Visible', 'off', 'Position', [100, 100, 1000, 420]);
    
    % 左图：Loss
    subplot(1, 2, 1);
    plot(1:n_iter, train_info.TrainingLoss, 'b-', 'LineWidth', 1.2); hold on;
    plot(val_x,    train_info.ValidationLoss, 'r-', 'LineWidth', 1.8);
    xlabel('Iteration'); ylabel('Loss');
    title('Training & Validation Loss');
    legend('Train Loss', 'Val Loss', 'Location', 'northeast');
    grid on; box on;
    
    % 右图：Accuracy
    subplot(1, 2, 2);
    plot(1:n_iter, train_info.TrainingAccuracy, 'b-', 'LineWidth', 1.2); hold on;
    plot(val_x,    train_info.ValidationAccuracy, 'r-', 'LineWidth', 1.8);
    xlabel('Iteration'); ylabel('Accuracy (%)');
    title('Training & Validation Accuracy');
    legend('Train Acc', 'Val Acc', 'Location', 'southeast');
    grid on; box on;
    
    sgtitle(sprintf('FlowPic ResNet  |  Best Val Acc: %.2f%%', ...
            max(train_info.ValidationAccuracy)), 'FontSize', 12);
    
    % 保存到实验目录（exp_dir 是 run_training_with_logging 里已有的变量）
    curve_path = fullfile(exp_dir, 'training_curves.png');
    exportgraphics(fig, curve_path, 'Resolution', 150);
    close(fig);
    fprintf('✓ 训练曲线已保存: %s\n', curve_path);

    training_time = toc(training_start);

    %% ====================== 5. 评估并保存所有结果 ======================
    Y_test_pred = classify(net, X_test);
    test_accuracy = mean(Y_test_pred == Y_test);

    % 保存训练曲线（若训练窗口存在）
    figs_after = findall(groot, 'Type', 'Figure');
    training_curve_saved = false;

    if isprop(options, 'Plots') && strcmpi(string(options.Plots), "training-progress")
        new_figs = setdiff(figs_after, figs_before);
        target_fig = [];

        if ~isempty(new_figs)
            target_fig = new_figs(1);
        elseif ~isempty(figs_after)
            target_fig = figs_after(1);
        end

        if ~isempty(target_fig) && isgraphics(target_fig)
            try
                exportgraphics(target_fig, fullfile(exp_dir, 'training_curves.png'));
                training_curve_saved = true;
            catch
                try
                    saveas(target_fig, fullfile(exp_dir, 'training_curves.png'));
                    training_curve_saved = true;
                catch
                    training_curve_saved = false;
                end
            end
        end
    end

    % 混淆矩阵
    fig_cm = figure('Visible', 'off');
    cm = confusionchart(Y_test, Y_test_pred);
    cm.Title = sprintf('测试集混淆矩阵 (Acc=%.2f%%)', test_accuracy * 100);
    try
        exportgraphics(fig_cm, fullfile(exp_dir, 'confusion_matrix.png'));
    catch
        saveas(fig_cm, fullfile(exp_dir, 'confusion_matrix.png'));
    end
    close(fig_cm);

    %% ====================== 6. 汇总结果并保存 ======================
    results = struct();
    results.test_accuracy = test_accuracy;
    results.training_time = training_time;
    results.class_names = class_names;
    results.class_counts_train = class_counts;
    results.class_weights = reshape(class_weights, 1, []);
    results.confusion_matrix = confusionmat(Y_test, Y_test_pred);
    results.hyperparameters = exp_info;
    results.training_curve_saved = training_curve_saved;

    save(fullfile(exp_dir, 'final_results.mat'), 'net', 'results');

    %% ====================== 7. 生成 Markdown 总结 ======================
    fid = fopen(fullfile(exp_dir, 'experiment_summary.md'), 'w');
    if fid == -1
        error('无法创建 experiment_summary.md，请检查输出目录权限。');
    end

    fprintf(fid, '# FlowPic ResNet 实验总结\n\n');
    fprintf(fid, '**时间**：%s  \n', exp_info.date);
    fprintf(fid, '**测试准确率**：%.2f%%  \n', test_accuracy * 100);
    fprintf(fid, '**训练时长**：%.1f 分钟\n\n', training_time / 60);
    
    fprintf(fid, '## 训练过程最优结果\n\n');
    fprintf(fid, '| 指标 | 轮次 | 数值 | 对应另一指标 |\n');
    fprintf(fid, '|------|------|------|--------------|\n');
    fprintf(fid, '| Loss 最低 | 第 %d 轮 | %.4f | 准确率 %.2f%% |\n', ...
            best_loss_epoch, best_loss_val, acc_at_best_loss);
    fprintf(fid, '| 准确率最高 | 第 %d 轮 | %.2f%% | Loss %.4f |\n\n', ...
            best_acc_epoch, best_acc_val, loss_at_best_acc);
    fprintf(fid, '> 当前保存模型来自 Loss 最低轮次（第 %d 轮）。\n', best_loss_epoch);
    fprintf(fid, '> 若两者不在同一轮，说明模型在该轮判对更多但部分错误更自信。\n\n');

    fprintf(fid, '## 改进内容\n\n');

    fprintf(fid, '## 超参数\n');
    fprintf(fid, '- Epochs: %d\n', exp_info.max_epochs);
    fprintf(fid, '- Batch Size: %d\n', exp_info.batch_size);
    fprintf(fid, '- Learning Rate: %.6f\n', exp_info.learning_rate);
    fprintf(fid, '- L2: %.6f\n', exp_info.l2_reg);
    if isfield(exp_info, 'dropout_fc1')
        fprintf(fid, '- Dropout FC1: %.2f\n', exp_info.dropout_fc1);
    end
    if isfield(exp_info, 'dropout_fc2')
        fprintf(fid, '- Dropout FC2: %.2f\n', exp_info.dropout_fc2);
    end

    fprintf(fid, '\n## 数据集分布\n');
    fprintf(fid, '详见 `dataset_distribution.csv`\n\n');

    fprintf(fid, '## 可视化\n');
    if training_curve_saved
        fprintf(fid, '![训练曲线](training_curves.png)\n');
    else
        fprintf(fid, '- 训练曲线图未成功捕获（不影响模型训练与结果保存）\n');
    end
    fprintf(fid, '![混淆矩阵](confusion_matrix.png)\n\n');

    fprintf(fid, '## 改进效果\n\n');

    fprintf(fid, '## 改进建议\n\n');

    fclose(fid);

    fprintf('\n🎉 所有训练结果已保存至： %s\n', exp_dir);
    fprintf('   推荐直接打开 experiment_summary.md 查看总结\n\n');
end
