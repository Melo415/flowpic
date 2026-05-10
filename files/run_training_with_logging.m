function [net, results, exp_dir] = run_training_with_logging(X_train, Y_train, X_val, Y_val, X_test, Y_test, ...
    lgraph, options, class_names, hyperparameters, class_weights)
% RUN_TRAINING_WITH_LOGGING
% 带完整实验记录的训练函数

    fprintf('\n【实验记录系统启动】\n');

    %% ── 1. 创建本次实验文件夹 ────────────────────────────────────
    results_root = "D:\Desktop\GitHub\flowpic\files\training_results";
    if ~exist(results_root, 'dir'), mkdir(results_root); end

    exp_dir = fullfile(results_root, ['FlowPic_ResNet_' datestr(now, 'yyyy-mm-dd_HH-MM-SS')]);
    mkdir(exp_dir);
    fprintf('本次实验记录文件夹：\n   %s\n\n', exp_dir);

    %% ── 2. 保存超参数 ────────────────────────────────────────────
    exp_info = hyperparameters;
    exp_info.date = datestr(now);
    exp_info.execution_env = 'unknown';
    if isprop(options, 'ExecutionEnvironment')
        exp_info.execution_env = options.ExecutionEnvironment;
    end
    exp_info.class_weights_used = ~isempty(class_weights);
    save(fullfile(exp_dir, 'experiment_info.mat'), 'exp_info');

    fid = fopen(fullfile(exp_dir, 'hyperparameters.txt'), 'w');
    if fid == -1, error('无法创建 hyperparameters.txt'); end
    fprintf(fid, 'FlowPic ResNet 实验记录\n时间: %s\n\n', exp_info.date);
    fprintf(fid, '=== 核心超参数 ===\n');
    fprintf(fid, 'Epochs               : %d\n',   exp_info.max_epochs);
    fprintf(fid, 'Batch Size           : %d\n',   exp_info.batch_size);
    fprintf(fid, 'Learning Rate        : %.6f\n', exp_info.learning_rate);
    fprintf(fid, 'L2 Regularization    : %.6f\n', exp_info.l2_reg);
    fprintf(fid, 'FlowPic Size         : %dx%dx%d\n', ...
            exp_info.flowpic_size, exp_info.flowpic_size, exp_info.num_channels);
    fprintf(fid, 'Execution Env        : %s\n',   exp_info.execution_env);
    fprintf(fid, 'Class Weights Used   : %s\n',   string(exp_info.class_weights_used));
    fclose(fid);
    fprintf('✓ 超参数已完整记录\n');

    %% ── 3. 保存训练集类别分布 ────────────────────────────────────
    class_names   = class_names(:);
    class_counts  = histcounts(double(Y_train), 1:(numel(class_names) + 1))';
    if nargin < 11 || isempty(class_weights)
        class_weights = ones(numel(class_names), 1);
    end
    class_weights = reshape(class_weights, [], 1);

    fprintf('调试信息：\n');
    fprintf('  numel(class_names)   = %d\n', numel(class_names));
    fprintf('  numel(class_counts)  = %d\n', numel(class_counts));
    fprintf('  numel(class_weights) = %d\n', numel(class_weights));

    assert(numel(class_names) == numel(class_counts),  'class_names 和 class_counts 长度不一致');
    assert(numel(class_names) == numel(class_weights), 'class_names 和 class_weights 长度不一致');

    writetable( ...
        table((1:numel(class_names))', class_names, class_counts, class_weights, ...
              'VariableNames', {'ClassID','ClassName','Count','Weight'}), ...
        fullfile(exp_dir, 'dataset_distribution.csv'));

    %% ── 4. 训练 ──────────────────────────────────────────────────
    fprintf('开始训练并记录...\n');
    training_start = tic;

    % 注入类别权重
    layer_names = string({lgraph.Layers.Name});
    if any(layer_names == "output")
        lgraph = replaceLayer(lgraph, 'output', classificationLayer( ...
            'Name', 'output', ...
            'Classes',      categorical(1:numel(class_names))', ...
            'ClassWeights', reshape(class_weights, 1, [])));
    else
        error('未找到名为 "output" 的分类层，请检查 create_flowpic_model.m。');
    end

    figs_before = findall(groot, 'Type', 'Figure');
    [net, train_info] = trainNetwork(X_train, Y_train, lgraph, options);
    training_time = toc(training_start);

    %% ── 5. 提取最优验证指标 ──────────────────────────────────────
    vl = train_info.ValidationLoss;
    va = train_info.ValidationAccuracy;

    % 用 Inf / 0 替换 NaN，确保 min/max 正确跳过无效位置
    vl_clean = vl;  vl_clean(isnan(vl_clean)) = Inf;
    va_clean = va;  va_clean(isnan(va_clean)) = 0;

    [~, best_loss_check] = min(vl_clean);
    [~, best_acc_check]  = max(va_clean);

    % 从原始数组取真实值
    best_loss_val    = vl(best_loss_check);
    best_acc_val     = va(best_acc_check);
    acc_at_best_loss = va(best_loss_check);
    loss_at_best_acc = vl(best_acc_check);

    % 判断 train_info 是按迭代还是按轮次存储，推算迭代次数
    val_freq = max(1, floor(length(Y_train) / hyperparameters.batch_size));
    if length(vl) > hyperparameters.max_epochs + 5
        % 数组长度远大于设定轮数 → 按迭代存储，下标即迭代次数
        best_loss_iter = best_loss_check;
        best_acc_iter  = best_acc_check;
    else
        % 按轮次存储 → 下标 × 每轮迭代数 = 迭代次数
        best_loss_iter = best_loss_check * val_freq;
        best_acc_iter  = best_acc_check  * val_freq;
    end

    fprintf('\n── 训练结果对比 ──────────────────────────────\n');
    fprintf('  Loss 最低  ：第 %d 次迭代  Loss=%.4f  对应准确率=%.2f%%\n', ...
            best_loss_iter, best_loss_val, acc_at_best_loss);
    fprintf('  准确率最高 ：第 %d 次迭代  Acc=%.2f%%  对应Loss=%.4f\n', ...
            best_acc_iter, best_acc_val, loss_at_best_acc);
    fprintf('──────────────────────────────────────────────\n\n');


    %% ── 6. 评估测试集 ────────────────────────────────────────────
    Y_test_pred   = classify(net, X_test);
    test_accuracy = mean(Y_test_pred == Y_test);
    
    % 新增：打印测试集结果
    fprintf('\n── 测试集评估结果 ────────────────────────────\n');
    fprintf('  测试集准确率：%.2f%%\n', test_accuracy * 100);
    fprintf('──────────────────────────────────────────────\n\n');

    %% ── 7. 保存训练曲线 ──────────────────────────────────────────
    training_curve_saved = false;
    if isprop(options, 'Plots') && strcmpi(string(options.Plots), "training-progress")
        figs_after = findall(groot, 'Type', 'Figure');
        new_figs   = setdiff(figs_after, figs_before);
        target_fig = [];
        if ~isempty(new_figs),   target_fig = new_figs(1);
        elseif ~isempty(figs_after), target_fig = figs_after(1); end

        if ~isempty(target_fig) && isgraphics(target_fig)
            try
                exportgraphics(target_fig, fullfile(exp_dir, 'training_curves.png'));
                training_curve_saved = true;
            catch
                try
                    saveas(target_fig, fullfile(exp_dir, 'training_curves.png'));
                    training_curve_saved = true;
                catch; end
            end
        end
    end

    %% ── 8. 混淆矩阵 ─────────────────────────────────────────────
    fig_cm = figure('Visible', 'off');
    cm = confusionchart(Y_test, Y_test_pred);
    cm.Title = sprintf('测试集混淆矩阵 (Acc=%.2f%%)', test_accuracy * 100);
    try
        exportgraphics(fig_cm, fullfile(exp_dir, 'confusion_matrix.png'));
    catch
        saveas(fig_cm, fullfile(exp_dir, 'confusion_matrix.png'));
    end
    close(fig_cm);

    %% ── 9. 汇总结果 ─────────────────────────────────────────────
    results = struct();
    results.test_accuracy        = test_accuracy;
    results.training_time        = training_time;
    results.class_names          = class_names;
    results.class_counts_train   = class_counts;
    results.class_weights        = reshape(class_weights, 1, []);
    results.confusion_matrix     = confusionmat(Y_test, Y_test_pred);
    results.hyperparameters      = exp_info;
    results.training_curve_saved = training_curve_saved;
    save(fullfile(exp_dir, 'final_results.mat'), 'net', 'results');

    %% ── 10. 生成 Markdown 总结 ───────────────────────────────────
    fid = fopen(fullfile(exp_dir, 'experiment_summary.md'), 'w');
    if fid == -1, error('无法创建 experiment_summary.md'); end

    fprintf(fid, '# FlowPic ResNet 实验总结\n\n');
    fprintf(fid, '**时间**：%s  \n',         exp_info.date);
    fprintf(fid, '**测试集准确率**：%.2f%%  \n', test_accuracy * 100);
    fprintf(fid, '**训练时长**：%.1f 分钟\n\n', training_time / 60);

    fprintf(fid, '## 训练过程最优结果\n\n');
    fprintf(fid, '| 指标 | 迭代次数 | 数值 | 对应另一指标 |\n');
    fprintf(fid, '|------|----------|------|--------------|\n');
    fprintf(fid, '| Loss 最低   | 第 %d 次迭代 | %.4f | 准确率 %.2f%% |\n', ...
            best_loss_iter, best_loss_val, acc_at_best_loss);
    fprintf(fid, '| 准确率最高  | 第 %d 次迭代 | %.2f%% | Loss %.4f |\n\n', ...
            best_acc_iter, best_acc_val, loss_at_best_acc);
    fprintf(fid, '> 当前保存模型来自 Loss 最低轮次（第 %d 次迭代）。\n',   best_loss_iter);
    fprintf(fid, '> 若两者不在同一迭代，说明模型在该轮判对更多但部分错误更自信。\n\n');

    fprintf(fid, '## 改进内容\n\n');

    fprintf(fid, '## 超参数\n');
    fprintf(fid, '- Epochs: %d\n',          exp_info.max_epochs);
    fprintf(fid, '- Batch Size: %d\n',      exp_info.batch_size);
    fprintf(fid, '- Learning Rate: %.6f\n', exp_info.learning_rate);
    fprintf(fid, '- L2: %.6f\n',            exp_info.l2_reg);

    fprintf(fid, '\n## 数据集分布\n详见 `dataset_distribution.csv`\n\n');

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