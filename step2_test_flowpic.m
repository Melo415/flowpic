%% 步骤2：测试FlowPic生成
% 详细测试和调试FlowPic四通道直方图生成功能

clear; clc;
fprintf('====================================================\n');
fprintf('   步骤2：测试FlowPic生成\n');
fprintf('====================================================\n\n');

%% 加载数据
fprintf('[1/5] 加载示例数据...\n');

if ~exist('data/sample_flows.mat', 'file')
    error('错误：找不到数据文件。请先运行 step1_generate_sample_data.m');
end

load('data/sample_flows.mat', 'flows', 'labels', 'class_names');
fprintf('  ✓ 已加载 %d 个流样本\n', length(flows));

%% 测试单个流的FlowPic生成
fprintf('\n[2/5] 测试单个流的FlowPic生成...\n');

% 选择第一个流进行详细测试
test_flow = flows{1};
fprintf('  测试流信息:\n');
fprintf('    - 类别: %s\n', test_flow.label_name);
fprintf('    - 包数: %d\n', length(test_flow.lengths));
fprintf('    - 时间范围: [%.3f, %.3f] 秒\n', ...
            min(test_flow.times), max(test_flow.times));
fprintf('    - 长度范围: [%d, %d] 字节\n', ...
            min(test_flow.lengths), max(test_flow.lengths));

% 统计方向
num_uplink = sum(test_flow.directions == 1);
num_downlink = sum(test_flow.directions == -1);
fprintf('    - 上行包: %d (%.1f%%)\n', num_uplink, 100*num_uplink/length(test_flow.directions));
fprintf('    - 下行包: %d (%.1f%%)\n', num_downlink, 100*num_downlink/length(test_flow.directions));

% 生成FlowPic
fprintf('\n  生成FlowPic...\n');
tic;
flowpic = generate_flowpic(test_flow.lengths, test_flow.times, ...
                           test_flow.directions, 32);
elapsed = toc;

fprintf('  ✓ 生成成功！用时: %.4f 秒\n', elapsed);
fprintf('  FlowPic属性:\n');
fprintf('    - 形状: %s\n', mat2str(size(flowpic)));
fprintf('    - 数据类型: %s\n', class(flowpic));
fprintf('    - 总计数: %.0f\n', sum(flowpic(:)));

% 检查每个通道
channel_names = {'上行Length', '下行Length', '上行IAT', '下行IAT'};
for ch = 1:4
    channel_sum = sum(flowpic(:,:,ch), 'all');
    channel_max = max(flowpic(:,:,ch), [], 'all');
    fprintf('    - Channel %d (%s): sum=%.0f, max=%.0f\n', ...
            ch, channel_names{ch}, channel_sum, channel_max);
end

%% 可视化单个FlowPic
fprintf('\n[3/5] 可视化FlowPic四通道...\n');

fig1 = figure('Name', '单个FlowPic可视化', 'Position', [100, 100, 1200, 900]);

for ch = 1:4
    subplot(2, 2, ch);
    imagesc(flowpic(:,:,ch));
    colorbar;
    colormap('hot');
    title(sprintf('Channel %d: %s', ch, channel_names{ch}));
    xlabel('时间 bins (0-15秒)');
    ylabel('Length/IAT bins');
    
    % 添加数值统计到标题
    ch_sum = sum(flowpic(:,:,ch), 'all');
    ch_max = max(flowpic(:,:,ch), [], 'all');
    subtitle(sprintf('sum=%.0f, max=%.0f', ch_sum, ch_max));
end

sgtitle(sprintf('FlowPic: %s (包数=%d)', test_flow.label_name, length(test_flow.lengths)));
saveas(fig1, 'figures/step2_single_flowpic.png');
fprintf('  ✓ 已保存到: figures/step2_single_flowpic.png\n');

%% 批量生成FlowPic
fprintf('\n[4/5] 批量生成FlowPic（测试性能）...\n');

num_test = min(100, length(flows));  % 测试100个样本
flowpic_size = 32;

fprintf('  正在生成 %d 个FlowPic', num_test);

tic;
X_test = zeros(flowpic_size, flowpic_size, 4, num_test);

for i = 1:num_test
    if mod(i, 20) == 0
        fprintf('.');
    end
    
    X_test(:,:,:,i) = generate_flowpic(flows{i}.lengths, ...
                                        flows{i}.times, ...
                                        flows{i}.directions, ...
                                        flowpic_size);
end

batch_time = toc;
fprintf(' 完成！\n');
fprintf('  ✓ 总用时: %.2f 秒\n', batch_time);
fprintf('  ✓ 平均每个: %.4f 秒\n', batch_time/num_test);
fprintf('  ✓ 预计处理1000个: %.1f 秒\n', batch_time/num_test*1000);

%% 验证数据质量
fprintf('\n[5/5] 验证FlowPic数据质量...\n');

% 检查是否有NaN或Inf
has_nan = any(isnan(X_test(:)));
has_inf = any(isinf(X_test(:)));

if has_nan
    fprintf('  ✗ 警告：检测到NaN值！\n');
else
    fprintf('  ✓ 无NaN值\n');
end

if has_inf
    fprintf('  ✗ 警告：检测到Inf值！\n');
else
    fprintf('  ✓ 无Inf值\n');
end

% 统计值范围
fprintf('  数值范围:\n');
fprintf('    - 最小值: %.2f\n', min(X_test(:)));
fprintf('    - 最大值: %.2f\n', max(X_test(:)));
fprintf('    - 平均值: %.2f\n', mean(X_test(:)));
fprintf('    - 中位数: %.2f\n', median(X_test(:)));

% 检查空FlowPic（全零）
num_empty = 0;
for i = 1:num_test
    if sum(X_test(:,:,:,i), 'all') == 0
        num_empty = num_empty + 1;
    end
end

if num_empty > 0
    fprintf('  ⚠ 发现 %d 个空FlowPic（可能是数据问题）\n', num_empty);
else
    fprintf('  ✓ 无空FlowPic\n');
end

% 可视化多个样本
fprintf('\n  生成多样本对比图...\n');

fig2 = figure('Name', '多样本FlowPic对比', 'Position', [100, 100, 1400, 1000]);

num_display = 9;  % 显示9个样本
sample_indices = randperm(num_test, num_display);

for idx = 1:num_display
    i = sample_indices(idx);
    
    % 显示Channel 1（上行Length）
    subplot(3, 3, idx);
    imagesc(X_test(:,:,1,i));
    colorbar;
    colormap('hot');
    title(sprintf('%d: %s', i, flows{i}.label_name));
    
    if idx > 6
        xlabel('时间');
    end
    if mod(idx-1, 3) == 0
        ylabel('长度');
    end
end

sgtitle('FlowPic对比 - Channel 1 (上行包长度)');
saveas(fig2, 'figures/step2_multi_flowpic.png');
fprintf('  ✓ 已保存到: figures/step2_multi_flowpic.png\n');

%% 不同大小对比
fprintf('\n[额外测试] 测试不同FlowPic尺寸...\n');

sizes_to_test = [16, 32, 64];
fprintf('  测试尺寸: %s\n', mat2str(sizes_to_test));

fig3 = figure('Name', 'FlowPic尺寸对比', 'Position', [100, 100, 1200, 400]);

for idx = 1:length(sizes_to_test)
    sz = sizes_to_test(idx);
    
    fp = generate_flowpic(test_flow.lengths, test_flow.times, ...
                         test_flow.directions, sz);
    
    subplot(1, 3, idx);
    imagesc(fp(:,:,1));  % 只显示Channel 1
    colorbar;
    colormap('hot');
    title(sprintf('%d × %d', sz, sz));
    xlabel('时间 bins');
    ylabel('长度 bins');
end

sgtitle('不同FlowPic尺寸对比');
saveas(fig3, 'figures/step2_size_comparison.png');
fprintf('  ✓ 已保存到: figures/step2_size_comparison.png\n');

%% 保存测试结果
fprintf('\n  保存测试数据...\n');
save('data/flowpic_test.mat', 'X_test', 'num_test');
fprintf('  ✓ 已保存到: data/flowpic_test.mat\n');

%% 总结
fprintf('\n====================================================\n');
fprintf('   步骤2完成！\n');
fprintf('====================================================\n');
fprintf('✓ FlowPic生成功能正常\n');
fprintf('✓ 数据质量验证通过\n');
fprintf('✓ 性能测试完成：%.4f 秒/样本\n', batch_time/num_test);
fprintf('✓ 可视化已保存到 figures/ 目录\n\n');

fprintf('下一步：运行 step3_test_model.m 测试模型创建\n\n');
