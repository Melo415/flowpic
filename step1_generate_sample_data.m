%% 步骤1：生成示例数据
% 创建模拟的网络流量数据用于测试和调试

clear; clc;
fprintf('====================================================\n');
fprintf('   步骤1：生成示例数据\n');
fprintf('====================================================\n\n');

%% 设置随机种子（保证可重复）
rng(42);

%% 参数设置
fprintf('[1/4] 设置参数...\n');

% 数据集配置
num_samples = 500;      % 总样本数（实际使用时可增加到几千）
num_classes = 5;        % 应用类别数
class_names = {'Facebook', 'Spotify', 'Skype', 'YouTube', 'WhatsApp'};

% 流量特征配置
packets_per_flow_range = [50, 300];    % 每个flow的包数范围
time_window = 15;                      % 时间窗口（秒）

fprintf('  - 样本数: %d\n', num_samples);
fprintf('  - 类别数: %d\n', num_classes);
fprintf('  - 包数范围: [%d, %d]\n', packets_per_flow_range(1), packets_per_flow_range(2));
fprintf('  - 时间窗口: %d 秒\n', time_window);

%% 生成流量数据
fprintf('\n[2/4] 生成网络流数据...\n');

flows = cell(num_samples, 1);
labels = zeros(num_samples, 1);
label_names = cell(num_samples, 1);

% 为不同类别定义不同的流量特征（模拟真实应用）
class_profiles = {
    % [mean_length, std_length, mean_iat, std_iat, uplink_ratio]
    [800, 400, 0.05, 0.03, 0.6];   % Facebook: 中等包长，均衡流量
    [400, 200, 0.02, 0.01, 0.5];   % Spotify: 小包，音频流
    [1200, 300, 0.03, 0.02, 0.4];  % Skype: 大包，视频流，下行多
    [1000, 500, 0.04, 0.025, 0.3]; % YouTube: 大包，视频下载为主
    [300, 150, 0.01, 0.005, 0.7];  % WhatsApp: 小包，上行多
};

fprintf('  正在生成数据');
for i = 1:num_samples
    if mod(i, 50) == 0
        fprintf('.');
    end
    
    % 随机选择一个类别
    class_id = randi(num_classes);
    labels(i) = class_id;
    label_names{i} = class_names{class_id};
    
    % 获取该类别的流量特征
    profile = class_profiles{class_id};
    mean_len = profile(1);
    std_len = profile(2);
    mean_iat = profile(3);
    std_iat = profile(4);
    uplink_ratio = profile(5);
    
    % 生成包数
    num_packets = randi(packets_per_flow_range);
    
    % 生成包长度（正态分布，截断到合理范围）
    lengths = round(normrnd(mean_len, std_len, num_packets, 1));
    lengths = max(40, min(1460, lengths));  % 限制在40-1460字节
    
    % 生成inter-arrival times（指数分布）
    iats = exprnd(mean_iat, num_packets, 1);
    iats = min(iats, 1);  % 限制最大IAT
    
    % 生成时间戳（累积IAT）
    times = cumsum([0; iats(1:end-1)]);
    
    % 只保留15秒内的包
    valid_idx = times <= time_window;
    lengths = lengths(valid_idx);
    times = times(valid_idx);
    num_packets = length(times);
    
    % 生成方向（根据uplink_ratio）
    directions = zeros(num_packets, 1);
    num_uplink = round(num_packets * uplink_ratio);
    uplink_indices = randperm(num_packets, num_uplink);
    directions(uplink_indices) = 1;
    directions(directions == 0) = -1;
    
    % 存储
    flows{i}.lengths = lengths;
    flows{i}.times = times;
    flows{i}.directions = directions;
    flows{i}.label = class_id;
    flows{i}.label_name = class_names{class_id};
end

fprintf(' 完成！\n');
fprintf('  ✓ 生成了 %d 个网络流\n', num_samples);

%% 数据统计
fprintf('\n[3/4] 数据统计分析...\n');

% 统计每个类别的样本数
fprintf('  类别分布:\n');
for c = 1:num_classes
    count = sum(labels == c);
    fprintf('    %d. %-10s: %3d 样本 (%.1f%%)\n', ...
            c, class_names{c}, count, 100*count/num_samples);
end

% 统计包数分布
packets_counts = cellfun(@(x) length(x.lengths), flows);
fprintf('\n  包数统计:\n');
fprintf('    平均: %.1f 包/流\n', mean(packets_counts));
fprintf('    范围: [%d, %d]\n', min(packets_counts), max(packets_counts));
fprintf('    中位数: %.1f\n', median(packets_counts));

% 统计包长度分布
all_lengths = cell2mat(cellfun(@(x) x.lengths, flows, 'UniformOutput', false));
fprintf('\n  包长度统计:\n');
fprintf('    平均: %.1f 字节\n', mean(all_lengths));
fprintf('    范围: [%d, %d]\n', min(all_lengths), max(all_lengths));

%% 保存数据
fprintf('\n[4/4] 保存数据...\n');

save('data/sample_flows.mat', 'flows', 'labels', 'label_names', 'class_names');
fprintf('  ✓ 已保存到: data/sample_flows.mat\n');

% 同时保存为CSV格式（方便查看）
fprintf('  正在导出CSV格式...\n');
csv_data = [];
for i = 1:min(10, num_samples)  % 只导出前10个flow的详细信息
    flow = flows{i};
    n = length(flow.lengths);
    
    flow_data = [
        repmat(i, n, 1), ...              % flow_id
        flow.times, ...                    % timestamp
        flow.lengths, ...                  % packet_length
        flow.directions, ...               % direction
        repmat(flow.label, n, 1)          % label
    ];
    
    csv_data = [csv_data; flow_data];
end

csv_table = array2table(csv_data, ...
    'VariableNames', {'flow_id', 'timestamp', 'packet_length', 'direction', 'label'});
writetable(csv_table, 'data/sample_flows_preview.csv');
fprintf('  ✓ 已保存预览到: data/sample_flows_preview.csv (前10个流)\n');

%% 可视化
fprintf('\n[可选] 生成可视化...\n');

fig = figure('Position', [100, 100, 1200, 600]);

% 子图1：类别分布
subplot(2, 3, 1);
histogram(categorical(label_names), 'FaceColor', [0.2 0.6 0.8]);
title('类别分布');
xlabel('应用类型');
ylabel('样本数');
grid on;

% 子图2：包数分布
subplot(2, 3, 2);
histogram(packets_counts, 20, 'FaceColor', [0.8 0.4 0.2]);
title('每流包数分布');
xlabel('包数');
ylabel('流数量');
grid on;

% 子图3：包长度分布
subplot(2, 3, 3);
histogram(all_lengths, 50, 'FaceColor', [0.4 0.8 0.2]);
title('包长度分布');
xlabel('长度 (字节)');
ylabel('频次');
grid on;

% 子图4-6：显示3个样本流
for idx = 1:3
    subplot(2, 3, 3 + idx);
    flow = flows{idx};
    
    % 上行和下行分开显示
    uplink_idx = flow.directions == 1;
    downlink_idx = flow.directions == -1;
    
    scatter(flow.times(uplink_idx), flow.lengths(uplink_idx), 30, 'r', 'filled');
    hold on;
    scatter(flow.times(downlink_idx), flow.lengths(downlink_idx), 30, 'b', 'filled');
    
    title(sprintf('样本 %d: %s', idx, flow.label_name));
    xlabel('时间 (秒)');
    ylabel('包长度 (字节)');
    legend('上行', '下行', 'Location', 'best');
    grid on;
    ylim([0, 1500]);
    xlim([0, time_window]);
end

saveas(fig, 'figures/step1_data_overview.png');
fprintf('  ✓ 可视化已保存到: figures/step1_data_overview.png\n');

%% 总结
fprintf('\n====================================================\n');
fprintf('   步骤1完成！\n');
fprintf('====================================================\n');
fprintf('✓ 已生成 %d 个样本，包含 %d 个类别\n', num_samples, num_classes);
fprintf('✓ 数据已保存到 data/ 目录\n');
fprintf('✓ 可视化已保存到 figures/ 目录\n\n');

fprintf('下一步：运行 step2_test_flowpic.m 测试FlowPic生成\n\n');
