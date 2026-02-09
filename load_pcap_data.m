function [flows, labels, class_names] = load_mirage_data(dataset_path)
% 从MIRAGE数据集加载网络流数据
% 
% 输入:
%   dataset_path: MIRAGE数据集路径（包含CSV文件的文件夹）
% 输出:
%   flows: cell array，每个元素是一个flow结构体
%   labels: 标签向量
%   class_names: 类别名称

fprintf('正在加载MIRAGE数据集...\n');

%% 方法1：如果数据集已转换为CSV格式
% MIRAGE数据集格式通常是：
% flow_id, timestamp, length, direction, app_label

% 查找所有CSV文件
csv_files = dir(fullfile(dataset_path, '*.csv'));

if isempty(csv_files)
    error('在 %s 中未找到CSV文件', dataset_path);
end

fprintf('找到 %d 个CSV文件\n', length(csv_files));

% 读取第一个CSV文件作为示例
data = readtable(fullfile(dataset_path, csv_files(1).name));

% 检查列名（根据实际数据集调整）
expected_cols = {'flow_id', 'timestamp', 'length', 'direction', 'label'};
% 或者可能是：{'FlowID', 'Time', 'PacketLength', 'Direction', 'Application'}

fprintf('数据集列名: %s\n', strjoin(data.Properties.VariableNames, ', '));

%% 提取flow数据
% 获取唯一的flow_id
flow_ids = unique(data.flow_id);
num_flows = length(flow_ids);

fprintf('总流数: %d\n', num_flows);

flows = cell(num_flows, 1);
label_strings = cell(num_flows, 1);

fprintf('正在处理流数据');
for i = 1:num_flows
    if mod(i, 1000) == 0
        fprintf('.');
    end
    
    % 提取当前flow的所有包
    idx = data.flow_id == flow_ids(i);
    
    flows{i}.lengths = data.length(idx);
    flows{i}.times = data.timestamp(idx);
    flows{i}.directions = data.direction(idx);
    
    % 标签（假设同一flow的所有包标签相同）
    label_strings{i} = data.label{find(idx, 1)};
end
fprintf(' 完成!\n');

%% 转换标签
[class_names, ~, labels] = unique(label_strings, 'stable');

fprintf('\n数据集统计:\n');
fprintf('  总流数: %d\n', num_flows);
fprintf('  类别数: %d\n', length(class_names));
fprintf('  类别: %s\n', strjoin(class_names, ', '));

% 显示每类的数量
fprintf('\n类别分布:\n');
for c = 1:length(class_names)
    count = sum(labels == c);
    fprintf('  %d. %-15s: %6d (%.1f%%)\n', ...
            c, class_names{c}, count, 100*count/num_flows);
end

end