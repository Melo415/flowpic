% load_mirage_json 是数据加载函数，不是专门的预处理函数
% 它的核心是把 JSON 文件中的原始流量数据读取为 MATLAB 可操作的结构体
% 仅附带 “让数据可用” 的基础预处理；

function [flows, labels, class_names] = load_mirage_json(dataset_path)
% 从MIRAGE-19 JSON文件加载网络流数据（适配L4_payload_bytes/packet_dir/iat字段）
% text 测试结果加载完成！总流数: 122007, 类别数: 18
% 输入:
%   dataset_path: MIRAGE数据集路径（包含JSON文件的文件夹）
% 输出:
%   flows: cell array，每个元素是一个flow结构体
%   labels: 标签向量
%   class_names: 类别名称

fprintf('正在加载MIRAGE-19 JSON数据集...\n');

%% 查找所有JSON文件
json_files = [dir(fullfile(dataset_path, '**', '*.json')); ...
              dir(fullfile(dataset_path, '*.json'))];

if isempty(json_files)
    error('在 %s 中未找到JSON文件', dataset_path);
end

fprintf('找到 %d 个JSON文件\n', length(json_files));

%% 读取所有JSON文件
flows = {};
label_strings = {};

fprintf('正在处理JSON文件');

for f = 1:length(json_files)
    if mod(f, 10) == 0
        fprintf('.');
    end
    
    json_path = fullfile(json_files(f).folder, json_files(f).name);
    
    try
        % 读取JSON
        fid = fopen(json_path, 'r');
        raw = fread(fid, inf, 'uint8=>char')';
        fclose(fid);
        data = jsondecode(raw);
        
        % 从文件名提取应用标签
        % 例如：1494596297_air.com.hypah.io.slither_MIRAGE-2019_traffic_dataset_labeled_biflows.json
        % 提取 "air.com.hypah.io.slither" 部分，简化为 "slither"
        [~, fname, ~] = fileparts(json_files(f).name);
        parts = split(fname, '_');
        if length(parts) >= 2
            % 提取包名部分（通常是第二部分）
            app_full = parts{2};
            % 提取最后一个词作为应用名
            app_parts = split(app_full, '.');
            app_label = app_parts{end};
        else
            app_label = parts{1};
        end
        
        % MIRAGE-19格式：每个字段是一个flow (五元组格式)
        flow_ids = fieldnames(data);
        
        if isempty(flow_ids)
            continue;
        end
        
        % 遍历每个flow
        for flow_idx = 1:length(flow_ids)
            flow_id = flow_ids{flow_idx};
            flow_data = data.(flow_id);
            
            % flow_data应该包含包信息
            % 适配实际字段：L4_payload_bytes/packet_dir/iat
            
            % 初始化变量
            lengths = [];
            times = [];
            directions = [];
            
            % 方式1：适配MIRAGE-2019新格式（你的数据结构）
            if isfield(flow_data, 'packet_data')
                pkt_data = flow_data.packet_data;
                
                % ========== 核心修改1：适配L4_payload_bytes（包长度） ==========
                if isfield(pkt_data, 'L4_payload_bytes')
                    lengths = pkt_data.L4_payload_bytes;
                elseif isfield(pkt_data, 'packet_sizes')
                    lengths = pkt_data.packet_sizes;
                elseif isfield(pkt_data, 'sizes')
                    lengths = pkt_data.sizes;
                elseif isfield(pkt_data, 'lengths')
                    lengths = pkt_data.lengths;
                else
                    continue;  % 无法提取长度，跳过此flow
                end
                
                % ========== 核心修改2：适配iat（计算绝对时间） ==========
                if isfield(pkt_data, 'iat')
                    % iat是包间间隔，需计算绝对时间（从0开始）
                    iat_1d = pkt_data.iat(:)';  % 转为一维行向量
                    if length(iat_1d) == length(lengths)-1
                        times = cumsum([0, iat_1d]);  % 0 + 第一个iat + 第二个iat...
                    else
                        times = cumsum([0, iat_1d(1:min(end, length(lengths)-1))]);
                    end
                elseif isfield(pkt_data, 'packet_times')
                    times = pkt_data.packet_times;
                elseif isfield(pkt_data, 'timestamps')
                    times = pkt_data.timestamps;
                elseif isfield(pkt_data, 'times')
                    times = pkt_data.times;
                else
                    % 生成默认时间戳
                    times = (0:length(lengths)-1)' * 0.001;
                end
                
                % ========== 核心修改3：适配packet_dir（方向） ==========
                if isfield(pkt_data, 'packet_dir')
                    directions = pkt_data.packet_dir;
                elseif isfield(pkt_data, 'packet_directions')
                    directions = pkt_data.packet_directions;
                elseif isfield(pkt_data, 'directions')
                    directions = pkt_data.directions;
                else
                    % 从长度符号推断
                    directions = sign(lengths);
                    lengths = abs(lengths);
                end
                
            % 方式2：直接字段格式（旧格式）
            elseif isfield(flow_data, 'ipPacketSizes') && isfield(flow_data, 'ipPacketTimestamps')
                % 标准格式
                lengths = flow_data.ipPacketSizes;
                times = flow_data.ipPacketTimestamps;
                
                % 方向信息
                if isfield(flow_data, 'ipPacketDirections')
                    directions = flow_data.ipPacketDirections;
                else
                    % 推断方向：正数=上行，负数=下行
                    directions = sign(lengths);
                    lengths = abs(lengths);
                end
                
            elseif isfield(flow_data, 'packets')
                % 包数组格式
                packets = flow_data.packets;
                n_packets = length(packets);
                
                times = zeros(n_packets, 1);
                lengths = zeros(n_packets, 1);
                directions = zeros(n_packets, 1);
                
                for i = 1:n_packets
                    pkt = packets(i);
                    
                    % 时间
                    if isfield(pkt, 'timestamp')
                        times(i) = pkt.timestamp;
                    elseif isfield(pkt, 'time')
                        times(i) = pkt.time;
                    else
                        times(i) = i * 0.001;
                    end
                    
                    % 长度
                    if isfield(pkt, 'length')
                        lengths(i) = abs(pkt.length);
                    elseif isfield(pkt, 'size')
                        lengths(i) = abs(pkt.size);
                    else
                        lengths(i) = 1000;
                    end
                    
                    % 方向
                    if isfield(pkt, 'direction')
                        directions(i) = pkt.direction;
                    else
                        directions(i) = sign(pkt.length);
                    end
                end
                
            else
                % 跳过无法识别格式的flow
                continue;
            end
            
            % ========== 核心修改4：统一维度（解决维度不一致问题） ==========
            % 转为列向量，确保维度统一
            times = times(:);
            lengths = lengths(:);
            directions = directions(:);
            
            % 确保方向是 1 或 -1
            directions(directions == 0) = 1;
            directions = sign(directions);
            
            % 归一化时间戳（从0开始）
            if length(times) > 0 && any(times > 0)
                times = times - min(times);
            end
            
            % 移除零长度包（TCP SYN / ACK / FIN）
            nonzero_idx = abs(lengths) > 0;
            lengths    = lengths(nonzero_idx);
            times      = times(nonzero_idx);
            directions = directions(nonzero_idx);
            
            % 限制有效包的长度范围到 [1, MTU]
            lengths = min(1460, abs(lengths));
            
            % 只保留前15秒的包
            valid_idx = times <= 15;
            times = times(valid_idx);
            lengths = lengths(valid_idx);
            directions = directions(valid_idx);
            
            % 至少要有10个包（可根据需求调整）
            if length(times) >= 10
                flow = struct();
                flow.lengths = lengths;
                flow.times = times;
                flow.directions = directions;
                
                flows{end+1} = flow;
                label_strings{end+1} = app_label;
            end
        end
        
    catch ME
        fprintf('\n警告: 无法处理文件 %s: %s\n', json_files(f).name, ME.message);
        continue;
    end
end

fprintf(' 完成!\n');

if isempty(flows)
    error('未能从JSON文件中提取任何流数据。请检查JSON格式。');
end

%% 转换为数组
flows = flows';
label_strings = label_strings';

%% 转换标签
[class_names, ~, labels] = unique(label_strings, 'stable');

fprintf('\n数据集统计:\n');
fprintf('  总流数: %d\n', length(flows));
fprintf('  类别数: %d\n', length(class_names));
fprintf('  类别: %s\n', strjoin(class_names, ', '));

% 显示每类的数量
fprintf('\n类别分布:\n');
for c = 1:length(class_names)
    count = sum(labels == c);
    fprintf('  %d. %-15s: %6d (%.1f%%)\n', ...
            c, class_names{c}, count, 100*count/length(flows));
end

% 显示包统计
all_packet_counts = cellfun(@(x) length(x.lengths), flows);
fprintf('\n包统计:\n');
fprintf('  平均包数/流: %.1f\n', mean(all_packet_counts));
fprintf('  包数范围: [%d, %d]\n', min(all_packet_counts), max(all_packet_counts));

end