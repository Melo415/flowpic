function [flows, labels, class_names] = load_mirage_json(dataset_path)
    % LOAD_MIRAGE_JSON 加载MIRAGE-19 JSON数据集
    % 输入: dataset_path - 数据集根目录
    % 输出: flows - 所有流数据(cell数组)
    %       labels - 对应标签(数值)
    %       class_names - 类别名称(cell数组)
    
    % 初始化变量
    flows = {};
    labels = [];
    class_names = {};
    label_map = containers.Map();
    current_label = 1;
    
    % 获取所有JSON文件
    json_files = dir(fullfile(dataset_path, '**', '*.json'));
    fprintf('找到 %d 个JSON文件\n', length(json_files));
    
    % 遍历所有文件
    for f_idx = 1:length(json_files)
        file_path = fullfile(json_files(f_idx).folder, json_files(f_idx).name);
        fprintf('处理文件: %s\n', file_path);
        
        % 读取JSON文件
        fid = fopen(file_path, 'r');
        if fid == -1
            warning('无法打开文件: %s', file_path);
            continue;
        end
        raw = fread(fid, inf, 'uint8=>char')';
        fclose(fid);
        data = jsondecode(raw);
        
        % 提取应用标签
        [~, fname, ~] = fileparts(file_path);
        parts = split(fname, '_');
        if length(parts) >= 2
            app_full = parts{2};
            app_parts = split(app_full, '.');
            app_label = app_parts{end};
        else
            app_label = parts{1};
        end
        
        % 映射标签到数值
        if ~isKey(label_map, app_label)
            label_map(app_label) = current_label;
            class_names{current_label} = app_label;
            current_label = current_label + 1;
        end
        numeric_label = label_map(app_label);
        
        % 遍历当前文件的所有flow
        flow_ids = fieldnames(data);
        for flow_idx = 1:length(flow_ids)
            flow_id = flow_ids{flow_idx};
            flow_data = data.(flow_id);
            
            % 初始化当前流的变量
            flow_info = struct();
            flow_info.lengths = [];
            flow_info.times = [];
            flow_info.directions = [];
            flow_info.label = numeric_label;
            flow_info.app_name = app_label;
            
            % 提取packet_data（核心修复：解决维度+运算符错误）
            if isfield(flow_data, 'packet_data')
                pkt_data = flow_data.packet_data;
                
                % 1. 提取包长度 (L4_payload_bytes) - 修复维度+运算符
                if isfield(pkt_data, 'L4_payload_bytes')
                    % 转为一维行向量，避免维度错误
                    flow_info.lengths = pkt_data.L4_payload_bytes(:)'; 
                elseif isfield(pkt_data, 'packet_sizes')
                    flow_info.lengths = pkt_data.packet_sizes(:)';
                elseif isfield(pkt_data, 'sizes')
                    flow_info.lengths = pkt_data.sizes(:)';
                end
                
                % 2. 提取时间戳 (iat) - 修复维度+运算符
                if isfield(pkt_data, 'iat')
                    % 关键修复：先转一维，再计算绝对时间（避免运算符错误）
                    iat_1d = pkt_data.iat(:)';
                    flow_info.times = cumsum([0, iat_1d]);
                elseif isfield(pkt_data, 'packet_times')
                    flow_info.times = pkt_data.packet_times(:)';
                elseif isfield(pkt_data, 'timestamps')
                    flow_info.times = pkt_data.timestamps(:)';
                end
                
                % 3. 提取方向 (packet_dir) - 修复维度+运算符
                if isfield(pkt_data, 'packet_dir')
                    flow_info.directions = pkt_data.packet_dir(:)';
                elseif isfield(pkt_data, 'packet_directions')
                    flow_info.directions = pkt_data.packet_directions(:)';
                elseif isfield(pkt_data, 'directions')
                    flow_info.directions = pkt_data.directions(:)';
                end
            end
            
            % 回退：旧格式兼容
            if isempty(flow_info.lengths) && isfield(flow_data, 'ipPacketSizes')
                flow_info.lengths = flow_data.ipPacketSizes(:)';
            end
            if isempty(flow_info.times) && isfield(flow_data, 'ipPacketTimestamps')
                flow_info.times = flow_data.ipPacketTimestamps(:)';
            end
            if isempty(flow_info.directions) && isfield(flow_data, 'ipPacketDirections')
                flow_info.directions = flow_data.ipPacketDirections(:)';
            end
            
            % 仅保留有效流（有包长度数据）
            if ~isempty(flow_info.lengths)
                flows{end+1} = flow_info;
                labels = [labels, numeric_label]; % 修复：确保数组拼接运算符正确
            end
        end
    end
    
    % 整理类别名称（排序）
    [~, sorted_idx] = sort(cell2mat(values(label_map)));
    class_names = class_names(sorted_idx);
    
    fprintf('加载完成！总流数: %d, 类别数: %d\n', length(flows), length(class_names));
end