function flowpic = generate_flowpic(lengths, times, directions, size)
% 生成四通道flowpic直方图
% 输入:
%   lengths: 数据包长度向量
%   times: 相对时间向量 (秒)
%   directions: 方向向量 (1=上行, -1=下行)
%   size: 直方图大小 (默认32)
% 输出:
%   flowpic: size×size×4 的多通道直方图

if nargin < 4
    size = 32;
end

% 参数设置
MAX_TIME = 15;          % 最大时间15秒
MTU = 1460;             % 最大传输单元
TIME_BINS = size;
LENGTH_BINS = size;

% 只取前15秒的数据包
valid_idx = times <= MAX_TIME;
lengths = lengths(valid_idx);
times = times(valid_idx);
directions = directions(valid_idx);

% 计算 inter-arrival times
iat = [0; diff(times)];  % 第一个包的IAT设为0

% 计算95th percentile作为IAT的上限
iat_max = prctile(iat, 95);
if iat_max == 0
    iat_max = 0.001;  % 防止除零
end

% 定义bins
time_edges = linspace(0, MAX_TIME, TIME_BINS + 1);
length_edges = linspace(0, MTU, LENGTH_BINS + 1);
iat_edges = linspace(0, iat_max, TIME_BINS + 1);

% 初始化四通道直方图
flowpic = zeros(LENGTH_BINS, TIME_BINS, 4);

% 分离上行和下行
uplink_idx = directions == 1;
downlink_idx = directions == -1;

% Channel 1: 上行packet length vs time
if any(uplink_idx)
    flowpic(:,:,1) = hist2d(lengths(uplink_idx), times(uplink_idx), ...
                            length_edges, time_edges);
end

% Channel 2: 下行packet length vs time  
if any(downlink_idx)
    flowpic(:,:,2) = hist2d(lengths(downlink_idx), times(downlink_idx), ...
                            length_edges, time_edges);
end

% Channel 3: 上行inter-arrival time vs time
if any(uplink_idx)
    flowpic(:,:,3) = hist2d(iat(uplink_idx), times(uplink_idx), ...
                            iat_edges, time_edges);
end

% Channel 4: 下行inter-arrival time vs time
if any(downlink_idx)
    flowpic(:,:,4) = hist2d(iat(downlink_idx), times(downlink_idx), ...
                            iat_edges, time_edges);
end

end

function H = hist2d(x, y, x_edges, y_edges)
% 生成2D直方图
% x: 行变量 (length 或 IAT)
% y: 列变量 (time)

n_x = length(x_edges) - 1;
n_y = length(y_edges) - 1;
H = zeros(n_x, n_y);

% 对每个数据点分配到对应的bin
for i = 1:length(x)
    x_bin = find(x(i) >= x_edges, 1, 'last');
    y_bin = find(y(i) >= y_edges, 1, 'last');
    
    % 边界检查
    if isempty(x_bin), x_bin = 1; end
    if isempty(y_bin), y_bin = 1; end
    if x_bin > n_x, x_bin = n_x; end
    if y_bin > n_y, y_bin = n_y; end
    
    H(x_bin, y_bin) = H(x_bin, y_bin) + 1;
end

end
