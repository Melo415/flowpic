% 原数据加载代码（示例）
function [imds, numClasses] = load_data()
    % 假设原代码错误统计了类别数为5，实际标签只有3类
    dataPath = 'D:\Desktop\基于深度学习的加密流量分类\part2\files\data\sample_flows_preview.csv';
    imds = imageDatastore(dataPath, 'IncludeSubfolders', true, 'LabelSource', 'foldernames');
    
    % 关键修正：获取真实的类别数
    labels = categories(imds.Labels);
    numClasses = length(labels); % 此时会得到真实的3类，而非5类
    fprintf('  ✓ 类别数: %d\n', numClasses);
end