%% 步骤0：环境检查和配置
% 运行此脚本检查MATLAB环境是否满足要求

clear; clc;
fprintf('====================================================\n');
fprintf('   FlowPic 环境检查工具\n');
fprintf('====================================================\n\n');

%% 1. MATLAB版本检查
fprintf('[1/6] 检查MATLAB版本...\n');
matlab_version = version('-release');
matlab_year = str2double(matlab_version(1:4));

if matlab_year >= 2020
    fprintf('  ✓ MATLAB版本: %s (满足要求 ≥ R2020a)\n', matlab_version);
else
    fprintf('  ✗ MATLAB版本: %s (需要 R2020a 或更高)\n', matlab_version);
    fprintf('    建议升级MATLAB版本\n');
end

%% 2. 检查必需的工具箱
fprintf('\n[2/6] 检查必需工具箱...\n');

% 修正：使用MATLAB官方标准的工具箱授权ID
required_toolboxes = {
    'Deep Learning Toolbox', 'Deep_Learning_Toolbox';          % 修正后的DL工具箱ID
    'Statistics and Machine Learning Toolbox', 'Statistics_Machine_Learning'; % 修正后的统计工具箱ID
};

all_installed = true;
for i = 1:size(required_toolboxes, 1)
    toolbox_name = required_toolboxes{i, 1};
    toolbox_id = required_toolboxes{i, 2};
    
    % license('test', id) 返回1表示授权可用，0表示不可用
    if license('test', toolbox_id)
        fprintf('  ✓ %s: 已安装\n', toolbox_name);
    else
        fprintf('  ✗ %s: 未安装\n', toolbox_name);
        all_installed = false;
    end
end

if ~all_installed
    fprintf('\n  警告：缺少必需工具箱，请安装后继续\n');
end
%% 3. 检查可选工具箱（GPU支持）
fprintf('\n[3/6] 检查GPU支持...\n');

if license('test', 'distrib_computing_toolbox')
    fprintf('  ✓ Parallel Computing Toolbox: 已安装\n');
    
    % 检查GPU设备
    try
        gpu_device = gpuDevice;
        fprintf('  ✓ GPU设备: %s\n', gpu_device.Name);
        fprintf('    - 显存: %.1f GB\n', gpu_device.AvailableMemory / 1024^3);
        fprintf('    - 计算能力: %.1f\n', gpu_device.ComputeCapability);
        fprintf('    建议在训练时使用GPU加速\n');
    catch
        fprintf('  ⚠ GPU设备未检测到，将使用CPU训练\n');
    end
else
    fprintf('  ⚠ Parallel Computing Toolbox: 未安装\n');
    fprintf('    将使用CPU训练（速度较慢）\n');
end

%% 4. 创建工作目录
fprintf('\n[4/6] 创建工作目录...\n');

folders = {'data', 'models', 'results', 'figures'};
for i = 1:length(folders)
    if ~exist(folders{i}, 'dir')
        mkdir(folders{i});
        fprintf('  ✓ 创建目录: %s/\n', folders{i});
    else
        fprintf('  - 目录已存在: %s/\n', folders{i});
    end
end

%% 5. 检查代码文件
fprintf('\n[5/6] 检查代码文件...\n');

required_files = {
    'generate_flowpic.m';
    'create_flowpic_model.m';
    'train_flowpic.m';
    'load_pcap_data.m';
    'quick_test.m';
};

all_files_exist = true;
for i = 1:length(required_files)
    if exist(required_files{i}, 'file')
        fprintf('  ✓ %s\n', required_files{i});
    else
        fprintf('  ✗ %s (缺失)\n', required_files{i});
        all_files_exist = false;
    end
end

if ~all_files_exist
    fprintf('\n  错误：缺少必需文件，请确保所有.m文件在当前目录\n');
end

%% 6. 测试基本功能
fprintf('\n[6/6] 测试基本功能...\n');

try
    % 测试数组操作
    test_data = rand(32, 32, 4, 10);
    fprintf('  ✓ 数组操作: 正常\n');
    
    % 测试直方图计算
    test_hist = histcounts(rand(100,1), 10);
    fprintf('  ✓ 直方图计算: 正常\n');
    
    % 测试图像显示
    fig = figure('Visible', 'off');
    imagesc(rand(32, 32));
    close(fig);
    fprintf('  ✓ 图像显示: 正常\n');
    
catch ME
    fprintf('  ✗ 基本功能测试失败: %s\n', ME.message);
end

%% 总结
fprintf('\n====================================================\n');
fprintf('   环境检查完成\n');
fprintf('====================================================\n');

if all_installed && all_files_exist
    fprintf('✓ 所有检查通过！可以开始使用。\n\n');
    fprintf('下一步操作：\n');
    fprintf('  1. 运行 step1_generate_sample_data.m 生成示例数据\n');
    fprintf('  2. 运行 step2_test_flowpic.m 测试FlowPic生成\n');
    fprintf('  3. 运行 step3_test_model.m 测试模型创建\n');
    fprintf('  4. 运行 step4_full_training.m 完整训练\n');
else
    fprintf('⚠ 存在问题，请先解决上述错误\n');
end

fprintf('\n');
