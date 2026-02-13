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
fprintf('[2/6] 检查必需工具箱...\n');

% 初始化状态标记（关键：定义后续逻辑需要的变量）
dl_ok = false;
stat_ok = false;
all_installed = false; % 定义原脚本需要的all_installed变量

% 检查Deep Learning Toolbox
try
    convolution2dLayer(3, 16);
    dl_ok = true;
    fprintf('  ✓ Deep Learning Toolbox: 已安装\n');
catch
    fprintf('  ✗ Deep Learning Toolbox: 未安装/不可用\n');
end

% 检查Statistics and Machine Learning Toolbox
try
    ttest(randn(10,1));
    stat_ok = true;
    fprintf('  ✓ Statistics and Machine Learning Toolbox: 已安装\n');
catch
    fprintf('  ✗ Statistics and Machine Learning Toolbox: 未安装/不可用\n');
end

% 核心：更新all_installed变量（匹配原脚本逻辑）
all_installed = dl_ok && stat_ok;

% 输出警告
if ~all_installed
    fprintf('\n  警告：缺少必需工具箱，请安装后继续\n');
else
    fprintf('\n  所有必需工具箱均已安装\n');
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
    fprintf('运行 step1_generate_sample_data.m 生成示例数据\n');

else
    fprintf('⚠ 存在问题，请先解决上述错误\n');
end

fprintf('\n');
