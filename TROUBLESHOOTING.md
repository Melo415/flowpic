# 故障排除指南 (Troubleshooting Guide)

## 🔴 步骤0：环境检查失败

### 问题1：MATLAB版本不足
```
✗ MATLAB版本: 2019b (需要 R2020a 或更高)
```

**解决方案：**
- 方案A：升级MATLAB到R2020a或更新版本
- 方案B：修改代码以兼容旧版本（不推荐）

**验证：**
```matlab
version('-release')  % 应显示 '2020a' 或更高
```

---

### 问题2：缺少Deep Learning Toolbox
```
✗ Deep Learning Toolbox: 未安装
```

**解决方案：**
1. 打开MATLAB
2. 点击 "主页" → "添加-Ons" → "获取添加-Ons"
3. 搜索 "Deep Learning Toolbox"
4. 安装

**验证：**
```matlab
license('test', 'deep_learning_toolbox')  % 应返回 1
```

---

### 问题3：GPU检测失败
```
⚠ GPU设备未检测到
```

**这不是错误**，可以继续使用CPU。

如需GPU加速：
1. 检查GPU驱动：`nvidia-smi`（Windows/Linux）
2. 安装CUDA Toolkit（与MATLAB版本兼容）
3. 重启MATLAB

**验证：**
```matlab
gpuDevice  % 应显示GPU信息
```

---

## 🟠 步骤1：数据生成问题

### 问题1：数据分布不均
```
1. Facebook  : 250 样本 (50.0%)  % 太多
2. Spotify   :  50 样本 (10.0%)  % 太少
```

**原因：** 随机种子问题

**解决方案：**
```matlab
% 在step1开头添加
rng(42);  % 固定随机种子
```

---

### 问题2：时间超过15秒
```
警告: 发现时间戳超过15秒的包
```

**解决方案：**
在step1中修改：
```matlab
% 截断时间
times = times(times <= 15);
lengths = lengths(1:length(times));
directions = directions(1:length(times));
```

---

### 问题3：包长度异常
```
警告: 发现包长度 > 1460 或 < 40
```

**解决方案：**
```matlab
% 限制包长度范围
lengths = max(40, min(1460, lengths));
```

---

## 🟡 步骤2：FlowPic生成问题

### 问题1：FlowPic全为0
```
总计数: 0  % 错误！应该等于包数
```

**可能原因：**
1. 时间戳全部为0
2. 包长度全部为0
3. binning问题

**调试：**
```matlab
% 检查输入数据
fprintf('包数: %d\n', length(lengths));
fprintf('时间范围: [%.3f, %.3f]\n', min(times), max(times));
fprintf('长度范围: [%d, %d]\n', min(lengths), max(lengths));

% 检查binning
fprintf('时间bins: %d\n', length(time_edges)-1);
fprintf('长度bins: %d\n', length(length_edges)-1);
```

**解决方案：**
检查generate_flowpic.m的hist2d函数

---

### 问题2：Channel sum不对
```
Channel 1 (上行Length): sum=150  % 正确
Channel 2 (下行Length): sum=0    % 错误！
```

**原因：** 方向标注错误

**检查：**
```matlab
unique(directions)  % 应该有 1 和 -1
sum(directions == 1)  % 上行包数
sum(directions == -1) % 下行包数
```

**解决方案：**
确保directions只包含1和-1

---

### 问题3：性能太慢
```
平均每个: 0.5000 秒  % 太慢！应该 < 0.01秒
```

**可能原因：** hist2d实现效率低

**优化：**
```matlab
% 使用MATLAB内置函数
H = histcounts2(x, y, x_edges, y_edges);
```

---

## 🔵 步骤3：模型创建问题

### 问题1：模型创建失败
```
错误：Undefined function 'additionLayer'
```

**原因：** MATLAB版本太旧或Deep Learning Toolbox未更新

**解决方案：**
```matlab
% 检查函数是否存在
which additionLayer  % 应显示路径

% 如果不存在，更新toolbox
```

---

### 问题2：网络连接错误
```
错误: 层 'res1_add' 的输入未连接
```

**调试：**
```matlab
% 查看所有连接
lgraph.Connections

% 查看未连接的层
disconnectedLayers = findDisconnectedLayers(lgraph);
```

**解决方案：**
检查create_flowpic_model.m中的connectLayers调用

---

### 问题3：参数量异常
```
总参数量: 5824500 (5824.50 K)  % 太多！应该约58K
```

**可能原因：** 层配置错误

**检查：**
```matlab
% 查看每层参数
for i = 1:length(lgraph.Layers)
    layer = lgraph.Layers(i);
    if isprop(layer, 'Weights')
        fprintf('%s: %d params\n', layer.Name, numel(layer.Weights));
    end
end
```

---

### 问题4：前向传播失败
```
错误: 层 'conv1' 输入大小不匹配
期望: 32x32x4, 实际: 32x32x1
```

**解决方案：**
```matlab
% 确保输入大小正确
input_size = [32, 32, 4];  % 必须是4通道
test_input = rand(32, 32, 4, 8);  % 注意第3维是4
```

---

## 🟢 步骤4：训练问题

### 问题1：内存溢出 (Out of Memory)
```
错误: Out of memory
```

**解决方案A：** 减小batch size
```matlab
options = trainingOptions('adam', ...
    'MiniBatchSize', 32, ...  % 从64改为32
    ...
);
```

**解决方案B：** 清理内存
```matlab
clear all;
close all;
clc;
```

**解决方案C：** 使用CPU而非GPU
```matlab
options = trainingOptions('adam', ...
    'ExecutionEnvironment', 'cpu', ...
    ...
);
```

---

### 问题2：训练不收敛
```
Epoch 50/50: Loss = 1.6094, Val Acc = 20.0%  % 随机猜测水平
```

**可能原因：**
1. 学习率太大或太小
2. 数据标签错误
3. 模型结构问题

**调试步骤：**

1. **检查数据：**
```matlab
% 查看几个样本
for i = 1:5
    fprintf('样本%d: 标签=%s, FlowPic sum=%.0f\n', ...
            i, char(Y_train(i)), sum(X_train(:,:,:,i), 'all'));
end
```

2. **调整学习率：**
```matlab
'InitialLearnRate', 0.01,  % 尝试更大
% 或
'InitialLearnRate', 0.0001,  % 尝试更小
```

3. **简化模型测试：**
```matlab
% 用10个样本过拟合测试
X_small = X_train(:,:,:,1:10);
Y_small = Y_train(1:10);
options_test = trainingOptions('adam', 'MaxEpochs', 100, 'Verbose', true);
net_test = trainNetwork(X_small, Y_small, lgraph, options_test);
% 应该能达到100%准确率
```

---

### 问题3：训练太慢
```
Epoch 1/50: 预计剩余时间 10小时  % 太慢
```

**加速方案：**

**方案A：** 使用GPU
```matlab
'ExecutionEnvironment', 'gpu',
```

**方案B：** 增大batch size
```matlab
'MiniBatchSize', 128,  % 如果内存允许
```

**方案C：** 减少数据
```matlab
% 只用200个样本测试
X_train_small = X_train(:,:,:,1:200);
Y_train_small = Y_train(1:200);
```

**方案D：** 减少epochs
```matlab
'MaxEpochs', 20,  % 先用20个epoch测试
```

---

### 问题4：过拟合严重
```
训练集准确率: 99.5%
验证集准确率: 65.0%
测试集准确率: 62.0%
```

**解决方案：**

1. **增加Dropout：**
```matlab
% 在create_flowpic_model.m中
dropoutLayer(0.5, 'Name', 'drop1')  % 从0.2改为0.5
```

2. **增加L2正则化：**
```matlab
'L2Regularization', 0.001,  % 从0.0001增加到0.001
```

3. **数据增强：**
```matlab
% 可以实现时间平移、缩放等
```

4. **增加训练数据：**
```matlab
% step1中增加num_samples
num_samples = 1000;  % 从500增加到1000
```

---

### 问题5：GPU错误
```
错误: GPU memory allocation failed
```

**解决方案：**
```matlab
% 清空GPU
gpuDevice([]);

% 重新选择GPU
gpuDevice(1);

% 或直接用CPU
'ExecutionEnvironment', 'cpu',
```

---

## 🟣 步骤5：预测问题

### 问题1：预测结果全错
```
1      Facebook        Spotify         23.45%     ✗ 错误
2      Spotify         YouTube         19.67%     ✗ 错误
3      YouTube         Skype           21.34%     ✗ 错误
```

**可能原因：** 模型未正确训练或新数据格式不对

**检查：**
```matlab
% 1. 检查模型在测试集上的表现
load('models/trained_model.mat');
fprintf('模型测试准确率: %.2f%%\n', results.test_accuracy * 100);
% 应该 > 70%

% 2. 检查新数据FlowPic
figure;
for i = 1:3
    subplot(1,3,i);
    imagesc(X_new(:,:,1,i));
    colorbar;
end
% 应该看到明显的模式

% 3. 检查类别映射
results.class_names
% 确保与训练时一致
```

---

### 问题2：置信度太低
```
预测: Facebook (置信度: 25.34%)  % < 40% 不可信
```

**原因：** 模型不确定

**解决方案：**
- 如果训练集准确率高：正常，新数据可能与训练数据分布不同
- 如果训练集准确率低：需要重新训练更好的模型

---

## 🔍 通用调试技巧

### 1. 启用详细输出
```matlab
% 在训练时
'Verbose', true,
'VerboseFrequency', 1,  % 每个iteration都输出
```

### 2. 使用断点
```matlab
keyboard  % 程序暂停，可以检查变量
% 输入 dbcont 继续
% 输入 dbquit 退出调试
```

### 3. 可视化中间结果
```matlab
% 每10个epoch保存一次
if mod(epoch, 10) == 0
    save(sprintf('checkpoint_epoch%d.mat', epoch), 'net');
end
```

### 4. 单元测试
```matlab
% 测试generate_flowpic
test_lengths = [100, 200, 300];
test_times = [0, 1, 2];
test_dirs = [1, -1, 1];
fp = generate_flowpic(test_lengths, test_times, test_dirs, 32);
assert(sum(fp(:)) == 3, 'FlowPic计数错误');
```

### 5. 对比基准
```matlab
% 使用简单模型作为baseline
% 如果复杂模型表现不如简单模型，说明有问题
```

---

## 📞 获取帮助

如果以上都无法解决：

1. **检查MATLAB版本和工具箱**
   ```matlab
   ver  % 显示所有已安装的产品
   ```

2. **查看完整错误堆栈**
   ```matlab
   try
       % 你的代码
   catch ME
       fprintf('错误: %s\n', ME.message);
       for i = 1:length(ME.stack)
           fprintf('  文件: %s, 行: %d\n', ME.stack(i).file, ME.stack(i).line);
       end
   end
   ```

3. **创建最小可复现示例**
   - 只保留出错的几行代码
   - 使用简单的测试数据
   - 记录完整错误信息

4. **参考论文原始代码**
   - 虽然是Python版本，但算法逻辑相同
   - GitHub: https://github.com/danielpoliakov/flowmind

5. **MATLAB官方文档**
   - Deep Learning Toolbox: https://www.mathworks.com/help/deeplearning/
   - 示例和教程

---

## ✅ 验证清单

在报告问题前，确认已完成：

- [ ] 运行了step0_check_environment.m
- [ ] MATLAB版本 ≥ R2020a
- [ ] Deep Learning Toolbox已安装
- [ ] 所有.m文件在当前目录
- [ ] 按顺序运行step1-5
- [ ] 检查了每步的输出是否正常
- [ ] 查看了本文档的相关章节
- [ ] 尝试了建议的解决方案

如果全部完成仍有问题，请提供：
1. MATLAB版本
2. 错误信息（完整）
3. 哪一步出错
4. 已尝试的解决方案
