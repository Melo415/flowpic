# 🎓 FlowPic真实数据训练 - 完整教程

## 📖 目录

1. [快速开始](#快速开始)
2. [系统架构](#系统架构)
3. [详细步骤](#详细步骤)
4. [参数调优](#参数调优)
5. [常见问题](#常见问题)
6. [进阶使用](#进阶使用)

---

## 🚀 快速开始

### 前置条件

✅ MATLAB R2020a或更高版本  
✅ Deep Learning Toolbox  
✅ MIRAGE-19数据集（JSON格式）  
✅ 至少8GB内存（推荐16GB）  
✅ GPU（可选，但强烈推荐）

### 3步快速运行

```matlab
% 步骤1: 修改数据路径
% 打开 COMPLETE_TRAINING.m，第23行
DATASET_PATH = 'D:\your\path\to\mirage';

% 步骤2: 运行主脚本
COMPLETE_TRAINING

% 步骤3: 等待训练完成（20分钟-数小时）
% 结果将自动保存到 models/ 和 figures/
```

---

## 🏗️ 系统架构

### 整体流程图

```
┌─────────────────┐
│  MIRAGE-19 JSON │  ← 真实网络流量数据
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  数据加载器     │  load_mirage_json.m
│  ├─ 解析JSON    │  - 提取流量五元组
│  ├─ 提取包信息  │  - 包长度、时间、方向
│  └─ 提取标签    │  - 应用分类标签
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  FlowPic生成    │  generate_flowpic.m
│  ├─ 创建直方图  │  - 32×32×4 多通道
│  ├─ 通道1-2     │  - 双向包长度
│  └─ 通道3-4     │  - 双向IAT
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  数据划分       │  80% / 10% / 10%
│  ├─ 训练集      │  - 用于训练模型
│  ├─ 验证集      │  - 调整超参数
│  └─ 测试集      │  - 最终评估
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  ResNet模型     │  create_flowpic_model.m
│  ├─ 输入层      │  - 32×32×4 FlowPic
│  ├─ ResBlock 1  │  - 64 filters
│  ├─ ResBlock 2  │  - 128 filters
│  └─ 分类器      │  - 全连接层 → Softmax
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  训练循环       │  trainNetwork
│  ├─ Adam优化器  │  - lr=0.001
│  ├─ 50 epochs   │  - dropout=0.2
│  └─ 验证监控    │  - early stopping
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  模型评估       │  classify & confusionmat
│  ├─ 准确率      │  - 整体 & 各类别
│  ├─ 混淆矩阵    │  - 错误分析
│  └─ 性能可视化  │  - 图表生成
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  保存结果       │  .mat & .png
│  ├─ 模型文件    │  - net + results
│  └─ 可视化图    │  - 混淆矩阵/性能图
└─────────────────┘
```

---

## 📚 详细步骤

### 第一部分：环境配置

**代码位置：** `COMPLETE_TRAINING.m` 第18-40行

**核心配置：**

```matlab
% ⭐ 必须修改
DATASET_PATH = 'your/path/to/mirage';  % MIRAGE数据路径

% ⭐ 根据需要修改
USE_SUBSET = true;     % 测试时true，正式训练false
SUBSET_SIZE = 2000;    % 子集大小

% ⭐ 高级选项（通常不需要改）
FLOWPIC_SIZE = 32;     % FlowPic大小
MAX_EPOCHS = 50;       % 训练轮数
BATCH_SIZE = 64;       % 批大小
```

**注意事项：**

- **首次运行**：建议 `USE_SUBSET = true` + `SUBSET_SIZE = 1000`，快速验证流程（10-20分钟）
- **正式训练**：设置 `USE_SUBSET = false`，使用全部数据（2-6小时）
- **内存不足**：减小 `BATCH_SIZE` 到 32 或 16

---

### 第二部分：数据加载

**代码位置：** 第42-82行

**功能：** 从MIRAGE-19 JSON文件加载网络流数据

**数据格式：**

```json
{
  "x192_168_20_105_35639_216_58_214_74_443_6": {
    "packet_data": {
      "packet_sizes": [120, -1460, 52, ...],
      "packet_times": [0.0, 0.015, 0.023, ...],
      "packet_directions": [1, -1, 1, ...]
    },
    ...
  },
  ...
}
```

**输出结果：**

```matlab
flows{i}.lengths     % 包长度向量
flows{i}.times       % 时间戳向量
flows{i}.directions  % 方向向量 (1=上行, -1=下行)
labels(i)            % 类别ID
class_names{j}       % 类别名称
```

**预期输出：**

```
✓ 数据加载成功！用时: 15.3 秒

数据集统计:
  总样本数: 8640
  类别数: 20
  类别: facebook, spotify, youtube, ...
  
  平均包数/流: 156.7
```

**故障排除：**

| 错误 | 原因 | 解决方案 |
|------|------|---------|
| 找不到JSON文件 | 路径错误 | 检查 `DATASET_PATH` |
| 无法识别字段 | JSON格式不同 | 运行 `test_mirage_json.m` 查看实际字段 |
| 加载太慢 | 文件太多 | 正常现象，耐心等待 |

---

### 第三部分：数据子集采样

**代码位置：** 第84-115行

**功能：** 从大数据集中采样子集用于快速测试

**采样策略：** 分层采样（保持类别比例）

```matlab
每个类别采样数 = floor(SUBSET_SIZE / num_classes)
```

**示例：**

```
原始: 8640 样本, 20 类
子集: 2000 样本
  → 每类约 100 样本 (2000/20)
```

**何时使用子集：**

- ✅ **首次测试流程**：1000-2000样本，10-30分钟
- ✅ **调试代码**：500-1000样本，5-15分钟
- ❌ **最终训练**：全部数据，达到最佳性能

---

### 第四部分：生成FlowPic

**代码位置：** 第117-148行

**核心函数：** `generate_flowpic.m`

**FlowPic定义：**

FlowPic是一个 32×32×4 的多通道直方图：

```
Channel 1: 上行包长度 vs 时间
  ├─ X轴: 0-15秒 (32 bins)
  └─ Y轴: 0-1460字节 (32 bins)

Channel 2: 下行包长度 vs 时间
  ├─ X轴: 0-15秒 (32 bins)
  └─ Y轴: 0-1460字节 (32 bins)

Channel 3: 上行IAT vs 时间
  ├─ X轴: 0-15秒 (32 bins)
  └─ Y轴: 0-95th percentile (32 bins)

Channel 4: 下行IAT vs 时间
  ├─ X轴: 0-15秒 (32 bins)
  └─ Y轴: 0-95th percentile (32 bins)
```

**示例：**

```matlab
flow.lengths = [120, 1460, 52, 800, ...];
flow.times = [0, 0.01, 0.015, 0.02, ...];
flow.directions = [1, -1, 1, -1, ...];

flowpic = generate_flowpic(lengths, times, directions, 32);
% → 输出: 32×32×4 直方图
```

**性能：**

- 典型速度：0.002-0.005秒/样本
- 1000样本：约3-5秒
- 10000样本：约30-50秒

---

### 第五部分：数据集划分

**代码位置：** 第150-180行

**划分比例：** 80% 训练 + 10% 验证 + 10% 测试

**实现：**

```matlab
% 第一次：80/20 分割
cv1 = cvpartition(N, 'HoldOut', 0.2);
train_val = training(cv1);
test = test(cv1);

% 第二次：训练/验证分割
cv2 = cvpartition(sum(train_val), 'HoldOut', 0.125);
train = training(cv2);  % 80% of total
val = test(cv2);         % 10% of total
```

**为什么这样划分？**

- **训练集 (80%)**：学习模式
- **验证集 (10%)**：调整超参数，防止过拟合
- **测试集 (10%)**：最终评估，模拟真实场景

**注意：** 确保每个类别在所有集合中都有代表

---

### 第六部分：构建模型

**代码位置：** 第182-204行

**核心函数：** `create_flowpic_model.m`

**网络架构：**

```
Input: 32×32×4
  ↓
Conv 3×3, 64 filters + BN + Dropout
  ↓
[Residual Block 1] ────────┐
│ Conv 3×3, 64            │
│ BN → ReLU               │
│ Conv 3×3, 64            │
│ BN → Dropout            │
└─────────── + ←──────────┘
  ↓ ReLU
[Residual Block 2] ────────┐
│ Conv 3×3, 128 (stride=2)│
│ BN → ReLU               │
│ Conv 3×3, 128           │
│ BN → Dropout            │
└─────────── + ←──────────┘
  ↓ ReLU  (shortcut: Conv 1×1, stride=2)
AvgPool 2×2
  ↓ (128×2×2 = 512)
FC 512→256 → ReLU → Dropout
  ↓
FC 256→num_classes
  ↓
Softmax → Output
```

**参数量：** 约58,000参数

---

### 第七部分：配置训练

**代码位置：** 第206-250行

**关键参数：**

| 参数 | 值 | 说明 |
|------|-----|------|
| Optimizer | Adam | 自适应学习率 |
| Initial LR | 0.001 | 初始学习率 |
| LR Schedule | Piecewise | 每20轮×0.1 |
| Batch Size | 64 | 每批样本数 |
| Epochs | 50 | 训练轮数 |
| L2 Reg | 0.0001 | 权重衰减 |
| Dropout | 0.2 | 防过拟合 |

**学习率变化：**

```
Epoch 1-20:  lr = 0.001
Epoch 21-40: lr = 0.0001  (×0.1)
Epoch 41-50: lr = 0.00001 (×0.1)
```

---

### 第八部分：训练模型

**代码位置：** 第252-281行

**训练监控：**

MATLAB会弹出一个训练进度窗口，显示：

1. **训练损失** (Training Loss)
2. **验证损失** (Validation Loss)
3. **验证准确率** (Validation Accuracy)

**正常训练曲线：**

```
Loss ↓
  │ \
  │  \___
  │      \___
  │          ----
  └────────────────> Epoch

Accuracy ↑
  │          ----
  │      ___/
  │  ___/
  │ /
  └────────────────> Epoch
```

**异常情况：**

| 现象 | 原因 | 解决方案 |
|------|------|---------|
| Loss不下降 | LR太小 | 增大学习率到0.01 |
| Loss震荡 | LR太大 | 减小学习率到0.0001 |
| 过拟合 | 训练数据少 | 增加Dropout或数据 |
| 内存溢出 | Batch太大 | 减小到32或16 |

**训练时间估算：**

| 硬件 | 1000样本 | 5000样本 | 全量(~10K) |
|------|----------|----------|-----------|
| GPU (RTX 3080) | 10分钟 | 40分钟 | 1.5小时 |
| GPU (GTX 1080) | 15分钟 | 1小时 | 2.5小时 |
| CPU (i7) | 1小时 | 5小时 | 10小时 |

---

### 第九部分：模型评估

**代码位置：** 第283-324行

**评估指标：**

1. **整体准确率**
   ```
   Accuracy = 正确分类数 / 总样本数
   ```

2. **各类准确率**
   ```
   Per-class Accuracy = 该类正确数 / 该类总数
   ```

3. **混淆矩阵**
   ```
   C[i,j] = 真实类别i被预测为j的次数
   ```

**结果解读：**

```
测试集准确率: 81.23%
  - 优秀: >80%  (达到论文水平)
  - 良好: 70-80% (可接受)
  - 一般: 60-70% (需要改进)
  - 较差: <60%  (检查数据/代码)
```

---

### 第十部分：保存结果

**代码位置：** 第326-400行

**保存内容：**

1. **模型文件** (`models/flowpic_mirage_model.mat`)
   ```matlab
   - net: 训练好的神经网络
   - results: 评估结果结构体
     ├─ train_accuracy
     ├─ val_accuracy
     ├─ test_accuracy
     ├─ class_accuracies
     ├─ confusion_matrix
     ├─ hyperparameters
     └─ dataset_info
   ```

2. **可视化图像**
   - `figures/flowpic_mirage_model_confusion.png` - 混淆矩阵
   - `figures/flowpic_mirage_model_performance.png` - 性能对比图

---

## 🎛️ 参数调优

### 快速调优指南

#### 问题1：准确率低（<60%）

**可能原因：**
- 数据太少
- 学习率不合适
- 训练不充分

**解决方案：**
```matlab
% 方案A：增加数据
USE_SUBSET = false;  % 使用全部数据

% 方案B：调整学习率
LEARNING_RATE = 0.01;  % 提高10倍

% 方案C：延长训练
MAX_EPOCHS = 100;  % 增加到100轮
```

#### 问题2：过拟合（训练准确率>>测试准确率）

**解决方案：**
```matlab
% 增加正则化
options = trainingOptions('adam', ...
    'L2Regularization', 0.001);  % 从0.0001增加到0.001

% 或修改模型中的dropout
% 在 create_flowpic_model.m 中
dropoutLayer(0.5)  % 从0.2增加到0.5
```

#### 问题3：训练太慢

**解决方案：**
```matlab
% 使用GPU
execution_env = 'gpu';

% 或增大batch size
BATCH_SIZE = 128;  % 如果显存足够

% 或减少epochs
MAX_EPOCHS = 30;  % 先用30轮快速测试
```

---

## ❓ 常见问题

### Q1: 如何加载保存的模型？

```matlab
% 加载模型
load('models/flowpic_mirage_model.mat', 'net', 'results');

% 查看结果
results.test_accuracy  % 测试准确率
results.class_names    % 类别名称

% 预测新数据
predictions = classify(net, new_flowpics);
```

### Q2: 如何在新数据上测试？

```matlab
% 1. 加载模型
load('models/flowpic_mirage_model.mat', 'net');

% 2. 准备新数据（同样的格式）
new_flowpic = generate_flowpic(lengths, times, directions, 32);

% 3. 预测
prediction = classify(net, new_flowpic);
[scores, labels] = predict(net, new_flowpic);

% 4. 查看结果
fprintf('预测类别: %s\n', char(prediction));
fprintf('置信度: %.2f%%\n', max(scores)*100);
```

### Q3: 不同数据量的预期性能？

| 训练样本数 | 预期准确率 | 训练时间(GPU) |
|-----------|-----------|--------------|
| 500-1000 | 60-70% | 10-20分钟 |
| 2000-5000 | 70-78% | 30-60分钟 |
| 全量(~10K) | 78-82% | 1.5-3小时 |

### Q4: 如何改进性能？

**优先级排序：**

1. **增加训练数据** ⭐⭐⭐⭐⭐
   - 效果最明显
   - 使用全部MIRAGE数据

2. **调整学习率** ⭐⭐⭐⭐
   - 尝试 0.01, 0.001, 0.0001
   - 观察训练曲线

3. **数据增强** ⭐⭐⭐
   - 时间平移
   - 包长度扰动

4. **调整网络结构** ⭐⭐
   - 增加filters
   - 添加residual block

5. **集成学习** ⭐
   - 训练多个模型
   - 投票或平均

---

## 🔬 进阶使用

### 1. 批量实验

创建 `batch_experiments.m`：

```matlab
% 测试不同超参数组合
learning_rates = [0.01, 0.001, 0.0001];
batch_sizes = [32, 64, 128];

for lr = learning_rates
    for bs = batch_sizes
        fprintf('Testing LR=%.4f, BS=%d\n', lr, bs);
        
        % 修改参数
        LEARNING_RATE = lr;
        BATCH_SIZE = bs;
        MODEL_NAME = sprintf('model_lr%.4f_bs%d', lr, bs);
        
        % 运行训练
        COMPLETE_TRAINING;
    end
end
```

### 2. 交叉验证

```matlab
% K-fold交叉验证
K = 5;
cv = cvpartition(num_samples, 'KFold', K);

accuracies = zeros(K, 1);

for k = 1:K
    train_idx = training(cv, k);
    test_idx = test(cv, k);
    
    % 训练和评估
    ...
    
    accuracies(k) = test_accuracy;
end

fprintf('平均准确率: %.2f%% ± %.2f%%\n', ...
        mean(accuracies)*100, std(accuracies)*100);
```

### 3. 特征可视化

```matlab
% 可视化学到的filter
layer = net.Layers(2);  % 第一个卷积层
weights = layer.Weights;

figure;
montage(weights, 'Size', [8 8]);
title('Learned Filters');
colormap('gray');
```

---

## 📊 完整文件清单

| 文件 | 说明 | 必需 |
|------|------|------|
| `COMPLETE_TRAINING.m` | 主训练脚本 | ✅ |
| `load_mirage_json.m` | JSON数据加载器 | ✅ |
| `generate_flowpic.m` | FlowPic生成 | ✅ |
| `create_flowpic_model.m` | ResNet模型 | ✅ |
| `test_mirage_json.m` | 数据测试工具 | ⭐推荐 |
| 本文档 | 使用说明 | 📖 |

---

## 🎯 总结

### 典型使用流程

```
1. 下载MIRAGE-19数据 → data/mirage/
2. 修改 COMPLETE_TRAINING.m 第23行路径
3. 首次运行：USE_SUBSET=true, SUBSET_SIZE=1000
4. 验证流程正常（10-20分钟）
5. 正式训练：USE_SUBSET=false
6. 等待训练完成（2-6小时）
7. 查看结果：models/ 和 figures/
8. 测试准确率 >75% → 成功！
```

### 关键要点

- ✅ **数据是关键**：更多数据 = 更好性能
- ✅ **先小后大**：先用子集测试，再全量训练
- ✅ **监控训练**：观察loss和accuracy曲线
- ✅ **GPU加速**：能用GPU就用GPU
- ✅ **保存结果**：每次实验都保存模型和结果

---

**祝训练顺利！达到81%准确率！🎉**
