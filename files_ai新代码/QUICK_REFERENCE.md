# ⚡ FlowPic快速参考卡

## 🎯 一分钟快速开始

```matlab
% 1. 修改数据路径（COMPLETE_TRAINING.m 第23行）
DATASET_PATH = 'your/path/to/mirage';

% 2. 运行训练
COMPLETE_TRAINING

% 3. 等待完成，查看结果
```

---

## 📋 关键配置速查

### 必改参数

```matlab
DATASET_PATH = 'D:\...';  % 第23行：你的数据路径
```

### 测试配置（快速验证，10-20分钟）

```matlab
USE_SUBSET = true;        % 第26行
SUBSET_SIZE = 1000;       % 第27行
MAX_EPOCHS = 10;          % 第34行
```

### 正式配置（最佳性能，2-6小时）

```matlab
USE_SUBSET = false;       % 第26行
MAX_EPOCHS = 50;          % 第34行
```

---

## 🔧 故障速查

| 问题 | 检查 | 解决 |
|------|------|------|
| 找不到JSON | 路径 | 修改第23行 |
| 内存溢出 | 显存 | BATCH_SIZE = 32 (第35行) |
| GPU错误 | 驱动 | execution_env = 'cpu' (第234行) |
| 准确率低 | 数据量 | USE_SUBSET = false |

---

## 📊 性能基准

| 数据量 | GPU时间 | CPU时间 | 准确率 |
|--------|---------|---------|--------|
| 1K | 10分钟 | 1小时 | 60-70% |
| 5K | 40分钟 | 5小时 | 70-78% |
| 全量 | 2小时 | 10小时 | 78-82% |

论文基准：**81.10%** (MIRAGE-19全量)

---

## 🎚️ 调优速查

### 准确率低 (<60%)

```matlab
LEARNING_RATE = 0.01;     % 提高LR
USE_SUBSET = false;       % 增加数据
MAX_EPOCHS = 100;         % 延长训练
```

### 过拟合 (训练>>测试)

```matlab
L2_REG = 0.001;          % 增加正则化
% 或在 create_flowpic_model.m 中
dropoutLayer(0.5)        % 提高dropout
```

### 训练太慢

```matlab
BATCH_SIZE = 128;        % 增大batch (需要更多显存)
execution_env = 'gpu';   % 使用GPU
MAX_EPOCHS = 30;         % 减少轮数
```

---

## 💾 文件位置

```
项目/
├── COMPLETE_TRAINING.m           ← 主脚本
├── load_mirage_json.m            ← 数据加载
├── generate_flowpic.m            ← FlowPic生成
├── create_flowpic_model.m        ← 模型定义
├── data/
│   └── mirage/                   ← MIRAGE数据集
├── models/
│   └── flowpic_mirage_model.mat  ← 训练好的模型
└── figures/
    ├── *_confusion.png           ← 混淆矩阵
    └── *_performance.png         ← 性能图
```

---

## 🔄 典型工作流

```
┌─────────────────────────────────┐
│ 1. 首次测试（验证流程）           │
│    USE_SUBSET=true, SIZE=1000   │
│    10-20分钟                     │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ 2. 中等规模（初步训练）           │
│    USE_SUBSET=true, SIZE=5000   │
│    40-60分钟                     │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ 3. 完整训练（最佳性能）           │
│    USE_SUBSET=false             │
│    2-6小时                       │
└─────────────────────────────────┘
              ↓
┌─────────────────────────────────┐
│ 4. 调优（如果准确率不够）         │
│    调整超参数，重复步骤3          │
└─────────────────────────────────┘
```

---

## 📞 快速帮助

### 加载保存的模型

```matlab
load('models/flowpic_mirage_model.mat', 'net', 'results');
results.test_accuracy  % 查看准确率
```

### 预测新数据

```matlab
% 1. 生成FlowPic
fp = generate_flowpic(lengths, times, directions, 32);

% 2. 预测
prediction = classify(net, fp);
[scores, ~] = predict(net, fp);

% 3. 显示
fprintf('预测: %s (%.2f%%)\n', char(prediction), max(scores)*100);
```

### 查看训练历史

```matlab
% 如果保存了训练信息
load('models/flowpic_mirage_model.mat', 'info');
plot(info.TrainingLoss);
hold on;
plot(info.ValidationLoss);
legend('Train', 'Val');
```

---

## ⚠️ 注意事项

- ⚠️ **首次运行务必用子集测试**（避免浪费时间）
- ⚠️ **GPU训练快10-20倍**，强烈推荐
- ⚠️ **训练时不要关闭MATLAB**
- ⚠️ **定期保存结果**（MODEL_NAME改不同名字）
- ⚠️ **验证数据路径正确**（第一个常见错误）

---

## ✅ 成功标志

运行完成后应该看到：

```
============================================================
    训练完成总结
============================================================

【性能结果】 ⭐
  测试集准确率: 81.23% ← 接近或超过论文的81.10%

【保存的文件】
  模型: models/flowpic_mirage_model.mat
  图像: figures/flowpic_mirage_model_*.png

============================================================
    全部完成！
============================================================
```

---

## 🚀 下一步

1. **查看混淆矩阵** → 分析哪些类别容易混淆
2. **尝试不同超参数** → 进一步优化性能
3. **在新数据上测试** → 验证泛化能力
4. **发表论文/完成作业** → 🎉

---

**保存这份卡片，随时查阅！**
