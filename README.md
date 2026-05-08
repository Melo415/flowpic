# FlowPic 加密流量分类器

基于二维直方图表示（FlowPic）与自定义 ResNet 架构的加密网络流量分类系统，使用 MATLAB 实现。

> **测试准确率：70.99%**　·　MIRAGE-19 数据集　·　18 个流量类别　·　102,269 条网络流

---

## 项目简介

随着 TLS/QUIC 协议的普及，传统深度包检测（DPI）方法已无法有效分析加密流量。本项目采用**载荷无关**的方法：将每条网络流转换为包长度与包间隔时间（IAT）的二维联合直方图，生成 64×64×4 的"流量指纹"图像（FlowPic），再通过残差卷积神经网络完成分类。

```
原始网络流  →  FlowPic（64×64×4）  →  ResNet  →  流量类别
（长度+IAT）      二维直方图            3个残差块    （18类）
```

### 三大核心创新

| 创新点 | 说明 |
|--------|------|
| **完整概率分布表示** | 用直方图捕获包长度的完整分布形态，而非仅依赖均值、方差等统计量 |
| **二维联合直方图** | 同时编码包长度与 IAT 的相关性，保留了一维特征向量所丢失的时空联合信息 |
| **带跳跃连接的深度 ResNet** | 通过 `output = F(x) + x` 的残差结构解决梯度消失，实现对直方图细粒度空间模式的学习 |

---

## 网络架构

```
输入层（64×64×4）
     │
  Conv 7×7, 64通道 → BN → ReLU → MaxPool（stride=2）
     │
  残差块 1 — 64 filters
     │
  残差块 2 — 128 filters（stride=2 下采样）
     │
  残差块 3 — 256 filters（stride=2 下采样）
     │
  Global Average Pooling
     │
  FC(256) → BN → ReLU → Dropout(0.5)
     │
  FC(18) → Softmax
```

每个残差块结构：`Conv→BN→ReLU→Dropout(0.15)→Conv→BN → + shortcut → ReLU`，其中下采样块的 shortcut 分支使用 1×1 卷积对齐通道数。

---

## 环境要求

- **MATLAB** R2021b 及以上
- **Deep Learning Toolbox**
- （可选）支持 CUDA 的 GPU — CPU 也可运行，但训练时间将显著增加

---

## 项目结构

```
flowpic/
├── COMPLETE_TRAINING.m            # 主入口，直接运行此文件
├── create_flowpic_model.m         # ResNet 模型定义
├── generate_flowpic.m             # 网络流 → 64×64×4 FlowPic 转换
├── load_mirage_json.m             # MIRAGE-19 JSON 数据加载
├── run_training_with_logging.m    # 训练循环 + 实验日志记录
├── figures/                       # 自动生成的训练曲线图
├── models/                        # 模型检查点保存目录
└── training_results/              # 各次实验的日志与汇总报告
```

---

## 快速开始

**第一步：下载数据集**

获取 [MIRAGE-19 数据集](http://traffic.comics.unina.it/mirage/) 并解压到：
```
data/mirage/
```

**第二步：配置数据路径**

打开 `COMPLETE_TRAINING.m`，修改第一部分中的路径：
```matlab
DATASET_PATH = 'your/path/to/data/mirage';
CACHE_FILE   = 'your/path/to/data/flowpic_cache.mat';
```

**第三步：运行训练**

```matlab
% 在 MATLAB 命令窗口中：
cd your/path/to/flowpic
COMPLETE_TRAINING
```

首次运行会自动生成 FlowPic 并保存缓存文件（约 22 秒），后续运行直接加载缓存跳过此步骤。**修改 `FLOWPIC_SIZE` 或归一化方式后，需手动删除缓存文件重新生成。**

---

## 参数配置

所有超参数集中在 `COMPLETE_TRAINING.m` 第一部分配置：

```matlab
%% FlowPic 配置
FLOWPIC_SIZE   = 64;       % 分辨率：32 | 48 | 64（64需要至少8GB可用内存）
NUM_CHANNELS   = 4;        % 通道：上行包长、下行包长、上行IAT、下行IAT

%% 训练超参数
MAX_EPOCHS     = 150;
BATCH_SIZE     = 64;       % GPU显存不足时改为32
LEARNING_RATE  = 0.0005;
L2_REG         = 0.0001;

%% 数据增强（仅训练集，保留时序语义）
USE_AUGMENTATION = true;
AUG_NOISE_PROB   = 0.35;  % 弱高斯噪声触发概率
AUG_NOISE_STD    = 0.02;  % 噪声强度
```

> **内存说明：** 64×64 全量数据集约需 6.7 GB（single 精度）。内存不足时可改用 `FLOWPIC_SIZE = 48`（约 3.8 GB）或 `32`（约 1.7 GB）。

---

## 数据集

**MIRAGE-19**：102,269 条标注网络流，覆盖 18 个移动应用类别：

| ID | 应用 | 样本数 | ID | 应用 | 样本数 |
|----|------|--------|----|------|--------|
| 1 | slither | 2,040 | 10 | comics | 4,227 |
| 2 | groupon | 1,408 | 11 | pinterest | 2,882 |
| 3 | tripadvisor | 2,441 | 12 | trello | 1,516 |
| 4 | android | 14,014 | 13 | foursquared | 4,800 |
| 5 | duolingo | 5,939 | 14 | katana | 3,364 |
| 6 | subito | 5,807 | 15 | orca | 1,836 |
| 7 | voip | 2,062 | 16 | wish | 4,308 |
| 8 | music | 4,458 | 17 | youtube | 3,165 |
| 9 | iliga | 8,352 | 18 | waze | 9,197 |

数据集存在**类别不均衡**（android 14,014 vs. groupon 1,408，约 10:1）。训练时采用逆频率类别权重进行补偿。

**数据划分（分层抽样，保证各类比例一致）：**

| 子集 | 比例 | 样本数（约） |
|------|------|--------|
| 训练集 | 80% | 81,815 |
| 验证集 | 10% | 10,227 |
| 测试集 | 10% | 10,227 |

---

## 实验结果

在独立测试集上的评估结果（与上一版 32×32 对比）：

| 配置 | 测试准确率 | 训练时长 |
|------|-----------|--------|
| ResNet，32×32×4 | 65.94% | 18.4 分钟 |
| ResNet，64×64×4（当前） | **70.99%** | 373.4 分钟 |

**各类别表现（基于混淆矩阵）：**

| 表现 | 类别 | 召回率估算 |
|------|------|-----------|
| 优秀 | waze、music、duolingo、comics | > 85% |
| 良好 | iliga、subito、foursquared、wish | 70–85% |
| 较弱 | tripadvisor、groupon、orca | < 65% |

主要混淆对：class 14（katana）↔ class 15（orca）存在明显相互误判，两者在当前分辨率下 FlowPic 特征高度相似。

---

## FlowPic 原理

一条网络流包含一系列数据包，每个包有：
- `lengths`：包的字节大小
- `times`：相邻包之间的时间间隔（IAT）
- `directions`：上行（设备→服务器）或下行（服务器→设备）

`generate_flowpic.m` 将上行/下行数据分别统计为包长×IAT 的二维直方图，生成 4 通道图像，每个通道对应一种流量维度的分布。

**归一化流程：**
```
原始计数值  →  log1p(x)  →  逐通道 min-max 归一化到 [0, 1]
```

log1p 变换用于压缩直方图中极度不均匀的计数分布（大量零值 + 少量大值），使网络训练更稳定。

---

## 实验记录

每次训练结束后，`run_training_with_logging.m` 会自动在 `training_results/` 下生成带时间戳的实验文件夹：

```
training_results/
└── exp_YYYYMMDD_HHMMSS/
    ├── experiment_summary.md    # 实验总结（准确率、时长、超参数）
    ├── confusion_matrix.png     # 测试集混淆矩阵
    ├── hyperparameters.txt      # 超参数记录
    ├── dataset_distribution.csv # 数据集分布与类别权重
    ├── experiment_info.mat      # 完整实验元信息
    └── final_results.mat        # 模型与评估结果
```

---

## 参考文献

- Shapira, T. & Shavitt, Y. (2021). *FlowPic: A Generic Representation for Encrypted Traffic Classification and Applications Identification.* IEEE Transactions on Network and Service Management.
- Luxemburk, J. & Čejka, T. (2023). *Fine-grained TLS services classification with reject option.* Computer Networks.
- MIRAGE-19 数据集：[traffic.comics.unina.it/mirage](http://traffic.comics.unina.it/mirage/)

---

## License

MIT License — 详见 [LICENSE](LICENSE) 文件。
