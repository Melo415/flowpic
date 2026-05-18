# 基于深度学习的加密流量分类

基于 FlowPic 框架与深度残差卷积网络的加密网络流量分类系统，参考 Poliakov et al. (2025) 论文实现，使用 MATLAB 深度学习工具箱完成全流程开发。

**在 MIRAGE-19 数据集（18类应用）上测试集准确率达到 73.94%**

---

## 项目简介

HTTPS、VPN、Tor 等协议的广泛普及使传统深度包检测（DPI）方法几乎完全失效——载荷已被加密，无法直接解析。本项目在**不解密流量内容**的前提下，仅利用数据包的长度、到达时间间隔（IAT）和传输方向等元数据，实现对加密流量的多类别识别。

核心思路是将每条流量会话转换为**四通道二维直方图图像**，再送入端到端训练的 ResNet 分类器进行识别。

---

## 方法

### 流量表示

取每条流量会话的**前 15 秒**数据，按传输方向（上行 / 下行）分组，分别针对两种特征构建二维直方图：

- **包长度 × 时间** → 二维直方图
- **包间时间间隔（IAT）× 时间** → 二维直方图

上行 + 下行 × 两种特征，共生成 **H × W × 4** 的四通道张量作为模型输入。

对直方图数值先做 **log1p 变换**，再进行逐通道 **min-max 归一化**到 [0, 1]，以消除数值量纲差异并压缩长尾分布。

### 模型架构

```
输入: 32×32×4

Stem:  Conv(7×7, 64) → BN → ReLU → MaxPool(stride=2)         →  16×16×64

Res1:  Conv(3×3, 64)  → BN → ReLU → Dropout(0.35)
       Conv(3×3, 64)  → BN → (+shortcut) → ReLU               →  16×16×64

Res2:  Conv(3×3, 128, stride=2) → BN → ReLU → Dropout(0.35)
       Conv(3×3, 128)            → BN → (+shortcut_conv) → ReLU →  8×8×128

AvgPool(4×4, stride=4)  →  2×2×128  →  展平 512 维

FC(512→256) → ReLU → Dropout(0.5)
FC(256→18)  → Softmax
```

### 训练配置

| 参数 | 值 |
|------|-----|
| 优化器 | Adam |
| 初始学习率 | 0.001 |
| 学习率衰减 | 第 40 轮以因子 0.3 衰减 |
| 最大训练轮数 | 80 |
| Batch Size | 128 |
| L2 正则化系数 | 0.0005 |
| 梯度裁剪阈值 | 1.0 |
| Early Stopping patience | 20（基于验证集损失） |
| 数据集划分 | 训练 80% / 验证 10% / 测试 10%（分层抽样） |

**数据增强说明：** 未使用翻转、平移等常规图像增强操作——此类操作会破坏直方图中包长和时间的物理语义。仅对训练集施加弱高斯噪声（概率 0.5，标准差 0.05）以提升鲁棒性。

---

## 数据集

使用 **MIRAGE-19** 公开数据集。

| 属性 | 详情 |
|------|------|
| 总流数 | 122,007 |
| 应用类别数 | 18 |
| 数据格式 | JSON（每文件对应一个应用） |
| 最小包数阈值 | 10 个包/流 |

数据集官方地址：[MIRAGE-2019](http://traffic.comics.unina.it/mirage/MIRAGE-2019.html)

下载后将数据放置于 `data/mirage/` 目录，脚本会自动扫描所有 JSON 文件。

---

## 实验结果

### 最优实验配置

| 指标 | 数值 |
|------|------|
| 测试集准确率 | **73.94%** |
| 验证集 Loss 最低值 | 0.9205（第 17958 次迭代） |
| 对应验证准确率 | 73.43% |
| 验证集准确率峰值 | 73.66%（第 19710 次迭代） |
| 训练时长 | 约 48 分钟（单 GPU） |

> 当前保存的模型来自验证集 Loss 最低的轮次。验证损失最低点与准确率最高点不重合，是因为两者衡量的目标不同；以损失最低点保存模型，泛化性通常更稳定。

### 混淆矩阵

![混淆矩阵](confusion_matrix.png)

大多数类别分类效果良好，少数样本量较小的类别存在一定混淆，与类别分布不均衡有关。

### 训练曲线

![训练进度](training_curves.png)

学习率在第 40 轮下降后，训练集与验证集的准确率差距有所扩大，存在轻度过拟合现象。

---

## 项目结构

```
flowpic/
├── data/
│   ├── mirage/                  # MIRAGE-19 原始 JSON 数据（需自行下载）
│   └── flowpic_cache.mat        # 预处理缓存（首次运行自动生成）
├── COMPLETE_TRAINING.m          # 主训练脚本（一键运行）
├── create_flowpic_model.m       # ResNet 模型定义
├── generate_flowpic.m           # 四通道 FlowPic 直方图生成
├── load_mirage_json.m           # MIRAGE-19 数据加载与预处理
├── run_training_with_logging.m  # 训练封装 + 实验记录
└── files/
    └── training_results/        # 每次实验自动生成结果文件夹
        ├── experiment_summary.md
        ├── hyperparameters.txt
        ├── confusion_matrix.png
        ├── training_curves.png
        └── final_results.mat
```

---

## 快速开始

### 环境要求

- MATLAB R2021a 及以上
- Deep Learning Toolbox
- （可选）Parallel Computing Toolbox（用于 GPU 加速）

### 运行步骤

**1. 准备数据**

下载 MIRAGE-19 数据集，解压至 `data/mirage/`。

**2. 修改路径**

打开 `COMPLETE_TRAINING.m`，将以下两行改为本地实际路径：

```matlab
DATASET_PATH = 'your/path/to/data/mirage';
CACHE_FILE   = 'your/path/to/data/flowpic_cache.mat';
```

**3. 运行训练**

直接运行 `COMPLETE_TRAINING.m`，脚本将按顺序完成：

- 数据加载与 FlowPic 生成（首次运行约需数分钟，结果自动缓存）
- 数据集划分与增强
- 模型构建与训练
- 测试集评估与结果保存

训练结束后，混淆矩阵、训练曲线、超参数记录等结果自动保存至 `files/training_results/` 下对应时间戳的文件夹中。

> ⚠️ **注意：** 若修改了 `FLOWPIC_SIZE` 或归一化方式，需手动删除 `flowpic_cache.mat` 重新生成缓存，否则旧缓存会导致结果异常。

---

## 主要结论

- **数据增强需与语义相符：** 常规图像翻转、平移等操作会破坏流量直方图中包长和时间的物理含义，屏蔽此类操作后准确率提升约 10%。
- **验证损失最低 ≠ 验证准确率最高：** 两者出现在不同训练步，以损失最低点保存模型泛化性更稳定。
- **类别不均衡影响少数类：** 混淆矩阵显示性能较弱的类别集中于样本量较少的应用，后续可通过更精细的采样策略加以改善。
- **轻度过拟合仍有改善空间：** 学习率衰减后训练集与验证集差距扩大，可进一步探索更强正则化或 Mixup 等增强策略。

---

## 参考文献

- Poliakov et al. (2025). *FlowPic: Encrypted Internet Traffic Classification is as Easy as Image Recognition.*
- MIRAGE-2019 Dataset. University of Naples Federico II. http://traffic.comics.unina.it/mirage/
