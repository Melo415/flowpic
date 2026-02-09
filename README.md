# FlowPic 多通道加密流量分类 - MATLAB复现

基于论文：**Multichannel Histograms for Flow Classification** (NOMS 2025)  
作者：Daniel Poliakov et al.

## 📋 文件说明

1. **generate_flowpic.m** - 生成四通道flowpic直方图
2. **create_flowpic_model.m** - 创建ResNet风格的CNN模型
3. **train_flowpic.m** - 完整训练脚本
4. **load_pcap_data.m** - 数据加载辅助函数

## 🚀 快速开始

### 前置要求
- MATLAB R2020a 或更高版本
- Deep Learning Toolbox
- （可选）Parallel Computing Toolbox（GPU加速）

### 使用步骤

#### 步骤1：准备数据

你需要网络流量数据，包含：
- 数据包长度（bytes）
- 时间戳（秒）
- 方向（上行/下行）
- 标签（应用类别）

**数据格式示例（CSV）：**
```csv
flow_id,timestamp,packet_length,direction,label
1,0.000,120,1,Facebook
1,0.015,1460,-1,Facebook
1,0.023,52,1,Facebook
2,0.000,800,1,Spotify
...
```

**如果你有PCAP文件：**
```bash
# 使用tshark转换为CSV
tshark -r input.pcap -T fields \
  -e frame.time_relative \
  -e frame.len \
  -e ip.src \
  -e ip.dst \
  -E header=y -E separator=, > output.csv
```

#### 步骤2：加载数据

```matlab
% 修改 train_flowpic.m 中的数据加载部分
[flows, labels] = load_pcap_data('data_folder', 'labels.csv');
```

#### 步骤3：运行训练

```matlab
% 直接运行训练脚本
run train_flowpic.m
```

## 🔧 核心函数详解

### 1. generate_flowpic()

```matlab
flowpic = generate_flowpic(lengths, times, directions, size)
```

**功能**：将网络流转换为32×32×4的多通道直方图

**通道说明**：
- Channel 1: 上行packet length vs time
- Channel 2: 下行packet length vs time
- Channel 3: 上行inter-arrival time vs time
- Channel 4: 下行inter-arrival time vs time

**参数**：
- `size`: 直方图大小（默认32）
- `MAX_TIME`: 15秒时间窗口
- `MTU`: 1460字节（最大传输单元）

### 2. create_flowpic_model()

```matlab
lgraph = create_flowpic_model([32, 32, 4], num_classes)
```

**功能**：创建带残差连接的2D CNN

**架构**：
```
Input (32×32×4)
  ↓
Conv 3×3 (64 filters)
  ↓
Residual Block 1 (64 filters)
  ↓
Residual Block 2 (128 filters, stride=2)
  ↓
Average Pooling
  ↓
FC (256) → FC (num_classes)
  ↓
Softmax
```

## 📊 超参数设置

与论文保持一致：

| 参数 | 值 |
|------|-----|
| Optimizer | AdamW (MATLAB中使用Adam) |
| Learning Rate | 0.001 (cosine annealing) |
| Weight Decay | 0.0001 |
| Batch Size | 64 |
| Epochs | 50 |
| Dropout | 0.2 |
| Train/Val/Test | 80:10:10 |

## 📈 预期结果

论文在两个数据集上的结果：

| 数据集 | 单通道 | 双通道 | 四通道 |
|--------|--------|--------|--------|
| MIRAGE-19 | 77.02% | 80.21% | **81.10%** |
| MIRAGE-22 | 95.36% | 95.96% | **96.71%** |

## 🔍 调试技巧

### 1. 检查数据分布
```matlab
% 可视化生成的flowpic
figure;
for ch = 1:4
    subplot(2,2,ch);
    imagesc(flowpic(:,:,ch));
    title(['Channel ' num2str(ch)]);
    colorbar;
end
```

### 2. 监控训练
```matlab
% 在training options中添加
'Plots', 'training-progress'
```

### 3. 验证数据预处理
```matlab
% 检查时间范围
assert(all(times <= 15), '存在超过15秒的数据包');

% 检查长度范围
assert(all(lengths <= 1460), '存在超过MTU的数据包');
```

## 🐛 常见问题

**Q1: 内存不足？**
- 减少batch size
- 使用16×16直方图（性能略降但节省内存）
- 分批处理数据

**Q2: 训练很慢？**
- 使用GPU: `'ExecutionEnvironment', 'gpu'`
- 减少epochs或使用early stopping

**Q3: 准确率很低？**
- 检查数据标签是否正确
- 确认数据预处理（时间归一化、方向标注）
- 尝试调整学习率

## 📚 数据集

论文使用的公开数据集：
1. **MIRAGE-19** - 移动应用流量（短流）
2. **MIRAGE-22** - 视频会议流量（长流）

可从以下链接获取：
- https://traffic.comics.unina.it/mirage/

## 🔗 参考资源

- 论文GitHub（原始Python代码）: https://github.com/danielpoliakov/flowmind
- FlowPic原始论文: Shapira & Shavitt, INFOCOM 2019
- MATLAB Deep Learning文档: https://www.mathworks.com/help/deeplearning/

## 📝 引用

如果使用此代码，请引用原始论文：

```bibtex
@inproceedings{poliakov2025multichannel,
  title={Multichannel Histograms for Flow Classification},
  author={Poliakov, Daniel and Jeřábek, Kamil and Kolář, Dušan and Čejka, Tomáš},
  booktitle={2025 IEEE/IFIP Network Operations and Management Symposium (NOMS)},
  year={2025}
}
```

## ⚖️ 许可证

本实现仅供学术研究使用。商业使用请联系原作者。
