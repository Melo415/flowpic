# ✅ FlowPic训练前检查清单

在运行 `COMPLETE_TRAINING` 之前，请逐项检查：

---

## 📋 第一步：环境准备

### MATLAB环境

- [ ] MATLAB版本 ≥ R2020a
  ```matlab
  version('-release')  % 应该显示 '2020a' 或更高
  ```

- [ ] Deep Learning Toolbox已安装
  ```matlab
  license('test', 'deep_learning_toolbox')  % 应该返回 1
  ```

- [ ] Statistics Toolbox已安装
  ```matlab
  license('test', 'statistics_toolbox')  % 应该返回 1
  ```

### 硬件要求

- [ ] 内存 ≥ 8GB（推荐16GB）
  ```matlab
  memory  % 查看可用内存
  ```

- [ ] 磁盘空间 ≥ 5GB
  ```matlab
  % 检查当前目录可用空间
  ```

- [ ] GPU（可选但强烈推荐）
  ```matlab
  gpuDevice  % 如果有GPU，显示GPU信息
  ```

---

## 📂 第二步：数据准备

### MIRAGE-19数据集

- [ ] 已下载MIRAGE-19数据集
  - 下载地址：https://traffic.comics.unina.it/mirage/

- [ ] 数据已解压到指定位置
  ```
  例如：D:\Desktop\GitHub\flowpic\data\mirage\
  ```

- [ ] 文件夹结构正确
  ```
  mirage/
  ├── folder1/
  │   ├── xxx.json
  │   └── yyy.json
  └── folder2/
      └── ...
  ```

- [ ] 至少有10个JSON文件
  ```matlab
  json_files = dir('data/mirage/**/*.json');
  length(json_files)  % 应该 > 10
  ```

### 数据验证

- [ ] 运行数据测试脚本
  ```matlab
  % 修改test_mirage_json.m中的路径后运行
  test_mirage_json
  ```

- [ ] 测试输出显示成功
  ```
  ✓ 该文件包含 XX 个flows
  ✓ 数据加载成功！
  ```

---

## 📝 第三步：代码文件

### 必需文件（全部在当前目录）

- [ ] `COMPLETE_TRAINING.m` - 主训练脚本
- [ ] `load_mirage_json.m` - 数据加载器
- [ ] `generate_flowpic.m` - FlowPic生成函数
- [ ] `create_flowpic_model.m` - 模型定义

验证：
```matlab
% 检查所有文件是否存在
required_files = {
    'COMPLETE_TRAINING.m';
    'load_mirage_json.m';
    'generate_flowpic.m';
    'create_flowpic_model.m';
};

for i = 1:length(required_files)
    if exist(required_files{i}, 'file')
        fprintf('✓ %s\n', required_files{i});
    else
        fprintf('✗ %s (缺失)\n', required_files{i});
    end
end
```

### 目录结构

- [ ] 创建必要的目录
  ```matlab
  % 自动创建（训练脚本会自动创建，但提前创建更好）
  mkdir('data');
  mkdir('models');
  mkdir('figures');
  mkdir('results');
  ```

---

## ⚙️ 第四步：配置参数

### 打开 `COMPLETE_TRAINING.m`

- [ ] 第23行：修改数据路径
  ```matlab
  DATASET_PATH = 'D:\Desktop\GitHub\flowpic\data\mirage';
  % 改为你的实际路径
  ```

- [ ] 第26-27行：首次运行使用子集
  ```matlab
  USE_SUBSET = true;    % ✓ 首次测试设为true
  SUBSET_SIZE = 1000;   % ✓ 1000样本快速测试
  ```

- [ ] 第34行：首次可以减少epochs
  ```matlab
  MAX_EPOCHS = 10;  % 首次测试用10轮（可选）
  ```

### 检查路径是否正确

- [ ] 数据路径存在
  ```matlab
  DATASET_PATH = 'D:\Desktop\GitHub\flowpic\data\mirage';
  exist(DATASET_PATH, 'dir')  % 应该返回 7 (目录存在)
  ```

- [ ] 路径中有JSON文件
  ```matlab
  json_files = dir(fullfile(DATASET_PATH, '**', '*.json'));
  length(json_files)  % 应该 > 0
  fprintf('找到 %d 个JSON文件\n', length(json_files));
  ```

---

## 🎯 第五步：首次测试运行

### 运行前确认

- [ ] 所有文件在当前目录
  ```matlab
  pwd  % 显示当前目录
  ls   % 列出所有文件
  ```

- [ ] 配置参数已修改
  ```matlab
  % 再次确认 DATASET_PATH 正确
  ```

- [ ] 保存了所有文件
  ```matlab
  % 如果修改了代码，记得保存
  ```

### 启动测试

- [ ] 清空工作区
  ```matlab
  clear all
  close all
  clc
  ```

- [ ] 运行主脚本
  ```matlab
  COMPLETE_TRAINING
  ```

### 观察输出

- [ ] 看到"第一部分：环境配置"
- [ ] 看到"第二部分：加载真实数据"
- [ ] 看到数据加载成功
  ```
  ✓ 数据加载成功！用时: XX 秒
  数据集统计:
    总样本数: XXXX
  ```

如果到这一步都正常，说明准备工作完成！

---

## ⏱️ 第六步：时间预估

### 首次测试运行（验证流程）

配置：
```matlab
USE_SUBSET = true
SUBSET_SIZE = 1000
MAX_EPOCHS = 10
```

预计时间：
- GPU: **10-15分钟**
- CPU: **30-60分钟**

### 完整训练运行

配置：
```matlab
USE_SUBSET = false  % 全部数据
MAX_EPOCHS = 50
```

预计时间：
- GPU (RTX 3080): **1.5-2.5小时**
- GPU (GTX 1080): **2.5-4小时**
- CPU: **8-12小时** ⚠️

---

## 🚨 常见问题预检

### 问题1：路径错误

症状：`找不到JSON文件`

检查：
```matlab
DATASET_PATH = 'xxx';
dir(fullfile(DATASET_PATH, '*.json'))
% 如果为空，说明路径错误
```

解决：仔细检查路径，使用绝对路径

### 问题2：内存不足

症状：`Out of memory`

检查：
```matlab
memory  % 查看可用内存
```

解决：
```matlab
BATCH_SIZE = 32;  % 或16
SUBSET_SIZE = 500;  % 减小数据量
```

### 问题3：GPU错误

症状：`GPU device error`

检查：
```matlab
gpuDevice  % 查看GPU状态
```

解决：
```matlab
% 在COMPLETE_TRAINING.m中强制使用CPU
execution_env = 'cpu';
```

---

## ✅ 最终检查

### 全部准备就绪

- [ ] ✅ MATLAB环境正确
- [ ] ✅ 数据集已下载并解压
- [ ] ✅ 所有代码文件在当前目录
- [ ] ✅ 数据路径已配置
- [ ] ✅ 参数已设置为测试模式
- [ ] ✅ 时间预估已知晓
- [ ] ✅ 准备好等待训练完成

### 开始训练

```matlab
% 深呼吸，运行主脚本
COMPLETE_TRAINING
```

---

## 📊 训练完成后检查

训练结束后，应该看到：

- [ ] 显示"训练完成总结"
- [ ] 测试准确率 > 60%（子集）或 > 75%（全量）
- [ ] `models/` 目录下有 `.mat` 文件
- [ ] `figures/` 目录下有 `.png` 图像
- [ ] 混淆矩阵显示正常

如果全部✅，恭喜完成！🎉

---

## 🆘 需要帮助？

如果遇到问题：

1. **查看错误信息** - 完整复制错误
2. **检查本清单** - 逐项确认
3. **查看文档** - `COMPLETE_TRAINING_GUIDE.md`
4. **查看快速参考** - `QUICK_REFERENCE.md`

---

**准备好了吗？开始训练吧！** 🚀
