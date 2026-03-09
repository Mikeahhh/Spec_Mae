# Sound-UAV 项目 TODO 清单

## 📋 当前状态：文件结构已创建 ✓

---

## 🎯 阶段1：环境配置与依赖安装

### 1.1 创建requirements.txt
```bash
# 待添加的依赖包：
torch>=2.0.0
torchaudio>=2.0.0
torchvision>=0.15.0
librosa>=0.10.0
scipy>=1.10.0
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tensorboard>=2.13.0
pyyaml>=6.0
timm>=0.9.0
scikit-learn>=1.3.0
tqdm>=4.65.0
```

### 1.2 创建.gitignore
```bash
# 待添加的忽略规则：
__pycache__/
*.pyc
*.pth
*.npy
*.wav
checkpoints/
results/
data/raw/
data/processed/
.vscode/
.idea/
```

### 1.3 创建setup.py
```bash
# 用于pip install -e .安装项目
```

---

## 🎯 阶段2：数据准备（你的数据集较多）

### 2.1 整理现有数据
- [ ] 将无人机噪音文件分类到对应目录
  - [ ] 沙漠场景：`data/raw/desert/train/drone_noise/`
  - [ ] 森林场景：`data/raw/forest/train/drone_noise/`
  - [ ] 海洋场景：`data/raw/ocean/train/drone_noise/`

- [ ] 将环境噪音文件分类到对应目录
  - [ ] 沙漠环境音：`data/raw/desert/train/ambient_noise/`
  - [ ] 森林环境音：`data/raw/forest/train/ambient_noise/`
  - [ ] 海洋环境音：`data/raw/ocean/train/ambient_noise/`

### 2.2 生成混合数据
- [ ] 编写音频混合脚本：`scripts/utils/mix_audio.py`
- [ ] 生成训练集（无人机+环境噪音）
  - [ ] 沙漠场景混合数据
  - [ ] 森林场景混合数据
  - [ ] 海洋场景混合数据

### 2.3 生成测试集
- [ ] 准备人声呼救音频
- [ ] 生成不同SNR的异常样本
  - [ ] SNR = -10, -5, 0, 5, 10, 15, 20 dB
  - [ ] 每个场景至少50个样本

### 2.4 数据集统计
- [ ] 编写统计脚本：`scripts/utils/compute_stats.py`
- [ ] 计算每个场景的mean和std
- [ ] 更新配置文件中的归一化参数

---

## 🎯 阶段3：工具函数开发

### 3.1 音频处理 (scripts/utils/audio_processing.py)
- [ ] `load_audio()` - 音频加载
- [ ] `resample_audio()` - 重采样
- [ ] `mix_audio()` - 音频混合（不同SNR）
- [ ] `segment_audio()` - 音频分段
- [ ] 单元测试

### 3.2 特征提取 (scripts/utils/feature_extraction.py)
- [ ] `extract_logmel()` - Log-Mel特征提取
- [ ] `normalize_features()` - 特征归一化
- [ ] `split_into_patches()` - Patch切分
- [ ] 单元测试

### 3.3 数据加载器 (scripts/utils/data_loader.py)
- [ ] `AudioDataset` - PyTorch Dataset类
- [ ] `get_dataloader()` - DataLoader工厂函数
- [ ] 支持多场景数据加载
- [ ] 单元测试

### 3.4 日志工具 (scripts/utils/logger.py)
- [ ] `setup_logger()` - 日志配置
- [ ] `TensorBoardLogger` - TensorBoard集成
- [ ] `ExperimentTracker` - 实验跟踪

---

## 🎯 阶段4：模型开发

### 4.1 基础组件
- [ ] `models/specmae/patch_embed.py` - Patch Embedding
  - [ ] 参考AST的实现
  - [ ] 支持16×16 patch
  - [ ] 单元测试

- [ ] `models/specmae/pos_embed.py` - 位置编码
  - [ ] 参考AudioMAE的实现
  - [ ] 2D正弦位置编码
  - [ ] 单元测试

### 4.2 Encoder & Decoder
- [ ] `models/specmae/encoder.py` - Transformer Encoder
  - [ ] 参考AudioMAE的encoder
  - [ ] 只处理visible patches
  - [ ] 12层Transformer Block
  - [ ] 单元测试

- [ ] `models/specmae/decoder.py` - Transformer Decoder
  - [ ] 参考AudioMAE的decoder
  - [ ] 4层Transformer Block
  - [ ] 重建masked patches
  - [ ] 单元测试

### 4.3 完整模型
- [ ] `models/specmae/specmae_model.py` - SpecMAE主模型
  - [ ] 整合所有组件
  - [ ] 前向传播逻辑
  - [ ] 损失计算（只在masked patches上）
  - [ ] 异常检测接口
  - [ ] 单元测试

### 4.4 模型测试
- [ ] 测试模型能否正常前向传播
- [ ] 测试输入输出维度是否正确
- [ ] 测试掩码机制是否正常
- [ ] 测试损失计算是否正确

---

## 🎯 阶段5：训练脚本开发

### 5.1 单场景训练
- [ ] `scripts/train/train_single_scenario.py`
  - [ ] 加载配置文件
  - [ ] 初始化模型
  - [ ] 训练循环
  - [ ] 验证循环
  - [ ] 保存checkpoint
  - [ ] TensorBoard日志

### 5.2 交叉验证
- [ ] `scripts/train/train_cross_validation.py`
  - [ ] K-fold数据划分
  - [ ] 训练每个fold
  - [ ] 汇总结果
  - [ ] 选择最优超参数

### 5.3 多场景训练
- [ ] `scripts/train/train_multi_scenario.py`
  - [ ] 多场景数据加载
  - [ ] 场景加权采样
  - [ ] 自适应掩码率（可选）
  - [ ] 训练循环

---

## 🎯 阶段6：测试脚本开发

### 6.1 异常检测测试
- [ ] `scripts/test/test_anomaly_detection.py`
  - [ ] 加载训练好的模型
  - [ ] 测试正常样本
  - [ ] 测试异常样本
  - [ ] 计算异常分数
  - [ ] 保存结果

### 6.2 定位精度测试
- [ ] `scripts/test/test_localization.py`
  - [ ] 实现GCC-PHAT算法
  - [ ] 实现TDOA估计
  - [ ] 实现DOA计算
  - [ ] 测试不同SNR
  - [ ] 计算定位误差

### 6.3 完整系统测试
- [ ] `scripts/test/test_full_system.py`
  - [ ] 模拟飞行轨迹
  - [ ] 哨兵模式检测
  - [ ] 响应者模式定位
  - [ ] 状态切换可视化
  - [ ] 系统日志记录

---

## 🎯 阶段7：评估与可视化

### 7.1 指标计算
- [ ] `scripts/eval/compute_metrics.py`
  - [ ] AUC / pAUC
  - [ ] Precision / Recall / F1
  - [ ] 定位误差统计
  - [ ] 生成性能报告

### 7.2 结果绘图
- [ ] `scripts/eval/plot_results.py`
  - [ ] Fig 3A: 异常检测曲线
  - [ ] Fig 3B: 定位精度对比
  - [ ] Fig 4: 掩码率实验
  - [ ] Fig 5: SNR vs 误差
  - [ ] 3D飞行轨迹图

### 7.3 性能分析
- [ ] `scripts/eval/analyze_performance.py`
  - [ ] 跨场景性能对比
  - [ ] 泛化能力分析
  - [ ] 失败案例分析
  - [ ] 生成分析报告

---

## 🎯 阶段8：实验执行

### 8.1 单场景实验
- [ ] 沙漠场景
  - [ ] 交叉验证选择超参数
  - [ ] 全量训练最终模型
  - [ ] 测试异常检测
  - [ ] 测试定位精度

- [ ] 森林场景
  - [ ] 交叉验证
  - [ ] 全量训练
  - [ ] 测试

- [ ] 海洋场景
  - [ ] 交叉验证
  - [ ] 全量训练
  - [ ] 测试

### 8.2 多场景实验
- [ ] 多场景联合训练
- [ ] 跨场景泛化测试
- [ ] 性能对比

### 8.3 关键实验（论文需要）
- [ ] 掩码率对比实验（50%, 60%, 70%, 75%, 80%, 90%）
- [ ] SNR vs 定位误差实验（-10 to 20 dB）
- [ ] 环形缓冲区对比实验（有/无）
- [ ] 能耗分析实验

---

## 🎯 阶段9：论文图表生成

### 9.1 论文所需图表
- [ ] Fig 1: 系统架构图（手绘或PPT）
- [ ] Fig 2: SpecMAE架构图（手绘或PPT）
- [ ] Fig 3A: 异常检测曲线（Python生成）
- [ ] Fig 3B: 定位精度对比（Python生成）
- [ ] Fig 4: 掩码率实验（Python生成）
- [ ] Fig 5: SNR vs 误差（Python生成）
- [ ] 3D飞行轨迹图（Python生成）

### 9.2 Pre-train Strategy图
- [ ] 参考AudioMAE的多decoder架构图
- [ ] 绘制多场景预训练流程图

---

## 🎯 阶段10：文档完善

### 10.1 代码文档
- [ ] 为所有函数添加docstring
- [ ] 生成API文档
- [ ] 添加使用示例

### 10.2 实验文档
- [ ] 记录所有实验配置
- [ ] 记录实验结果
- [ ] 记录失败案例与解决方案

### 10.3 README更新
- [ ] 添加安装说明
- [ ] 添加快速开始指南
- [ ] 添加实验复现步骤
- [ ] 添加引用信息

---

## 📊 进度追踪

### 当前进度
- [x] 项目结构创建
- [x] 文档框架搭建
- [ ] 代码开发（0%）
- [ ] 数据准备（0%）
- [ ] 实验执行（0%）

### 预计时间线
- 阶段1-2：1-2天（环境配置+数据准备）
- 阶段3-4：3-5天（工具函数+模型开发）
- 阶段5-6：3-5天（训练+测试脚本）
- 阶段7-8：5-7天（评估+实验执行）
- 阶段9-10：2-3天（图表生成+文档）

**总计：约2-3周**

---

## 🚨 优先级标记

### 🔴 高优先级（必须完成）
- 数据准备与整理
- 核心模型开发（SpecMAE）
- 单场景训练与测试
- 论文关键实验（掩码率、SNR vs 误差）
- 论文图表生成

### 🟡 中优先级（重要但可延后）
- 多场景训练
- 完整系统测试
- 性能分析报告

### 🟢 低优先级（可选）
- 基线模型对比
- Jupyter notebooks
- 详细的API文档

---

## 📝 注意事项

1. **数据集较多**：优先整理和标注数据
2. **实验记录**：每次实验都要记录配置和结果
3. **版本控制**：及时commit代码
4. **备份重要文件**：定期备份模型和数据
5. **参考基线代码**：充分利用AudioMAE、AST、DCASE的代码

---

## 🎯 下一步行动

**立即开始：**
1. 创建requirements.txt和.gitignore
2. 整理你的音频数据集
3. 开始编写工具函数（audio_processing.py）

**需要帮助时：**
- 参考`PROJECT_STRUCTURE.md`查看完整结构
- 参考`models/README.md`了解模型设计
- 参考`scripts/README.md`了解脚本功能
- 参考`data/README.md`了解数据组织

---

生成时间：2026-03-09
最后更新：2026-03-09
