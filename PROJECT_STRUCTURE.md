# Sound-UAV 项目完整文件结构

生成时间: 2026-03-09

## 📁 完整目录树

```
E:\model_train_example\Spec_Mae/
│
├── README.md                          # 项目总览
├── requirements.txt                   # Python依赖（待创建）
├── setup.py                          # 安装脚本（待创建）
│
├── data/                             # 数据集目录
│   ├── README.md                     # 数据集说明文档 ✓
│   │
│   ├── raw/                          # 原始音频数据
│   │   ├── desert/                   # 沙漠场景
│   │   │   ├── train/
│   │   │   │   ├── drone_noise/     # 无人机噪音
│   │   │   │   │   ├── hover_*.wav
│   │   │   │   │   ├── cruise_*.wav
│   │   │   │   │   └── maneuver_*.wav
│   │   │   │   ├── ambient_noise/   # 环境噪音
│   │   │   │   │   ├── wind_*.wav
│   │   │   │   │   └── background_*.wav
│   │   │   │   └── mixed_normal/    # 混合正常噪音
│   │   │   │       └── normal_*.wav
│   │   │   ├── test/
│   │   │   │   ├── normal/          # 正常样本
│   │   │   │   │   └── test_normal_*.wav
│   │   │   │   └── anomaly/         # 异常样本（含人声）
│   │   │   │       ├── test_anomaly_snr_minus10_*.wav
│   │   │   │       ├── test_anomaly_snr_minus5_*.wav
│   │   │   │       ├── test_anomaly_snr_0_*.wav
│   │   │   │       ├── test_anomaly_snr_5_*.wav
│   │   │   │       ├── test_anomaly_snr_10_*.wav
│   │   │   │       ├── test_anomaly_snr_15_*.wav
│   │   │   │       └── test_anomaly_snr_20_*.wav
│   │   │   └── validation/          # 验证集（交叉验证用）
│   │   │       ├── fold_1/
│   │   │       ├── fold_2/
│   │   │       ├── fold_3/
│   │   │       ├── fold_4/
│   │   │       └── fold_5/
│   │   │
│   │   ├── forest/                   # 森林场景（结构同desert）
│   │   │   ├── train/
│   │   │   ├── test/
│   │   │   └── validation/
│   │   │
│   │   ├── ocean/                    # 海洋场景（结构同desert）
│   │   │   ├── train/
│   │   │   ├── test/
│   │   │   └── validation/
│   │   │
│   │   └── multi_scenario/           # 多场景混合
│   │       ├── train/
│   │       ├── test/
│   │       └── validation/
│   │
│   ├── processed/                    # 预处理后的数据
│   │   ├── desert/
│   │   │   ├── train_logmel.npy
│   │   │   ├── train_labels.npy
│   │   │   └── train_metadata.json
│   │   ├── forest/
│   │   └── ocean/
│   │
│   └── stats/                        # 数据集统计信息
│       ├── desert_stats.json
│       ├── forest_stats.json
│       ├── ocean_stats.json
│       └── multi_scenario_stats.json
│
├── models/                           # 模型定义
│   ├── README.md                     # 模型说明文档 ✓
│   │
│   ├── specmae/                      # SpecMAE核心模型
│   │   ├── __init__.py              # ✓
│   │   ├── specmae_model.py         # 主模型类 ✓
│   │   ├── encoder.py               # Transformer Encoder ✓
│   │   ├── decoder.py               # Transformer Decoder ✓
│   │   ├── patch_embed.py           # Patch Embedding ✓
│   │   └── pos_embed.py             # 位置编码 ✓
│   │
│   └── baseline/                     # 基线模型（参考）
│       ├── __init__.py              # ✓
│       ├── dcase_ae.py              # DCASE Autoencoder ✓
│       ├── audiomae.py              # AudioMAE参考 ✓
│       └── ast.py                   # AST参考 ✓
│
├── scripts/                          # 实验脚本
│   ├── README.md                     # 脚本说明文档 ✓
│   │
│   ├── train/                        # 训练脚本
│   │   ├── __init__.py              # ✓
│   │   ├── train_single_scenario.py # 单场景训练 ✓
│   │   ├── train_multi_scenario.py  # 多场景训练 ✓
│   │   └── train_cross_validation.py # 交叉验证 ✓
│   │
│   ├── test/                         # 测试脚本
│   │   ├── __init__.py              # ✓
│   │   ├── test_anomaly_detection.py # 异常检测测试 ✓
│   │   ├── test_localization.py     # 定位测试 ✓
│   │   └── test_full_system.py      # 完整系统测试 ✓
│   │
│   ├── eval/                         # 评估脚本
│   │   ├── __init__.py              # ✓
│   │   ├── compute_metrics.py       # 计算指标 ✓
│   │   ├── plot_results.py          # 绘制图表 ✓
│   │   └── analyze_performance.py   # 性能分析 ✓
│   │
│   └── utils/                        # 工具函数
│       ├── __init__.py              # ✓
│       ├── audio_processing.py      # 音频处理 ✓
│       ├── feature_extraction.py    # 特征提取 ✓
│       ├── data_loader.py           # 数据加载 ✓
│       └── logger.py                # 日志记录 ✓
│
├── configs/                          # 配置文件
│   ├── base_config.yaml             # 基础配置 ✓
│   ├── desert_config.yaml           # 沙漠场景配置 ✓
│   ├── forest_config.yaml           # 森林场景配置 ✓
│   ├── ocean_config.yaml            # 海洋场景配置 ✓
│   ├── multi_scenario_config.yaml   # 多场景配置 ✓
│   └── experiment_config.yaml       # 实验配置 ✓
│
├── checkpoints/                      # 模型检查点
│   ├── desert/
│   │   ├── best_model.pth
│   │   ├── last_model.pth
│   │   └── epoch_*.pth
│   ├── forest/
│   ├── ocean/
│   └── multi_scenario/
│
├── results/                          # 实验结果
│   ├── figures/                     # 图表
│   │   ├── fig3a_anomaly_detection.pdf
│   │   ├── fig3b_localization_comparison.pdf
│   │   ├── fig4_mask_ratio.pdf
│   │   ├── fig5_snr_vs_error.pdf
│   │   └── 3d_trajectory.png
│   │
│   ├── logs/                        # 日志文件
│   │   ├── desert_train.log
│   │   ├── forest_train.log
│   │   └── multi_scenario_train.log
│   │
│   └── metrics/                     # 评估指标
│       ├── desert_metrics.json
│       ├── forest_metrics.json
│       ├── cv_results/
│       └── analysis_report.md
│
├── notebooks/                        # Jupyter notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_feature_visualization.ipynb
│   ├── 03_model_analysis.ipynb
│   └── 04_results_visualization.ipynb
│
├── docs/                            # 文档
│   ├── architecture.md              # 架构说明
│   ├── experiments.md               # 实验设计
│   ├── api_reference.md             # API文档
│   └── troubleshooting.md           # 故障排除
│
└── research/                        # 论文与会议记录
    ├── SPAWC_UAV.txt                # 论文文本 ✓
    └── 会议list.txt                  # 会议记录 ✓
```

---

## 📊 文件统计

### 已创建文件
- ✓ 标记的文件已创建（空文件，待填充代码）
- 总计：~40个文件

### 待创建文件
- Python代码文件：待编写
- 配置文件：部分已创建
- 文档文件：部分已创建

---

## 🎯 下一步行动

### 1. 数据准备阶段
```bash
# 将你的音频数据按照以下规则放入对应目录：
data/raw/desert/train/drone_noise/     # 沙漠场景的无人机噪音
data/raw/desert/train/ambient_noise/   # 沙漠场景的环境噪音
data/raw/forest/train/drone_noise/     # 森林场景的无人机噪音
# ... 以此类推
```

### 2. 环境配置
```bash
# 创建虚拟环境
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 安装依赖（待创建requirements.txt）
pip install torch torchvision torchaudio
pip install librosa scipy numpy pandas
pip install timm transformers
pip install matplotlib seaborn
pip install tensorboard
pip install pyyaml
```

### 3. 代码开发顺序
1. **utils/** - 工具函数（音频处理、特征提取）
2. **models/specmae/** - 模型组件（从底层到顶层）
3. **scripts/train/** - 训练脚本
4. **scripts/test/** - 测试脚本
5. **scripts/eval/** - 评估脚本

### 4. 实验执行顺序
1. 数据预处理与统计
2. 交叉验证选择超参数
3. 单场景模型训练
4. 多场景模型训练
5. 异常检测测试
6. 定位精度测试
7. 完整系统测试
8. 结果分析与可视化

---

## 📝 命名规范

### 音频文件命名
```
<scenario>_<type>_<state>_<index>.wav

示例：
- desert_drone_hover_0001.wav
- forest_ambient_wind_0042.wav
- ocean_test_anomaly_snr_5_0123.wav
```

### 模型文件命名
```
<scenario>_<model>_<config>_<timestamp>.pth

示例：
- desert_specmae_base_20260309_143022.pth
- multi_scenario_specmae_best.pth
```

### 结果文件命名
```
<experiment>_<scenario>_<metric>_<timestamp>.csv

示例：
- anomaly_detection_desert_scores_20260309.csv
- localization_forest_errors_20260309.csv
```

---

## 🔧 配置文件说明

### base_config.yaml
所有场景共享的基础配置（模型架构、训练参数等）

### desert_config.yaml / forest_config.yaml / ocean_config.yaml
特定场景的配置（数据路径、归一化参数、掩码率等）

### multi_scenario_config.yaml
多场景联合训练的配置（场景权重、自适应策略等）

### experiment_config.yaml
具体实验的配置（用于复现特定实验）

---

## 📚 参考资源

### 基线模型代码
- AudioMAE: `E:\model_train_example\AudioMAE-main\`
- AST: `E:\model_train_example\ast-master\`
- DCASE: `E:\model_train_example\dcase2023_task2_baseline_ae-main\`

### 论文与文档
- 论文: `research/SPAWC_UAV.txt`
- 会议记录: `research/会议list.txt`
- 主README: `README.md`

---

## ⚠️ 注意事项

1. **数据集较多**：确保有足够的存储空间
2. **命名一致性**：严格遵循命名规范
3. **版本控制**：使用git管理代码（建议添加.gitignore）
4. **实验记录**：每次实验都要记录配置和结果
5. **备份重要文件**：定期备份模型和结果

---

## 📞 联系方式

如有问题，请参考：
- `docs/troubleshooting.md` - 故障排除指南
- `docs/api_reference.md` - API文档
- 会议记录中的TODO列表
