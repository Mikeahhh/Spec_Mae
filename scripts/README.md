# 实验脚本组织说明

## 目录结构

```
scripts/
├── train/                        # 训练脚本
│   ├── __init__.py
│   ├── train_single_scenario.py    # 单场景训练
│   ├── train_multi_scenario.py     # 多场景训练
│   └── train_cross_validation.py   # 交叉验证
│
├── test/                         # 测试脚本
│   ├── __init__.py
│   ├── test_anomaly_detection.py   # 测试异常检测
│   ├── test_localization.py        # 测试定位精度
│   └── test_full_system.py         # 测试完整系统
│
├── eval/                         # 评估脚本
│   ├── __init__.py
│   ├── compute_metrics.py          # 计算评估指标
│   ├── plot_results.py             # 绘制结果图表
│   └── analyze_performance.py      # 性能分析
│
└── utils/                        # 工具函数
    ├── __init__.py
    ├── audio_processing.py         # 音频处理
    ├── feature_extraction.py       # 特征提取
    ├── data_loader.py              # 数据加载器
    └── logger.py                   # 日志记录
```

---

## 训练脚本 (train/)

### train_single_scenario.py
**单场景模型训练**

功能：
- 训练特定场景的SpecMAE模型（沙漠/森林/海洋）
- 支持从头训练或从checkpoint恢复
- 自动保存最佳模型

使用方法：
```bash
python scripts/train/train_single_scenario.py \
    --scenario desert \
    --config configs/desert_config.yaml \
    --data_dir data/raw/desert \
    --output_dir checkpoints/desert \
    --epochs 100 \
    --batch_size 32 \
    --mask_ratio 0.75
```

输出：
- `checkpoints/desert/best_model.pth` - 最佳模型
- `checkpoints/desert/last_model.pth` - 最后一个epoch
- `results/logs/desert_train.log` - 训练日志

参考：
- DCASE的01_train.sh
- AudioMAE的pretrain_audioset2M.sh

---

### train_multi_scenario.py
**多场景联合训练**

功能：
- 使用多个场景的数据联合训练
- 学习跨场景的鲁棒特征
- 支持场景加权采样

使用方法：
```bash
python scripts/train/train_multi_scenario.py \
    --scenarios desert forest ocean \
    --config configs/multi_scenario_config.yaml \
    --data_dirs data/raw/desert data/raw/forest data/raw/ocean \
    --output_dir checkpoints/multi_scenario \
    --epochs 150 \
    --batch_size 32
```

输出：
- `checkpoints/multi_scenario/best_model.pth`
- `results/logs/multi_scenario_train.log`

关键参数：
- `--scenario_weights` - 各场景的采样权重
- `--adaptive_mask` - 是否使用自适应掩码率

---

### train_cross_validation.py
**交叉验证训练**

功能：
- K-fold交叉验证
- 选择最优超参数（掩码率、阈值等）
- 评估模型泛化能力

使用方法：
```bash
python scripts/train/train_cross_validation.py \
    --scenario desert \
    --config configs/desert_config.yaml \
    --data_dir data/raw/desert \
    --n_folds 5 \
    --mask_ratios 0.5 0.6 0.7 0.75 0.8 0.9 \
    --output_dir results/cv_results
```

输出：
- `results/cv_results/fold_1_metrics.json`
- `results/cv_results/fold_2_metrics.json`
- ...
- `results/cv_results/cv_summary.csv` - 汇总结果

参考：
- DCASE的交叉验证逻辑
- 会议记录中的交叉验证要求

---

## 测试脚本 (test/)

### test_anomaly_detection.py
**测试异常检测性能**

功能：
- 测试SpecMAE的异常检测能力
- 计算AUC、pAUC、F1-score等指标
- 生成异常分数曲线（论文Fig 3A）

使用方法：
```bash
python scripts/test/test_anomaly_detection.py \
    --model checkpoints/desert/best_model.pth \
    --test_data data/raw/desert/test \
    --output_dir results/anomaly_detection \
    --threshold 0.001
```

输出：
- `results/anomaly_detection/anomaly_scores.csv` - 每个样本的异常分数
- `results/anomaly_detection/metrics.json` - 评估指标
- `results/figures/anomaly_curve.png` - 异常检测曲线

测试内容：
- 正常样本的重建损失分布
- 异常样本的重建损失分布
- ROC曲线
- 不同阈值下的性能

---

### test_localization.py
**测试定位精度**

功能：
- 测试GCC-PHAT定位算法
- 评估不同SNR下的定位误差
- 对比有/无环形缓冲区的性能（论文Fig 3B）

使用方法：
```bash
python scripts/test/test_localization.py \
    --test_data data/raw/desert/test/anomaly \
    --snr_values -10 -5 0 5 10 15 20 \
    --output_dir results/localization \
    --use_ring_buffer
```

输出：
- `results/localization/localization_errors.csv` - 定位误差
- `results/figures/snr_vs_error.png` - SNR vs 误差曲线

测试内容：
- 不同SNR下的DOA误差
- 不同场景下的定位精度
- 环形缓冲区的影响

---

### test_full_system.py
**测试完整系统**

功能：
- 端到端测试哨兵-响应者系统
- 模拟完整的SAR任务
- 生成3D飞行轨迹可视化

使用方法：
```bash
python scripts/test/test_full_system.py \
    --sentinel_model checkpoints/desert/best_model.pth \
    --test_scenario desert \
    --flight_path configs/flight_path.yaml \
    --output_dir results/full_system
```

输出：
- `results/full_system/trajectory.json` - 飞行轨迹数据
- `results/figures/3d_trajectory.png` - 3D轨迹图
- `results/full_system/system_log.csv` - 系统状态日志

测试内容：
- 哨兵模式检测延迟
- 响应者模式唤醒时间
- 完整系统的能耗估算

---

## 评估脚本 (eval/)

### compute_metrics.py
**计算评估指标**

功能：
- 计算AUC、pAUC、Precision、Recall、F1
- 计算定位误差统计（均值、中位数、90分位）
- 生成性能报告

使用方法：
```bash
python scripts/eval/compute_metrics.py \
    --predictions results/anomaly_detection/anomaly_scores.csv \
    --ground_truth data/raw/desert/test/labels.csv \
    --output results/metrics/desert_metrics.json
```

参考：
- DCASE的评估指标计算

---

### plot_results.py
**绘制结果图表**

功能：
- 生成论文所需的所有图表
- Fig 3A: 异常检测曲线
- Fig 3B: 定位精度对比
- Fig 4: 掩码率实验
- Fig 5: SNR vs 误差

使用方法：
```bash
python scripts/eval/plot_results.py \
    --results_dir results/ \
    --output_dir results/figures/ \
    --paper_style
```

输出：
- `results/figures/fig3a_anomaly_detection.pdf`
- `results/figures/fig3b_localization_comparison.pdf`
- `results/figures/fig4_mask_ratio.pdf`
- `results/figures/fig5_snr_vs_error.pdf`

---

### analyze_performance.py
**性能分析**

功能：
- 跨场景性能对比
- 泛化能力分析
- 失败案例分析

使用方法：
```bash
python scripts/eval/analyze_performance.py \
    --results_dir results/ \
    --output results/analysis_report.md
```

---

## 工具函数 (utils/)

### audio_processing.py
**音频处理工具**

功能：
- 音频加载与重采样
- 音频混合（不同SNR）
- 音频分段

关键函数：
```python
def load_audio(file_path, sr=48000)
def resample_audio(audio, orig_sr, target_sr)
def mix_audio(signal, noise, snr_db)
def segment_audio(audio, segment_length)
```

参考：
- DCASE的common.py
- AST的dataloader.py

---

### feature_extraction.py
**特征提取工具**

功能：
- Log-Mel特征提取
- 特征归一化
- Patch切分

关键函数：
```python
def extract_logmel(audio, sr, n_mels, n_fft, hop_length)
def normalize_features(features, mean, std)
def split_into_patches(spectrogram, patch_size)
```

参考：
- AST的特征提取
- AudioMAE的预处理

---

### data_loader.py
**数据加载器**

功能：
- PyTorch Dataset类
- 数据增强（SpecAugment）
- 批处理

关键类：
```python
class AudioDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, scenario, split)
    def __getitem__(self, idx)
    def __len__(self)
```

---

### logger.py
**日志记录工具**

功能：
- 训练日志记录
- TensorBoard集成
- 实验结果保存

---

## 实验执行流程

### 完整实验流程（按顺序执行）

```bash
# 1. 计算数据集统计量
python scripts/utils/compute_stats.py --scenario desert

# 2. 交叉验证选择超参数
python scripts/train/train_cross_validation.py --scenario desert --n_folds 5

# 3. 全量训练最终模型
python scripts/train/train_single_scenario.py --scenario desert --epochs 100

# 4. 测试异常检测
python scripts/test/test_anomaly_detection.py --model checkpoints/desert/best_model.pth

# 5. 测试定位精度
python scripts/test/test_localization.py --snr_values -10 -5 0 5 10 15 20

# 6. 测试完整系统
python scripts/test/test_full_system.py --sentinel_model checkpoints/desert/best_model.pth

# 7. 计算指标
python scripts/eval/compute_metrics.py --results_dir results/

# 8. 绘制图表
python scripts/eval/plot_results.py --results_dir results/ --output_dir results/figures/

# 9. 生成分析报告
python scripts/eval/analyze_performance.py --results_dir results/
```

---

## Shell脚本封装（可选）

可以创建以下shell脚本简化执行：

```bash
scripts/
├── run_all_experiments.sh       # 运行所有实验
├── run_single_scenario.sh       # 运行单场景实验
├── run_multi_scenario.sh        # 运行多场景实验
└── generate_paper_figures.sh    # 生成论文图表
```

---

## 注意事项

1. **所有脚本都应支持命令行参数**
2. **使用配置文件管理超参数**
3. **记录所有实验的随机种子**
4. **保存完整的实验日志**
5. **结果文件使用时间戳命名**
