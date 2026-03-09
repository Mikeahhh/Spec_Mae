# 🚀 Sound-UAV 快速参考卡片

## 📁 关键目录

```bash
data/raw/desert/train/          # 沙漠训练数据
data/raw/forest/train/          # 森林训练数据
models/specmae/                 # SpecMAE模型代码
scripts/train/                  # 训练脚本
scripts/test/                   # 测试脚本
configs/                        # 配置文件
checkpoints/                    # 模型保存
results/                        # 实验结果
```

## 📝 关键文档

| 文档 | 用途 |
|------|------|
| `PROJECT_STRUCTURE.md` | 完整项目结构说明 |
| `TODO.md` | 详细任务清单 |
| `data/README.md` | 数据集组织说明 |
| `models/README.md` | 模型架构说明 |
| `scripts/README.md` | 脚本使用说明 |
| `research/SPAWC_UAV.txt` | 论文文本 |
| `research/会议list.txt` | 会议记录 |

## 🎯 数据命名规范

### 训练数据
```
<scenario>_<type>_<state>_<index>.wav

示例：
desert_drone_hover_0001.wav
forest_ambient_wind_0042.wav
```

### 测试数据
```
<scenario>_test_<label>_snr_<value>_<index>.wav

示例：
desert_test_normal_0001.wav
forest_test_anomaly_snr_5_0042.wav
```

## 🔧 模型配置

### 轻量级配置（推荐）
```yaml
embed_dim: 768
encoder_depth: 12
decoder_depth: 4
num_heads: 12
patch_size: 16
mask_ratio: 0.75
```

### 音频配置
```yaml
sample_rate: 48000
duration: 1.0
n_mels: 128
n_fft: 1024
hop_length: 480
```

## 🚀 快速开始

### 1. 数据准备
```bash
# 将音频文件放入对应目录
data/raw/desert/train/drone_noise/*.wav
data/raw/desert/train/ambient_noise/*.wav
```

### 2. 计算统计量
```bash
python scripts/utils/compute_stats.py --scenario desert
```

### 3. 训练模型
```bash
python scripts/train/train_single_scenario.py \
    --scenario desert \
    --config configs/desert_config.yaml \
    --epochs 100
```

### 4. 测试模型
```bash
python scripts/test/test_anomaly_detection.py \
    --model checkpoints/desert/best_model.pth \
    --test_data data/raw/desert/test
```

## 📊 论文实验

### 实验1：掩码率对比
```bash
python scripts/train/train_cross_validation.py \
    --mask_ratios 0.5 0.6 0.7 0.75 0.8 0.9
```

### 实验2：SNR vs 误差
```bash
python scripts/test/test_localization.py \
    --snr_values -10 -5 0 5 10 15 20
```

### 实验3：多场景泛化
```bash
python scripts/train/train_multi_scenario.py \
    --scenarios desert forest ocean
```

## 🎨 论文图表

| 图表 | 脚本 | 输出 |
|------|------|------|
| Fig 3A | `test_anomaly_detection.py` | `fig3a_anomaly_detection.pdf` |
| Fig 3B | `test_localization.py` | `fig3b_localization_comparison.pdf` |
| Fig 4 | `train_cross_validation.py` | `fig4_mask_ratio.pdf` |
| Fig 5 | `test_localization.py` | `fig5_snr_vs_error.pdf` |

## 📚 参考资源

### 基线代码
```bash
E:\model_train_example\AudioMAE-main\          # AudioMAE
E:\model_train_example\ast-master\             # AST
E:\model_train_example\dcase2023_task2_baseline_ae-main\  # DCASE
```

### 关键文件
```bash
AudioMAE-main/models_mae.py                    # MAE架构
ast-master/src/models/ast_models.py            # AST模型
dcase2023_task2_baseline_ae-main/baseline.yaml # DCASE配置
```

## 🔍 常用命令

### 查看项目结构
```bash
cd E:\model_train_example\Spec_Mae
cat PROJECT_STRUCTURE.md
```

### 查看TODO
```bash
cat TODO.md
```

### 查看数据说明
```bash
cat data/README.md
```

### 查看模型说明
```bash
cat models/README.md
```

## ⚠️ 重要提醒

1. **数据集较多**：确保有足够存储空间（建议>50GB）
2. **命名规范**：严格遵循命名规范，便于批处理
3. **实验记录**：每次实验都要记录配置和结果
4. **版本控制**：及时commit代码
5. **备份数据**：定期备份重要文件

## 🆘 遇到问题？

1. 查看`docs/troubleshooting.md`（待创建）
2. 查看会议记录：`research/会议list.txt`
3. 参考基线代码实现
4. 检查配置文件是否正确

## 📞 下一步

**立即开始：**
1. ✅ 项目结构已创建
2. 📦 整理你的音频数据集
3. 🔧 创建requirements.txt
4. 💻 开始编写工具函数

**预计时间：2-3周完成所有实验**

---

生成时间：2026-03-09
版本：v1.0
