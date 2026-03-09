# SpecMAE 在 M4 Pro MacBook 上训练完整指南

> 目标机器：MacBook Pro M4 Pro 低配版（12核CPU / 18核GPU / 24GB 统一内存）
>
> 最后更新：2026-03-10

---

## 0. 为什么可以用 M4 Pro 跑

SpecMAE-Base 模型参数量 ~86M，单精度下模型本身约 330MB。训练时 batch_size=32 的峰值显存约 2–3GB。M4 Pro 的 24GB 统一内存对 CPU 和 GPU 共享，远超需求。PyTorch 通过 MPS（Metal Performance Shaders）后端支持 Apple Silicon GPU 加速，实测 M4 Pro 的 MPS 训练速度约为 RTX 3060 的 40–60%，跑完全部实验大约需要几个小时。

---

## 1. 环境配置

### 1.1 安装 Python（推荐 3.11 或 3.12）

```bash
# 如果还没装，用 Homebrew
brew install python@3.12

# 或者用 conda / miniforge（推荐，对 Apple Silicon 原生支持更好）
brew install miniforge
conda create -n specmae python=3.12
conda activate specmae
```

### 1.2 安装 PyTorch（MPS 版本）

```bash
# PyTorch 官方从 2.0 起原生支持 Apple Silicon MPS
# 直接 pip 安装即可（不需要指定特殊 index-url）
pip install torch torchvision torchaudio
```

### 1.3 安装项目依赖

```bash
pip install librosa soundfile numpy scipy matplotlib scikit-learn tqdm pyyaml
```

### 1.4 验证 MPS 可用

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
if torch.backends.mps.is_available():
    x = torch.randn(2, 2, device='mps')
    print(f'MPS tensor test: OK ({x.device})')
"
```

预期输出：
```
PyTorch: 2.x.x
MPS available: True
MPS built: True
MPS tensor test: OK (mps:0)
```

如果 `MPS available: False`，说明 PyTorch 版本太旧或不是 ARM 原生版本，需重装。

---

## 2. 拷贝项目到 Mac

### 方式一：直接拷贝整个目录

将 Windows 上的 `E:\model_train_example\Spec_Mae\` 整个文件夹拷贝到 Mac，例如：

```
~/Projects/model_train_example/Spec_Mae/
```

需要拷贝的内容：
```
Spec_Mae/
├── models/          # 模型代码（必须）
├── scripts/         # 训练/测试脚本（必须）
├── configs/         # 配置文件
├── data/desert/     # 沙漠场景数据（必须，约 140MB）
│   ├── train/normal/     (1000 个 WAV)
│   ├── val/normal/       (200 个 WAV)
│   └── test/             (410 个 WAV)
├── results/         # 可以不拷贝，会自动创建
└── checkpoints/     # 可以不拷贝
```

### 方式二：Git / 网盘同步

如果项目已在 Git 管理，直接 clone 即可。data 目录下的 WAV 文件需要单独传输（建议压缩后传，~140MB 左右）。

---

## 3. 代码适配（重要！）

当前代码的 `get_device()` 函数只检测 CUDA，**不识别 MPS**。需要修改以下文件中的设备检测逻辑。

### 3.1 需要修改的文件及位置

以下 4 个文件都有 `get_device()` 函数，改法相同：

- `scripts/train/train_single_scenario.py` — 第 90 行
- `scripts/train/train_cross_validation.py` — 第 89 行
- `scripts/train/train_multi_scenario.py` — 第 96 行
- `scripts/test/test_anomaly_detection.py` — 第 256 行（`main()` 里的 `torch.device(...)` 行）

将所有 `get_device()` 函数统一改为：

```python
def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
```

对于 `test_anomaly_detection.py` 和 `test_full_system.py` 中直接写的 `torch.device("cuda" if ...)` 行，同样改为上述三段式。

### 3.2 AMP（混合精度）在 MPS 上不可用

当前代码中 AMP 相关逻辑已经用 `device.type == "cuda"` 做了保护，MPS 设备会自动走全精度路径，**不需要额外修改**。不要在 Mac 上加 `--amp` 参数。

### 3.3 num_workers 注意事项

macOS 上 `num_workers > 0` 使用 `fork` 可能导致挂起。建议：

```bash
# macOS 上保持默认 --num_workers 0
# 或者最多设为 2-4（如果确认没问题再调大）
```

### 3.4 快速验证修改是否正确

```bash
cd ~/Projects/model_train_example

python -c "
import sys; sys.path.insert(0, '.')
from Spec_Mae.models.specmae import specmae_vit_base_patch16
import torch

device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
print(f'Device: {device}')

model = specmae_vit_base_patch16(mask_ratio=0.75).to(device)
x = torch.randn(4, 1, 128, 101, device=device)
loss, pred, mask = model(x)
print(f'Forward pass OK — loss={loss.item():.4f}, pred={pred.shape}, mask={mask.shape}')

score = model.compute_anomaly_score(x)
print(f'Anomaly score OK — shape={score.shape}, mean={score.mean().item():.4f}')
"
```

如果输出 `Device: mps` 且没有报错，说明模型在 MPS 上运行正常。

---

## 4. 训练命令

### 4.1 工作目录

**所有命令都在项目根目录执行**（即 `Spec_Mae` 的父目录）：

```bash
cd ~/Projects/model_train_example
```

### 4.2 第一步：交叉验证（选最佳 mask_ratio）

```bash
python Spec_Mae/scripts/train/train_cross_validation.py \
    --scenario desert \
    --data_dir Spec_Mae/data/desert \
    --out_dir  Spec_Mae/results/cv_desert \
    --auto_norm \
    --cv_epochs 30 \
    --final_epochs 100 \
    --batch_size 32 \
    --mask_ratios 0.50 0.60 0.70 0.75 0.80 0.90 \
    --num_workers 0
```

**参数说明：**

| 参数 | 值 | 说明 |
|------|-----|------|
| `--auto_norm` | | 自动计算数据集均值/标准差，替代占位值 |
| `--cv_epochs` | 30 | 每个 fold 训练 30 个 epoch（足够区分 mask_ratio） |
| `--final_epochs` | 100 | 选出最佳 mask_ratio 后全量训练 100 epoch |
| `--batch_size` | 32 | M4 Pro 24GB 跑 32 没问题；若内存紧张可降到 16 |
| `--mask_ratios` | 6 个值 | 完整搜索网格 |
| `--num_workers` | 0 | macOS 安全选择 |

**预估时间：**
- 6 个 mask_ratio × 5 folds × 30 epochs = 900 个 epoch 的 CV
- 加上 100 epoch 的 final training
- M4 Pro MPS: 每个 epoch ~10–20 秒 → 总计约 **3–5 小时**
- 如果用 CPU: 每个 epoch ~30–60 秒 → 总计约 **10–15 小时**

### 4.3 第二步：正式训练（如果 CV 结果不满意，可单独跑）

```bash
python Spec_Mae/scripts/train/train_single_scenario.py \
    --scenario desert \
    --data_dir Spec_Mae/data/desert \
    --cv_dir   Spec_Mae/results/cv_desert \
    --out_dir  Spec_Mae/results/train_desert \
    --auto_norm \
    --epochs 200 \
    --batch_size 32 \
    --patience 30 \
    --num_workers 0
```

这一步会自动从 `cv_desert/cv_summary.json` 读取最佳 mask_ratio。也可以用 `--mask_ratio 0.75` 手动指定。

### 4.4 第三步：评估（异常检测指标）

```bash
python Spec_Mae/scripts/test/test_anomaly_detection.py \
    --checkpoint Spec_Mae/results/cv_desert/best_model.pth \
    --data_dir   Spec_Mae/data/desert \
    --out_dir    Spec_Mae/results/eval_desert \
    --num_workers 0
```

输出：AUC、pAUC、F1 及 per-SNR 分解 + ROC 曲线图 + 分数分布图。

### 4.5 第四步：定位测试（GCC-PHAT，纯 NumPy，不需要 GPU）

```bash
python Spec_Mae/scripts/test/test_localization.py \
    --data_dir Spec_Mae/data/desert \
    --out_dir  Spec_Mae/results/localization \
    --n_trials 200
```

### 4.6 第五步：生成论文图表

```bash
python Spec_Mae/scripts/eval/plot_results.py \
    --metrics_json Spec_Mae/results/eval_desert/metrics.json \
    --scores_csv   Spec_Mae/results/eval_desert/anomaly_scores.csv \
    --cv_json      Spec_Mae/results/cv_desert/cv_summary.json \
    --out_dir      Spec_Mae/results/figures
```

---

## 5. MPS 常见问题与解决

### 5.1 "MPS backend out of memory"

M4 Pro 24GB 是 CPU/GPU 共享内存。如果同时开了很多应用，可用内存不够。

解决：
- 关闭 Chrome 等内存大户
- 降低 `--batch_size` 到 16 或 8
- 用 `Activity Monitor` 监控内存使用

### 5.2 MPS 上某些操作不支持（fallback to CPU）

PyTorch MPS 后端仍有少量算子未实现，会自动回退到 CPU。如果看到警告：
```
UserWarning: MPS: no support for ... falling back to CPU
```
这是正常的，不影响训练正确性，只是那个操作会稍慢。

### 5.3 训练速度对比参考

| 设备 | 每 epoch 预估时间 | 总 CV 时间 |
|------|------------------|-----------|
| M4 Pro (MPS) | ~10–20 秒 | 3–5 小时 |
| M4 Pro (CPU only) | ~30–60 秒 | 10–15 小时 |
| RTX 5060 8GB (CUDA) | ~3–8 秒 | 1–2 小时 |

如果 MPS 出问题，可以完全不改代码，直接用 CPU 跑（速度也可接受）。

### 5.4 强制使用 CPU（如果 MPS 有 bug）

如果 MPS 训练出现 NaN 或结果异常，可以临时环境变量禁用：

```bash
PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 python Spec_Mae/scripts/train/...
```

或直接改 `get_device()` 强制返回 `torch.device("cpu")`。

---

## 6. 训练完成后：将结果拷回 Windows

训练完成后，将以下目录拷贝回 Windows 机器：

```
Spec_Mae/results/
├── cv_desert/
│   ├── cv_summary.json      ← mask_ratio 选择结果
│   ├── cv_results.csv        ← 每个 fold 详细数据
│   ├── best_model.pth        ← 最终模型（~1.2GB）
│   ├── norm_stats.json       ← 归一化参数
│   └── training_curve.png    ← 训练曲线
├── eval_desert/              ← 评估结果
├── localization/             ← 定位测试结果
└── figures/                  ← 论文图表
```

最重要的是 `best_model.pth` 和 `cv_summary.json`。

---

## 7. 一键脚本（可选）

将以下内容保存为 `run_all_desert.sh`，放在项目根目录，一次跑完全部流程：

```bash
#!/bin/bash
set -e

echo "=========================================="
echo "SpecMAE Desert Full Pipeline"
echo "=========================================="

cd "$(dirname "$0")"

# Step 1: Cross-validation
echo ""
echo "[Step 1/4] Cross-validation ..."
python Spec_Mae/scripts/train/train_cross_validation.py \
    --scenario desert \
    --data_dir Spec_Mae/data/desert \
    --out_dir  Spec_Mae/results/cv_desert \
    --auto_norm \
    --cv_epochs 30 \
    --final_epochs 100 \
    --batch_size 32 \
    --num_workers 0

# Step 2: Evaluation
echo ""
echo "[Step 2/4] Anomaly detection evaluation ..."
python Spec_Mae/scripts/test/test_anomaly_detection.py \
    --checkpoint Spec_Mae/results/cv_desert/best_model.pth \
    --data_dir   Spec_Mae/data/desert \
    --out_dir    Spec_Mae/results/eval_desert \
    --num_workers 0

# Step 3: Localization
echo ""
echo "[Step 3/4] Localization test ..."
python Spec_Mae/scripts/test/test_localization.py \
    --data_dir Spec_Mae/data/desert \
    --out_dir  Spec_Mae/results/localization \
    --n_trials 200

# Step 4: Plot
echo ""
echo "[Step 4/4] Generating figures ..."
python Spec_Mae/scripts/eval/plot_results.py \
    --metrics_json Spec_Mae/results/eval_desert/metrics.json \
    --scores_csv   Spec_Mae/results/eval_desert/anomaly_scores.csv \
    --cv_json      Spec_Mae/results/cv_desert/cv_summary.json \
    --out_dir      Spec_Mae/results/figures

echo ""
echo "=========================================="
echo "All done! Check Spec_Mae/results/"
echo "=========================================="
```

使用：
```bash
chmod +x run_all_desert.sh
./run_all_desert.sh
```

---

## 8. 检查清单

在 Mac 上开始之前，确认以下所有项：

- [ ] Python 3.11+ 已安装
- [ ] `torch.backends.mps.is_available()` 返回 `True`
- [ ] librosa, soundfile, numpy, scipy, matplotlib, scikit-learn 已安装
- [ ] `Spec_Mae/data/desert/` 已拷贝到 Mac（含 1610 个 WAV 文件）
- [ ] `get_device()` 已修改为支持 MPS（共 4 个文件）
- [ ] 模型 forward pass 验证通过（第 3.4 节的测试命令）
- [ ] 关闭不必要的应用以释放内存
