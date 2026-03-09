# SpecMAE 训练前严谨性改进指南 (Pre-training Rigor & Optimization Guide) - v3.0

本指南综合对标 `ast-master` (MIT)、`AudioMAE-main` (Meta) 及 `DCASE 2023 Task 2` 行业基准，并结合项目内部技术审查意见编写。为确保 `Spec_Mae` 在正式训练前达到最高工业标准并消除所有逻辑隐患，请落实以下改进。

---

## 1. 数据归一化严谨性 (Data Normalization Rigor) —— **[极高优先级]**

**对比基准：** `ast-master` 和 `AudioMAE` 均有显式的 `norm_stats` 硬性检查。
**当前问题：** `AudioConfig` 中的 `norm_mean` (-6.0) 和 `norm_std` (5.0) 仅为占位值。沙漠、海洋等场景的能量分布差异巨大（可达 3-8dB），使用错误参数会导致训练初期的收敛方向失真。
**改进建议：** 
- **操作：** 在启动 `train_single_scenario.py` 前，必须运行 `compute_dataset_stats()`。
- **验证：** 确认计算出的真实 Mean/Std 已写入配置文件并随 checkpoint 导出。

---

## 2. 配置系统工程化 (Experimental Management Rigor) —— **[极高优先级]**

**对比基准：** `DCASE` 基准使用分层 YAML 严格管理所有超参数。
**当前问题：** `configs/` 目录下大量 YAML 文件（如 `desert_config.yaml`）为 **0 字节空文件**。这在多场景实验中会导致参数管理的极度混乱，甚至导致加载了错误默认值而不自知。
**改进建议：**
- **补全配置：** 填充各场景子配置，显式定义该场景的特征提取参数（`fmin/fmax`）和 `mask_ratio`。
- **参数备份：** 脚本启动时必须将当前生效的 `.yaml` 备份至 `out_dir`，确保每一场实验均可完美复现。

---

## 3. 数据合成与标签严谨性 (Data & SNR Rigor) —— **[高优先级]**

**对比基准：** `DCASE 2023 Task 2` 官方基准在混合异常信号时会考虑有效信号段功率。
**当前问题：** `mix_desert_data.py` 使用全局 RMS 计算 SNR。若异常信号（如人声）短促，全局 RMS 会拉低其在混合片段中的表现，导致**实际有效段的瞬时 SNR 远高于标签设定**。
**改进建议：**
- **精细化混合：** 修改 `mix_snr` 函数，支持基于信号活动检测（VAD）或局部能量的功率对齐，确保 SNR 标签在感知上是真实的。

---

## 4. 模型初始化精准对标 (Architecture Rigor) —— **[高优先级]**

**对比基准：** `AudioMAE` 对 Transformer 内部线性层使用 `trunc_normal_(std=0.02)` 以确保权重分布紧凑。
**当前问题：** `Spec_Mae` 统一使用 `xavier_uniform_`。这在 ViT 这种初值敏感的架构中，可能导致深层梯度的方差控制不够精准。
**改进建议：**
- 修改 `encoder.py` 和 `decoder.py` 的 `_init_weights`：
  - 对 `Linear` 层改用 `trunc_normal_(std=0.02)`。
  - 对 `cls_token` 采用更保守的正态分布初始化。

---

## 5. 训练性能与分布式 (Scalability & DDP) —— **[中优先级]**

**对比基准：** `AudioMAE` 原生支持分布式训练（DDP）和梯度累积。
**当前问题：** 目前仅支持单卡。虽然 `prepare_desert_data.py` 预处理解决了重采样瓶颈，但单卡无法在有限显存下模拟 MAE 所需的大 Batch（如 256/512）。
**改进建议：**
- **引入 DDP：** 在训练脚本中加入 `torch.distributed`。
- **梯度累积：** 在单卡环境下通过 `accum_iter` 模拟大 Batch 收敛动态。

---

## 6. 特殊说明：关于数据增强 (Augmentation Note) —— **[已修正]**

**修正逻辑：** 经对标 `AudioMAE` (Meta) 原始论文，在 **自监督预训练阶段** 额外叠加 SpecAugment 会导致可见信息过少，干扰重建任务。
**最终决定：** **撤回** 在预训练阶段引入 SpecAugment 的建议。仅在后续的 **下游分类 Fine-tuning 阶段** 考虑引入该增强。

---

**结论：** 落实上述改进（尤其是前 4 项）后，`Spec_Mae` 将完全对齐顶级开源项目的质量标准，确保您的训练过程不仅“能跑”，而且“科学且严谨”。
