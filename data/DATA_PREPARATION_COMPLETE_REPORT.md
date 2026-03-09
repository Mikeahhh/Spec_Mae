# 数据准备完整报告

更新时间：2026-03-09
状态：已确认所有数据来源，准备开始数据转换

---

## ✅ 数据来源确认

### 1. 无人机噪音 ✓
- **来源**: Kaggle - Drone Sound Classification Dataset
- **链接**: https://www.kaggle.com/datasets/anmishajadi/drone-sound-classification
- **数量**: 1332个文件
- **采样率**: 16kHz → 需要转换为48kHz

### 2. 人声数据 ✓
- **来源**: ASVP-ESD (Speech & Non-Speech Emotional Sound)
- **数量**: 1919个文件（1097儿童 + 822男性）
- **采样率**: 16kHz → 需要转换为48kHz
- **时长**: 0.6-8.8秒 → 需要标准化为1秒

### 3. 环境噪音 ❌
- **沙漠环境音**: 待获取（100个）
- **森林环境音**: 待获取（100个）
- **海洋环境音**: 待获取（100个）

---

## ⚠️ 关键发现

### 发现1：所有数据都是16kHz
```
无人机噪音：16kHz ⚠️
人声数据：16kHz ⚠️
目标采样率：48kHz

结论：需要批量转换所有数据的采样率
```

### 发现2：人声数据时长不一致
```
儿童哭声：0.6-5.0秒
男性呼救：0.8-8.8秒
目标时长：1秒

结论：mix_audio.py中的pad_or_trim函数会自动处理
```

### 发现3：数据量非常充足
```
无人机噪音：1332个（需要500-1000个）✓✓
人声数据：1919个（需要10-20个）✓✓✓
环境噪音：0个（需要100个）❌

结论：只缺环境噪音，其他数据充足
```

---

## 🎯 数据转换计划

### 阶段1：转换无人机噪音（必须，约30分钟）

```bash
# 步骤1：检查原始采样率
python scripts/utils/convert_sample_rate.py \
    --mode check \
    --input_dir data/raw/drone_noise_original

# 步骤2：转换为48kHz
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/drone_noise_original \
    --output_dir data/raw/drone_noise_48k \
    --target_sr 48000

# 预计时间：约30分钟（1332个文件）
# 输出：data/raw/drone_noise_48k/ 目录下1332个48kHz文件
```

### 阶段2：转换人声数据（必须，约1小时）

```bash
# 步骤1：转换儿童哭声
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/human_voice/Child_Cry_400_600Hz \
    --output_dir data/raw/human_voice_48k/Child_Cry \
    --target_sr 48000

# 步骤2：转换男性呼救声
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/human_voice/Male_Rescue_100_300Hz \
    --output_dir data/raw/human_voice_48k/Male_Rescue \
    --target_sr 48000

# 预计时间：约1小时（1919个文件）
# 输出：data/raw/human_voice_48k/ 目录下1919个48kHz文件
```

### 阶段3：获取环境噪音（1-2天）

```bash
# 选项A：从FreeSound.org手动下载
# 1. 访问 https://freesound.org/search/?q=desert+wind
# 2. 筛选：License = CC0 或 CC BY
# 3. 下载100个音频文件
# 4. 放到 data/raw/desert_ambient/

# 选项B：使用ESC-50数据集
# 1. 下载 ESC-50: https://github.com/karolpiczak/ESC-50
# 2. 提取相关类别
# 3. 切分为1秒片段
```

---

## 📊 转换后的数据结构

```
data/raw/
├── drone_noise_original/          # 原始16kHz（备份）
│   └── *.wav (1332个)
│
├── drone_noise_48k/               # 转换后48kHz（使用）
│   └── *.wav (1332个)
│
├── human_voice/                   # 原始16kHz（备份）
│   ├── Child_Cry_400_600Hz/
│   │   └── *.wav (1097个)
│   └── Male_Rescue_100_300Hz/
│       └── *.wav (822个)
│
├── human_voice_48k/               # 转换后48kHz（使用）
│   ├── Child_Cry/
│   │   └── *.wav (1097个)
│   └── Male_Rescue/
│       └── *.wav (822个)
│
└── desert_ambient/                # 环境噪音（待获取）
    └── *.wav (100个)
```

---

## 🚀 完整执行流程

### Day 1：数据转换（今天）

```bash
# 上午：转换无人机噪音（30分钟）
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/drone_noise_original \
    --output_dir data/raw/drone_noise_48k \
    --target_sr 48000

# 下午：转换人声数据（1小时）
# 儿童哭声
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/human_voice/Child_Cry_400_600Hz \
    --output_dir data/raw/human_voice_48k/Child_Cry \
    --target_sr 48000

# 男性呼救声
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/human_voice/Male_Rescue_100_300Hz \
    --output_dir data/raw/human_voice_48k/Male_Rescue \
    --target_sr 48000

# 验证转换结果
python scripts/utils/convert_sample_rate.py \
    --mode check \
    --input_dir data/raw/drone_noise_48k

python scripts/utils/convert_sample_rate.py \
    --mode check \
    --input_dir data/raw/human_voice_48k/Child_Cry
```

### Day 2：获取环境噪音

```bash
# 从FreeSound.org下载100个沙漠环境音
# 放到 data/raw/desert_ambient/

# 如果需要转换采样率
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/desert_ambient \
    --output_dir data/raw/desert_ambient_48k \
    --target_sr 48000
```

### Day 3：生成训练和测试数据

```bash
# 生成训练数据（1000个样本）
python scripts/utils/mix_audio.py \
    --mode train \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient_48k \
    --output_dir data/raw/desert/train/mixed_normal \
    --n_samples 1000

# 生成测试数据（使用男性呼救声）
python scripts/utils/mix_audio.py \
    --mode test \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient_48k \
    --human_dir data/raw/human_voice_48k/Male_Rescue \
    --output_dir data/raw/desert/test

# 可选：生成测试数据（使用儿童哭声）
python scripts/utils/mix_audio.py \
    --mode test \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient_48k \
    --human_dir data/raw/human_voice_48k/Child_Cry \
    --output_dir data/raw/desert/test_child
```

---

## 📋 数据质量检查清单

### 转换后检查

- [ ] 无人机噪音：1332个文件，全部48kHz
- [ ] 儿童哭声：1097个文件，全部48kHz
- [ ] 男性呼救：822个文件，全部48kHz
- [ ] 沙漠环境音：100个文件，全部48kHz

### 生成数据检查

- [ ] 训练集：1000个混合样本，全部1秒，48kHz
- [ ] 测试集（正常）：100个样本
- [ ] 测试集（异常）：105个样本（7个SNR × 15个）

### 音频质量检查

- [ ] 无削波（max < 0.99）
- [ ] 无静音（RMS > 1e-5）
- [ ] 时长准确（1.0秒 ± 0.01秒）

---

## 💾 存储空间估算

### 原始数据（16kHz）
```
无人机噪音：1332个 × ~16KB = ~21MB
人声数据：1919个 × ~32KB = ~61MB
总计：~82MB
```

### 转换后数据（48kHz）
```
无人机噪音：1332个 × ~48KB = ~64MB
人声数据：1919个 × ~96KB = ~184MB
总计：~248MB
```

### 生成的混合数据
```
训练集：1000个 × ~96KB = ~96MB
测试集：205个 × ~96KB = ~20MB
总计：~116MB
```

### 总存储需求
```
原始数据：~82MB
转换数据：~248MB
生成数据：~116MB
总计：~446MB（约0.5GB）

建议：预留1GB空间
```

---

## 📝 论文数据章节草稿

### Data Collection

We collected audio data from three sources:

1. **Drone Noise**: We used the Drone Sound Classification Dataset from Kaggle [1], containing 1,332 one-second audio clips of various drone models (Bebop, Membo, etc.) recorded at 16 kHz.

2. **Human Distress Signals**: We extracted 1,919 audio clips from the ASVP-ESD (Speech & Non-Speech Emotional Sound) dataset [2], including 1,097 child crying sounds (400-600 Hz) and 822 male rescue calls (100-300 Hz), originally sampled at 16 kHz.

3. **Environmental Noise**: We collected 100 one-second ambient sound clips for each scenario (desert, forest, ocean) from FreeSound.org [3], licensed under CC0 or CC BY.

### Data Preprocessing

All audio data was resampled to 48 kHz using librosa [4] to match the UAV's onboard microphone sampling rate. Human voice clips with varying durations (0.6-8.8 seconds) were randomly cropped or zero-padded to 1 second. We generated training samples by mixing drone noise with environmental noise, and test samples by further mixing human distress signals at various SNR levels (-10 to +20 dB).

### References
[1] Jadi, A. (2023). Drone Sound Classification Dataset. Kaggle.
[2] ASVP-ESD: Speech & Non-Speech Emotional Sound Dataset.
[3] FreeSound.org. https://freesound.org
[4] McFee, B., et al. (2015). librosa: Audio and music signal analysis in python.

---

## ✅ 完成状态

### 已完成 ✓
- [x] 确认无人机噪音来源（Kaggle）
- [x] 确认人声数据来源（ASVP-ESD）
- [x] 检查所有数据的采样率（16kHz）
- [x] 检查人声数据的时长（0.6-8.8秒）
- [x] 删除重复文件
- [x] 创建数据来源文档

### 进行中 🔄
- [ ] 转换无人机噪音采样率
- [ ] 转换人声数据采样率

### 待完成 ❌
- [ ] 获取沙漠环境音
- [ ] 获取森林环境音
- [ ] 获取海洋环境音
- [ ] 生成训练数据
- [ ] 生成测试数据

---

## 🎯 下一步：立即开始转换

**现在就可以开始转换采样率！**

```bash
# 第一步：转换无人机噪音（约30分钟）
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/drone_noise_original \
    --output_dir data/raw/drone_noise_48k \
    --target_sr 48000
```

需要我帮你运行这个命令吗？或者你想先做什么？
