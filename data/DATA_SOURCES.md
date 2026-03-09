# 数据来源声明

本文档记录Sound-UAV项目中使用的所有音频数据来源，用于论文引用和版权声明。

---

## 1. 无人机噪音数据

### 数据集信息
- **名称**: Drone Sound Classification Dataset
- **来源**: Kaggle
- **链接**: https://www.kaggle.com/datasets/anmishajadi/drone-sound-classification
- **作者**: Anmisha Jadi
- **许可证**: [需要在Kaggle页面确认]

### 数据详情
- **文件数量**: 1332个WAV文件
- **原始采样率**: 16000 Hz
- **转换后采样率**: 48000 Hz
- **声道**: 单声道 (mono)
- **时长**: 约1秒/文件
- **无人机型号**: Bebop, Membo等

### 存放位置
```
data/raw/drone_noise_original/     # 原始16kHz文件（备份）
data/raw/drone_noise_48k/          # 转换后的48kHz文件（使用）
```

### 处理说明
- 使用librosa重采样从16kHz转换为48kHz
- 保持单声道
- 未进行其他预处理

### 论文引用格式
```
Jadi, A. (2023). Drone Sound Classification Dataset.
Kaggle. https://www.kaggle.com/datasets/anmishajadi/drone-sound-classification
```

---

## 2. 人声呼救数据

### 数据集信息
- **名称**: ASVP-ESD (Speech & Non-Speech Emotional Sound)
- **来源**: ASVP-ESD Dataset
- **类型**: 情感语音和非语音声音数据集
- **许可证**: [需要确认具体许可证]

### 数据详情

#### 2.1 儿童哭声 (Child Cry)
- **目录**: `data/raw/human_voice/Child_Cry_400_600Hz/`
- **文件数量**: 1097个WAV文件
- **频率范围**: 400-600 Hz
- **原始采样率**: 16000 Hz
- **转换后采样率**: 48000 Hz（需要转换）
- **时长**: 0.6-5.0秒（不等长）
- **情感类型**: 哭泣、悲伤

#### 2.2 男性呼救声 (Male Rescue)
- **目录**: `data/raw/human_voice/Male_Rescue_100_300Hz/`
- **文件数量**: 822个WAV文件
- **频率范围**: 100-300 Hz
- **原始采样率**: 16000 Hz
- **转换后采样率**: 48000 Hz（需要转换）
- **时长**: 0.8-8.8秒（不等长）
- **情感类型**: 呼救、恐慌

### 总体统计
- **总文件数**: 1919个WAV文件
- **类型**: 儿童哭声 + 男性呼救声
- **原始采样率**: 16000 Hz ⚠️
- **目标采样率**: 48000 Hz
- **时长范围**: 0.6-8.8秒（需要裁剪/填充为1秒）
- **声道**: 单声道 (mono)

### 处理需求
1. **采样率转换**: 16kHz → 48kHz
2. **时长标准化**: 裁剪或填充为1秒
3. **随机选择**: 从1919个文件中随机选择用于测试

### 使用说明
- 用于生成测试集的异常样本
- 按不同SNR混合到背景噪音中
- SNR范围: -10dB 到 +20dB
- 可以分别测试儿童哭声和成人呼救的检测效果

### 论文引用格式
```
ASVP-ESD: Speech & Non-Speech Emotional Sound Dataset.
[需要补充完整引用信息]
```

---

## 3. 环境噪音数据

### 3.1 沙漠环境音
- **目录**: `data/raw/desert_ambient/`
- **状态**: ❌ 待获取
- **需要数量**: 100-200个1秒片段
- **推荐来源**:
  - FreeSound.org (搜索: "desert wind", "sand storm")
  - ESC-50数据集
- **许可证**: 建议使用CC0或CC BY

### 3.2 森林环境音
- **目录**: `data/raw/forest_ambient/`
- **状态**: ❌ 待获取
- **需要数量**: 100-200个1秒片段
- **推荐来源**:
  - FreeSound.org (搜索: "forest ambience", "bird chirping")
  - ESC-50数据集
- **许可证**: 建议使用CC0或CC BY

### 3.3 海洋环境音
- **目录**: `data/raw/ocean_ambient/`
- **状态**: ❌ 待获取
- **需要数量**: 100-200个1秒片段
- **推荐来源**:
  - FreeSound.org (搜索: "ocean waves", "sea ambience")
  - ESC-50数据集
- **许可证**: 建议使用CC0或CC BY

---

## 4. 数据处理流程

### 4.1 无人机噪音处理
```python
# 采样率转换
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/drone_noise_original \
    --output_dir data/raw/drone_noise_48k \
    --target_sr 48000
```

### 4.2 训练数据生成
```python
# 混合无人机噪音 + 环境噪音
python scripts/utils/mix_audio.py \
    --mode train \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient \
    --output_dir data/raw/desert/train/mixed_normal \
    --n_samples 1000
```

### 4.3 测试数据生成
```python
# 混合无人机噪音 + 环境噪音 + 人声（不同SNR）
python scripts/utils/mix_audio.py \
    --mode test \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient \
    --human_dir data/raw/human_voice/Male_Rescue_100_300Hz \
    --output_dir data/raw/desert/test
```

---

## 5. 版权与许可声明

### 使用声明
本项目中使用的所有音频数据仅用于学术研究目的。

### 引用要求
- 无人机噪音数据：必须引用Kaggle数据集
- 人声数据：[待补充来源后确定]
- 环境噪音：根据具体来源的许可证要求

### 数据分发
- 原始数据不包含在本项目代码仓库中
- 用户需要自行从原始来源下载
- 生成的混合数据仅供项目内部使用

---

## 6. 数据统计

### 当前状态（2026-03-09）

| 数据类型 | 文件数量 | 状态 | 来源 |
|---------|---------|------|------|
| 无人机噪音 | 1332 | ✓ 已有 | Kaggle |
| 儿童哭声 | 1097 | ✓ 已有 | [待补充] |
| 男性呼救声 | 822 | ✓ 已有 | [待补充] |
| 沙漠环境音 | 0 | ❌ 缺失 | 待获取 |
| 森林环境音 | 0 | ❌ 缺失 | 待获取 |
| 海洋环境音 | 0 | ❌ 缺失 | 待获取 |

### 数据充足性评估
- **无人机噪音**: ✓ 充足（1332个）
- **人声数据**: ✓ 非常充足（1919个）
- **环境噪音**: ❌ 需要获取

---

## 7. 待办事项

- [ ] 确认人声数据的具体来源和许可证
- [ ] 检查人声数据的采样率和时长
- [ ] 获取沙漠环境音（100个）
- [ ] 获取森林环境音（100个）
- [ ] 获取海洋环境音（100个）
- [ ] 记录所有环境噪音的来源
- [ ] 更新论文中的数据来源章节

---

## 8. 联系方式

如有数据来源相关问题，请联系：
- 项目负责人：[待补充]
- 邮箱：[待补充]

---

最后更新：2026-03-09
更新人：Claude (AI Assistant)
