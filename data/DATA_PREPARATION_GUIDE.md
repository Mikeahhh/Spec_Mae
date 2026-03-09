# 数据准备指南 - 针对"大量1秒无人机噪音"

## ✅ 你的数据情况

你有：**大量1秒的纯粹无人机噪音**

这是**完美的**！因为：
- 论文设计的输入长度就是1秒（48000 samples @ 48kHz）
- 1秒音频 → 128×100 Log-Mel → 切分成16×16 patches
- 不需要再切分或拼接

---

## 📋 数据准备清单

### 第一步：整理你的无人机噪音

```bash
# 1. 将你的无人机噪音文件放到临时目录
mkdir -p data/raw/drone_noise_original

# 2. 检查文件
# - 采样率是48kHz吗？
# - 是单声道吗？
# - 是1秒长度吗？

# 3. 如果需要转换，使用ffmpeg或sox
# 示例：转换为48kHz单声道
for file in data/raw/drone_noise_original/*.wav; do
    ffmpeg -i "$file" -ar 48000 -ac 1 "data/raw/drone_noise_converted/$(basename $file)"
done
```

### 第二步：获取环境噪音

你需要获取环境噪音（也是1秒片段）：

#### 选项A：使用公开数据集（推荐）

**1. FreeSound.org**
```
搜索关键词：
- "desert wind" - 沙漠风声
- "forest ambience" - 森林环境音
- "ocean waves" - 海浪声
- "bird chirping" - 鸟叫声

下载后切分成1秒片段
```

**2. ESC-50数据集**
```bash
# 包含50类环境声音
# 下载地址：https://github.com/karolpiczak/ESC-50
# 已经是5秒片段，可以切分成5个1秒片段
```

**3. AudioSet**
```
# 包含大量环境声音
# 但需要从YouTube下载，比较麻烦
```

#### 选项B：录制或生成

```python
# 使用白噪声模拟风声
import numpy as np
import soundfile as sf

sr = 48000
duration = 1.0
white_noise = np.random.randn(int(sr * duration))
# 低通滤波模拟风声
# ...
sf.write("wind_noise.wav", white_noise, sr)
```

### 第三步：获取人声呼救

你需要少量人声（用于生成测试集）：

```
需要的人声类型：
- 呼救声："Help!", "救命!"
- 尖叫声
- 呼喊声

数量：10-20个不同的1秒片段即可
来源：
- 录制
- FreeSound.org搜索"scream", "help", "shout"
- 电影音效库
```

---

## 🛠️ 使用混合脚本生成数据

我已经创建了`scripts/utils/mix_audio.py`脚本。

### 生成训练数据

```bash
# 沙漠场景训练数据
python scripts/utils/mix_audio.py \
    --mode train \
    --scenario desert \
    --drone_dir data/raw/drone_noise_converted \
    --ambient_dir data/raw/desert_ambient \
    --output_dir data/raw/desert/train/mixed_normal \
    --n_samples 1000

# 森林场景训练数据
python scripts/utils/mix_audio.py \
    --mode train \
    --scenario forest \
    --drone_dir data/raw/drone_noise_converted \
    --ambient_dir data/raw/forest_ambient \
    --output_dir data/raw/forest/train/mixed_normal \
    --n_samples 1000
```

### 生成测试数据

```bash
# 沙漠场景测试数据
python scripts/utils/mix_audio.py \
    --mode test \
    --scenario desert \
    --drone_dir data/raw/drone_noise_converted \
    --ambient_dir data/raw/desert_ambient \
    --human_dir data/raw/human_voice \
    --output_dir data/raw/desert/test

# 这会生成：
# - 100个正常样本（无人机+环境噪音）
# - 105个异常样本（7个SNR × 15个样本）
```

---

## 📊 最小数据量需求

### 快速开始（单场景）

```
沙漠场景：
├── 无人机噪音：你已有（假设500个1秒片段）
├── 沙漠环境音：需要获取（100个1秒片段）
├── 人声呼救：需要获取（10个1秒片段）
└── 生成：
    ├── 训练集：500个混合样本
    └── 测试集：100个正常 + 70个异常

总计需要：
- 无人机噪音：500个 ✓（你已有）
- 环境噪音：100个
- 人声：10个
```

### 完整实验（三场景）

```
三个场景（沙漠、森林、海洋）：
├── 无人机噪音：你已有（共享，1500个）
├── 沙漠环境音：100个
├── 森林环境音：100个
├── 海洋环境音：100个
├── 人声呼救：10个（共享）
└── 生成：
    ├── 训练集：每个场景1000个 = 3000个
    └── 测试集：每个场景170个 = 510个

总计需要：
- 无人机噪音：1500个 ✓（你已有）
- 环境噪音：300个（3种×100）
- 人声：10个
```

---

## 🎯 推荐的数据准备流程

### 阶段1：最小可行数据（1-2天）

```bash
# 1. 整理你的无人机噪音
# 选择500个质量好的1秒片段
cp data/raw/drone_noise_original/*.wav data/raw/drone_noise_selected/

# 2. 获取沙漠环境音
# 从FreeSound下载100个1秒风声片段
# 放到 data/raw/desert_ambient/

# 3. 获取人声
# 录制或下载10个1秒呼救声
# 放到 data/raw/human_voice/

# 4. 生成训练和测试数据
python scripts/utils/mix_audio.py --mode train ...
python scripts/utils/mix_audio.py --mode test ...

# 5. 开始训练！
```

### 阶段2：扩展到多场景（3-5天）

```bash
# 1. 获取森林环境音
# 下载鸟叫、树叶声等

# 2. 获取海洋环境音
# 下载海浪声

# 3. 生成多场景数据
# 重复阶段1的步骤

# 4. 多场景训练
```

---

## 📝 数据来源记录模板

**重要**：论文需要声明数据来源！

创建文件：`data/DATA_SOURCES.md`

```markdown
# 数据来源声明

## 无人机噪音
- 来源：[真实录音/数据集名称/其他]
- 数量：[X]个1秒片段
- 采样率：48kHz
- 录音设备：[设备型号]
- 录音环境：[室内/室外]
- 许可证：[许可证类型]

## 环境噪音
### 沙漠环境音
- 来源：FreeSound.org
- 文件ID：[列出文件ID]
- 许可证：CC BY 4.0
- 处理：切分为1秒片段

### 森林环境音
- 来源：[...]
- ...

## 人声数据
- 来源：[录制/数据集]
- 数量：[X]个1秒片段
- 内容：呼救声、尖叫声
- 许可证：[...]
```

---

## 🔍 数据质量检查

生成数据后，运行质量检查：

```python
# 创建检查脚本：scripts/utils/check_data_quality.py

import librosa
import numpy as np
from pathlib import Path

def check_audio_file(file_path):
    """检查单个音频文件"""
    audio, sr = librosa.load(file_path, sr=None)

    # 检查采样率
    assert sr == 48000, f"Wrong sample rate: {sr}"

    # 检查长度
    duration = len(audio) / sr
    assert 0.99 <= duration <= 1.01, f"Wrong duration: {duration}"

    # 检查是否有削波
    max_val = np.abs(audio).max()
    if max_val > 0.99:
        print(f"Warning: Clipping detected in {file_path}")

    # 检查是否静音
    rms = np.sqrt(np.mean(audio ** 2))
    if rms < 1e-5:
        print(f"Warning: Silent audio in {file_path}")

    return True

# 批量检查
data_dir = Path("data/raw/desert/train/mixed_normal")
for file in data_dir.glob("*.wav"):
    try:
        check_audio_file(file)
    except Exception as e:
        print(f"Error in {file}: {e}")
```

---

## ❓ 常见问题

### Q1: 我的无人机噪音不是48kHz怎么办？
```bash
# 使用ffmpeg转换
ffmpeg -i input.wav -ar 48000 output.wav
```

### Q2: 我的无人机噪音不是1秒怎么办？
```python
# 如果更长：随机裁剪1秒
# 如果更短：循环填充或丢弃

# 使用mix_audio.py中的pad_or_trim函数会自动处理
```

### Q3: 我需要多少个无人机噪音片段？
```
最少：500个（单场景快速实验）
推荐：1000-1500个（单场景完整实验）
理想：3000+个（多场景实验）

你说有"大量"，应该足够了！
```

### Q4: 环境噪音和无人机噪音的比例？
```python
# 在mix_audio.py中已设置：
drone_audio = normalize_audio(drone_audio, target_db=-20.0)
ambient_audio = normalize_audio(ambient_audio, target_db=-25.0)
mixed = drone_audio + ambient_audio * 0.5

# 环境噪音比无人机噪音弱约6dB
# 可以根据实际情况调整
```

### Q5: 测试集的SNR值为什么是-10到20dB？
```
SNR = -10dB：人声很弱（困难）
SNR = 0dB：人声和噪音相当
SNR = 20dB：人声很强（容易）

这个范围可以测试模型在不同信噪比下的鲁棒性
```

---

## 🚀 下一步

1. **立即行动**：
   - 告诉我你的无人机噪音在哪里
   - 告诉我大约有多少个1秒片段
   - 我帮你规划具体的数据准备步骤

2. **获取环境噪音**：
   - 我可以帮你找合适的公开数据集
   - 或者帮你编写音频切分脚本

3. **开始生成数据**：
   - 使用`mix_audio.py`脚本
   - 我可以帮你调试

需要我帮你做什么？
