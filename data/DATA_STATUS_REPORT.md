# 数据目录检查报告

生成时间：2026-03-09

---

## ✅ 已有数据

### 1. 无人机噪音（已有）
```
位置：data/raw/desert/train/drone_noise/
      data/raw/drone_noise_original/

文件数量：1332个WAV文件
文件格式：RIFF WAVE, 16-bit PCM, mono
采样率：16000 Hz ⚠️ 需要转换为48000 Hz
长度：约1秒（需要验证）

文件命名示例：
- B_S2_D1_067-bebop_000_.wav
- B_S2_D1_068-bebop_001_.wav
- mixed_membo_9-membo_003_.wav

状态：✓ 数据充足（1332个片段）
问题：⚠️ 采样率是16kHz，需要转换为48kHz
```

### 2. 人声数据（部分）
```
位置：data/raw/human_voice/

文件数量：2个文件（README.md, QUICK_START.md）
实际音频：0个 ⚠️

状态：❌ 缺少人声音频文件
需要：10-20个1秒的呼救声/尖叫声WAV文件
```

---

## ❌ 缺少的数据

### 1. 环境噪音（全部缺失）

#### 沙漠环境音
```
位置：data/raw/desert_ambient/
状态：❌ 空目录
需要：100-200个1秒的沙漠环境音（风声、沙尘）
```

#### 森林环境音
```
位置：data/raw/forest_ambient/
状态：❌ 空目录
需要：100-200个1秒的森林环境音（鸟叫、树叶、风声）
```

#### 海洋环境音
```
位置：data/raw/ocean_ambient/
状态：❌ 空目录
需要：100-200个1秒的海洋环境音（海浪、海风）
```

---

## 🔄 重复的目录（可以删除）

### 重复1：无人机噪音
```
data/raw/desert/train/drone_noise/     ← 1332个文件
data/raw/drone_noise_original/         ← 1332个文件（重复）

建议：
- 保留 drone_noise_original/ 作为原始备份
- desert/train/drone_noise/ 中的文件可以删除
- 或者反过来，看你的偏好
```

### 重复2：场景目录结构
```
data/raw/desert/train/drone_noise/     ← 有数据
data/raw/desert/train/ambient_noise/   ← 空
data/raw/desert/train/mixed_normal/    ← 空

data/raw/forest/train/...              ← 全空
data/raw/ocean/train/...               ← 全空

建议：
- 这些空目录保留（后续会用到）
- 不算重复，只是还没填充数据
```

---

## ⚠️ 需要处理的问题

### 问题1：采样率不匹配
```
当前：16000 Hz
需要：48000 Hz

解决方案：
使用ffmpeg批量转换
```

### 问题2：文件位置混乱
```
无人机噪音同时存在于两个位置：
- data/raw/desert/train/drone_noise/
- data/raw/drone_noise_original/

建议整理：
- 原始文件统一放在 drone_noise_original/
- 转换后的文件放在各场景的 train/drone_noise/
```

---

## 📋 数据准备优先级

### 🔴 高优先级（立即需要）

1. **转换无人机噪音采样率**
   ```bash
   # 从16kHz转换为48kHz
   # 需要处理1332个文件
   ```

2. **获取沙漠环境音**
   ```
   需要：100个1秒片段
   来源：FreeSound.org 或其他
   ```

3. **获取人声呼救**
   ```
   需要：10-20个1秒片段
   来源：FreeSound.org 或自己录制
   ```

### 🟡 中优先级（单场景实验后）

4. **获取森林环境音**
   ```
   需要：100个1秒片段
   ```

5. **获取海洋环境音**
   ```
   需要：100个1秒片段
   ```

### 🟢 低优先级（可选）

6. **清理重复文件**
   ```
   删除重复的无人机噪音
   ```

7. **组织文件结构**
   ```
   统一命名规范
   ```

---

## 🎯 推荐的行动计划

### 阶段1：准备最小可行数据（1-2天）

```bash
# 步骤1：转换无人机噪音采样率
cd data/raw/drone_noise_original
mkdir ../drone_noise_48k

for file in *.wav; do
    ffmpeg -i "$file" -ar 48000 "../drone_noise_48k/${file}"
done

# 步骤2：获取沙漠环境音
# 从FreeSound.org下载100个风声片段
# 放到 data/raw/desert_ambient/

# 步骤3：获取人声
# 下载或录制10个呼救声
# 放到 data/raw/human_voice/

# 步骤4：生成训练数据
python scripts/utils/mix_audio.py \
    --mode train \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient \
    --output_dir data/raw/desert/train/mixed_normal \
    --n_samples 1000

# 步骤5：生成测试数据
python scripts/utils/mix_audio.py \
    --mode test \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient \
    --human_dir data/raw/human_voice \
    --output_dir data/raw/desert/test
```

### 阶段2：扩展到多场景（3-5天）

```bash
# 重复阶段1的步骤，针对森林和海洋场景
```

---

## 📊 数据量估算

### 当前状态
```
✓ 无人机噪音：1332个（充足）
✓ 沙漠环境音：0个（需要100个）
✓ 森林环境音：0个（需要100个）
✓ 海洋环境音：0个（需要100个）
✓ 人声呼救：0个（需要10-20个）
```

### 最小可行实验（单场景）
```
需要准备：
- 无人机噪音：500个（从1332个中选择）✓
- 沙漠环境音：100个 ❌
- 人声呼救：10个 ❌

生成：
- 训练集：500个混合样本
- 测试集：100个正常 + 105个异常
```

### 完整实验（三场景）
```
需要准备：
- 无人机噪音：1332个 ✓
- 沙漠环境音：100个 ❌
- 森林环境音：100个 ❌
- 海洋环境音：100个 ❌
- 人声呼救：10个 ❌

生成：
- 训练集：3000个混合样本
- 测试集：600个（每场景200个）
```

---

## 🔧 需要的工具脚本

### 已创建 ✓
- `scripts/utils/mix_audio.py` - 音频混合脚本

### 需要创建
- `scripts/utils/convert_sample_rate.py` - 批量转换采样率
- `scripts/utils/check_audio_quality.py` - 检查音频质量
- `scripts/utils/organize_files.py` - 整理文件结构

---

## 📝 数据来源记录

### 无人机噪音
```
来源：[需要你补充]
文件数量：1332个
原始采样率：16kHz
文件命名：bebop, membo等型号
```

### 需要记录的信息
1. 无人机噪音的来源（数据集名称/录音设备）
2. 录音环境（室内/室外）
3. 无人机型号
4. 许可证信息

---

## ❓ 需要你回答的问题

1. **无人机噪音来源**：
   - 这些数据是从哪里获取的？
   - 是什么数据集？
   - 有许可证信息吗？

2. **文件位置**：
   - 你想保留哪个目录的无人机噪音？
   - drone_noise_original/ 还是 desert/train/drone_noise/？

3. **场景选择**：
   - 你想先做单场景（沙漠）还是直接做三场景？
   - 建议先做单场景，快速验证

4. **数据获取**：
   - 需要我帮你找环境噪音数据集吗？
   - 需要我帮你编写采样率转换脚本吗？

---

## 🚀 下一步建议

**立即行动：**

1. **回答上面的问题**
2. **转换采样率**（我可以帮你写脚本）
3. **获取沙漠环境音**（我可以推荐数据集）
4. **获取人声数据**（我可以提供链接）

**然后：**
- 生成训练和测试数据
- 开始训练模型
- 运行实验

需要我帮你做什么？
