# 数据状态更新报告

更新时间：2026-03-09
状态：已删除重复文件，已记录数据来源

---

## ✅ 已完成的工作

### 1. 删除重复文件 ✓
```
已删除：data/raw/desert/train/drone_noise/ 中的1332个重复文件
保留：data/raw/drone_noise_original/ 中的1332个原始文件
```

### 2. 记录数据来源 ✓
```
创建文件：data/DATA_SOURCES.md
记录了：
- 无人机噪音来源（Kaggle数据集）
- 人声数据详情（儿童哭声 + 男性呼救声）
- 待获取的环境噪音来源
```

---

## 📊 当前数据状态

### ✅ 已有数据（充足）

| 数据类型 | 位置 | 数量 | 状态 |
|---------|------|------|------|
| **无人机噪音** | `drone_noise_original/` | 1332个 | ✓ 充足 |
| **儿童哭声** | `human_voice/Child_Cry_400_600Hz/` | 1097个 | ✓ 非常充足 |
| **男性呼救声** | `human_voice/Male_Rescue_100_300Hz/` | 822个 | ✓ 非常充足 |

**总计**：3251个音频文件 ✓

### ❌ 缺少的数据

| 数据类型 | 位置 | 需要数量 | 优先级 |
|---------|------|----------|--------|
| **沙漠环境音** | `desert_ambient/` | 100个 | 🔴 高 |
| **森林环境音** | `forest_ambient/` | 100个 | 🟡 中 |
| **海洋环境音** | `ocean_ambient/` | 100个 | 🟡 中 |

---

## ⚠️ 需要处理的问题

### 问题1：采样率转换（必须）
```
当前：无人机噪音是16kHz
需要：转换为48kHz

解决方案：
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/drone_noise_original \
    --output_dir data/raw/drone_noise_48k \
    --target_sr 48000
```

### 问题2：人声数据检查（建议）
```
需要检查：
- 采样率是多少？
- 文件长度是多少？
- 是否需要预处理？

检查命令：
python scripts/utils/convert_sample_rate.py \
    --mode check \
    --input_dir data/raw/human_voice/Child_Cry_400_600Hz
```

### 问题3：人声数据来源（待补充）
```
需要补充：
- Child_Cry_400_600Hz 的具体来源
- Male_Rescue_100_300Hz 的具体来源
- 许可证信息

请在 data/DATA_SOURCES.md 中补充
```

---

## 🎯 下一步行动计划

### 阶段1：准备现有数据（今天）

```bash
# 步骤1：转换无人机噪音采样率（约30分钟）
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/drone_noise_original \
    --output_dir data/raw/drone_noise_48k \
    --target_sr 48000

# 步骤2：检查人声数据（约5分钟）
python scripts/utils/convert_sample_rate.py \
    --mode check \
    --input_dir data/raw/human_voice/Child_Cry_400_600Hz

python scripts/utils/convert_sample_rate.py \
    --mode check \
    --input_dir data/raw/human_voice/Male_Rescue_100_300Hz

# 步骤3：如果人声不是48kHz，也需要转换
# （根据检查结果决定）
```

### 阶段2：获取环境噪音（1-2天）

```bash
# 选项A：从FreeSound.org下载
# 1. 访问 https://freesound.org
# 2. 搜索 "desert wind"
# 3. 筛选：License = CC0 或 CC BY
# 4. 下载100个音频文件
# 5. 放到 data/raw/desert_ambient/

# 选项B：使用ESC-50数据集
# 1. 下载 ESC-50
# 2. 提取相关类别的音频
# 3. 切分为1秒片段
```

### 阶段3：生成训练和测试数据（1天）

```bash
# 生成沙漠场景训练数据
python scripts/utils/mix_audio.py \
    --mode train \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient \
    --output_dir data/raw/desert/train/mixed_normal \
    --n_samples 1000

# 生成沙漠场景测试数据
python scripts/utils/mix_audio.py \
    --mode test \
    --scenario desert \
    --drone_dir data/raw/drone_noise_48k \
    --ambient_dir data/raw/desert_ambient \
    --human_dir data/raw/human_voice/Male_Rescue_100_300Hz \
    --output_dir data/raw/desert/test
```

---

## 📋 数据充足性评估

### 无人机噪音 ✓
```
已有：1332个
需要：500-1000个（单场景）
评估：✓ 充足，可以支持多场景实验
```

### 人声数据 ✓✓
```
已有：1919个（1097儿童 + 822男性）
需要：10-20个
评估：✓✓ 非常充足，远超需求
优势：
- 数量充足，可以增加测试集多样性
- 包含不同性别和年龄，测试泛化能力
- 可以分别测试儿童哭声和成人呼救的检测效果
```

### 环境噪音 ❌
```
已有：0个
需要：100个（单场景）/ 300个（三场景）
评估：❌ 缺失，需要获取
优先级：🔴 高（阻塞训练）
```

---

## 💡 数据使用建议

### 建议1：充分利用人声数据多样性
```python
# 可以分别测试两种人声的检测效果
# 测试1：使用儿童哭声
python scripts/utils/mix_audio.py \
    --human_dir data/raw/human_voice/Child_Cry_400_600Hz \
    ...

# 测试2：使用男性呼救声
python scripts/utils/mix_audio.py \
    --human_dir data/raw/human_voice/Male_Rescue_100_300Hz \
    ...

# 测试3：混合使用（随机选择）
# 需要修改mix_audio.py支持多个人声目录
```

### 建议2：分阶段实验
```
阶段1：单场景（沙漠）+ 单人声（男性）
- 快速验证模型架构
- 需要数据：无人机噪音 + 沙漠环境音 + 男性呼救声

阶段2：单场景（沙漠）+ 多人声（男性+儿童）
- 测试泛化能力
- 需要数据：同上 + 儿童哭声

阶段3：多场景 + 多人声
- 完整实验
- 需要数据：所有数据
```

### 建议3：数据增强
```
由于人声数据非常充足（1919个），可以：
1. 增加测试集的样本数量
2. 测试更多的SNR级别
3. 测试不同的混合比例
4. 进行消融实验（ablation study）
```

---

## 📝 待补充信息

请补充以下信息到 `data/DATA_SOURCES.md`：

1. **人声数据来源**：
   - Child_Cry_400_600Hz 是从哪里获取的？
   - Male_Rescue_100_300Hz 是从哪里获取的？
   - 有许可证信息吗？

2. **人声数据详情**：
   - 采样率是多少？
   - 文件长度是多少？
   - 是否需要预处理？

---

## 🎉 总结

### 已完成 ✓
- [x] 删除重复的无人机噪音文件
- [x] 记录数据来源（Kaggle数据集）
- [x] 确认人声数据充足（1919个）

### 进行中 🔄
- [ ] 转换无人机噪音采样率（16kHz → 48kHz）
- [ ] 检查人声数据属性

### 待完成 ❌
- [ ] 获取沙漠环境音（100个）
- [ ] 获取森林环境音（100个）
- [ ] 获取海洋环境音（100个）
- [ ] 补充人声数据来源信息

### 优先级排序
1. 🔴 转换无人机噪音采样率（阻塞）
2. 🔴 获取沙漠环境音（阻塞训练）
3. 🟡 检查人声数据属性
4. 🟡 获取森林/海洋环境音
5. 🟢 补充数据来源信息

---

**下一步：立即转换无人机噪音采样率！**

```bash
python scripts/utils/convert_sample_rate.py \
    --mode convert \
    --input_dir data/raw/drone_noise_original \
    --output_dir data/raw/drone_noise_48k \
    --target_sr 48000
```

需要我帮你运行这个命令吗？
