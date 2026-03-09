# 数据集组织说明

## 目录结构

### raw/ - 原始音频数据

#### 场景分类
- **desert/** - 沙漠场景（低混响，风噪为主）
- **forest/** - 森林场景（中等混响，鸟叫、树叶声）
- **ocean/** - 海洋场景（高混响，海浪声）
- **multi_scenario/** - 多场景混合数据

#### 每个场景的子目录

```
<scenario>/
├── train/                        # 训练集（只包含正常噪音）
│   ├── drone_noise/             # 无人机噪音
│   │   ├── hover_*.wav          # 悬停状态
│   │   ├── cruise_*.wav         # 巡航状态
│   │   └── maneuver_*.wav       # 机动状态
│   ├── ambient_noise/           # 环境噪音
│   │   ├── wind_*.wav           # 风噪
│   │   └── background_*.wav     # 背景噪音
│   └── mixed_normal/            # 混合的正常噪音
│       └── normal_*.wav         # 无人机+环境噪音
│
├── test/                         # 测试集
│   ├── normal/                  # 正常样本（无异常信号）
│   │   └── test_normal_*.wav
│   └── anomaly/                 # 异常样本（含人声呼救）
│       ├── test_anomaly_snr_minus10_*.wav
│       ├── test_anomaly_snr_minus5_*.wav
│       ├── test_anomaly_snr_0_*.wav
│       ├── test_anomaly_snr_5_*.wav
│       ├── test_anomaly_snr_10_*.wav
│       ├── test_anomaly_snr_15_*.wav
│       └── test_anomaly_snr_20_*.wav
│
└── validation/                   # 验证集（用于交叉验证）
    ├── fold_1/
    ├── fold_2/
    ├── fold_3/
    ├── fold_4/
    └── fold_5/
```

### processed/ - 预处理后的数据

```
processed/
├── desert/
│   ├── train_logmel.npy         # Log-Mel特征
│   ├── train_labels.npy         # 标签
│   └── train_metadata.json      # 元数据
├── forest/
└── ocean/
```

### stats/ - 数据集统计信息

```
stats/
├── desert_stats.json            # 沙漠场景统计
│   ├── mean: -4.27
│   ├── std: 4.57
│   └── duration: 3600s
├── forest_stats.json
├── ocean_stats.json
└── multi_scenario_stats.json
```

## 音频规格

- **采样率**: 48000 Hz
- **声道数**: 单声道（哨兵模式）/ 8声道（响应者模式）
- **帧长**: 1秒（48000 samples）
- **格式**: WAV (16-bit PCM)

## 数据命名规范

### 训练数据
```
<scenario>_<type>_<state>_<index>.wav

示例:
- desert_drone_hover_0001.wav
- forest_ambient_wind_0042.wav
- ocean_mixed_normal_0123.wav
```

### 测试数据
```
<scenario>_test_<label>_snr_<value>_<index>.wav

示例:
- desert_test_normal_0001.wav
- forest_test_anomaly_snr_5_0042.wav
```

## 数据生成记录

### 合成方法
1. 无人机噪音：从真实录音中提取
2. 环境噪音：从公开数据集获取（需标注来源）
3. 混合方法：使用Python代码按不同SNR混合

### 数据来源
- 无人机噪音：[TODO: 添加来源]
- 沙漠环境音：[TODO: 添加来源]
- 森林环境音：[TODO: 添加来源]
- 人声呼救：[TODO: 添加来源]

## 数据集统计

| 场景 | 训练样本 | 测试样本（正常） | 测试样本（异常） | 总时长 |
|------|----------|------------------|------------------|--------|
| 沙漠 | TBD      | TBD              | TBD              | TBD    |
| 森林 | TBD      | TBD              | TBD              | TBD    |
| 海洋 | TBD      | TBD              | TBD              | TBD    |

## 注意事项

1. **训练集不包含人声**：只用于学习正常噪音流形
2. **测试集包含不同SNR**：评估不同信噪比下的检测性能
3. **交叉验证**：使用5-fold交叉验证选择超参数
4. **数据增强**：可选使用SpecAugment（训练时）
