# 模型代码组织说明

## 目录结构

```
models/
├── specmae/                      # SpecMAE核心模型
│   ├── __init__.py              # 模块初始化
│   ├── specmae_model.py         # 主模型类
│   ├── encoder.py               # Transformer Encoder
│   ├── decoder.py               # Transformer Decoder
│   ├── patch_embed.py           # Patch Embedding层
│   └── pos_embed.py             # 位置编码
│
└── baseline/                     # 基线模型（参考用）
    ├── __init__.py
    ├── dcase_ae.py              # DCASE 2023 Autoencoder
    ├── audiomae.py              # AudioMAE参考实现
    └── ast.py                   # AST参考实现
```

## 文件功能说明

### specmae/specmae_model.py
**主模型类，整合所有组件**

功能：
- SpecMAE完整模型定义
- 前向传播逻辑
- 损失计算（只在masked patches上）
- 异常检测接口

关键类：
```python
class SpecMAE(nn.Module):
    def __init__(self, ...):
        # 初始化encoder, decoder, patch_embed等

    def forward(self, x, mask_ratio=0.75):
        # 前向传播

    def compute_loss(self, original, reconstructed, mask):
        # 计算重建损失

    def detect_anomaly(self, x, threshold):
        # 异常检测接口
```

参考：
- AudioMAE的models_mae.py
- 论文Section III

### specmae/encoder.py
**Transformer Encoder实现**

功能：
- 只处理未遮盖的patches（25%）
- Multi-head Self-Attention
- Feed-Forward Network
- Layer Normalization

关键类：
```python
class TransformerEncoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        # 初始化多层Transformer Block

    def forward(self, x, mask):
        # 只处理visible patches
```

参考：
- AudioMAE的encoder实现
- timm库的ViT实现

### specmae/decoder.py
**Transformer Decoder实现**

功能：
- 重建masked patches
- 轻量级设计（比encoder浅）
- 输出重建的spectrogram

关键类：
```python
class TransformerDecoder(nn.Module):
    def __init__(self, embed_dim, depth, num_heads):
        # 初始化decoder layers

    def forward(self, latent_features, mask_tokens):
        # 重建完整spectrogram
```

参考：
- AudioMAE的decoder实现

### specmae/patch_embed.py
**Patch Embedding层**

功能：
- 将Log-Mel spectrogram切分成patches
- 线性投影到embedding空间
- 支持16×16或其他patch size

关键类：
```python
class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim):
        # 初始化卷积层

    def forward(self, x):
        # 输入: [B, 1, F, T] (Log-Mel)
        # 输出: [B, N, D] (Patch embeddings)
```

参考：
- AST的patch embedding
- AudioMAE的PatchEmbed类

### specmae/pos_embed.py
**位置编码**

功能：
- 2D正弦位置编码
- 适配音频的时频特性

关键函数：
```python
def get_2d_sincos_pos_embed(embed_dim, grid_size):
    # 生成2D位置编码
    # 返回: [grid_size*grid_size, embed_dim]
```

参考：
- AudioMAE的pos_embed.py
- AST的位置编码

---

## 基线模型说明

### baseline/dcase_ae.py
**DCASE 2023 Autoencoder参考**

用途：
- 对比实验的基线
- 参考异常检测逻辑
- 参考动态阈值策略

不直接使用，仅作参考。

### baseline/audiomae.py
**AudioMAE参考实现**

用途：
- 参考完整的MAE架构
- 参考预训练策略
- 参考损失函数设计

可以复用部分代码。

### baseline/ast.py
**AST参考实现**

用途：
- 参考音频特征提取
- 参考Patch处理
- 参考数据增强

可以复用特征提取部分。

---

## 模型参数配置

### 推荐配置（轻量级，适合DSP）

```python
# ViT-Base配置
embed_dim = 768          # Embedding维度
encoder_depth = 12       # Encoder层数
decoder_depth = 4        # Decoder层数（更浅）
num_heads = 12           # 注意力头数
mlp_ratio = 4.0          # MLP扩展比例

# 输入配置
img_size = (128, 1024)   # (F, T) = (128 mel bins, 1024 frames)
patch_size = 16          # 16×16 patches
in_chans = 1             # 单声道

# 训练配置
mask_ratio = 0.75        # 默认掩码率
batch_size = 32
learning_rate = 1e-4
epochs = 100
```

### 更轻量级配置（如果DSP算力不足）

```python
# ViT-Tiny配置
embed_dim = 384
encoder_depth = 6
decoder_depth = 2
num_heads = 6
```

---

## 模型输入输出

### 输入
```python
# 训练时
x: torch.Tensor          # [B, 1, F, T] Log-Mel spectrogram
mask_ratio: float        # 掩码率（0.75）

# 推理时（异常检测）
x: torch.Tensor          # [B, 1, F, T] 实时音频的Log-Mel
threshold: float         # 异常阈值γ_th
```

### 输出
```python
# 训练时
reconstructed: torch.Tensor  # [B, 1, F, T] 重建的spectrogram
loss: torch.Tensor           # 重建损失

# 推理时
is_anomaly: bool             # 是否检测到异常
anomaly_score: float         # 异常分数（Lrec）
```

---

## 开发顺序建议

1. **patch_embed.py** - 最基础的组件
2. **pos_embed.py** - 位置编码
3. **encoder.py** - Encoder实现
4. **decoder.py** - Decoder实现
5. **specmae_model.py** - 整合所有组件
6. **测试** - 单元测试每个组件

---

## 参考资源

- AudioMAE代码: `E:\model_train_example\AudioMAE-main\`
- AST代码: `E:\model_train_example\ast-master\`
- DCASE代码: `E:\model_train_example\dcase2023_task2_baseline_ae-main\`
- 论文: `E:\model_train_example\Spec_Mae\research\SPAWC_UAV.txt`
