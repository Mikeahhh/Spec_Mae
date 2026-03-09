# Sound-UAV: Energy-Efficient Search and Rescue via Biomimetic Sentinel-Responder Sensing

> ✅ **Security Status:** All dependencies verified and secure (Last checked: 2026-03-09)

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd E:\model_train_example\Spec_Mae
pip install -r requirements.txt
```

### 2. Run Security Check (Recommended)
```bash
python check_security.py
```

### 3. Start Training
```bash
# See docs/ for detailed instructions
python scripts/train/train_single_scenario.py --scenario desert
```

## 🔒 Security

This project follows security best practices:
- ✅ All dependencies are scanned for CVEs
- ✅ Secure checkpoint loading with integrity verification
- ✅ Input validation and path sanitization
- 📚 See [SECURITY_SUMMARY.md](SECURITY_SUMMARY.md) for details

**Security Tools:**
- `check_security.py` - Automated security checks
- `scripts/utils/security_utils.py` - Security utilities
- Full audit report: [SECURITY_AUDIT_REPORT.md](SECURITY_AUDIT_REPORT.md)

## Project Structure

```
Spec_Mae/
├── data/                          # 数据集目录
│   ├── raw/                       # 原始音频数据
│   │   ├── desert/               # 沙漠场景
│   │   ├── forest/               # 森林场景
│   │   ├── ocean/                # 海洋场景
│   │   └── multi_scenario/       # 多场景混合
│   ├── processed/                # 预处理后的数据
│   └── stats/                    # 数据集统计信息
│
├── models/                        # 模型定义
│   ├── specmae/                  # SpecMAE模型
│   └── baseline/                 # 基线模型
│
├── scripts/                       # 实验脚本
│   ├── train/                    # 训练脚本
│   ├── test/                     # 测试脚本
│   ├── eval/                     # 评估脚本
│   └── utils/                    # 工具函数
│
├── configs/                       # 配置文件
├── checkpoints/                   # 模型检查点
├── results/                       # 实验结果
├── notebooks/                     # Jupyter notebooks
├── docs/                         # 文档
└── research/                     # 论文与会议记录
```

## Quick Start

TODO: 添加使用说明

## Citation

TODO: 添加引用信息
