# EdgeCloud-SEC 环境设置指南

## 🚀 快速开始

### 1. 克隆仓库
```bash
git clone <your-repo-url>
cd edgecloud-sec
```

### 2. 创建conda环境
使用提供的environment.yml文件创建完全相同的环境：

```bash
# 创建环境
conda env create -f environment.yml

# 激活环境
conda activate sec-gpu
```

### 3. 数据设置
由于`data/`目录被gitignore忽略，您需要：

#### 选项A：下载数据（推荐）
```bash
# 创建数据目录
mkdir -p data/raw data/processed data/cache data/external

# 下载必要的数据文件（需要提供下载链接）
# 例如：
# wget <data-url> -O data/raw/mer2024_dataset.zip
# unzip data/raw/mer2024_dataset.zip -d data/raw/
```

#### 选项B：使用示例数据
```bash
# 创建示例数据文件用于测试
mkdir -p data/processed/mer2024
touch data/processed/mer2024/sample_data.json
```

## 🔧 环境详情

### 关键依赖包
- **Python**: 3.10.18
- **PyTorch**: 2.6.0+cu124 (CUDA 12.4)
- **Transformers**: 4.52.3
- **CUDA**: 12.1
- **音频处理**: librosa 0.11.0, soundfile
- **数据处理**: pandas, numpy, scipy
- **评估**: nltk, pycocoevalcap

### GPU支持
环境已配置CUDA支持：
- CUDA 12.1
- cuDNN 9.1.0.70
- PyTorch CUDA 12.4版本

## 🏃‍♂️ 运行代码

### 基本运行命令
```bash
# 激活环境
conda activate sec-gpu

# 运行实验
python experiments/runs/run_speculative_decoding_cpu_limited.py \
    --dataset_path data/processed/mer2024/final-EMER-reason.csv \
    --model_name your_model_name \
    --output_dir experiments/results/
```

### 必需参数
- `--dataset_path`: 数据集路径（必须提供）
- `--model_name`: 模型名称
- `--output_dir`: 输出目录

## 📁 目录结构

```
edgecloud-sec/
├── src/                    # 源代码
│   ├── data/              # 数据处理模块
│   ├── models/            # 模型定义
│   ├── evaluation/        # 评估工具
│   └── utils/             # 工具函数
├── experiments/           # 实验脚本
│   └── runs/             # 运行脚本
├── data/                 # 数据目录（需要手动创建）
│   ├── raw/              # 原始数据
│   ├── processed/        # 处理后数据
│   ├── cache/            # 缓存文件
│   └── external/         # 外部数据
├── MERTools/             # MER工具集
├── requirements.txt      # Python依赖
├── environment.yml       # Conda环境配置
└── .gitignore           # Git忽略规则
```

## ⚠️ 注意事项

### 被Git忽略的重要文件
- `data/` - 整个数据目录
- `__pycache__/` - Python缓存
- `venv/`, `.venv/` - 虚拟环境
- `.DS_Store` - macOS系统文件

### 数据获取
由于数据文件较大，被gitignore忽略。您需要：
1. 从原始数据源下载数据集
2. 或联系项目维护者获取数据访问权限
3. 或使用提供的示例数据进行测试

### 环境兼容性
- 推荐使用Linux系统
- 需要NVIDIA GPU支持CUDA 12.1+
- 至少8GB GPU内存（推荐16GB+）

## 🐛 故障排除

### 常见问题

1. **CUDA版本不匹配**
   ```bash
   # 检查CUDA版本
   nvidia-smi
   # 如果版本不匹配，修改environment.yml中的CUDA版本
   ```

2. **数据文件不存在**
   ```bash
   # 确保数据目录存在
   mkdir -p data/{raw,processed,cache,external}
   # 检查数据集路径是否正确
   ```

3. **内存不足**
   ```bash
   # 减少batch size或使用CPU模式
   python script.py --device cpu
   ```

## 📞 支持

如果遇到问题，请：
1. 检查环境是否正确激活
2. 确认所有依赖已正确安装
3. 验证数据文件路径
4. 查看错误日志获取详细信息
