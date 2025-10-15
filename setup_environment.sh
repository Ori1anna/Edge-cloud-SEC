#!/bin/bash

# EdgeCloud-SEC 环境快速设置脚本
# 使用方法: bash setup_environment.sh

set -e  # 遇到错误立即退出

echo "🚀 开始设置 EdgeCloud-SEC 环境..."

# 检查conda是否安装
if ! command -v conda &> /dev/null; then
    echo "❌ Conda 未安装，请先安装 Anaconda 或 Miniconda"
    exit 1
fi

# 检查environment.yml是否存在
if [ ! -f "environment.yml" ]; then
    echo "❌ environment.yml 文件不存在"
    exit 1
fi

# 创建conda环境
echo "📦 创建conda环境 sec-gpu..."
conda env create -f environment.yml

# 激活环境并验证
echo "✅ 环境创建完成！"
echo ""
echo "📋 接下来的步骤："
echo "1. 激活环境: conda activate sec-gpu"
echo "2. 创建数据目录: mkdir -p data/{raw,processed,cache,external}"
echo "3. 下载或准备数据文件"
echo "4. 运行实验脚本"
echo ""
echo "🔧 环境验证："
echo "conda activate sec-gpu"
echo "python -c 'import torch; print(f\"PyTorch版本: {torch.__version__}\"); print(f\"CUDA可用: {torch.cuda.is_available()}\")'"
echo ""
echo "📖 详细说明请查看 SETUP_GUIDE.md"
