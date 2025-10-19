#!/bin/bash
#SBATCH -p gpu-a100
#SBATCH -J cloud-opt-baseline
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

# Load environment
module purge
source /data/gpfs/projects/punim2341/$USER/apps/mambaforge/etc/profile.d/conda.sh
conda activate sec

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Navigate to project directory
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec

# Run Cloud Optimized Baseline experiment
echo "Starting Cloud Optimized Baseline experiment..."
echo "Timestamp: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "CUDA Version: $(nvcc --version | grep "release")"

python experiments/runs/run_cloud_optimized_baseline.py \
    --model_path "Qwen/Qwen2.5-Omni-7B" \
    --test_data "/data/gpfs/projects/punim2341/jiajunlu/datasets/mer_audio_emotion_chinese/test.json" \
    --output_dir "./experiments/results" \
    --num_samples 332 \
    --max_new_tokens 64 \
    --target_sentences 2 \
    --min_chars 90 \
    --min_new_tokens_sc 48 \
    --prompt_type "default" \
    --batch_size 1 \
    --device "cuda"

echo "Cloud Optimized Baseline experiment completed!"
echo "End timestamp: $(date)"
