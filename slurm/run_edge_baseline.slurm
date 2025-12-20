#!/bin/bash

#SBATCH --partition=gpu-a100
#SBATCH --account=punim2341
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --job-name="edge-baseline-mer-en"
#SBATCH -o "slurm-%N.%j.out" # STDOUT
#SBATCH -e "slurm-%N.%j.err" # STDERR
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --mail-user=jiajunlu@unimelb.edu.au
#SBATCH --mail-type=FAIL
#SBATCH --mail-type=BEGIN
#SBATCH --mail-type=END
#SBATCH --time=6:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8

# Navigate to project directory
cd /data/gpfs/projects/punim2341/jiajunlu/edgecloud-sec || exit 1

# Load modules
module purge
module load CUDA/12.4.1
module load cuDNN/9.6.0.74-CUDA-12.4.1

# Activate conda environment
source /data/gpfs/projects/punim2341/jiajunlu/apps/miniconda3/etc/profile.d/conda.sh
export CONDA_ENVS_PATH=/data/gpfs/projects/punim2341/jiajunlu/conda/envs
conda activate sec-gpu

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Print system info
echo "=========================================="
echo "Edge Baseline Experiment (CPU Limited)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_JOB_NODELIST"
echo "Start time: $(date)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits)"
echo "CUDA Version: $(nvcc --version | grep 'release')"
echo "=========================================="

# Run Edge Baseline experiment
echo "Starting Edge Baseline experiment..."

python experiments/runs/run_edge_baseline_cpu_limited.py \
    --dataset_type unified \
    --dataset_path data/processed/mer2024/manifest_audio_text_augmented_v5.json \
    --caption_type audio_only \
    --language english \
    --prompt_type detailed \
    --input_modality audio_only \
    --max_cpu_cores 2 \
    --max_memory_gb 16 \
    --verbose \
    --output_name edge_baseline_mer_english_job2

echo "=========================================="
echo "Edge Baseline experiment completed!"
echo "End time: $(date)"
echo "=========================================="




