#!/bin/bash
#SBATCH --job-name=edge_baseline
#SBATCH --output=experiments/logs/edge_baseline_%j.out
#SBATCH --error=experiments/logs/edge_baseline_%j.err
#SBATCH --time=12:00:00
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

# Load modules
module load anaconda3/2023.09

# Activate conda environment
source activate sec-gpu

# Set Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create logs directory
mkdir -p experiments/logs
mkdir -p experiments/results

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Starting edge-only baseline experiment..."

# Run edge-only baseline experiment
python experiments/run_edge_baseline.py

echo "Edge-only baseline experiment completed!"
