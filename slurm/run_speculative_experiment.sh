#!/bin/bash
#SBATCH --job-name=speculative_exp
#SBATCH --output=experiments/logs/speculative_exp_%j.out
#SBATCH --error=experiments/logs/speculative_exp_%j.err
#SBATCH --time=36:00:00
#SBATCH --partition=gpu-a100-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G

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
echo "Starting speculative decoding experiment..."

# Run speculative decoding experiment
python experiments/run_speculative_experiment.py

echo "Speculative decoding experiment completed!"
