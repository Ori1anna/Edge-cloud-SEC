#!/bin/bash
#SBATCH --job-name=sec_experiment
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=experiments/logs/slurm_%j.out
#SBATCH --error=experiments/logs/slurm_%j.err

# Load modules (adjust for your HPC system)
module load anaconda3
module load cuda/11.8

# Activate conda environment
source activate sec-gpu

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Create logs directory
mkdir -p experiments/logs

# Run experiment
echo "Starting experiment at $(date)"
python experiments/run_baseline.py

echo "Experiment completed at $(date)"
