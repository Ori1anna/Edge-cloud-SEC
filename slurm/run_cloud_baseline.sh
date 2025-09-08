#!/bin/bash
#SBATCH --partition=feit-gpu-a100
#SBATCH --account=punim2341
#SBATCH --qos=feit
#SBATCH --nodes=1
#SBATCH --job-name=sec_cloud_baseline
#SBATCH --output=experiments/logs/cloud_baseline_%j.out
#SBATCH --error=experiments/logs/cloud_baseline_%j.err
#SBATCH --time=24:00:00


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
echo "Starting cloud-only baseline experiment..."

# Run cloud-only baseline experiment
python experiments/run_cloud_baseline.py

echo "Cloud-only baseline experiment completed!"
