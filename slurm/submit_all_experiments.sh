#!/bin/bash
# Submit all baseline experiments in sequence

echo "Submitting baseline experiments..."

# Submit edge-only baseline
echo "Submitting edge-only baseline experiment..."
edge_job_id=$(sbatch slurm/run_edge_baseline.sh | awk '{print $4}')
echo "Edge-only job submitted with ID: $edge_job_id"

# Submit cloud-only baseline (depends on edge-only completion)
echo "Submitting cloud-only baseline experiment..."
cloud_job_id=$(sbatch --dependency=afterok:$edge_job_id slurm/run_cloud_baseline.sh | awk '{print $4}')
echo "Cloud-only job submitted with ID: $cloud_job_id"

# Submit speculative decoding experiment (depends on cloud-only completion)
echo "Submitting speculative decoding experiment..."
spec_job_id=$(sbatch --dependency=afterok:$cloud_job_id slurm/run_speculative_experiment.sh | awk '{print $4}')
echo "Speculative decoding job submitted with ID: $spec_job_id"

echo "All experiments submitted!"
echo "Job IDs:"
echo "  Edge-only: $edge_job_id"
echo "  Cloud-only: $cloud_job_id"
echo "  Speculative: $spec_job_id"
echo ""
echo "Monitor jobs with:"
echo "  squeue -u $USER"
echo ""
echo "Check logs in: experiments/logs/"
