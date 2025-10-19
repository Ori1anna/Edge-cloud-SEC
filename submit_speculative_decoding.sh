#!/bin/bash

# Script to submit Speculative Decoding experiments to SLURM
# Usage: ./submit_speculative_decoding.sh [test|english|chinese|all]

echo "=========================================="
echo "Speculative Decoding Job Submission"
echo "=========================================="

case "$1" in
    "test")
        echo "Submitting test job (5 samples)..."
        sbatch slurm/run_speculative_decoding_test.slurm
        ;;
    "english")
        echo "Submitting English experiment (50 samples)..."
        sbatch slurm/run_speculative_decoding.slurm
        ;;
    "chinese")
        echo "Submitting Chinese experiment (50 samples)..."
        sbatch slurm/run_speculative_decoding_chinese.slurm
        ;;
    "all")
        echo "Submitting all experiments..."
        echo "1. Test job (5 samples)..."
        sbatch slurm/run_speculative_decoding_test.slurm
        sleep 2
        echo "2. English experiment (50 samples)..."
        sbatch slurm/run_speculative_decoding.slurm
        sleep 2
        echo "3. Chinese experiment (50 samples)..."
        sbatch slurm/run_speculative_decoding_chinese.slurm
        ;;
    *)
        echo "Usage: $0 [test|english|chinese|all]"
        echo ""
        echo "Options:"
        echo "  test     - Submit test job (5 samples, 2 hours)"
        echo "  english  - Submit English experiment (50 samples, 12 hours)"
        echo "  chinese  - Submit Chinese experiment (50 samples, 12 hours)"
        echo "  all      - Submit all experiments"
        echo ""
        echo "Examples:"
        echo "  ./submit_speculative_decoding.sh test"
        echo "  ./submit_speculative_decoding.sh english"
        echo "  ./submit_speculative_decoding.sh all"
        exit 1
        ;;
esac

echo ""
echo "Job(s) submitted! Check status with: squeue --me"
echo "Cancel job with: scancel <job_id>"
echo "=========================================="
