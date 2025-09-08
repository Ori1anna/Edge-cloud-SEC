#!/bin/bash
#SBATCH -p interactive
#SBATCH -J hello-cpu
#SBATCH --time=0-00:20:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

module purge
source /data/gpfs/projects/punim2341/$USER/apps/miniconda3/etc/profile.d/conda.sh
conda activate sec

python - <<'PY'
print("Hello from CPU job!")
PY
