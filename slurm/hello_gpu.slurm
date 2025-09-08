#!/bin/bash
#SBATCH -p gpu-a100
#SBATCH -J hello-gpu
#SBATCH --gres=gpu:1
#SBATCH --time=0-00:03:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH -o %x.%j.out
#SBATCH -e %x.%j.err

module purge
source /data/gpfs/projects/punim2341/$USER/apps/mambaforge/etc/profile.d/conda.sh
conda activate sec

python - <<'PY'
import torch, platform
print("Host:", platform.node())
print("Torch:", torch.__version__, "CUDA ok?", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY
