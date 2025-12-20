5. Submit a task

If you think your code is ready you can submit a task to run (enjoy your coffee or paper writing :)).

First find out which gpu you want to submit: spartan-weather

You will see:Then do: vim your_task_name.slurm

write: 

```
\#!/bin/bash

\#SBATCH --partition=feit-gpu-a100

\#SBATCH --account=punim2243

\#SBATCH --qos=feit

\#SBATCH --nodes=1

\#SBATCH --job-name="fla-tts"

\#SBATCH -o "slurm-%N.%j.out" # STDOUT

\#SBATCH -e "slurm-%N.%j.err" # STDERR#SBATCH --ntasks=1

\#SBATCH --gres=gpu:1

\#SBATCH --mail-user=yourusername@unimelb.edu.au

\#SBATCH --mail-type=FAIL

\#SBATCH --time=0-12:00:00

\#SBATCH --mem=64G

\# Send yourself an email when the job:

\# aborts abnormally (fails)

\#SBATCH --mail-type=FAIL

\# begins

\#SBATCH --mail-type=BEGIN

\# ends successfully

\#SBATCH --mail-type=END

cd /data/gpfs/projects/project_name/your_project || exit 1 # Exit if cd fails

module purge

\# Load CUDA version compatible with the cuDNN module

module load CUDA/12.4.1

module load cuDNN/9.6.0.74-CUDA-12.4.1

\# Activate the bash virtual environment

conda activate your_environment

\# run the code

python yourscript.py
```

You MUST need to 1. cd to your *.py folder 2. activate your env 

before submit slurm.

If you everything is ready. Then:

sbatch your_task_name.slurm

you can check your submitted task ID via:

squeue --me

And if you want to stop:

scancle task_ID