#!/bin/bash

#SBATCH --job-name=ml_clinical

#SBATCH --nodes=1

#SBATCH --ntasks=1

#SBATCH --cpus-per-task=64

#SBATCH --mem=200gb

#SBATCH --time=08:00:00 # example 8 hrs

#SBATCH --output=output_%j/ml_clinical_%j.out   # Standard output (%j = job ID)

#SBATCH --error=output_%j/ml_clinical_%j.err    # Standard error

hostname;date;pwd

export XDG_RUNTIME_DIR=${SLURM_TMPDIR}

# HPG install of vscode. Comment the next line out with '#' if using a personal install
module load conda
conda init
conda env list
conda activate my_env_2025_4_25 

# Strongly recommended: pin BLAS threads to 1 for each rank
# export OMP_NUM_THREADS=1
# export OPENBLAS_NUM_THREADS=1
# export MKL_NUM_THREADS=1
# export NUMEXPR_NUM_THREADS=1

WORKSPACE=$1
cd $WORKSPACE
echo "Workspace is $(realpath "$WORKSPACE")"

python -m ml_clinical_trial.main_internal
# Use srun or mpirun (pick the one your cluster prefers)
# srun will set local rank env vars; mpirun works too if available.
#srun -n ${SLURM_NTASKS} python -m ml_clinical_trial.main
# OR:
# mpirun -np ${SLURM_NTASKS} python -m ml_clinical_trial.main
