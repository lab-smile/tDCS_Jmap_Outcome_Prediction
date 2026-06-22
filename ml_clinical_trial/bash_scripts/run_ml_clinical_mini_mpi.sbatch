#!/bin/bash
#SBATCH --job-name=ml_clinical
#SBATCH --nodes=4                 # <— scale this up/down
#SBATCH --ntasks-per-node=8       # ranks per node (total ranks = nodes * ntasks-per-node)
#SBATCH --cpus-per-task=1         # keep 1 (code pins threads)
#SBATCH --mem=200G                   # use all memory on each node; or set --mem=200G per node if you prefer
#SBATCH --time=08:00:00
#SBATCH --hint=nomultithread      # avoid SMT/HT; steadier perf
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

echo "Job $SLURM_JOB_ID on $(hostname)"
date; pwd

# Ensure log dir exists (since SLURM won't create parent dirs automatically)
mkdir -p logs

# Optional: create a per-job work dir on local scratch for faster IO
export WORKDIR="${SLURM_TMPDIR:-/tmp}/${SLURM_JOB_ID}"
mkdir -p "$WORKDIR"

# Environment (one time per task on each node)
module load conda            # or your site’s module stack
conda activate b200

# Keep inner math single-threaded per rank (matches your Python)
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

# Your project workspace comes in as the script arg
WORKSPACE="$1"
echo "Workspace is $(realpath "$WORKSPACE")"

# If data are large and read-only, copying to node-local scratch can help:
# rsync -a --info=progress2 "$WORKSPACE/" "$WORKDIR/"
# cd "$WORKDIR"
# Otherwise, run in place:
cd "$WORKSPACE"

# Launch: one Python process per MPI rank across all nodes
# --cpu-bind=cores gives clean core affinity
srun --cpu-bind=cores -n $SLURM_NTASKS python -m ml_clinical_trial.main

# If your site prefers mpirun, this also works:
# mpirun -np $SLURM_NTASKS python -m ml_clinical_trial.main
