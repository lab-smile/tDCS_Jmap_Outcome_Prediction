#!/bin/bash
#SBATCH --job-name=install_env
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=8gb
#SBATCH --time=08:00:00 # example 8 hrs

# === logging setup ===
logdir="installation_log"
mkdir -p "$logdir"
#SBATCH --output=installation_log/slurm_installation_%j.out
#SBATCH --error=installation_log/slurm_installation_%j.err

# === Arguments ===
ENV_FILE=$1         # Path to environment.yml
ENV_NAME=$2         # Name of environment to create

if [ -z "$ENV_FILE" ] || [ -z "$ENV_NAME" ]; then
    echo "Usage: sbatch install_env.sbatch <environment.yml path> <env name>"
    exit 1
fi

echo "=== Job started at $(date) on $(hostname) ==="
echo "Using environment file: $ENV_FILE"
echo "Target environment name: $ENV_NAME"
echo "Logs will be stored in: $logdir"

# === Conda setup (adjust for your cluster) ===
module load anaconda   # or module load miniconda
# source ~/miniconda3/etc/profile.d/conda.sh

# === Environment creation ===
echo "Step 1: Creating conda environment '$ENV_NAME' from $ENV_FILE ..."
conda env create -f "$ENV_FILE" -n "$ENV_NAME"

if [ $? -eq 0 ]; then
    echo "Step 1 complete: Environment created successfully."
else
    echo "Step 1 FAILED: Could not create environment."
    exit 1
fi

# === Environment activation & checks ===
echo "Step 2: Activating environment..."
source activate "$ENV_NAME"   # or: conda activate "$ENV_NAME"
echo "Environment '$ENV_NAME' activated."

echo "Step 3: Checking Python version..."
python -V

echo "Step 4: Listing installed packages..."
conda list

echo "=== Job finished at $(date) ==="

