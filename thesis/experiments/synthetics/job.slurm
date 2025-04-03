#!/bin/bash
#SBATCH --job-name=copy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --partition=pli-c
#SBATCH --account=ehazan
#SBATCH --error=jax_scan_attn_copy_%j.err
#SBATCH --output=jax_scan_attn_copy_%j.out

# Logging
log_info() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# Error handling
set -e
trap 'log_info "Error on line $LINENO"; exit 1' ERR

# Activate your virtual environment accordingly
module purge
module load anaconda3/2024.6
module load cudatoolkit/12.4
conda activate thesis

# Log start of training run
log_info "Starting training run..."

# Run the script using torchrun
srun python jax_scan_attn_copy.py

# Log end of training run
log_info "Job finished."
