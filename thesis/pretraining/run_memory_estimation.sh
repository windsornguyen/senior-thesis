#!/usr/bin/bash

NGPU=${NGPU:-"1"}
NNODES=${NNODES:-"1"}
CONFIG_FILE=${CONFIG_FILE:-"/scratch/gpfs/mn4560/thesis/thesis/pretraining/configs/mamba_debug.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

# Calculate WORLD_SIZE as the product of NGPU and NNODES
# Export WORLD_SIZE and LOCAL_RANK
export WORLD_SIZE=$((NGPU * NNODES))
export LOCAL_RANK=0
python estimation.py --job.config_file ${CONFIG_FILE} --memory_estimation.enabled $overrides
