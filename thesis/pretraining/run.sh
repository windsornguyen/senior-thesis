#!/usr/bin/bash

NGPU=${NGPU:-"1"}
LOG_RANK=${LOG_RANK:-0}
CONFIG_FILE=${CONFIG_FILE:-"/scratch/gpfs/mn4560/thesis/thesis/pretraining/configs/mamba_debug.toml"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" \
torchrun --nproc_per_node=${NGPU} --rdzv_backend c10d --rdzv_endpoint="localhost:0" \
--local-ranks-filter ${LOG_RANK} --role rank --tee 3 \
train.py --job.config_file ${CONFIG_FILE} $overrides
