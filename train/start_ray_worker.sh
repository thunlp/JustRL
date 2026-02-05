#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hbx_ck

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN

ray stop --force 2>/dev/null || true
sleep 2
ray start --address='11.11.18.2:6379' --num-gpus=8 --disable-usage-stats --block
