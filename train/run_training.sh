#!/bin/bash
set -x

# CRITICAL: Activate conda environment to ensure all paths are correct
source ~/miniconda3/etc/profile.d/conda.sh
conda activate hbx_ck

export PYTHONUNBUFFERED=1
export PROJECT_NAME='justrl'
export PROJECT_PATH=/home/test/test06/hbx/JustRL/train
export DATA_DIR=$PROJECT_PATH/data

export EXPERIMENT_NAME=JustRL-DeepSeek-1.5B-$(date +%Y-%m-%d_%H-%M-%S)

export TRAIN_DATASET=$DATA_DIR/DAPO/dapo-math-17k.parquet
# export TRAIN_DATASET=$DATA_DIR/DAPO/dapo-math-17k-sample-10k.parquet
export TEST_AIME24=$DATA_DIR/AIME24/test.parquet
export TEST_AIME25=$DATA_DIR/AIME25/test.parquet
export TEST_AMC23=$DATA_DIR/AMC23/test.parquet

# export TEST_DATASET="['$TEST_AIME24']"
export TEST_DATASET="['$TEST_AIME24', '$TEST_AIME25', '$TEST_AMC23']"

# Model and training paths
# export ACTOR_MODEL_PATH=/home/test/testdata/models/OpenMath-Nemotron-1.5B
export ACTOR_MODEL_PATH=/home/test/testdata/models/DeepSeek-R1-Distill-Qwen-1.5B


export PARALLEL_SIZE=1
export CKPT_PATH=${PROJECT_PATH}/checkpoints
export OUTLINES_CACHE_DIR=~/.cache/outlines/$(uuidgen)

# Environment settings
export NCCL_DEBUG=WARN
export WANDB_API_KEY='7e69f789501e2f5153bf315454c1f1a414b06c55'
export TOKENIZERS_PARALLELISM=true
export WANDB_MODE=offline
export WANDB_DIR=${PROJECT_PATH}/wandb/
export TENSORBOARD_DIR=${PROJECT_PATH}/tensorboard/$PROJECT_NAME/$EXPERIMENT_NAME
export HYDRA_FULL_ERROR=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

# IMPORTANT: Set Ray address to connect to existing cluster
export RAY_ADDRESS="11.11.18.2:6379"

cd $PROJECT_PATH

# Set PYTHONPATH to include JustRL directory so verl module can be imported without installation
export PYTHONPATH="${PROJECT_PATH}:${PYTHONPATH}"

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.kl_ctrl.kl_coef=0.0 \
    data.train_files="$TRAIN_DATASET" \
    data.val_files="$TEST_DATASET" \
    data.train_batch_size=256 \
    data.val_batch_size=6312 \
    data.max_prompt_length=1024 \
    data.max_response_length=15360 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    +data.use_boxed_suffix_prompt=True \
    actor_rollout_ref.model.path=$ACTOR_MODEL_PATH \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$PARALLEL_SIZE \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    +actor_rollout_ref.rollout.val_kwargs.max_new_tokens=31744 \
    actor_rollout_ref.rollout.val_kwargs.n=32 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.7 \
    actor_rollout_ref.rollout.val_kwargs.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.85 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    reward_model.enable=False \
    reward_model.reward_manager=dapo \
    +reward_model.reward_kwargs.overlong_buffer_cfg.enable=False \
    +reward_model.reward_kwargs.overlong_buffer_cfg.len=4096 \
    +reward_model.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 \
    trainer.val_before_train=True \
    "trainer.logger=['console','tensorboard']" \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=4 \
    trainer.save_freq=50 \
    trainer.test_freq=50 \
    trainer.total_epochs=1 \
    trainer.default_local_dir="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME" \
    trainer.validation_data_dir="$CKPT_PATH/$PROJECT_NAME/$EXPERIMENT_NAME/validation"
