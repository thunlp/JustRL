#!/bin/bash

# 4-node Ray cluster setup for nodes g48, g49, g86, g87
# All nodes are on account test06 with shared filesystem

# Configure your nodes
# NODE1="g48"  # Head node
# NODE2="g49"  # Worker 1
# NODE3="g89"  # Worker 2  
# NODE4="g90"  # Worker 3

NODE1="g48"  # Head node
NODE2="g49"  # Worker 1
NODE3="g86"  # Worker 2  
NODE4="g87"  # Worker 3

HEAD_PORT=6379
BASE_DIR="/home/test/test06/hbx"
SCRIPT_DIR="${BASE_DIR}/JustRL/train"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="${SCRIPT_DIR}/logs/scaling/ray_${TIMESTAMP}"

# Conda environment name
CONDA_ENV="hbx_ck"

# Function to run command with conda environment
run_with_conda() {
    local node=$1
    shift
    local cmd="$@"
    ssh $node "source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && $cmd"
}

echo "=========================================="
echo "Starting 4-node Ray cluster"
echo "Nodes: $NODE1 (head), $NODE2, $NODE3, $NODE4"
echo "Conda environment: $CONDA_ENV"
echo "Timestamp: $TIMESTAMP"
echo "=========================================="

# Create log directory (only need to do once due to shared filesystem)
mkdir -p $LOG_DIR

# Step 1: Check GPU availability on all nodes first
# echo ""
# echo "Checking GPU availability on all nodes..."
# for node in $NODE1 $NODE2 $NODE3 $NODE4; do
#     echo -n "  $node: "
#     ssh $node "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits | head -1" 2>/dev/null || echo "Unable to check"
# done

# Step 3: Stop any existing Ray instances
echo ""
echo "Cleaning up existing Ray instances..."
for node in $NODE1 $NODE2 $NODE3 $NODE4; do
    echo "  Stopping Ray on $node..."
    run_with_conda $node "ray stop --force" 2>/dev/null || true
done
sleep 5

# Step 4: Start Ray head on NODE1
echo ""
echo "Starting Ray head on $NODE1..."
run_with_conda $NODE1 "
    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    export NCCL_DEBUG=WARN
    ray stop --force 2>/dev/null || true
    sleep 2
    ray start --head --port=$HEAD_PORT --num-gpus=8 --disable-usage-stats
" > ${LOG_DIR}/${NODE1}_ray_start.log 2>&1

# Get head node IP
sleep 5
HEAD_IP=$(ssh $NODE1 "hostname -I | awk '{print \$1}'")
RAY_ADDRESS="${HEAD_IP}:${HEAD_PORT}"
echo "Ray head started at ${RAY_ADDRESS}"

# Step 5: Create worker startup script (shared filesystem, create once)
cat > ${SCRIPT_DIR}/start_ray_worker.sh << EOF
#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export NCCL_DEBUG=WARN

ray stop --force 2>/dev/null || true
sleep 2
ray start --address='${RAY_ADDRESS}' --num-gpus=8 --disable-usage-stats --block
EOF
chmod +x ${SCRIPT_DIR}/start_ray_worker.sh

# Step 6: Start Ray workers on other nodes
echo ""
echo "Starting Ray workers..."
for node in $NODE2 $NODE3 $NODE4; do
    echo "  Starting worker on $node..."
    ssh $node "nohup ${SCRIPT_DIR}/start_ray_worker.sh > ${LOG_DIR}/${node}_ray_worker.log 2>&1 &"
    sleep 3
done

# Step 7: Wait for cluster to form
echo ""
echo "Waiting for cluster to form (30 seconds)..."
for i in {1..30}; do
    echo -n "."
    sleep 1
done
echo ""

# Step 8: Check cluster status
echo ""
echo "Ray cluster status:"
run_with_conda $NODE1 "ray status"

# Step 9: Create training script (shared filesystem, create once)
echo ""
echo "Creating training script..."
cat > ${SCRIPT_DIR}/run_training.sh << 'EOF'
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
export RAY_ADDRESS="${RAY_ADDRESS}"

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
EOF
chmod +x ${SCRIPT_DIR}/run_training.sh

# Step 9a: Update the run_training.sh script with RAY_ADDRESS
sed -i "s|export RAY_ADDRESS=\"\${RAY_ADDRESS}\"|export RAY_ADDRESS=\"${RAY_ADDRESS}\"|g" ${SCRIPT_DIR}/run_training.sh

# Step 10: Start training on head node with proper environment
echo ""
echo "Starting training on head node $NODE1..."
# Use nohup but ensure conda environment is activated
ssh $NODE1 "cd ${SCRIPT_DIR} && nohup bash -c 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate ${CONDA_ENV} && bash ${SCRIPT_DIR}/run_training.sh' > ${LOG_DIR}/training.log 2>&1 &"

# Step 11: Get the training PID for monitoring
sleep 5
TRAINING_PID=$(ssh $NODE1 "pgrep -f 'verl.trainer.main_ppo' | head -1")

echo ""
echo "=========================================="
echo "✓ Training started successfully!"
echo "=========================================="
echo ""
echo "Important information:"
echo "  - Log directory: ${LOG_DIR}"
echo "  - TensorBoard dir: ${SCRIPT_DIR}/tensorboard/justrl/"
echo "  - Worker script: ${SCRIPT_DIR}/start_ray_worker.sh"
echo "  - Training script: ${SCRIPT_DIR}/run_training.sh"
if [ ! -z "$TRAINING_PID" ]; then
    echo "  - Training process PID: $TRAINING_PID"
fi
echo ""
echo "Monitor commands:"
echo "  - Training log:    tail -f ${LOG_DIR}/training.log"
echo "  - TensorBoard:     tensorboard --logdir ${SCRIPT_DIR}/tensorboard/justrl/"
echo "  - Check TB files:  ls -la ${SCRIPT_DIR}/tensorboard/justrl/"
echo "  - Ray status:      ssh $NODE1 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && ray status'"
echo ""
echo "Worker logs:"
echo "  - $NODE2: tail -f ${LOG_DIR}/${NODE2}_ray_worker.log"
echo "  - $NODE3: tail -f ${LOG_DIR}/${NODE3}_ray_worker.log"
echo "  - $NODE4: tail -f ${LOG_DIR}/${NODE4}_ray_worker.log"
echo ""
echo "To stop the training:"
echo "  1. Kill training: ssh $NODE1 'pkill -f verl.trainer.main_ppo'"
echo "  2. Stop Ray cluster:"
for node in $NODE1 $NODE2 $NODE3 $NODE4; do
    echo "     ssh $node 'source ~/miniconda3/etc/profile.d/conda.sh && conda activate $CONDA_ENV && ray stop --force'"
done
echo ""
echo "Checking initial training status (waiting 10 seconds)..."
sleep 10

# Check if training is running
if ssh $NODE1 "ps -p ${TRAINING_PID:-0} > /dev/null 2>&1"; then
    echo "✓ Training process is running"
    echo ""
    echo "First few lines of training log:"
    ssh $NODE1 "head -20 ${LOG_DIR}/training.log 2>/dev/null || echo 'Log not ready yet'"
    echo ""
    echo "Checking TensorBoard directory:"
    ssh $NODE1 "ls -la ${SCRIPT_DIR}/tensorboard/justrl/ 2>/dev/null || echo 'TensorBoard directory not created yet'"
else
    echo "⚠ Note: Training process check inconclusive, please verify manually"
fi