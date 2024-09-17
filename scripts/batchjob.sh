#!/bin/bash
#SBATCH --job-name=llm
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=a3mixed
#SBATCH --gres=gpu:1                 # number of gpus
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive

######################
### Set enviroment ###
######################
cd /home/niklas/OLMoE
source /env/bin/start-ctx-user
conda activate llmoe
export WANDB_PROJECT="olmoe"
# Training setup
set -euo pipefail
export GPUS_PER_NODE=1
export NODENAME=$(hostname -s)
export MASTER_ADDR=$(scontrol show hostnames | head -n 1)
export MASTER_PORT=39594
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID
export FS_LOCAL_RANK=$SLURM_PROCID
export LOCAL_WORLD_SIZE=$SLURM_NTASKS_PER_NODE
export LOCAL_RANK=$SLURM_LOCALID
export NODE_RANK=$((($RANK - $LOCAL_RANK) / $LOCAL_WORLD_SIZE))
export R2_PROFILE=r2
export R2_ENDPOINT_URL=YOUR_URL
export AWS_ACCESS_KEY_ID=YOUR_KEY
export AWS_SECRET_ACCESS_KEY=YOUR_KEY
export HF_DATASETS_OFFLINE=1
#export CUDA_LAUNCH_BLOCKING=1
export TRITON_CACHE_DIR=/data/niklas/tcache

echo "World size: $WORLD_SIZE"
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

python -u scripts/train.py configs/olmoe17/olmoe17-8x1b-fullshard-swiglu-wrapb-ecg-k2.yml \
"--save_folder=/data/niklas/llm/checkpoints/${SLURM_JOB_ID}/" \
--eval_interval=1000 \
--save_interval=5000 \
--global_train_batch_size=16 \
--device_train_microbatch_size=2 \
--save_overwrite=True \
--evaluators=[]

# srun \
# --distribution=block:block \
# --kill-on-bad-exit \
# scripts/run_with_environment.sh \
# python -u scripts/train.py configs/olmoe17/olmoe-8x1b-newhp-newds-s3.yml \
# "--save_folder=/data/niklas/llm/checkpoints/${SLURM_JOB_ID}/" \
# --save_overwrite \
# --fsdp.sharding_strategy=HYBRID_SHARD \
# --device_train_microbatch_size=2
