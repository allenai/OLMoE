#!/bin/bash
#SBATCH --job-name=llm
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=8
#SBATCH --hint=nomultithread         # we get physical cores not logical
#SBATCH --partition=a3
#SBATCH --gres=gpu:8                 # number of gpus
#SBATCH --output=/data/niklas/jobs/%x-%j.out           # output file name
#SBATCH --exclusive

######################
### Set enviroment ###
######################
cd /home/niklas/OLMo
source /env/bin/start-ctx-user
conda activate llm
export WANDB_PROJECT="olmo-small"
# Training setup
set -euo pipefail
export GPUS_PER_NODE=8
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
export R2_ENDPOINT_URL=XXX
export AWS_ACCESS_KEY_ID=XXX
export AWS_SECRET_ACCESS_KEY=XXX
export HF_DATASETS_OFFLINE=1
echo "World size: $WORLD_SIZE"
######################

######################
#### Set network #####
######################
head_node_ip=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
######################

srun \
--distribution=block:block \
--kill-on-bad-exit \
scripts/run_with_environment.sh \
python -u scripts/train.py configs/mitchish1-s3.yaml \
--activation_checkpointing=fine_grained \
--canceled_check_interval=50 \
--gen1_gc_interval=1 \
--device_train_microbatch_size=8 \
--global_train_batch_size=512 \
--run_name=mitchish1 \
--wandb.group=mitchish1 \
--model.flash_attention=true \
--fsdp.wrapping_strategy=null \
--fsdp.sharding_strategy=SHARD_GRAD_OP \
--fused_loss=true \
'--load_path=${path.last_checkpoint:${remote_save_folder}}' \
"--save_folder=/data/niklas/llm/checkpoints/${SLURM_JOB_ID}/"
