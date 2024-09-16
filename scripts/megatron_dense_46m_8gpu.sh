#!/bin/bash

EXP_DIR=$1

# scaling law: 1B tokesn @ 125m = 2k steps.
#
# 512 * 1k * 400k = 200b tokens.
# 512 * 1k * 200k = 100b tokens.
# 512 * 1k * 100k = 50b tokens (default).
# 512 * 1k * 20k = 10b tokens.
TRAINING_STEPS=100000
if [ -n "${2}" ]; then
    TRAINING_STEPS=$2;
fi

##
### Pre-training for GPT2 46M parameter.
##

# Distributed hyperparameters.
DISTRIBUTED_ARGUMENTS="\
--nproc_per_node 8 \
--nnodes 1 \
--node_rank 0 \
--master_addr localhost \
--master_port 6000"

# Model hyperparameters.
MODEL_ARGUMENTS="\
--num-layers 6 \
--hidden-size 512 \
--num-attention-heads 8 \
--seq-length 1024 \
--max-position-embeddings 1024"

# Training hyperparameters.
TRAINING_ARGUMENTS="\
--micro-batch-size 64 \
--global-batch-size 512 \
--train-iters ${TRAINING_STEPS} \
--lr-decay-iters ${TRAINING_STEPS} \
--lr 0.0006 \
--min-lr 0.00006 \
--lr-decay-style cosine \
--lr-warmup-fraction 0.01 \
--clip-grad 1.0 \
--init-method-std 0.01"

C4_DATASET="/data/niklas/c4-subsets/55b/gpt2tok_c4_en_55B_text_document"

# NOTE: We don't train for enough tokens for the
# split to matter.
DATA_ARGUMENTS="\
--data-path ${C4_DATASET} \
--vocab-file /data/niklas/vocab.json \
--merge-file /data/niklas/merges.txt \
--make-vocab-size-divisible-by 1024 \
--split 969,30,1"

COMPUTE_ARGUMENTS="\
--bf16 \
--DDP-impl local \
--no-async-tensor-model-parallel-allreduce \
--use-flash-attn"

CHECKPOINT_ARGUMENTS="\
--save-interval 2000 \
--save ./${EXP_DIR}"

EVALUATION_ARGUMENTS="\
--eval-iters 100 \
--log-interval 100 \
--eval-interval 1000"

torchrun ${DISTRIBUTED_ARGUMENTS} \
       third_party/Megatron-LM/pretrain_gpt.py \
       ${MODEL_ARGUMENTS} \
       ${TRAINING_ARGUMENTS} \
       ${DATA_ARGUMENTS} \
       ${COMPUTE_ARGUMENTS} \
       ${CHECKPOINT_ARGUMENTS} \
       ${EVALUATION_ARGUMENTS} |& tee ./${EXP_DIR}/train.log
