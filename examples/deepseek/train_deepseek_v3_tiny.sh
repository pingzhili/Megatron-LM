#!/bin/bash

export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export WANDB_API_KEY=2b60f655a687ad1161d31f0002256865e1ace428

GPUS_PER_NODE=8
WORLD_SIZE=8

CHECKPOINT_PATH=$1
DATA_PATH=$2

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes 1
    --node_rank 0
    --master_addr localhost
    --master_port 6000
)

MODEL_ARGS=(
    --use-mcore-models
    --disable-bias-linear
    --seq-length 2048
    --max-position-embeddings 2048
    --num-layers 12
    --hidden-size 512
    --ffn-hidden-size 2048
    --num-attention-heads 8
    --kv-channels 64
    --init-method-std 0.006
    --normalization RMSNorm
    --position-embedding-type rope
    --untie-embeddings-and-output-weights
    --no-position-embedding
    --rotary-base 10000
    --rotary-scaling-factor 40
    --rotary-percent 1.0
    --masked-softmax-fusion
    --apply-rope-fusion
    --recompute-granularity full
    --recompute-activations
    --multi-latent-attention
    --kv-lora-rank 64
    --q-lora-rank 192
    --qk-head-dim 64
    --qk-layernorm
    --qk-pos-emb-head-dim 32
)

MOE_ARGS=(
    --num-experts 16
    --moe-layer-freq "[0]*2+[1]*10"
    --moe-ffn-hidden-size 256
    --moe-router-score-function sigmoid
    --moe-router-bias-update-rate 0.001
    --moe-router-enable-expert-bias
    --moe-router-topk 4
    --moe-router-pre-softmax
    --moe-router-topk-scaling-factor 2.5
    --moe-shared-expert-overlap
    --moe-shared-expert-intermediate-size 512
    --moe-router-load-balancing-type seq_aux_loss
    --moe-aux-loss-coeff 1e-4
    --moe-token-dispatcher-type alltoall
    --moe-token-drop-policy probs
)

DATA_ARGS=(
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model "deepseek-ai/DeepSeek-V3"
    --data-path $DATA_PATH
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 8
    --lr 0.001
    --train-iters 20000
    --lr-decay-iters 20000
    --lr-decay-style cosine
    --min-lr 0.0001
    --weight-decay 0.1
    --lr-warmup-fraction 0.01
    --clip-grad 1.0
    --bf16
    --save-interval 100 \
    --eval-interval 100 \
    --eval-iters 10 \
    --save $CHECKPOINT_PATH \
    --tensorboard-dir "${CHECKPOINT_PATH}/tensorboard" \
    --no-load-optim \
    --no-load-rng
)

LOGGING_ARGS=(
    --log-interval 1 \
    --wandb-project "megatron-lm" \
    --wandb-exp-name "DeepSeek-V3-Tiny" \
)

# shellcheck disable=SC2068
torchrun ${DISTRIBUTED_ARGS[@]} pretrain_gpt.py \
    ${MODEL_ARGS[@]} \
    ${MOE_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${LOGGING_ARGS[@]}






