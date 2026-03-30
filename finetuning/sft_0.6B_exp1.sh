#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-0.6B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="sft_output/Qwen3-TTS-12Hz-0.6B-Base/exp1"


# 4GPU H100 settings
N_GPUS=4
BATCH_SIZE=32
MAX_SEQ_LENGTH=999
MAX_TOKENS=4000
LR=2e-6
EPOCHS=8

export NCCL_NET=Socket  # NCCL 2.27.3+ NET Plugin 초기화 실패 방지

torchrun --nproc_per_node=${N_GPUS} sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --max_seq_length ${MAX_SEQ_LENGTH} \
  --max_tokens ${MAX_TOKENS}
  