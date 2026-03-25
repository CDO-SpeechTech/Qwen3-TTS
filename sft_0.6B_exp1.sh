#!/usr/bin/env bash
set -e

DEVICE="cuda:0"
TOKENIZER_MODEL_PATH="Qwen/Qwen3-TTS-Tokenizer-12Hz"
INIT_MODEL_PATH="Qwen/Qwen3-TTS-12Hz-0.6B-Base"

RAW_JSONL="train_raw.jsonl"
TRAIN_JSONL="train_with_codes.jsonl"
OUTPUT_DIR="sft_output/Qwen3-TTS-12Hz-0.6B-Base/exp1"

BATCH_SIZE=64
LR=1e-5
EPOCHS=8
SPEAKER_NAME="f_bomi"

# python finetuning/prepare_data.py \
#   --device ${DEVICE} \
#   --tokenizer_model_path ${TOKENIZER_MODEL_PATH} \
#   --input_jsonl ${RAW_JSONL} \
#   --output_jsonl ${TRAIN_JSONL}

python finetuning/sft_12hz.py \
  --init_model_path ${INIT_MODEL_PATH} \
  --output_model_path ${OUTPUT_DIR} \
  --train_jsonl ${TRAIN_JSONL} \
  --batch_size ${BATCH_SIZE} \
  --lr ${LR} \
  --num_epochs ${EPOCHS} \
  --speaker_name ${SPEAKER_NAME}
  