#!/bin/bash
source path.sh
set -e

log_root="logs_conv_only/logs_convonly_Lib_1g4r_cos_from_head"
input_training_file="../../Lib_resources/LibriTTS/train.lst"
input_validation_file="../../Lib_resources/LibriTTS/dev-clean_part.lst"

mode=debug
# mode=train
export CUDA_VISIBLE_DEVICES=3


if [ "${mode}" == "debug" ]; then
  ## debug
  echo "Debug"
  log_root=${log_root}_debug
  python ${BIN_DIR}/train.py \
    --config config_24k_320d_conv_1g4r_8g3k1s.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 10 \
    --summary_interval 10 \
    --validation_interval 10 \

elif [ "$mode" == "train" ]; then
  ## train
  echo "Train model..."
  python ${BIN_DIR}/train.py \
    --config config_24k_320d_conv_1g4r_8g3k1s.json \
    --checkpoint_path ${log_root} \
    --input_training_file ${input_training_file} \
    --input_validation_file ${input_validation_file} \
    --checkpoint_interval 5000 \
    --summary_interval 100 \
    --validation_interval 5000 \
    --num_ckpt_keep 100
fi
