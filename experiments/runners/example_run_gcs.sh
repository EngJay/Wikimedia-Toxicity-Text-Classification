#!/usr/bin/env bash

#!/bin/bash

exp_num=1
exp_name="EXP_NAME_HERE"

run_num=1
num_runs=10
while [[ run_num -le num_runs ]]
do
  echo "Beginning training run #${run_num} of ${num_runs}..."

  python3 cnn_text.py --use_tpu=False \
                      --exp_name=${exp_name} \
                      --exp_num=${exp_num} \
                      --decoding_dic_path=${STORAGE_BUCKET}/data/ID2WORD_DICT \
                      --data_dir=${STORAGE_BUCKET}/data/TEXT_DATA \
                      --embeddings_dir=${STORAGE_BUCKET}/data/EMBEDDINGS_NPY \
                      --model_dir=${STORAGE_BUCKET}/output/${exp_name}-${exp_num} \
                      --train_enabled=True \
                      --eval_enabled=True \
                      --predict_enabled=False \
                      --train_steps=1000 \
                      --eval_steps=50 \
                      --batch_size=128 \
                      --max_doc_length=400 \
                      --num_classes=2 \
                      --save_model=False \
                      --export_dir_base=saved_models \
                      --use_slack=False

  ((run_num++))
done

echo "Finished run #${run_num} of ${num_runs} of ${exp_name}-${exp_num}!"
