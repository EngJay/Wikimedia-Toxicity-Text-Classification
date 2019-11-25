#!/usr/bin/env bash

#!/bin/bash

exp_num=1
exp_name="GPU-WM-PA-Bin-Baseline-TF-1.13"

run_num=1
num_runs=1
while [[ run_num -le num_runs ]]
do
  echo "Beginning training run #${run_num} of ${num_runs}..."

  python3 cnn_text.py --use_tpu=False \
			              --exp_name=${exp_name} \
                          --exp_num=${exp_num} \
                          --decoding_dic_path=WM-PA-Gao-300-id2word.bin \
                          --data_dir=WM-PA-Bin-Threshold-5-Gao-300-data.bin \
                          --embeddings_dir=WM-PA-Gao-300-embeddings.npy \
                          --model_dir=output/${exp_name}-${exp_num} \
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

echo Finished!
