#!/bin/bash

# Define the model names
declare -a model_names=("wo-multi-scale" "w-pure-query-selection" "wo-denoising" "wo-label-flipping" "wo-2stage" "main_model")
epoch=$1

# Loop over the model names
for model_name in "${model_names[@]}"
do
    python evaluation/generate_preds.py --model_name $model_name --threshold 0 --epoch $epoch

    python evaluation/evaluate.py --model_name "$model_name" --epoch 0012
done