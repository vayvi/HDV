#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

source "$ROOT_DIR"/venv/bin/activate

# Define the model names
declare -a model_names=("wo-multi-scale" "w-pure-query-selection" "wo-denoising" "wo-label-flipping" "wo-2stage" "main_model")
epoch=$1

# Loop over the model names
for model_name in "${model_names[@]}"
do
    python "$ROOT_DIR"/evaluation/generate_preds.py --model_name "$model_name" --threshold 0 --epoch "$epoch"

    python "$ROOT_DIR"/evaluation/evaluate.py --model_name "$model_name" --epoch 0012
done