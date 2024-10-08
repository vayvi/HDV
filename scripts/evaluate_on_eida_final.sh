#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
DATA_DIR="$ROOT_DIR"/data
#MODEL_DIR="$ROOT_DIR"/logs

source "$ROOT_DIR"/venv/bin/activate

model_name=${1:-"main_model"}
epoch=${2:-0036}
data_dir=${3:-"eida_dataset"}

if [ ! -f "$DATA_DIR/$data_dir/valid.json" ]; then
    echo -e "\n\033[1;93mProcessing ground truth for $data_dir\033[0m\n"
    bash scripts/process_annotated_data.sh "$data_dir"
fi

if [ ! -d "$DATA_DIR/$data_dir/valid_labels" ]; then
    echo -e "\n\033[1;93mGenerating ground truth for $data_dir\033[0m\n"
    python "$ROOT_DIR"/src/evaluation/generate_gt.py --data_root "$data_dir"
fi

if [ ! -d "$DATA_DIR/$data_dir/npz_preds_$model_name$epoch" ]; then
    echo -e "\n\033[1;93mGenerating predictions for $data_dir\033[0m\n"
    python "$ROOT_DIR"/src/evaluation/generate_preds.py --model_name "$model_name" --threshold 0 --epoch "$epoch" --data_folder_name "$data_dir"
fi

echo -e "\n\033[1;93mEvaluating predictions on $data_dir with $model_name$epoch\033[0m\n"
python "$ROOT_DIR"/src/evaluation/evaluate.py --model_name "$model_name" --epoch "$epoch" --data_folder_name "$data_dir"
