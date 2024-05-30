#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
MODEL_DIR="$ROOT_DIR/logs"

ground_truth=$1
model_name=$2
device_nb=${3:-0}
batch_size=${4:-2}
max_size=${5:-1000} # reduce to prevent torch.cuda.OutOfMemoryError

source "$ROOT_DIR/venv/bin/activate"
export CUDA_VISIBLE_DEVICES=$device_nb

if [ -z "$ground_truth" ]; then
    echo "No ground truth provided"
    echo "Usage: $0 <ground_truth> <model_name?> <device_nb?> <batch_size?> <max_size?>"
    exit 1
fi

data_dir="$ROOT_DIR/data/$ground_truth"

if [ ! -d "$data_dir" ]; then
    echo "Data directory $data_dir does not exist"
    exit 1
fi

if [ -z "$model_name" ]; then
    model_dirs=("$MODEL_DIR"/*/)
else
    model_dirs=("$MODEL_DIR/$model_name")
fi

evalModel() {
    model_dir="$1"
    gt_dir="$2"
    config_file=$(ls "$model_dir"/config_cfg.py 2>/dev/null)
    models=$(ls "$model_dir"/checkpoint*.pth 2>/dev/null)

    if [ -z "$models" ]; then
        echo "No model was found inside $model_dir"
        return
    fi

    if [ -z "$config_file" ]; then
        config_file="$ROOT_DIR/src/config/DINO_4scale.py"
    fi

    python "$ROOT_DIR/src/main.py" \
        --eval_all \
        --config_file "$config_file" \
        --output_dir "$model_dir" \
        --coco_path "$gt_dir" \
        --options batch_size="$batch_size" \
                on_the_fly=False \
                on_the_fly_val=False \
                data_aug_max_size="$max_size" \
                use_wandb=True
}

for dir in "${model_dirs[@]}"; do
    if [ -d "$dir" ]; then
        evalModel "$dir" "$data_dir"
    fi
done