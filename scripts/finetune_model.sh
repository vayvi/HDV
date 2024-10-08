#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

source "$ROOT_DIR"/venv/bin/activate

model_name=${1:-"main_model"}
ground_truth=$2
device_nb=${3:-0}
batch_size=${4:-2}
max_size=${5:-1000} # reduce to prevent torch.cuda.OutOfMemoryError
learning_rate=${6:-0.0001}
epoch_nb=${7:-50}

export CUDA_VISIBLE_DEVICES=$device_nb

if [ -z "$ground_truth" ]; then
    echo "No ground truth provided"
    echo "Usage: $0 <model_name> <ground_truth> <device_nb?> <batch_size?> <max_size?> <learning_rate?> <epoch_nb?>"
    exit 1
fi

model_dir="$ROOT_DIR"/logs/"$model_name"

if [ ! -d "$model_dir" ]; then
    echo "Model directory $model_dir does not exist"
    exit 1
fi

models=$(ls "$model_dir"/checkpoint*.pth 2>/dev/null)
config_file=$(ls "$model_dir"/config_cfg.py 2>/dev/null)

if [ -z "$models" ]; then
    echo "No model was found inside $model_dir"
    exit 1
fi
if [ -z "$config_file" ]; then
    config_file="$ROOT_DIR"/src/config/DINO_4scale.py
fi

highest_epoch=$(printf "%s\n" "$models" | grep -o 'checkpoint[0-9]\+' | grep -o '[0-9]\+' | sort -n | tail -n 1)
model_file="checkpoint${highest_epoch}.pth"

echo "Using model $model_file"

data_dir="$ROOT_DIR"/data/"$ground_truth"

if [ ! -d "$data_dir" ]; then
    echo "Data directory $data_dir does not exist"
    exit 1
fi

echo -e "\n\033[1;93mFinetuning $model_file with $data_dir\033[0m\n"

finetuned_model="$model_name"_finetuned

output_dir="$ROOT_DIR"/logs/"$finetuned_model"
if [ -d "$output_dir" ]; then
    output_dir="$output_dir"_"$(date +'%Y-%m-%d')"
fi

cd "$ROOT_DIR"/src
python main.py \
    --pretrain_model_path "$model_dir/$model_file" \
    --config_file "$config_file" \
    --output_dir "$output_dir" \
    --coco_path "$data_dir" \
    --options batch_size="$batch_size" \
            on_the_fly=False \
            on_the_fly_val=False \
            data_aug_max_size="$max_size" \
            lr="$learning_rate" \
            epochs="$epoch_nb" \
            use_wandb=True
