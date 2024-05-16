#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

source "$ROOT_DIR"/venv/bin/activate

model_name=${1:-"main_model"}
groundtruth=$2
device_nb=${3:-0}
batch_size=${4:-2}

export CUDA_VISIBLE_DEVICES=$device_nb

if [ -z "$groundtruth" ]; then
    echo "No groundtruth provided"
    echo "Usage: $0 <model_name> <groundtruth> <device_nb?> <batch_size?>"
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
    echo "No config file was found inside $model_dir"
    exit 1
fi

highest_epoch=$(printf "%s\n" "$models" | grep -o 'checkpoint[0-9]\+' | grep -o '[0-9]\+' | sort -n | tail -n 1)
model_file="checkpoint${highest_epoch}.pth"

echo "Using model $model_file"

data_dir="$ROOT_DIR"/data/"$groundtruth"

if [ ! -d "$data_dir" ]; then
    echo "Data directory $data_dir does not exist"
    exit 1
fi

echo -e "\n\033[1;93mFinetuning $model_file with $data_dir\033[0m\n"

output_dir="$ROOT_DIR"/logs/"$model_name"_finetuned
if [ -d "$output_dir" ]; then
    output_dir="$output_dir"_"$(date +'%Y-%m-%d_%H-%M')"
fi

cd "$ROOT_DIR"/src
python main.py --pretrain_model_path "$model_dir/$model_file" --config_file "$config_file" --output_dir "$output_dir" --coco_path "$data_dir" --options batch_size=$batch_size on_the_fly=False --use_wandb
