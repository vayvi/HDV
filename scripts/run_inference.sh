#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$SCRIPT_DIR"/..

model_name=$1 || "main_model"
epoch=$2 || "0036"
data_dir=$3 || "eida_final"

echo -e "\n\033[1;93mPredicting primitives in $data_dir images\033[0m\n"
python "$ROOT_DIR"/src/inference.py --model_name $model_name --epoch $epoch --data_folder_name $data_dir