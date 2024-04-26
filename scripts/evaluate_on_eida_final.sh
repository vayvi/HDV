#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$SCRIPT_DIR"/..

model_name=$1
epoch=$2

python src/evaluation/generate_preds.py --model_name $model_name --threshold 0 --epoch $epoch
python src/evaluation/evaluate.py --model_name $model_name --epoch $epoch
