#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$SCRIPT_DIR"/..

cd "$ROOT_DIR"/src
python main.py --config_file config/DINO_4scale.py --output_dir "$ROOT_DIR"/logs/main_model_4scale --coco_path ""