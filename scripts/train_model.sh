#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$SCRIPT_DIR"/src
python main.py --config_file config/DINO_4scale.py --output_dir ../logs/main_model_4scale --coco_path ""