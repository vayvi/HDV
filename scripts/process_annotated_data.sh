#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

source "$ROOT_DIR"/venv/bin/activate

data_dir=${1:-"eida_dataset"}
python "$ROOT_DIR"/src/svg_helper/clean_svg_folder.py --input_folder "$data_dir"
python "$ROOT_DIR"/src/svg_helper/parse_svg.py --input_folder "$data_dir"
