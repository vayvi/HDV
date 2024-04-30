#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$SCRIPT_DIR"/..

folder_name=$1 || "eida_dataset"
python "$ROOT_DIR"/svg_helper/clean_svg_folder.py --input_folder "$folder_name"
python "$ROOT_DIR"/svg_helper/parse_svg.py --input_folder "$folder_name"
