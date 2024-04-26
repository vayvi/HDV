#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

folder_name=$1
python "$SCRIPT_DIR"/svg_helper/clean_svg_folder.py --input_folder $folder_name
python "$SCRIPT_DIR"/svg_helper/parse_svg.py --input_folder $folder_name
