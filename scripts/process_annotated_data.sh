#!/bin/bash

folder_name=$1
python svg_helper/clean_svg_folder.py --input_folder $folder_name
python svg_helper/parse_svg.py --input_folder $folder_name 
