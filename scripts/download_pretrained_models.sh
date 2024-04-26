#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$SCRIPT_DIR"/..
cd "$ROOT_DIR"

pip install gdown

MODEL_DIR=logs

mkdir -p "$MODEL_DIR"/
cd "$MODEL_DIR"/

mkdir -p main_model/
cd main_model/

# checkpoint 0012
gdown 15xY1bKqlZ07oaAALegahYkZR3t8YDIwq
# checkpoint 0036
gdown 1ia6aNNSyBIt721hfYvSro0nH8EWACOCw
# checkpoint 0044
gdown 1W-TiftAX40r7RvSWeBr3N1SvONdQTYhc

# config
gdown 1rZ2OvGskJKAbBgB4ccdjrbvm15c7Tem-
