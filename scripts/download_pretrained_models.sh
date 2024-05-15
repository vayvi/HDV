#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

pip install gdown

MODEL_DIR="$ROOT_DIR"/logs/

mkdir -p logs/
cd "$MODEL_DIR"

mkdir -p main_model/
cd main_model/

# checkpoint 0012
gdown 15xY1bKqlZ07oaAALegahYkZR3t8YDIwq
# checkpoint 0036
gdown 1ia6aNNSyBIt721hfYvSro0nH8EWACOCw
# config (num_queries=900 / two_stage_learn_wh=True)
gdown 12zY5560OJL0e-lmTjrYSY-jO3VeAZosk

cd "$MODEL_DIR"

mkdir -p eida_demo_model/
cd eida_demo_model/

# checkpoint 0044
gdown 1W-TiftAX40r7RvSWeBr3N1SvONdQTYhc
# config (num_queries=1400 / two_stage_learn_wh=False)
gdown 1rZ2OvGskJKAbBgB4ccdjrbvm15c7Tem-
