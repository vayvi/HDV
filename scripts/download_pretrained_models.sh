#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$SCRIPT_DIR"/..

pip install gdown

mkdir -p models/
cd models/
# checkpoint 0012
gdown 15xY1bKqlZ07oaAALegahYkZR3t8YDIwq
# checkpoint 0036
gdown 1ia6aNNSyBIt721hfYvSro0nH8EWACOCw
# checkpoint 0044
gdown 1W-TiftAX40r7RvSWeBr3N1SvONdQTYhc

# config
gdown 1rZ2OvGskJKAbBgB4ccdjrbvm15c7Tem-
