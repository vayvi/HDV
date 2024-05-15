#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

set -e
wget 'https://www.dropbox.com/s/tiqqb166f5ygzx2/synthetic_resource.zip?dl=0' --output-document synthetic_resource.zip
unzip synthetic_resource.zip -d data/
rm synthetic_resource.zip
