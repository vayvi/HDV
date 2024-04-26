#!/bin/bash

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

cd "$SCRIPT_DIR"

pip install gdown
sudo apt install unzip

gdown 1U-ArV5IIVfssmxHf8WT-E6cohgxZSDdN
unzip -q eida_dataset.zip

cd ../
mkdir -p data/
mv scripts/eida_dataset data/

rm scripts/eida_dataset.zip
rm -r scripts/__MACOSX