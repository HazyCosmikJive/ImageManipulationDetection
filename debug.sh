#!/usr/bin/env bash

CONF=$1
DEVICE=$2

echo "Running with debug mode."

python main.py --config $CONF --gpu $DEVICE --debug