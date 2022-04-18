#!/bin/bash

export PYTHONPATH="./"

if [[ "$1" =~ ^(train_mono|train_mono_fs|validate_mono|validate_mono_fs)$ ]]; then
    python3 src/$1.py -c src/$1_config.json
else
    echo "Unknown scenario '$1'"
fi