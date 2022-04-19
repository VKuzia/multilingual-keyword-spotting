#!/bin/bash

export PYTHONPATH="./"

if [[ "$1" =~ ^(train|train_fs|validate|validate_fs)$ ]]; then
    python3 src/$1.py -c src/$1_config.json
elif [[ "$1" =~ ^(profile_train|profile_train_fs)$ ]]; then
    python3 -m cProfile -o stats.txt src/$1.py -c src/$1_config.json
    python3 profiling/stats.py
elif [[ "$1" =~ ^(mswc_csv_formating|mswc_opus_to_wav|mswc_wav_to_spec)$ ]]; then
    python3 src/$1.py
else
    echo "Unknown scenario '$1'"
fi