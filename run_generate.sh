#!/bin/bash
# Specify the dataset split for which to generate predictions
SPLIT="val"

export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 apps/generate.py --split=${SPLIT}