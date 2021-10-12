#!/bin/bash
# Specify the dataset split for which to generate predictions
SPLIT="VALIDATION"

export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 apps/generate.py --generate_split=${SPLIT} --deform_net.no-threshold_mask_predictions --deform_net.gn_max_matches_eval=100000