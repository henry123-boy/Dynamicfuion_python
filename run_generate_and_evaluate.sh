#!/bin/bash
SPLIT="TEST"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo
echo "GENERATE..."
python3 apps/generate.py --generate_split=${SPLIT}  --deform_net.no-threshold_mask_predictions --deform_net.gn_max_matches_eval=100000

echo
echo "EVALUATE..."
python3 apps/evaluate.py --evaluate_split=${SPLIT}