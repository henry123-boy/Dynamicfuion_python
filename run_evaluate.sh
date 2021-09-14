
# Specify the dataset split for which to run evaluation
SPLIT="val"

export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 apps/evaluate.py --split=${SPLIT}