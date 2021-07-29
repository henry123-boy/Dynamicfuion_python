
# Specify the dataset split for which to run evaluation
SPLIT="test"

export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 apps/evaluate.py --split=${SPLIT}