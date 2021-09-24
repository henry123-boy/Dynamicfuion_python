
# Specify the dataset split for which to run evaluation
SPLIT="VALIDATION"

export PYTHONPATH="$(pwd):$PYTHONPATH"
python3 apps/evaluate.py --evaluate_split=${SPLIT}