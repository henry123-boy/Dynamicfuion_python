
SPLIT="test"

export PYTHONPATH="$(pwd):$PYTHONPATH"

echo
echo "GENERATE..."
python3 apps/generate.py --split=${SPLIT}

echo
echo "EVALUATE..."
python3 apps/evaluate.py --split=${SPLIT}