
# Name of the train and val datasets
train_dir="train_graphs"
val_dir="val_graphs"

# Give a name to your experiment
experiment="debug_flownet"
echo ${experiment}

GPU=${1:-0}

export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=${GPU} python3 train.py --train_dir="${train_dir}" \
                                            --val_dir="${val_dir}" \
                                            --experiment="${experiment}"