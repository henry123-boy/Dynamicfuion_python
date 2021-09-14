
# Name of the train and val datasets
train_dir="train"
val_dir="val"

# Give a name to your experiment
experiment="debug_flownet"
echo ${experiment}

GPU=${1:-0}

export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=${GPU} python3 apps/train.py --train_dir="${train_dir}" \
                                            --val_dir="${val_dir}" \
                                            --experiment="${experiment}"