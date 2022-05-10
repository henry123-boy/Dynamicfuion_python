#!/bin/bash
# Name of the train and val datasets
train_labels_name="train_graphs"
validation_labels_name="val_graphs"

# Give a name to your experiment
experiment="debug_flownet"
echo ${experiment}

GPU=${1:-0}

export PYTHONPATH="$(pwd):$PYTHONPATH"
CUDA_VISIBLE_DEVICES=${GPU} python3 apps/train.py --training.train_labels_name="${train_labels_name}" \
                                            --training.validation_labels_name="${validation_labels_name}" \
                                            --training.experiment="${experiment}"