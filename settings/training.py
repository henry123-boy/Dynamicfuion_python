#  ================================================================
#  Created by Gregory Kramida (https://github.com/Algomorph).
#  Copyright (c) 2021 Gregory Kramida
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at

#  http://www.apache.org/licenses/LICENSE-2.0

#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#  ================================================================
from datetime import datetime

from ext_argparse import ParameterEnum, Parameter
from typing import Type


# TODO: distinction between "Training" & "Learning" is unclear. Rethink categories & reorganize.
class LearningParameters(ParameterEnum):
    # TODO: replace with enum
    use_adam = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Use Adam to train instead of SGD.")
    batch_size = \
        Parameter(default=4, arg_type=int,
                  arg_help="Size of each batch during training")
    evaluation_frequency = \
        Parameter(default=2000, arg_type=int,
                  arg_help="Period of validations.")
    epochs = \
        Parameter(default=15, arg_type=int,
                  arg_help="Total number of training epochs.")
    learning_rate = \
        Parameter(default=1e-5, arg_type=float,
                  arg_help="Learning rate during training.")
    use_lr_scheduler = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Whether to use the learning rate scheduler.")
    step_lr = \
        Parameter(default=1000, arg_type=int,
                  arg_help="Period of learning rate decay.")
    weight_decay = \
        Parameter(default=0.0, arg_type=float,
                  arg_help="Weight decay used by the training optimizer.")
    momentum = \
        Parameter(default=0.9, arg_type=float,
                  arg_help="Momentum used by the SGD training optimizer.")


class LossParameters(ParameterEnum):
    use_flow_loss = Parameter(default=True, arg_type='bool_flag',
                              arg_help="Switch that enables/disables flow linear_loss during training.")
    # TODO: Change to Enum
    flow_loss_type = \
        Parameter(default="RobustL1", arg_type=str,
                  arg_help="Type of flow linear_loss to use during training. May be one of: ['RobustL1', 'L2']")
    lambda_flow = \
        Parameter(default=5.0, arg_type=float,
                  arg_help="Weight of the flow linear_loss during neural network training.")
    use_graph_loss = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Switch that enables/disables graph linear_loss (BatchGraphL2 in alignment/linear_loss.py).")
    lambda_graph = \
        Parameter(default=2.0, arg_type=float,
                  arg_help="Weight of the graph linear_loss during neural network training.")
    use_warp_loss = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Switch that enables/disables warp linear_loss (L2_Warp in alignment/linear_loss.py).")
    lambda_warp = \
        Parameter(default=2.0, arg_type=float,
                  arg_help="Weight of the warp linear_loss during neural network training.")
    use_mask_loss = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Switch that enables/disables the mask linear_loss (weighted binary cross-entropy, "
                           "see DeformLoss.mask_bce_loss in alignment/linear_loss.py).")
    lambda_mask = \
        Parameter(default=1000.0, arg_type=float,
                  arg_help="Weight of the warp linear_loss during neural network training.")

    use_fixed_mask_loss_neg_wrt_pos_weight = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Controls the behaviour of the weighting of the BCE linear_loss on masks. "
                           "If set to true, mask_neg_wrt_pos_weight is used for weighing as opposed "
                           "to a dynamic weight based on ratio of positive & negative mask pixels. "
                           "For details, see DeformLoss.mask_bce_loss in alignment/linear_loss.py")
    mask_neg_wrt_pos_weight = \
        Parameter(default=0.05, arg_type=float,
                  arg_help="Fixed weight for the negative mask residuals in the BCE linear_loss on masks. "
                           "To be used, 'mask_neg_wrt_pos_weight' has to be set to True. For details, see "
                           "alignment/linear_loss.py.")


class BaselineComparisonParameters(ParameterEnum):
    min_neg_flowed_source_to_target_dist = \
        Parameter(default=0.3, arg_type=float,
                  arg_help="(Minimal) threshold for distance between correspondence match and ground truth flow point "
                           "for marking correspondence as 'negative'.")
    max_pos_flowed_source_to_target_dist = \
        Parameter(default=0.1, arg_type=float,
                  arg_help="(Maximal) threshold for distance between correspondence match and ground truth flow point "
                           "for marking correspondence as 'positive'.")


class TrainingParameters(ParameterEnum):
    train_labels_name = \
        Parameter(default="train_graphs", arg_type=str,
                  arg_help="The name (sans extension) of the json file with labels for the training data.")
    validation_labels_name = \
        Parameter(default="val_graphs", arg_type=str,
                  arg_help="The name (sans extension) of the json file with labels for the validation data.")
    experiment = \
        Parameter(default="debug_flow", arg_type=str,
                  arg_help="Training experiment name.")
    timestamp = \
        Parameter(default=datetime.now().strftime('%y-%m-%d-%H-%M-%S'), arg_type=str,
                  arg_help="Timestamp in the format \"y-m-d-H-M-S\" (if you do not want to use the current time).")
    use_pretrained_model = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Resume training from a pretrained model rather than from scratch.")
    num_worker_threads = \
        Parameter(default=6, arg_type=int,
                  arg_help="Passed to num_workers parameter of torch.utils.data.DataLoader constructor during training.")
    num_threads = \
        Parameter(default=4, arg_type=int,
                  arg_help="Number of threads used for intraop parallelism in PyTorch on the CPU "
                           "during training (passed to torch.set_num_threads).")
    num_samples_eval = \
        Parameter(default=700, arg_type=int,
                  arg_help="Number of samples used for evaluation (linear_loss computation) during training.")
    # TODO: probably should be grouped with evaluation_frequency and the above 'num_samples_eval' in something like ValidationParameters
    do_validation = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Evaluate trained model on validation split and print metrics during training.")
    shuffle = \
        Parameter(default=False, arg_type='bool_flag', arg_help="Shuffle each batch during training.")
    gn_invalidate_too_far_away_translations = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Invalidate for too-far-away estimations, since they can produce noisy gradient information.")
    gn_max_mean_translation_error = \
        Parameter(default=0.5, arg_type=float,
                  arg_help="What kind of estimation (point match) is considered too far away during the training.")

    learning: Type[LearningParameters] = LearningParameters
    loss: Type[LossParameters] = LossParameters
    baseline: Type[BaselineComparisonParameters] = BaselineComparisonParameters
