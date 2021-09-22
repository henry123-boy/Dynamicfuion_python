from ext_argparse import ParameterEnum, Parameter


class LearningParameters(ParameterEnum):
    # TODO: replace with enum
    use_adam = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Use Adam to train instead of SGD.")
    use_batch_norm = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Use batch normalization.")
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
                              arg_help="Switch that enables/disables flow loss during training.")
    # TODO: Change to Enum
    flow_loss_type = \
        Parameter(default="RobustL1", arg_type=str,
                  arg_help="Type of flow loss to use during training. May be one of: ['RobustL1', 'L2']")
    lambda_flow = \
        Parameter(default=5.0, arg_type=float,
                  arg_help="Weight of the flow loss during neural network training.")
    use_graph_loss = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Switch that enables/disables graph loss (BatchGraphL2 in alignment/loss.py).")
    lambda_graph = \
        Parameter(default=2.0, arg_type=float,
                  arg_help="Weight of the graph loss during neural network training.")
    use_warp_loss = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Switch that enables/disables warp loss (L2_Warp in alignment/loss.py).")
    lambda_warp = \
        Parameter(default=2.0, arg_type=float,
                  arg_help="Weight of the warp loss during neural network training.")
    use_mask_loss = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Switch that enables/disables the mask loss (weighted binary cross-entropy, "
                           "see DeformLoss.mask_bce_loss in alignment/loss.py).")
    lambda_mask = \
        Parameter(default=1000.0, arg_type=float,
                  arg_help="Weight of the warp loss during neural network training.")
    use_fixed_mask_loss_neg_wrt_pos_weight = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Controls the behaviour of the weighting of the BCE loss on masks. "
                           "If set to true, mask_neg_wrt_pos_weight is used for weighing as opposed "
                           "to a dynamic weight based on ratio of positive & negative mask pixels. "
                           "For details, see DeformLoss.mask_bce_loss in alignment/loss.py")
    mask_neg_wrt_pos_weight = \
        Parameter(default=0.05, arg_type=float,
                  arg_help="Fixed weight for the negative mask values in the BCE loss on masks. "
                           "To be used, 'mask_neg_wrt_pos_weight' has to be set to True. For details, see "
                           "alignment/loss.py.")


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
    use_pretrained_model = Parameter(default=False, arg_type='bool_flag',
                                     arg_help="Resume training from a pretrained model rather than from scratch.")
    num_worker_threads = Parameter(default=6, arg_type=int,
                                   arg_help="Passed to num_workers parameter of torch.utils.data.DataLoader constructor "
                                            "during training.")
    num_threads = Parameter(default=4, arg_type=int,
                            arg_help="Number of threads used for intraop parallelism in PyTorch on the CPU "
                                     "during training (passed to torch.set_num_threads).")
    num_samples_eval = Parameter(default=700, arg_type=int,
                                 arg_help="Number of samples used for evaluation (loss computation) during training.")
    do_validation = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Evaluate trained model on validation split and print metrics during training.")

    shuffle = Parameter(default=False, arg_type='bool_flag', arg_help="Shuffle each batch during training.")

    gn_invalidate_too_far_away_translations = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Invalidate for too-far-away estimations, since they can produce noisy gradient information.")

    gn_max_mean_translation_error = \
        Parameter(default=0.5, arg_type=float,
                  arg_help="What kind of estimation (point match) is considered too far away during the training.")

    learning = LearningParameters
    loss = LossParameters
    baseline = BaselineComparisonParameters
