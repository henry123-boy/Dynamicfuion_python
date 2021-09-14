from ext_argparse import ParameterEnum, Parameter


# TODO: combine with related TrackingParameters -- reorganize
class GraphParameters(ParameterEnum):
    node_coverage = \
        Parameter(default=0.05, arg_type=float,
                  arg_help="This is the maximum distance between any point in the source point cloud and at least one "
                           "of the resulting graph nodes. Allows to control graph sparsity and influences a number of "
                           "other operations that have to do with node influence over surrounding points.")
    # TODO: refactor all parameters with 'graph_' prefix -- remove the prefix

    graph_debug = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Show & print debug output during graph generation.")
    # TODO: refactor to max_mesh_triangle_edge_length
    graph_max_triangle_distance = \
        Parameter(default=0.05, arg_type=float,
                  arg_help="This is actually the maximum edge length allowed for any triangles generated from an"
                           "RGB-D image pair / resulting point cloud during graph construction.")
    graph_erosion_num_iterations = \
        Parameter(default=4, arg_type=int,
                  arg_help="Number of erosion iterations applied to the graph during generation.")
    graph_erosion_min_neighbors = \
        Parameter(default=4, arg_type=int,
                  arg_help="While the graph is being eroded (during generation), the nodes not having the required"
                           "minimum neighbor count will be removed.")
    graph_use_only_valid_vertices = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Whether to use eroded nodes during sampling or not.")
    # TODO: refactor to max_neighbor_count
    graph_neighbor_count = \
        Parameter(default=8, arg_type=int,
                  arg_help="Maximum possible number of neighbors each node in the graph has after generation. "
                           "Corresponds to the width of the edge table/2d array.")
    graph_enforce_neighbor_count = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Whether to enforce the neighbor count during graph generation. If set to true,"
                           "even neighbors beyond the maximal edge influence (2*node_coverage) will be filled "
                           "in the edge table, so that each node has exactly neighbor_count neighbors. ")
    graph_sample_random_shuffle = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Whether to use random node shuffling during node sampling for graph generation.")
    graph_remove_nodes_with_too_few_neighbors = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Whether to remove nodes with \"too few\" neighbors before rolling out the graph. "
                           "Currently, the \"too few\" condition is hard-coded as \"one or fewer\".")


# Info for a saved alignment
# - In train.py, this info is only used if use_pretrained_model=True
# - In generate.py, evaluate.py or example_viz.py, it is used regardless of the value of use_pretrained_model
class ModelParameters(ParameterEnum):
    # TODO: switch to an Enum parameter
    model_module_to_load = \
        Parameter(default="full_model", arg_type=str,
                  arg_help="Must be set to one of ['only_flow_net', 'full_model']. Dictates whether the model will be"
                           "loaded in full or only the flow_net part will be loaded.")
    model_name = \
        Parameter(default="model_A", arg_type=str,
                  arg_help="Name of the pre-trained model to use.")
    model_iteration = \
        Parameter(default=0, arg_type=int,
                  arg_help="Iteration number of the model to load.")


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


class BaselineComparisonParameters(ParameterEnum):
    min_neg_flowed_source_to_target_dist = \
        Parameter(default=0.3, arg_type=float,
                  arg_help="(Minimal) threshold for distance between correspondence match and ground truth flow point "
                           "for marking correspondence as 'negative'.")
    max_pos_flowed_source_to_target_dist = \
        Parameter(default=0.1, arg_type=float,
                  arg_help="(Maximal) threshold for distance between correspondence match and ground truth flow point "
                           "for marking correspondence as 'positive'.")


# TODO: remove the prefix 'gn'
class SolverParameters(ParameterEnum):
    # TODO: use Enum instead
    gn_depth_sampling_mode = \
        Parameter(default="bilinear", arg_type=str,
                  arg_help="Sampling mode to use within the Gauss-Newton solver. Can be one of ['bilinear', 'nearest']")
    gn_max_depth = \
        Parameter(default=6.0, arg_type=float,
                  arg_help="Far clipping distance for a point to be considered by the Gauss-Newton solver during alignment.")
    gn_min_nodes = \
        Parameter(default=4, arg_type=int,
                  arg_help="The minimum number of nodes in graph required for the Gauss-Newton solver to work.")
    gn_max_nodes = \
        Parameter(default=300, arg_type=int,
                  arg_help="Number of nodes in graph not to be exceeded for the Gauss-Newton solver to work.")
    gn_max_matches_train = \
        Parameter(default=10000, arg_type=int,
                  arg_help="Maximum number of matching points when working on a sequence in the 'train' split of the "
                           "DeepDeform dataset. If the actual number of matches exceeds, excess matches will be "
                           "filtered out and discarded at random.")
    gn_max_matches_eval = \
        Parameter(default=10000, arg_type=int,
                  arg_help="Maximum number of matching points when working on a sequence in the 'val' or 'test' split "
                           "of the DeepDeform dataset. If the actual number of matches exceeds, excess matches will be "
                           "filtered out and discarded at random.")
    # TODO: why is 'train' in the name here? Is this not used equally regardless of the split?
    gn_max_matches_train_per_batch = \
        Parameter(default=100000, arg_type=int,
                  arg_help="Maximum number of matching points per batch. If the actual number of matches exceeds, "
                           "the program will terminate with an error.")
    gn_max_warped_points = \
        Parameter(default=100000, arg_type=int,
                  arg_help="Maximal number of deformed points. Usually at or greater than gn_max_matches_train_per_batch "
                           "and gn_max_matches_train/gn_max_matches_eval. This actually dictates the total size of the "
                           "tensor used to store the deformed points.")
    gn_debug = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Print telemetry info during the Gauss-Newton optimization.")
    gn_print_timings = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Print timing information for each term and the composite during the Gauss-Newton optimization.")
    gn_use_edge_weighting = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Use edge weight information within the ARAP energy term during the Gauss-Newton optimization.")
    gn_check_condition_num = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Check the determinant/condition number of the 'A' matrix after every optimization iteration. "
                           "If the condition number is unstable, break the optimization (unless breaking is disabled by"
                           "the corresponding argument).")
    gn_break_on_condition_num = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Whether to break the optimization if the condition number is unstable (only used if"
                           "--check_condition_num flag is passed in / set to 'True' in the configuration file). "
                           "The exact threshold can be controlled by --max_condition_num")
    gn_max_condition_num = \
        Parameter(default=1e6, arg_type=float,
                  arg_help="Minimum value of the condition number considered unstable.")
    gn_remove_clusters_with_few_matches = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="Remove clusters with too few matches during the Gauss-Newton optimization. "
                           "Further tuned by the min_num_correspondences_per_cluster parameter.")
    gn_min_num_correspondences_per_cluster = \
        Parameter(default=2000, arg_type=int,
                  arg_help="When used in conjunction with the remove_clusters_with_few_matches parameter, "
                           "defines the threshold below which the cluster is removed wholly from the rest of the "
                           "computation.")
    gn_num_iter = \
        Parameter(default=3, arg_type=int,
                  arg_help="Total number of Gauss-Newton solver iterations to run.")
    gn_data_flow = \
        Parameter(default=0.001, arg_type=float,
                  arg_help="Data term coefficient used in the 'flow' part of the Jacobian computations "
                           "within the Gauss-Newton solver.")
    gn_data_depth = \
        Parameter(default=1.0, arg_type=float,
                  arg_help="Data term coefficient used in the 'depth' part of the Jacobian computations "
                           "within the Gauss-Newton solver.")
    gn_arap = \
        Parameter(default=1.0, arg_type=float,
                  arg_help="ARAP term coefficient used in the Jacobian computations within the Gauss-Newton solver.")
    gn_lm_factor = \
        Parameter(default=0.1, arg_type=float,
                  arg_help="Small damping factor applied to the A=J^TJ matrix during the optimization.")


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


class AlignmentParameters(ParameterEnum):
    alignment_image_width = \
        Parameter(default=640, arg_type=int,
                  arg_help="Input image/point cloud height for the non-rigid alignment portion of the algorithm. "
                           "The actual image / point cloud will be cropped down to this height and intrinsic matrix "
                           "adjusted accordingly.")
    alignment_image_height = \
        Parameter(default=448, arg_type=int,
                  arg_help="Input image/point cloud width for the non-rigid alignment portion of the algorithm. "
                           "The actual image / point cloud will be cropped down to this width and intrinsic matrix "
                           "adjusted accordingly.")
    # TODO: depth_scale should be part of dataset & loaded from disk!
    depth_scale = \
        Parameter(default=1000.0, arg_type=float,
                  arg_help="Scale factor to multiply depth units in the depth image with in order to get meters.")
    freeze_optical_flow_net = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Freeze/disable OpticalFlowNet during alignment.")
    freeze_mask_net = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Freeze/disable MaskNet during alignment.")
    skip_solver = \
        Parameter(default=False, arg_type='bool_flag', arg_help="Skip Gauss-Newton optimization during alignment.")
    max_boundary_dist = \
        Parameter(default=0.10, arg_type=float,
                  arg_help="Used in marking up boundaries within an incoming RGB-D image pair. When neighboring pixel "
                           "points within the point cloud based on the depth image exceed this distance from each other, "
                           "the boundaries are drawn along the break.")
    # TODO: need to replace threshold_mask_predictions & patchwise_threshold_mask_predictions with an enum specifying
    #  masking mode (i.e. [NO_THRESHOLD, HARD_THRESHOLD, PATCHWISE_THRESHOLD].
    #  Original code had:
    #  assert not (threshold_mask_predictions and patchwise_threshold_mask_predictions)
    threshold_mask_predictions = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="During alignment, keep only those matches for which the mask prediction is above a "
                           "threshold (Only applies if evaluating, must be disabled for generation).")
    threshold = \
        Parameter(default=0.30, arg_type=float,
                  arg_help="During alignment, keep only those matches for which the mask prediction is above this "
                           "threshold (Only applies if evaluating, must be disabled for generation). "
                           "Used only when threshold_mask_predictions is passed in / set to True")
    patchwise_threshold_mask_predictions = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Use patch-wise threshold when applying mask during the alignment process instead of the "
                           "hard threshold.")
    patch_size = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Mask patch size when the patch-wise threshold is used during mask application in "
                           "the alignment.")
