from ext_argparse import ParameterEnum, Parameter
from enum import Enum


class DepthSamplingMode(Enum):
    BILINEAR = "bilinear"
    NEAREST = "nearest"


# TODO: figure out which DeformNet parameters *truly* refer to the GaussNewton solver,
#  remove the prefix 'gn' and move the parameters to GaussNewtonParameters instead
class DeformNetParameters(ParameterEnum):
    depth_sampling_mode = \
        Parameter(default=DepthSamplingMode.BILINEAR, arg_type=DepthSamplingMode,
                  arg_help="Sampling mode to use within the Gauss-Newton solver")
    gn_max_depth = \
        Parameter(default=6.0, arg_type=float,
                  arg_help="Far clipping distance for a point to be considered during alignment.")
    gn_min_nodes = \
        Parameter(default=4, arg_type=int,
                  arg_help="The minimum number of nodes in graph required for the Gauss-Newton solver to work.",
                  shorthand="maxnc")
    gn_max_nodes = \
        Parameter(default=300, arg_type=int,
                  arg_help="Number of nodes in graph not to be exceeded for the Gauss-Newton solver to work.",
                  shorthand="minnc")
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

    # TODO: depth_scale should be part of dataset & loaded from disk!
    depth_scale = \
        Parameter(default=1000.0, arg_type=float,
                  arg_help="Scale factor to multiply depth units in the depth image with in order to get meters.")
    freeze_optical_flow_net = \
        Parameter(default=False, arg_type='bool_flag', arg_help="Freeze/disable OpticalFlowNet during alignment.")
    freeze_mask_net = \
        Parameter(default=False, arg_type='bool_flag', arg_help="Freeze/disable MaskNet during alignment.")
    skip_solver = \
        Parameter(default=False, arg_type='bool_flag', arg_help="Skip Gauss-Newton optimization during alignment.")

    # TODO: need to replace threshold_mask_predictions & patchwise_threshold_mask_predictions with an enum specifying
    #  masking mode (i.e. [NO_THRESHOLD, HARD_THRESHOLD, PATCHWISE_THRESHOLD].
    #  Original code had:
    #  assert not (threshold_mask_predictions and patchwise_threshold_mask_predictions)
    threshold_mask_predictions = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="During alignment, keep only those matches for which the mask prediction is above a "
                           "threshold (Only applies if evaluating, must be disabled for generation).")
    threshold = \
        Parameter(default=0.35, arg_type=float,
                  arg_help="During alignment, keep only those matches for which the mask prediction is above this "
                           "threshold (Only applies if evaluating, must be disabled for generation). "
                           "Used only when threshold_mask_predictions is passed in / set to True")
    enforce_bidirectional_consistency = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Re-weight the correspondences based on bidirectional consistency (see "
                           "bidirectional_consistency_threshold).")
    bidirectional_consistency_threshold = \
        Parameter(default=0.20, arg_type=float,
                  arg_help="In meters, the bidirectional consistency error which will be used to filter out a "
                           "correspondence. Each point is transformed using keyframe/canonical<--->current frame "
                           "correspondence, then a backward correspondence is computed from the current to the "
                           "keyframe/canonical frame for all projected points. Ideally, each point should end up"
                           "at its origin. If the back-projection results in a displacement from the origin higher"
                           "than this threshold, the correspondence is rejected.")
    patchwise_threshold_mask_predictions = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Use patch-wise threshold when applying mask during the alignment process instead of the "
                           "hard threshold.")
    patch_size = \
        Parameter(default=False, arg_type='bool_flag',
                  arg_help="Mask patch size when the patch-wise threshold is used during mask application in "
                           "the alignment.")
    use_mask = \
        Parameter(default=True, arg_type='bool_flag',
                  arg_help="DeformNet will use correspondence masking via MaskNet if enabled.")
    use_batch_norm = \
        Parameter(default=False, arg_type='bool_flag', arg_help="Use batch normalization.")
