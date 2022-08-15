from ext_argparse import ParameterEnum, Parameter


class TsdfParameters(ParameterEnum):
    voxel_size = Parameter(default=0.005, arg_type=float, arg_help="Voxel resolution, in meters.")
    # TODO: move to another setting struct, e.g. VolumetricIntegrationParameters
    sdf_truncation_distance = Parameter(default=0.025, arg_type=float, arg_help="SDF truncation distance, in meters.")
    block_resolution = Parameter(default=16, arg_type=int,
                                 arg_help="SDF voxel block size (in voxels) used in the spatial hash.")
    initial_block_count = Parameter(default=1000, arg_type=int,
                                    arg_help="Initial number of blocks in the TSDF spatial hash.")
