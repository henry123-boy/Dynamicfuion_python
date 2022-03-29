import nnrt
import open3d as o3d
from settings.tsdf import TsdfParameters



def make_default_tsdf_voxel_grid(device: o3d.core.Device) -> nnrt.geometry.WarpableTSDFVoxelGrid:
    return nnrt.geometry.WarpableTSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=TsdfParameters.voxel_size.value,
        sdf_trunc=TsdfParameters.sdf_truncation_distance.value,
        block_resolution=TsdfParameters.block_resolution.value,
        block_count=TsdfParameters.initial_block_count.value,
        device=device)
