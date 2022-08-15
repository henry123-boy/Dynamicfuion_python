import nnrt
import open3d as o3d
from settings.tsdf import TsdfParameters


def make_default_tsdf_voxel_grid(device: o3d.core.Device) -> nnrt.geometry.NonRigidSurfaceVoxelBlockGrid:
    return nnrt.geometry.NonRigidSurfaceVoxelBlockGrid(
        ['tsdf', 'weight', 'color'],
        [o3d.core.Dtype.Float32, o3d.core.Dtype.UInt16, o3d.core.Dtype.UInt16],
        [o3d.core.SizeVector(1), o3d.core.SizeVector(1), o3d.core.SizeVector(3)],
        voxel_size=TsdfParameters.voxel_size.value,
        block_resolution=TsdfParameters.block_resolution.value,
        block_count=TsdfParameters.initial_block_count.value,
        device=device)
