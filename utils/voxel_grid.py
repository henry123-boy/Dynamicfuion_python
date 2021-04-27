import settings.settings_tsdf as tsdf_settings


def make_default_tsdf_voxel_grid(device):
    import open3d as o3d
    return o3d.t.geometry.TSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=tsdf_settings.voxel_size,
        sdf_trunc=tsdf_settings.sdf_truncation_distance,
        block_resolution=tsdf_settings.block_resolution,
        block_count=tsdf_settings.initial_block_count,
        device=device)
