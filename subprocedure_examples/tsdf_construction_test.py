
import sys
import nnrt
import open3d as o3d
import open3d.cuda.pybind.geometry


PROGRAM_EXIT_SUCCESS = 0


def main():
    # x = open3d.t.geometry.TSDFVoxelGrid(
    #     {
    #         'tsdf': o3d.core.Dtype.Float32,
    #         'weight': o3d.core.Dtype.UInt16,
    #         'color': o3d.core.Dtype.UInt16
    #     },
    #     voxel_size=0.005,
    #     sdf_trunc=0.025,
    #     block_resolution=16,
    #     block_count=1000,
    #     device=o3d.core.Device('cuda:0'))
    x = nnrt.geometry.WarpableTSDFVoxelGrid(
        {
            'tsdf': o3d.core.Dtype.Float32,
            'weight': o3d.core.Dtype.UInt16,
            'color': o3d.core.Dtype.UInt16
        },
        voxel_size=0.005,
        sdf_trunc=0.025,
        block_resolution=16,
        block_count=1000,
        device=o3d.core.Device('cuda:0'))

    return PROGRAM_EXIT_SUCCESS


if __name__ == "__main__":
    sys.exit(main())
