import argparse
import cProfile
import sys

from apps.fusion.fusion_pipeline import FusionPipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Basic Fusion Pipeline based on Neural Non-Rigid Tracking + Fusion4D + Open3D spatial hashing")
    parser.add_argument("--profile", action='store_true')
    args = parser.parse_args()
    pipeline = FusionPipeline()
    if args.profile:
        cProfile.run('tsdf_management.run()')
    else:
        sys.exit(pipeline.run())