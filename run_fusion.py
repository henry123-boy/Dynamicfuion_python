#!/usr/bin/python3
import cupy # temp hack around cupy-114 and torch==1.9.1+cu111 compatibility issue

import cProfile
import os
from pathlib import Path

import sys
from settings import process_arguments, Parameters

from apps.fusion.pipeline import FusionPipeline

if __name__ == "__main__":
    process_arguments(help_header="A 3D reconstruction pipeline based on Neural Non-Rigid Tracking + "
                                  "DynamicFusion/Fusion4D + Open3D Spatial Hashing")
    settings_path = os.path.join(Path(__file__).parent.resolve(), "configuration_files/nnrt_fusion_parameters.yaml")

    pipeline = FusionPipeline()
    if Parameters.profile.value:
        cProfile.run('pipeline.run()')
    else:
        sys.exit(pipeline.run())
