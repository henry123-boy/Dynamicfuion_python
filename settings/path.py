import os
from collections import namedtuple
from pathlib import Path
from ext_argparse import ParameterEnum, Parameter

LocalPathCollection = namedtuple("LocalPathCollection", "deep_deform_root output")

# to add your own root DeepDeform data directory, run the sha256 cypher on your MAC address and add the hash &
# local directory as a key/value pair to the dict below
REPOSITORY_ROOT = Path(__file__).parent.parent.resolve().absolute()

DEFAULT_OUTPUT_DIRECTORY = os.path.join(REPOSITORY_ROOT, "output")
# TODO: since we now have a more-legitimate output folder, rename the experiments to something better than the legacy
#  "experiments". It's a folder holding trained neural network data.
DEFAULT_EXPERIMENTS_DIRECTORY = os.path.join(REPOSITORY_ROOT, "experiments")


class PathParameters(ParameterEnum):
    dataset_base_directory = \
        Parameter(default="datasets/DeepDeform", arg_type=str,
                  arg_help="Path to the base of the DeepDeform dataset root.")
    output_directory = \
        Parameter(default=DEFAULT_OUTPUT_DIRECTORY, arg_type=str,
                  arg_help="Path to the directory where reconstruction output will be placed.")
    experiments_directory = \
        Parameter(default=DEFAULT_EXPERIMENTS_DIRECTORY, arg_type=str,
                  arg_help="Path to the directory where trained DeformNet models are stored.")



