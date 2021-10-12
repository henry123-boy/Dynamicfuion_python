import os
from typing import Type
from pathlib import Path

from ext_argparse import ParameterEnum, Parameter

from settings.deform_net import DeformNetParameters
from settings.path import PathParameters
from settings.graph import GraphParameters
from settings.model import ModelParameters
from settings.training import LearningParameters, BaselineComparisonParameters, LossParameters, TrainingParameters
from settings.tsdf import TsdfParameters
from settings.fusion import TrackingParameters, IntegrationParameters, VisualizationParameters, \
    LoggingParameters, TelemetryParameters, FusionParameters

from settings.split import Split


class Parameters(ParameterEnum):
    model: Type[ModelParameters] = ModelParameters
    deform_net: Type[DeformNetParameters] = DeformNetParameters
    path: Type[PathParameters] = PathParameters
    tsdf: Type[TsdfParameters] = TsdfParameters
    fusion: Type[FusionParameters] = FusionParameters
    training: Type[TrainingParameters] = TrainingParameters
    graph: Type[GraphParameters] = GraphParameters

    generate_split = Parameter(default=Split.VALIDATION, arg_type=Split,
                               arg_help="Specify the dataset split for which to generate predictions.")
    evaluate_split = Parameter(default=Split.VALIDATION, arg_type=Split,
                               arg_help="Specify the dataset split for which to compute evaluation metrics. Assumes predictions have been generated.")
    profile = Parameter(default=False, arg_type="bool_flag",
                        arg_help="Run the profiler to determine bottlenecks (where relevant).")


def process_arguments(help_header="A Neural Non-Rigid Fusion Application"):
    import ext_argparse
    default_configuration_path = os.path.join(Path(__file__).parent.parent.resolve(), "configuration_files/nnrt_fusion_parameters.yaml")
    return ext_argparse.process_arguments(Parameters, help_header, default_settings_file=default_configuration_path,
                                          generate_default_settings_if_missing=True)


def read_settings_file():
    import ext_argparse
    default_configuration_path = os.path.join(Path(__file__).parent.parent.resolve(), "configuration_files/nnrt_fusion_parameters.yaml")
    return ext_argparse.process_settings_file(Parameters, default_configuration_path, generate_default_settings_if_missing=True)
