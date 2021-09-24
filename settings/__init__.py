import os
from typing import Type
from pathlib import Path

from ext_argparse import ParameterEnum

from settings.deform_net import DeformNetParameters
from settings.path import PathParameters
from settings.graph import GraphParameters
from settings.model import ModelParameters
from settings.training import LearningParameters, BaselineComparisonParameters, LossParameters, TrainingParameters
from settings.tsdf import TsdfParameters
from settings.fusion import TrackingParameters, IntegrationParameters, VisualizationParameters, \
    LoggingParameters, TelemetryParameters, FusionParameters


class Parameters(ParameterEnum):
    model: Type[ModelParameters] = ModelParameters
    deform_net: Type[DeformNetParameters] = DeformNetParameters
    path: Type[PathParameters] = PathParameters
    tsdf: Type[TsdfParameters] = TsdfParameters
    fusion: Type[FusionParameters] = FusionParameters
    training: Type[TrainingParameters] = TrainingParameters
    graph: Type[GraphParameters] = GraphParameters


def process_arguments(help_header="A Neural Non-Rigid Fusion Application"):
    import ext_argparse
    default_configuration_path = os.path.join(Path(__file__).parent.parent.resolve(), "configuration_files/nnrt_fusion_parameters.yaml")
    ext_argparse.process_arguments(Parameters, help_header, default_settings_file=default_configuration_path,
                                   generate_default_settings_if_missing=True)
