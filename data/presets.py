from enum import Enum
from data.frame import StandaloneFrameDataset, DatasetType, DataSplit
from data.frame_pair import FramePairDataset
from data.frame_sequence import FrameSequenceDataset, StaticFrameSequenceDataset


class FramePairPreset(Enum):
    # uncomment only after edges become available
    # GREY_SHIRT_TEST_LOCAL = FramePairDataset(300, 600, 17, DataSplit.TEST, DatasetType.LOCAL, has_masks=False)
    # GREY_SHIRT_TEST = FramePairDataset(300, 600, 17, DataSplit.TEST, has_masks=False)
    # GREY_SHIRT_TRAIN_LOCAL = FramePairDataset(0, 110, 258, DataSplit.TRAIN, DatasetType.LOCAL, has_masks=False)
    RED_SHORTS_200_400 = FramePairDataset(200, 400, 14, DataSplit.VALIDATION, has_masks=True)


class StandaloneFramePreset(Enum):
    RED_SHORTS_0 = StandaloneFrameDataset(0, 14, DataSplit.VALIDATION, has_masks=True)
    RED_SHORTS_200 = StandaloneFrameDataset(200, 14, DataSplit.VALIDATION, has_masks=True)
    RED_SHORTS_400 = StandaloneFrameDataset(400, 14, DataSplit.VALIDATION, has_masks=True)
    BERLIN_0 = StandaloneFrameDataset(0, 70, DataSplit.TRAIN, has_masks=True)


class FrameSequencePreset(Enum):
    RED_SHORTS_40 = FrameSequenceDataset(14, DataSplit.VALIDATION, frame_count=40, has_masks=False)
    RED_SHORTS = FrameSequenceDataset(14, DataSplit.VALIDATION, has_masks=False)
    BERLIN_50 = FrameSequenceDataset(70, DataSplit.TRAIN, frame_count=50, has_masks=False)
    BERLIN = FrameSequenceDataset(70, DataSplit.TRAIN, has_masks=False)
    BERLIN_STATIC = StaticFrameSequenceDataset(70, DataSplit.TRAIN, frame_count=6, has_masks=False)
    # The BERLIN OFFSET, ROTATION, SCALE, ETC. sequences can be generated from the BERLIN_0 sequence using scripts
    # such as pipeline/data_generation/animate_berlin_x_offset.py
    # (comment these out if you get an error here)
    BERLIN_OFFSET_X = FrameSequenceDataset(
        base_dataset_type=DatasetType.CUSTOM,
        custom_frame_directory="/home/algomorph/Workbench/NeuralTracking/output/berlin_x_offset_sequence")
    BERLIN_OFFSET_XY = FrameSequenceDataset(
        base_dataset_type=DatasetType.CUSTOM,
        custom_frame_directory="/home/algomorph/Workbench/NeuralTracking/output/berlin_xy_offset_sequence")
    BERLIN_ROTATION_Z = FrameSequenceDataset(
        base_dataset_type=DatasetType.CUSTOM,
        custom_frame_directory="/home/algomorph/Workbench/NeuralTracking/output/berlin_z_rotation_sequence")
    BERLIN_STRETCH_Y = FrameSequenceDataset(
        base_dataset_type=DatasetType.CUSTOM,
        custom_frame_directory="/home/algomorph/Workbench/NeuralTracking/output/berlin_y_stretch_sequence")
    # This sequence is part of VolumeDeform data
    # MINION = FrameSequenceDataset(base_dataset_type=DatasetType.CUSTOM,
    #                               custom_frame_directory="/mnt/Data/Reconstruction/real_data/minion/data")
