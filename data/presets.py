from enum import Enum
from data.frame import StandaloneFrameDataset, DatasetType, DataSplit
from data.frame_pair import FramePairDataset
from data.frame_sequence import FrameSequenceDataset


class FramePairPreset(Enum):
    GREY_SHIRT_TEST_LOCAL = FramePairDataset(300, 600, 17, DataSplit.TEST, DatasetType.LOCAL, has_masks=False)
    GREY_SHIRT_TEST = FramePairDataset(300, 600, 17, DataSplit.TEST, has_masks=False)
    GREY_SHIRT_TRAIN_LOCAL = FramePairDataset(0, 110, 258, DataSplit.TRAIN, DatasetType.LOCAL, has_masks=False)
    RED_SHORTS_200_400 = FramePairDataset(200, 400, 14, DataSplit.VALIDATION, has_masks=True)


class StandaloneFramePreset(Enum):
    RED_SHORTS_200 = StandaloneFrameDataset(200, 14, DataSplit.VALIDATION, has_masks=True)
    RED_SHORTS_400 = StandaloneFrameDataset(400, 14, DataSplit.VALIDATION, has_masks=True)


class FrameSequencePreset(Enum):
    RED_SHORTS_40 = FrameSequenceDataset(14, DataSplit.VALIDATION, frame_count=40, has_masks=False)
    RED_SHORTS = FrameSequenceDataset(14, DataSplit.VALIDATION, has_masks=False)
# TODO: fix loading of custom sequences
#    MINION = FrameSequenceDataset(base_dataset_type=DatasetType.CUSTOM,
#                                  custom_frame_directory="/mnt/Data/Reconstruction/real_data/minion/data")
