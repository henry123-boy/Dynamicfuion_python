from enum import Enum
from data.frame_pair import FramePairDataset, DatasetType, DataSplit


class FramePairPreset(Enum):
    GREY_SHIRT_TEST_LOCAL = FramePairDataset(300, 600, 17, DataSplit.TEST, DatasetType.LOCAL, has_masks=False)
    GREY_SHIRT_TEST = FramePairDataset(300, 600, 17, DataSplit.TEST, has_masks=False)
    GREY_SHIRT_TRAIN_LOCAL = FramePairDataset(0, 110, 258, DataSplit.TRAIN, DatasetType.LOCAL, has_masks=False)
    RED_SHORTS = FramePairDataset(200, 400, 14, DataSplit.VALIDATION, has_masks=True)
