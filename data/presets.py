from enum import Enum

from data.frame import StandaloneFrameDataset, DataSplit
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


# SOD == salient object detection
# Please consult media/SOD_Generation_Instructions.md for instructions on how to run SOD to obtain the masks in "sod"
# folders

class FrameSequencePreset(Enum):
    RED_SHORTS = FrameSequenceDataset(14, DataSplit.VALIDATION, has_masks=False, far_clipping_distance=1.2)
    RED_SHORTS_SOD_MASKS = FrameSequenceDataset(14, DataSplit.VALIDATION, has_masks=True, masks_subfolder="sod",
                                                mask_lower_threshold=1)

    ALJAZ_FROM_ARTICLE_SOD_MASKS = \
        FrameSequenceDataset(9, DataSplit.TEST, start_frame_index=0, far_clipping_distance=1.5, has_masks=True,
                             masks_subfolder="sod", mask_lower_threshold=1)

    BERLIN = FrameSequenceDataset(70, DataSplit.TRAIN, has_masks=False, far_clipping_distance=2.4)
    BERLIN_STATIC = StaticFrameSequenceDataset(70, DataSplit.TRAIN, frame_count=6, has_masks=False,
                                               far_clipping_distance=2.4)
    BERLIN_SOD_MASKS = FrameSequenceDataset(70, DataSplit.TRAIN, start_frame_index=0, has_masks=True,
                                            masks_subfolder="sod", mask_lower_threshold=1, far_clipping_distance=2.4)

    BLUE_MAN_SOD_MASKS = FrameSequenceDataset(76, DataSplit.TRAIN, start_frame_index=0, far_clipping_distance=2.4,
                                              has_masks=True, masks_subfolder="sod", mask_lower_threshold=1)

    DOG_TRAINING_1_SOD_MASKS = FrameSequenceDataset(0, DataSplit.TRAIN, start_frame_index=0, far_clipping_distance=3.5,
                                                    has_masks=True, masks_subfolder="sod", mask_lower_threshold=1)

    DOG_TRAINING_2_SOD_MASKS = FrameSequenceDataset(2, DataSplit.TRAIN, start_frame_index=0, far_clipping_distance=3.5,
                                                    has_masks=True, masks_subfolder="sod", mask_lower_threshold=1)

    BLUE_BACKPACK_FLIP_SOD_MASKS = \
        FrameSequenceDataset(sequence_id=32, split=DataSplit.TRAIN, start_frame_index=0, far_clipping_distance=1.5,
                             has_masks=True, masks_subfolder="sod", mask_lower_threshold=1)

    BLUE_BACKPACK_FLIP = \
        FrameSequenceDataset(sequence_id=32, split=DataSplit.TRAIN, start_frame_index=0, far_clipping_distance=1.5,
                             has_masks=False)
