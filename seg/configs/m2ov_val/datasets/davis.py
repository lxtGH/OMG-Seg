from mmengine.config import read_base

from mmdet.models import BatchFixedSizePad

from seg.models.data_preprocessor import VideoSegDataPreprocessor
from seg.models.utils import NO_OBJ

with read_base():
    from ..._base_.default_runtime import *
    from ..._base_.datasets.davis import *
    from ..._base_.schedules.schedule_12e import *

batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=(image_size[1], image_size[0]),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=NO_OBJ
    )
]
data_preprocessor = dict(
    type=VideoSegDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=NO_OBJ,
    batch_augments=batch_augments
)

num_things_classes = 80
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

ov_datasets_name = 'CocoOVDataset'
