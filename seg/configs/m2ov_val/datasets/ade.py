from mmdet.models import BatchFixedSizePad
from mmengine import read_base

from seg.models.data_preprocessor import VideoSegDataPreprocessor

with read_base():
    from ..._base_.default_runtime import *
    from ..._base_.schedules.schedule_12e import *
    from ..._base_.datasets.ade_panoptic_ov import train_dataloader, image_size
    from ..._base_.datasets.ade_panoptic import val_dataloader, val_evaluator, test_dataloader, test_evaluator
    from ..._base_.datasets.joint_dataset import train_dataloader as training_loader

batch_augments = [
    dict(
        type=BatchFixedSizePad,
        size=(image_size[1], image_size[0]),
        img_pad_value=0,
        pad_mask=True,
        mask_pad_value=0,
        pad_seg=True,
        seg_pad_value=255
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
    seg_pad_value=255,
    batch_augments=batch_augments
)

num_things_classes = 100
num_stuff_classes = 50
num_classes = num_things_classes + num_stuff_classes

ov_datasets_name = 'ADEPanopticOVDataset'
