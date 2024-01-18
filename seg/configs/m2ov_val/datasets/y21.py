from mmengine.config import read_base

from mmdet.models import BatchFixedSizePad

from seg.models.data_preprocessor import VideoSegDataPreprocessor

with read_base():
    from ..._base_.default_runtime import *
    from ..._base_.datasets.youtube_vis_2021 import *
    from ..._base_.schedules.schedule_12e import *
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

num_things_classes = 40
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

ov_datasets_name = 'YouTubeVISDataset_2021'
default_hooks.update(
    logger=dict(type=LoggerHook, interval=1),
)
