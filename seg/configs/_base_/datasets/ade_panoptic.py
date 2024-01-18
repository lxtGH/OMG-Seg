# dataset settings
from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler

from mmdet.datasets import AspectRatioBatchSampler
from mmdet.datasets.transforms import LoadPanopticAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize
from mmdet.evaluation import CocoPanopticMetric

from mmdet.datasets.ade20k import ADE20KPanopticDataset


data_root = 'data/ade/'
backend_args = None
image_size = (1024, 1024)

train_pipeline = [
    dict(
        type=LoadImageFromFile,
        to_float32=True,
        backend_args=backend_args),
    dict(
        type=LoadPanopticAnnotations,
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=backend_args),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomResize,
        resize_type=Resize,
        scale=image_size,
        ratio_range=(0.1, 2.0),
        keep_ratio=True,
    ),
    dict(
        type=RandomCrop,
        crop_size=image_size,
        crop_type='absolute',
        recompute_bbox=True,
        allow_negative_crop=True),
    dict(type=PackDetInputs)
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=ADE20KPanopticDataset,
        data_root=data_root,
        ann_file='ADEChallengeData2016/ade20k_panoptic_train.json',
        data_prefix=dict(img='ADEChallengeData2016/images/training/',
                         seg='ADEChallengeData2016/ade20k_panoptic_train/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(2560, 640), keep_ratio=True),
    dict(type=LoadPanopticAnnotations, backend_args=backend_args),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=ADE20KPanopticDataset,
        data_root=data_root,
        ann_file='ADEChallengeData2016/ade20k_panoptic_val.json',
        data_prefix=dict(img='ADEChallengeData2016/images/validation/',
                         seg='ADEChallengeData2016/ade20k_panoptic_val/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoPanopticMetric,
    ann_file=data_root + 'ADEChallengeData2016/ade20k_panoptic_val.json',
    seg_prefix=data_root + 'ADEChallengeData2016/ade20k_panoptic_val/',
    backend_args=backend_args
)
test_evaluator = val_evaluator
