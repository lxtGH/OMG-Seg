# dataset settings
from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler

from mmdet.datasets import AspectRatioBatchSampler
from mmdet.datasets.transforms import RandomFlip, RandomCrop, PackDetInputs, Resize

from seg.datasets.pipeliens.loading import LoadPanopticAnnotationsHB
from seg.datasets.cityscapes import CityscapesPanopticDataset
from seg.evaluation.metrics.cityscapes_panoptic_metric import CityscapesPanopticMetric

data_root = 'data/cityscapes/'
backend_args = None
image_size = (1024, 1024)

train_pipeline = [
    dict(
        type=LoadImageFromFile,
        to_float32=True,
        backend_args=backend_args),
    dict(
        type=LoadPanopticAnnotationsHB,
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=backend_args),
    dict(type=RandomFlip, prob=0.5),
    dict(
        type=RandomResize,
        resize_type=Resize,
        scale=image_size,
        ratio_range=(.8, 1.5),
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
        type=CityscapesPanopticDataset,
        data_root=data_root,
        ann_file='annotations/cityscapes_panoptic_train_trainId.json',
        data_prefix=dict(img='leftImg8bit/train/',
                         seg='annotations/cityscapes_panoptic_train_trainId/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(2048, 1024), keep_ratio=True),
    dict(type=LoadPanopticAnnotationsHB, backend_args=backend_args),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor', 'instances')
    )
]
val_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=CityscapesPanopticDataset,
        data_root=data_root,
        ann_file='annotations/cityscapes_panoptic_val_trainId.json',
        data_prefix=dict(img='leftImg8bit/val/',
                         seg='annotations/cityscapes_panoptic_val_trainId/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CityscapesPanopticMetric,
    ann_file=data_root + 'annotations/cityscapes_panoptic_val_trainId.json',
    seg_prefix=data_root + 'annotations/cityscapes_panoptic_val_trainId/',
    backend_args=backend_args
)
test_evaluator = val_evaluator
