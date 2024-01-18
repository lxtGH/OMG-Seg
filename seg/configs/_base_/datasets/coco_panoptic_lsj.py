# dataset settings
from mmcv.transforms import LoadImageFromFile, RandomResize
from mmengine.dataset import DefaultSampler

from mmdet.datasets import AspectRatioBatchSampler, CocoPanopticDataset
from mmdet.datasets.transforms import LoadPanopticAnnotations, RandomFlip, RandomCrop, PackDetInputs, Resize
from mmdet.evaluation import CocoPanopticMetric, CocoMetric

from seg.datasets.coco_ov import CocoPanopticOVDataset
from seg.datasets.pipeliens.loading import LoadPanopticAnnotationsHB

data_root = 'data/coco/'
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
        type=CocoPanopticOVDataset,
        data_root=data_root,
        ann_file='annotations/panoptic_train2017.json',
        data_prefix=dict(img='train2017/', seg='annotations/panoptic_train2017/'),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
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
        type=CocoPanopticDataset,
        data_root=data_root,
        ann_file='annotations/panoptic_val2017.json',
        data_prefix=dict(img='val2017/', seg='annotations/panoptic_val2017/'),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = [
    dict(
        type=CocoPanopticMetric,
        ann_file=data_root + 'annotations/panoptic_val2017.json',
        seg_prefix=data_root + 'annotations/panoptic_val2017/',
        backend_args=backend_args
    ),
    dict(
        type=CocoMetric,
        ann_file=data_root + 'annotations/instances_val2017.json',
        metric=['segm'],
        backend_args=backend_args
    )
]
test_evaluator = val_evaluator
