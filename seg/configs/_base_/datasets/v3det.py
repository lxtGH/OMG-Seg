# dataset settings
from mmcv import LoadImageFromFile, RandomResize
from mmdet.datasets import AspectRatioBatchSampler
from mmdet.datasets.transforms import LoadAnnotations, RandomFlip, Resize, RandomCrop, PackDetInputs
from mmdet.evaluation import CocoMetric
from mmengine.dataset import DefaultSampler

from seg.datasets.pipeliens.loading import FilterAnnotationsHB
from seg.datasets.v3det import V3DetDataset

dataset_type = V3DetDataset
data_root = 'data/V3Det/'

backend_args = None


image_size = (1024, 1024)
train_pipeline = [
    dict(
        type=LoadImageFromFile,
        to_float32=True,
        backend_args=backend_args),
    dict(
        type=LoadAnnotations,
        with_bbox=True,
        with_mask=False,
        with_seg=False,
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
    dict(
        type=FilterAnnotationsHB,
        by_box=True,
        by_mask=False,
        min_gt_bbox_wh=(8, 8)
    ),
    dict(type=PackDetInputs)
]

test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=backend_args),
    dict(type=Resize, scale=(1333, 800), keep_ratio=True),
    # If you don't have a gt annotation, delete the pipeline
    dict(type=LoadAnnotations, with_bbox=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=AspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/v3det_2023_v1_train.json',
        data_prefix=dict(img=''),
        filter_cfg=dict(filter_empty_gt=True, min_size=4),
        pipeline=train_pipeline,
        backend_args=backend_args
    )
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='annotations/v3det_2023_v1_val.json',
        data_prefix=dict(img=''),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

val_evaluator = dict(
    type=CocoMetric,
    ann_file=data_root + 'annotations/v3det_2023_v1_val.json',
    metric='bbox',
    format_only=False,
    backend_args=backend_args,
    use_mp_eval=True,
    proposal_nums=[300]
)
test_evaluator = val_evaluator
