from mmcv import TransformBroadcaster, LoadImageFromFile, RandomResize
from mmdet.datasets.transforms import Resize, RandomFlip, RandomCrop
from mmengine.dataset import DefaultSampler

from seg.datasets.pipeliens.loading import LoadVideoSegAnnotations, ResizeOri
from seg.datasets.pipeliens.formatting import PackVidSegInputs
from seg.datasets.pipeliens.frame_sampling import VideoClipSample
from seg.datasets.samplers.batch_sampler import VideoSegAspectRatioBatchSampler
from seg.datasets.vipseg import VIPSegDataset
from seg.evaluation.metrics.vip_seg_metric import VIPSegMetric

dataset_type = VIPSegDataset
data_root = 'data/VIPSeg'

backend_args = None
image_size = (1280, 736)

# dataset settings
train_pipeline = [
    dict(
        type=VideoClipSample,
        num_selected=2,
        interval=2),
    dict(
        type=TransformBroadcaster,
        share_random_params=True,
        transforms=[
            dict(type=LoadImageFromFile, backend_args=backend_args),
            dict(type=LoadVideoSegAnnotations, with_bbox=True, with_label=True, with_mask=True, with_seg=True),
            dict(
                type=RandomResize,
                resize_type=Resize,
                scale=image_size,
                ratio_range=(.8, 2.),
                keep_ratio=True,
            ),
            dict(
                type=RandomCrop,
                crop_size=image_size,
                crop_type='absolute',
                recompute_bbox=True,
                allow_negative_crop=True),
            dict(type=RandomFlip, prob=0.5),
        ]),
    dict(type=PackVidSegInputs)
]

test_pipeline = [
    dict(
        type=TransformBroadcaster,
        transforms=[
            dict(type=LoadImageFromFile, backend_args=backend_args),
            dict(type=LoadVideoSegAnnotations, with_bbox=True, with_label=True, with_mask=True, with_seg=True),
            dict(type=Resize, scale=image_size, keep_ratio=True),
            dict(type=ResizeOri),
        ]),
    dict(type=PackVidSegInputs)
]

# dataloader
train_dataloader = dict(
    batch_size=2,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=VideoSegAspectRatioBatchSampler),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='train.txt',
        data_prefix=dict(img='imgs/', seg='panomasks/'),
        # check whether it is necessary.
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline))
val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type=DefaultSampler, shuffle=False, round_up=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file='val.txt',
        data_prefix=dict(img='imgs/', seg='panomasks/'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=VIPSegMetric,
    metric=['VPQ@1', 'VPQ@2', 'VPQ@4', 'VPQ@6'],
)
test_evaluator = val_evaluator
