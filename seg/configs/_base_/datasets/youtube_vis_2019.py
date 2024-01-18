from mmcv import TransformBroadcaster, LoadImageFromFile, RandomResize
from mmdet.datasets.transforms import LoadTrackAnnotations, Resize, RandomFlip, PackTrackInputs, RandomCrop
from mmdet.evaluation import YouTubeVISMetric
from mmengine.dataset import DefaultSampler

from seg.datasets.youtube_vis_dataset import YouTubeVISDatasetV2
from seg.datasets.pipeliens.formatting import PackVidSegInputs
from seg.datasets.pipeliens.frame_copy import AddSemSeg
from seg.datasets.pipeliens.frame_sampling import VideoClipSample
from seg.datasets.samplers.batch_sampler import VideoSegAspectRatioBatchSampler

dataset_type = YouTubeVISDatasetV2
data_root = 'data/youtube_vis_2019/'
dataset_version = data_root[-5:-1]  # 2019 or 2021

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
            dict(type=LoadTrackAnnotations, with_mask=True),
            dict(type=AddSemSeg),
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
            dict(type=Resize, scale=image_size, keep_ratio=True),
            dict(type=LoadTrackAnnotations, with_mask=True),
        ]),
    dict(type=PackTrackInputs)
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
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2019_train.json',
        data_prefix=dict(img_path='train/JPEGImages'),
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
        dataset_version=dataset_version,
        ann_file='annotations/youtube_vis_2019_valid.json',
        data_prefix=dict(img_path='valid/JPEGImages'),
        test_mode=True,
        pipeline=test_pipeline))
test_dataloader = val_dataloader

val_evaluator = dict(
    type=YouTubeVISMetric,
    metric='youtube_vis_ap',
    outfile_prefix='./youtube_vis_2019_results',
    format_only=True
)
test_evaluator = val_evaluator
