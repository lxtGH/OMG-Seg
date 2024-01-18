# dataset settings
# do not use this config for training, it is only used to create embedding.
from mmengine import read_base
from mmengine.dataset import DefaultSampler, RepeatDataset

from seg.datasets.concat_dataset import ConcatOVDataset
from seg.datasets.samplers.batch_sampler import VideoSegAspectRatioBatchSampler

with read_base():
    from .coco_panoptic_lsj import train_dataloader as _coco_vid_train_loader
    from .ade_panoptic_ov import train_dataloader as _ade_train_loader
    from .youtube_vis_2019 import train_dataloader as _yt19_train_loader
    from .youtube_vis_2021 import train_dataloader as _yt21_train_loader
    from .vipseg import train_dataloader as _vipseg_train_loader
    from .cityscapes_panoptic import train_dataloader as _city_train_loader
    from .coco_panoptic_lsj import val_dataloader, val_evaluator, test_dataloader, test_evaluator
    from .youtube_vis_2019 import image_size

train_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type=DefaultSampler, shuffle=True),
    batch_sampler=dict(type=VideoSegAspectRatioBatchSampler),
    dataset=dict(
        type=ConcatOVDataset,
        datasets=[
            dict(
                type=RepeatDataset,
                dataset=_coco_vid_train_loader.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,
                dataset=_ade_train_loader.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,
                dataset=_yt19_train_loader.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,
                dataset=_yt21_train_loader.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,
                dataset=_vipseg_train_loader.dataset,
                times=1,
            ),
            dict(
                type=RepeatDataset,
                dataset=_city_train_loader.dataset,
                times=1,
            ),
        ],
    )
)

