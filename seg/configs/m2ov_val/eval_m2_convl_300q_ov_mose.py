from mmengine import read_base

from seg.models.detectors import Mask2formerVideoMinVIS

with read_base():
    from .datasets.mose import *
    from .models.m2_convl_300q import *

model.update(
    data_preprocessor=data_preprocessor,
    type=Mask2formerVideoMinVIS,
    clip_size=5,
    clip_size_small=3,
    whole_clip_thr=0,
    small_clip_thr=15,
    overlap=0,
    panoptic_head=dict(
        ov_classifier_name=f'{ov_model_name}_{ov_datasets_name}',
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    ),
    panoptic_fusion_head=dict(
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
    ),
    test_cfg=dict(
        panoptic_on=False,
        semantic_on=False,
        instance_on=False,
        proposal_on=True,
        num_proposals=25,
    ),
)
