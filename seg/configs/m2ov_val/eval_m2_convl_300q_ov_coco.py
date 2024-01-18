from mmengine import read_base

with read_base():
    from .datasets.coco import *
    from .models.m2_convl_300q import *

model.update(
    data_preprocessor=data_preprocessor,
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
        panoptic_on=True,
        semantic_on=False,
        instance_on=True,
    ),
)
