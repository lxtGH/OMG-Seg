from mmengine import read_base

with read_base():
    from .datasets.coco_pan_point import *
    from .models.m2_convl_300q import *

model.update(
    data_preprocessor=data_preprocessor,
    inference_sam=True,
    panoptic_head=dict(
        enable_box_query=True,
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
        instance_on=True,
    ),
)
