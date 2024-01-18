from mmengine.config import read_base

from seg.evaluation.metrics.ins_cls_iou_metric import InsClsIoUMetric
from seg.models.data_preprocessor import OVSAMDataPreprocessor

with read_base():
    from ..._base_.default_runtime import *
    from ..._base_.datasets.coco_panoptic_lsj_sam import *
    from ..._base_.schedules.schedule_12e import *

data_preprocessor = dict(
    type=OVSAMDataPreprocessor,
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    bgr_to_rgb=True,
    pad_size_divisor=32,
    pad_mask=True,
    mask_pad_value=0,
    pad_seg=True,
    seg_pad_value=255,
    batch_augments=None,
    use_point_pseudo_box=True
)

num_things_classes = 80
num_stuff_classes = 0
num_classes = num_things_classes + num_stuff_classes

ov_datasets_name = 'CocoPanopticOVDataset'

val_evaluator = dict(
    type=InsClsIoUMetric,
    with_score=False,
)
test_evaluator = val_evaluator
