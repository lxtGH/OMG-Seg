import copy

import torch
import numpy as np
from mmdet.evaluation import INSTANCE_OFFSET

INSTANCE_OFFSET_HB = 10000


def mmpan2hbpan(pred_pan_map, num_classes):
    pan_seg_map = - np.ones_like(pred_pan_map)
    for itm in np.unique(pred_pan_map):
        if itm >= INSTANCE_OFFSET:
            # cls labels (from segmentation maps)
            cls = itm % INSTANCE_OFFSET
            # id labels (from tracking maps)
            ins = itm // INSTANCE_OFFSET
            pan_seg_map[pred_pan_map == itm] = cls * INSTANCE_OFFSET_HB + ins
        elif itm == num_classes:
            pan_seg_map[pred_pan_map == itm] = num_classes * INSTANCE_OFFSET_HB
        else:
            pan_seg_map[pred_pan_map == itm] = itm * INSTANCE_OFFSET_HB
    assert -1 not in pan_seg_map
    return pan_seg_map


def mmgt2hbpan(data_samples):
    pan_map = copy.deepcopy(data_samples.gt_sem_seg.sem_seg[0])
    pan_map = pan_map * INSTANCE_OFFSET_HB
    gt_instances = data_samples.gt_instances
    for idx in range(len(gt_instances)):
        mask = torch.tensor(gt_instances.masks.masks[idx], dtype=torch.bool)
        instance_id = gt_instances.instances_ids[idx].item()
        pan_map[mask] = instance_id

    return pan_map
