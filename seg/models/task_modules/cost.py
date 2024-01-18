from typing import Optional, Union

import torch
from mmdet.models.task_modules.assigners.match_cost import BaseMatchCost
from mmengine.structures import InstanceData
from torch import Tensor

from mmdet.registry import TASK_UTILS


@TASK_UTILS.register_module()
class FlexibleClassificationCost(BaseMatchCost):
    def __init__(self, weight: Union[float, int] = 1) -> None:
        super().__init__(weight=weight)

    def __call__(self,
                 pred_instances: InstanceData,
                 gt_instances: InstanceData,
                 img_meta: Optional[dict] = None,
                 **kwargs) -> Tensor:
        """Compute match cost.

        Args:
            pred_instances (:obj:`InstanceData`): ``scores`` inside is
                predicted classification logits, of shape
                (num_queries, num_class).
            gt_instances (:obj:`InstanceData`): ``labels`` inside should have
                shape (num_gt, ).
            img_meta (Optional[dict]): _description_. Defaults to None.

        Returns:
            Tensor: Match Cost matrix of shape (num_preds, num_gts).
        """
        _pred_scores = pred_instances.scores
        gt_labels = gt_instances.labels

        pred_scores = _pred_scores[..., :-1]
        iou_score = _pred_scores[..., -1:]

        pred_scores = pred_scores.softmax(-1)
        iou_score = iou_score.sigmoid()
        pred_scores = torch.cat([pred_scores, iou_score], dim=-1)
        cls_cost = -pred_scores[:, gt_labels]

        return cls_cost * self.weight
