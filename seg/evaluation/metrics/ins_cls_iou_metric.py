import os

import mmcv
import torch
from mmengine.dist import broadcast_object_list, collect_results, is_main_process

from typing import Dict, Optional, Sequence
from mmengine.evaluator import BaseMetric
from mmdet.registry import METRICS
from mmengine.evaluator.metric import _to_cpu
from mmengine.visualization import Visualizer


@METRICS.register_module()
class InsClsIoUMetric(BaseMetric):

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 base_classes=None,
                 novel_classes=None,
                 with_score=True,
                 output_failure=False,
                 ) -> None:

        super().__init__(collect_device=collect_device, prefix=prefix)
        self.scores = []
        self.iou_list = []

        self.base_scores = []
        self.novel_scores = []
        self.base_iou_list = []
        self.novel_iou_list = []

        self.with_score = with_score

        if base_classes is not None:
            assert novel_classes is not None
            num_classes = max(max(base_classes) + 1, max(novel_classes) + 1)
            self.base_novel_indicator = torch.zeros((num_classes,), dtype=torch.long)
            for clss in base_classes:
                self.base_novel_indicator[clss] = 1
            for clss in novel_classes:
                self.base_novel_indicator[clss] = 2
        else:
            self.base_novel_indicator = None

        self.output_failure = output_failure

    def get_iou(self, gt_masks, pred_masks):
        gt_masks = gt_masks
        n, h, w = gt_masks.shape
        intersection = (gt_masks & pred_masks).reshape(n, h * w).sum(dim=-1)
        union = (gt_masks | pred_masks).reshape(n, h * w).sum(dim=-1)
        ious = (intersection / (union + 1.e-8))
        return ious

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for data_sample in data_samples:
            gt_labels = data_sample['gt_instances']['labels']
            if len(gt_labels) == 0:
                score = gt_labels.new_zeros((0,), dtype=torch.float)
                ious = gt_labels.new_zeros((0,), dtype=torch.float)
            else:
                if self.with_score:
                    if self.base_novel_indicator is not None:
                        assert (self.base_novel_indicator[gt_labels.cpu()] > 0).all()
                    pred_labels = data_sample['pred_instances']['labels']
                    score = (pred_labels == gt_labels).to(dtype=torch.float) * 100
                if 'masks' in data_sample['pred_instances']:
                    pred_masks = data_sample['pred_instances']['masks']
                    if self.output_failure:
                        for idx, _score in enumerate(score.cpu().numpy().tolist()):
                            if _score == 0.:
                                img_path = data_sample['img_path']
                                vis = Visualizer()
                                rgb_img = mmcv.imread(img_path)
                                rgb_img = mmcv.bgr2rgb(rgb_img)
                                vis.set_image(rgb_img)
                                masks = pred_masks[idx]
                                # colors = [(0, 176, 237)]
                                colors = [(250, 177, 135)]
                                vis.draw_binary_masks(masks, alphas=.85, colors=colors)
                                vis_res = vis.get_image()
                                if vis_res is None:
                                    continue
                                img_name = os.path.basename(img_path)
                                mmcv.imwrite(
                                    mmcv.rgb2bgr(vis_res), os.path.join(
                                        'failure_lvis',
                                        img_name.split('.')[0] + '_' + str(idx) + '_' + str(int(gt_labels[idx]))
                                        + '_' + str(int(pred_labels[idx])) + '.jpg')
                                )
                    gt_masks = data_sample['gt_instances']['masks']
                    gt_masks = gt_masks.to_tensor(dtype=torch.bool, device=pred_masks.device)
                    ious = self.get_iou(gt_masks, pred_masks)
                else:
                    ious = gt_labels.new_tensor([0.])
            self.iou_list.append(ious.to(device='cpu'))
            if self.base_novel_indicator is not None:
                self.base_iou_list.append(ious[self.base_novel_indicator[gt_labels.cpu()] == 1].to(device='cpu'))
                self.novel_iou_list.append(ious[self.base_novel_indicator[gt_labels.cpu()] == 2].to(device='cpu'))
            if self.with_score:
                self.scores.append(score.to(device='cpu'))
                if self.base_novel_indicator is not None:
                    self.base_scores.append(score[self.base_novel_indicator[gt_labels.cpu()] == 1].to(device='cpu'))
                    self.novel_scores.append(score[self.base_novel_indicator[gt_labels.cpu()] == 2].to(device='cpu'))

    def compute_metrics(self, scores, ious,
                        base_scores, base_ious,
                        novel_scores, novel_ious) -> Dict[str, float]:

        iou = ious.mean().item()
        results = dict()
        results['miou'] = iou
        if self.base_novel_indicator is not None:
            results['base_iou'] = base_ious.mean().item()

            results['novel_iou'] = novel_ious.mean().item()
        if self.with_score:
            score = scores.mean().item()
            results['score'] = score
            if base_scores is not None:
                results['base_score'] = base_scores.mean().item()
                results['novel_score'] = novel_scores.mean().item()
        return results

    def evaluate(self, size: int) -> dict:
        _ious = collect_results(self.iou_list, size, self.collect_device)
        if self.base_novel_indicator is not None:
            _base_ious = collect_results(self.base_iou_list, size, self.collect_device)
            _novel_ious = collect_results(self.novel_iou_list, size, self.collect_device)
        if self.with_score:
            _scores = collect_results(self.scores, size, self.collect_device)
            if self.base_novel_indicator is not None:
                _base_scores = collect_results(self.base_scores, size, self.collect_device)

                _novel_scores = collect_results(self.novel_scores, size, self.collect_device)

        if is_main_process():
            if self.base_novel_indicator is not None:
                base_ious = torch.cat(_base_ious)
                novel_ious = torch.cat(_novel_ious)
            else:
                base_ious = None
                novel_ious = None
            if self.with_score:
                scores = torch.cat(_scores)
                scores = _to_cpu(scores)
                if self.base_novel_indicator is not None:
                    base_scores = torch.cat(_base_scores)
                    novel_scores = torch.cat(_novel_scores)
                else:
                    base_scores = None
                    novel_scores = None
            else:
                scores = None
                base_scores = None
                novel_scores = None
            ious = torch.cat(_ious)
            ious = _to_cpu(ious)
            _metrics = self.compute_metrics(
                scores, ious,
                base_scores, base_ious,
                novel_scores, novel_ious
            )
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore
        broadcast_object_list(metrics)
        return metrics[0]
