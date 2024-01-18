# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Union
import numpy as np
import torch
from mmdet.evaluation.metrics.coco_panoptic_metric import print_panoptic_table, parse_pq_results
from mmengine import print_log, mkdir_or_exist
from mmengine.dist import barrier, broadcast_object_list, is_main_process
from mmdet.registry import METRICS
from mmdet.evaluation.metrics.base_video_metric import BaseVideoMetric, collect_tracking_results
from panopticapi.evaluation import PQStat

from seg.models.utils import mmpan2hbpan, INSTANCE_OFFSET_HB, mmgt2hbpan
from seg.models.utils import cal_pq, NO_OBJ_ID, IoUObj


def parse_pan_map_hb(pan_map: np.ndarray, data_sample: dict, num_classes: int) -> dict:
    result = dict()
    result['video_id'] = data_sample['video_id']
    result['frame_id'] = data_sample['frame_id']

    # For video evaluation, each map may include several loads,
    # it is not efficient for saving an extra png map, especially
    # for machines not with high performance ssd.
    pan_labels = np.unique(pan_map)
    segments_info = []
    for pan_label in pan_labels:
        sem_label = pan_label // INSTANCE_OFFSET_HB
        if sem_label >= num_classes:
            continue
        mask = (pan_map == pan_label).astype(np.uint8)
        area = mask.sum()
        # _mask = maskUtils.encode(np.asfortranarray(mask))
        # _mask['counts'] = _mask['counts'].decode()
        segments_info.append({
            'id': int(pan_label),
            'category_id': sem_label,
            'area': int(area),
            'mask': mask
        })
    result['segments_info'] = segments_info

    return result


def parse_data_sample_gt(data_sample: dict, num_things: int, num_stuff: int) -> dict:
    num_classes = num_things + num_stuff
    result = dict()
    result['video_id'] = data_sample['video_id']
    result['frame_id'] = data_sample['frame_id']

    # For video evaluation, each map may include several loads,
    # it is not efficient for saving an extra png map, especially
    # for machines not with high performance ssd.
    gt_instances = data_sample['gt_instances']
    segments_info = []
    for thing_id in range(len(gt_instances['labels'])):
        mask = gt_instances['masks'].masks[thing_id].astype(np.uint8)
        area = mask.sum()
        pan_id = gt_instances['instances_ids'][thing_id]
        cat = int(gt_instances['labels'][thing_id])
        if cat >= num_things:
            raise ValueError(f"not reasonable value {cat}")
        # _mask = maskUtils.encode(np.asfortranarray(mask))
        # _mask['counts'] = _mask['counts'].decode()
        segments_info.append({
            'id': int(pan_id),
            'category_id': cat,
            'area': int(area),
            'mask': mask
        })

    gt_sem_seg = data_sample['gt_sem_seg']['sem_seg'][0].cpu().numpy()
    for stuff_id in np.unique(gt_sem_seg):
        if stuff_id < num_things:
            continue
        if stuff_id >= num_classes:
            assert stuff_id == NO_OBJ_ID // INSTANCE_OFFSET_HB
        _mask = (gt_sem_seg == stuff_id).astype(np.uint8)
        area = _mask.sum()
        cat = int(stuff_id)
        pan_id = cat * INSTANCE_OFFSET_HB
        segments_info.append({
            'id': int(pan_id),
            'category_id': cat,
            'area': int(area),
            'mask': _mask
        })

    if segments_info[-1]['id'] != NO_OBJ_ID:
        segments_info.append({
            'id': int(NO_OBJ_ID),
            'category_id': NO_OBJ_ID // INSTANCE_OFFSET_HB,
            'area': 0,
            'mask': np.zeros_like(gt_sem_seg, dtype=np.uint8)
        })
    result['segments_info'] = segments_info
    return result


@METRICS.register_module()
class VIPSegMetric(BaseVideoMetric):
    """mAP evaluation metrics for the VIS task.

    Args:
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `youtube_vis_ap`..
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonyms metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
    """

    default_prefix: Optional[str] = 'vip_seg'

    def __init__(self,
                 metric: Union[str, List[str]] = 'VPQ@1',
                 outfile_prefix: Optional[str] = None,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only: bool = False) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        # vis evaluation metrics
        self.metrics = metric if isinstance(metric, list) else [metric]
        self.format_only = format_only
        allowed_metrics = ['VPQ']
        for metric in self.metrics:
            if metric not in allowed_metrics and metric.split('@')[0] not in allowed_metrics:
                raise KeyError(
                    f"metric should be 'youtube_vis_ap', but got {metric}.")

        self.outfile_prefix = outfile_prefix
        self.per_video_res = []
        self.categories = {}
        self._vis_meta_info = defaultdict(list)  # record video and image infos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for track_data_sample in data_samples:
            video_data_samples = track_data_sample['video_data_samples']
            ori_video_len = video_data_samples[0].ori_video_length
            if ori_video_len == len(video_data_samples):
                # video process
                self.process_video(video_data_samples)
            else:
                # image process
                raise NotImplementedError

    def process_video(self, data_samples):
        video_length = len(data_samples)

        num_things = len(self.dataset_meta['thing_classes'])
        num_stuff = len(self.dataset_meta['stuff_classes'])
        num_classes = num_things + num_stuff
        for frame_id in range(video_length):
            img_data_sample = data_samples[frame_id].to_dict()
            # 0 is for dummy dimension in fusion head, not batch.
            pred = mmpan2hbpan(img_data_sample['pred_track_panoptic_seg']['sem_seg'][0], num_classes=num_classes)

            if self.format_only:
                vid_id = data_samples[frame_id].video_id
                gt = mmgt2hbpan(data_samples[frame_id])
                mkdir_or_exist('vipseg_output/gt/')
                mkdir_or_exist('vipseg_output/pred/')
                torch.save(gt.to(device='cpu'),
                           'vipseg_output/gt/{:06d}_{:06d}.pth'.format(vid_id, frame_id))
                torch.save(torch.tensor(pred, device='cpu'),
                           'vipseg_output/pred/{:06d}_{:06d}.pth'.format(vid_id, frame_id))
                continue

            pred_json = parse_pan_map_hb(pred, img_data_sample, num_classes=num_classes)
            gt_json = parse_data_sample_gt(img_data_sample, num_things=num_things, num_stuff=num_stuff)
            self.per_video_res.append((pred_json, gt_json))

        if self.format_only:
            return

        video_results = []
        for pred, gt in self.per_video_res:
            intersection_info = dict()
            gt_no_obj_info = gt['segments_info'][-1]
            for pred_seg_info in pred['segments_info']:
                intersection = int((gt_no_obj_info['mask'] * pred_seg_info['mask']).sum())
                union = pred_seg_info['area']
                intersection_info[gt_no_obj_info['id'], pred_seg_info['id']] = IoUObj(
                    intersection=intersection,
                    union=union
                )
            for pred_seg_info in pred['segments_info']:
                for gt_seg_info in gt['segments_info'][:-1]:
                    intersection = int((gt_seg_info['mask'] * pred_seg_info['mask']).sum())
                    union = gt_seg_info['area'] + pred_seg_info['area'] - \
                            intersection - intersection_info[NO_OBJ_ID, pred_seg_info['id']].intersection
                    intersection_info[gt_seg_info['id'], pred_seg_info['id']] = IoUObj(
                        intersection=intersection,
                        union=union
                    )
            video_results.append(intersection_info)
        self.per_video_res.clear()
        self.results.append(video_results)

    def compute_metrics(self, results: List) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        # split gt and prediction list
        eval_results = {}
        if self.format_only:
            return eval_results
        for metric in self.metrics:
            seq_len = int(metric.split('@')[-1])
            pq_stat = PQStat()
            cnt = 0
            for vid_idx, video_instances in enumerate(results):
                for frame_x in range(len(video_instances)):
                    if frame_x + seq_len > len(video_instances):
                        break
                    global_intersection_info = defaultdict(IoUObj)
                    for frame_offset in range(seq_len):
                        frame_info = video_instances[frame_x + frame_offset]
                        for gt_id, pred_id in frame_info:
                            global_intersection_info[gt_id, pred_id] += frame_info[gt_id, pred_id]
                    pq_stat += cal_pq(global_intersection_info, classes=self.dataset_meta['classes'])
                # global_intersection_info = defaultdict(IoUObj)
                # for frame_idx, frame_info in enumerate(video_instances):
                #     for gt_id, pred_id in frame_info:
                #         global_intersection_info[gt_id, pred_id] += frame_info[gt_id, pred_id]
                #     if frame_idx - seq_len >= 0:
                #         out_frame_info = video_instances[frame_idx - seq_len]
                #         for gt_id, pred_id in out_frame_info:
                #             global_intersection_info[gt_id, pred_id] -= out_frame_info[gt_id, pred_id]
                #             assert global_intersection_info[gt_id, pred_id].is_legal()
                #     if frame_idx - seq_len >= -1:
                #         pq_stat += cal_pq(global_intersection_info, classes=self.dataset_meta['classes'])
                #         cnt += 1
            print_log("Total calculated clips: " + str(cnt), logger='current')

            sub_metrics = [('All', None), ('Things', True), ('Stuff', False)]
            pq_results = {}

            for name, isthing in sub_metrics:
                pq_results[name], classwise_results = pq_stat.pq_average(
                    self.categories, isthing=isthing)
                if name == 'All':
                    pq_results['classwise'] = classwise_results

            # classwise_results = {
            #     k: v
            #     for k, v in zip(self.dataset_meta['classes'],
            #                     pq_results['classwise'].values())
            # }

            print_panoptic_table(pq_results, None, logger='current')
            metric_results = parse_pq_results(pq_results)
            for key in metric_results:
                eval_results[metric + f'_{key}'] = metric_results[key]
        return eval_results

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        # wait for all processes to complete prediction.
        barrier()

        cls_idx = 0
        for thing_cls in self.dataset_meta['thing_classes']:
            self.categories[cls_idx] = {'class': thing_cls, 'isthing': 1}
            cls_idx += 1
        for stuff_cls in self.dataset_meta['stuff_classes']:
            self.categories[cls_idx] = {'class': stuff_cls, 'isthing': 0}
            cls_idx += 1
        assert cls_idx == len(self.dataset_meta['classes'])

        if len(self.results) == 0:
            warnings.warn(
                f'{self.__class__.__name__} got empty `self.results`. Please '
                'ensure that the processed results are properly added into '
                '`self.results` in `process` method.')

        results = collect_tracking_results(self.results, self.collect_device)

        # # gather seq_info
        # gathered_seq_info = all_gather_object(self._vis_meta_info['videos'])
        # all_seq_info = []
        # for _seq_info in gathered_seq_info:
        #     all_seq_info.extend(_seq_info)
        # # update self._vis_meta_info
        # self._vis_meta_info = dict(videos=all_seq_info)

        if is_main_process():
            print_log(
                f"There are totally {len(results)} videos to be evaluated.",
                logger='current'
            )
            _metrics = self.compute_metrics(results)  # type: ignore
            # Add prefix to metric names
            if self.prefix:
                _metrics = {
                    '/'.join((self.prefix, k)): v
                    for k, v in _metrics.items()
                }
            metrics = [_metrics]
        else:
            metrics = [None]  # type: ignore

        broadcast_object_list(metrics)

        # reset the results list
        self.results.clear()
        # reset the vis_meta_info
        self._vis_meta_info.clear()
        return metrics[0]
