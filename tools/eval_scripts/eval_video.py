import collections
import os
import argparse
import sys
from functools import partial
from typing import Iterable

import mmengine
import numpy as np
import torch
from mmengine.utils.progressbar import init_pool, ProgressBar

from seg.models.utils import vpq_eval, stq

_EPSILON = 1e-15


def track_parallel_progress(func,
                            tasks,
                            nproc,
                            initializer=None,
                            initargs=None,
                            bar_width=50,
                            chunksize=1,
                            skip_first=False,
                            file=sys.stdout):
    if isinstance(tasks, tuple):
        assert len(tasks) == 2
        assert isinstance(tasks[0], Iterable)
        assert isinstance(tasks[1], int)
        task_num = tasks[1]
        tasks = tasks[0]
    elif isinstance(tasks, Iterable):
        task_num = len(tasks)
    else:
        raise TypeError(
            '"tasks" must be an iterable object or a (iterator, int) tuple')
    pool = init_pool(nproc, initializer, initargs)
    start = not skip_first
    task_num -= nproc * chunksize * int(skip_first)
    prog_bar = ProgressBar(task_num, bar_width, start, file=file)
    results = []
    gen = pool.starmap(func, tasks, chunksize)
    for result in gen:
        results.append(result)
        if skip_first:
            if len(results) < nproc * chunksize:
                continue
            elif len(results) == nproc * chunksize:
                prog_bar.start()
                continue
        prog_bar.update()
    prog_bar.file.write('\n')
    pool.close()
    pool.join()
    return results


def _update_dict_stats(stat_dict, id_array: np.ndarray):
    """Updates a given dict with corresponding counts."""
    ids, counts = np.unique(id_array, return_counts=True)
    for idx, count in zip(ids, counts):
        if idx in stat_dict:
            stat_dict[idx] += count
        else:
            stat_dict[idx] = count


class STQState:
    label_divisor = int(1e4)
    ins_divisor = int(1e7)
    max_ins = int(1e4)

    def __init__(self, num_classes):
        self._iou_matrix = collections.OrderedDict()
        self._predictions = collections.OrderedDict()
        self._ground_truth = collections.OrderedDict()
        self._intersections = collections.OrderedDict()
        self._sequence_length = collections.OrderedDict()

        self._include_indices = np.arange(num_classes)
        self.num_classes = num_classes + 1

    def update_state_stq(self, result, seq_id):
        if seq_id not in self._iou_matrix:
            self._iou_matrix[seq_id] = np.zeros(
                (self.num_classes, self.num_classes),
                dtype=np.int64)
            self._predictions[seq_id] = {}
            self._ground_truth[seq_id] = {}
            self._intersections[seq_id] = {}
            self._sequence_length[seq_id] = 0
        unique_idxs, counts = np.unique(result[0], return_counts=True)
        self._iou_matrix[seq_id][
            unique_idxs // self.label_divisor,
            unique_idxs % self.label_divisor] += counts
        self._sequence_length[seq_id] += 1

        _update_dict_stats(self._predictions[seq_id], result[1])
        _update_dict_stats(self._ground_truth[seq_id], result[2])
        _update_dict_stats(self._intersections[seq_id], result[3])

    def result(self):
        num_tubes_per_seq = [0] * len(self._ground_truth)
        aq_per_seq = [0] * len(self._ground_truth)
        iou_per_seq = [0] * len(self._ground_truth)
        id_per_seq = [''] * len(self._ground_truth)
        for index, sequence_id in enumerate(self._ground_truth):
            outer_sum = 0.0
            predictions = self._predictions[sequence_id]
            ground_truth = self._ground_truth[sequence_id]
            intersections = self._intersections[sequence_id]
            num_tubes_per_seq[index] = len(ground_truth)
            id_per_seq[index] = sequence_id

            for gt_id, gt_size in ground_truth.items():
                inner_sum = 0.0
                for pr_id, pr_size in predictions.items():
                    tpa_key = self.ins_divisor * gt_id + pr_id
                    if tpa_key in intersections:
                        tpa = intersections[tpa_key]
                        fpa = pr_size - tpa
                        fna = gt_size - tpa
                        inner_sum += tpa * (tpa / (tpa + fpa + fna))

                outer_sum += 1.0 / gt_size * inner_sum
            aq_per_seq[index] = outer_sum
        aq_mean = np.sum(aq_per_seq) / np.maximum(
            np.sum(num_tubes_per_seq), _EPSILON)
        aq_per_seq = aq_per_seq / np.maximum(num_tubes_per_seq, _EPSILON)

        # Compute IoU scores.
        # The rows correspond to ground-truth and the columns to predictions.
        # Remove fp from confusion matrix for the void/ignore class.
        total_confusion = np.zeros(
            (self.num_classes, self.num_classes),
            dtype=np.int64)
        for index, confusion in enumerate(
                self._iou_matrix.values()):
            removal_matrix = np.zeros_like(confusion)
            removal_matrix[self._include_indices, :] = 1.0
            confusion *= removal_matrix
            total_confusion += confusion

            # `intersections` corresponds to true positives.
            intersections = confusion.diagonal()
            fps = confusion.sum(axis=0) - intersections
            fns = confusion.sum(axis=1) - intersections
            unions = intersections + fps + fns

            num_classes = np.count_nonzero(unions)
            ious = (
                    intersections.astype(np.double) /
                    np.maximum(unions, 1e-15).astype(np.double))
            iou_per_seq[index] = np.sum(ious) / num_classes

        # `intersections` corresponds to true positives.
        intersections = total_confusion.diagonal()
        fps = total_confusion.sum(axis=0) - intersections
        fns = total_confusion.sum(axis=1) - intersections
        unions = intersections + fps + fns

        num_classes = np.count_nonzero(unions)
        ious = (
                intersections.astype(np.double) /
                np.maximum(unions, _EPSILON).astype(np.double))
        iou_mean = np.sum(ious) / num_classes

        st_quality = np.sqrt(aq_mean * iou_mean)
        st_quality_per_seq = np.sqrt(aq_per_seq * iou_per_seq)
        return {
            'STQ': st_quality,
            'AQ': aq_mean,
            'IoU': float(iou_mean),
            'STQ_per_seq': st_quality_per_seq,
            'AQ_per_seq': aq_per_seq,
            'IoU_per_seq': iou_per_seq,
            'ID_per_seq': id_per_seq,
            'Length_per_seq': list(self._sequence_length.values()),
        }


def evaluate_single_core(func, pred_name, gt_name, **kwargs):
    pred = torch.load(pred_name)
    gt_name = torch.load(gt_name)
    return func([pred, gt_name], **kwargs)


def evaluate_clip_single_core(func, pred_names, gt_names, **kwargs):
    pred = [torch.load(pred_name) for pred_name in pred_names]
    gt = [torch.load(gt_name) for gt_name in gt_names]
    pred = np.concatenate(pred, axis=1)
    gt = np.concatenate(gt, axis=1)
    return func([pred, gt], **kwargs)


def evaluate(eval_dir, eval_metrics, num_classes=124, num_things=None):
    gt_dir = os.path.join(eval_dir, 'gt')
    pred_dir = os.path.join(eval_dir, 'pred')

    gt_names = list(mmengine.scandir(gt_dir))
    gt_names = sorted(list(filter(lambda x: '.pth' in x and not x.startswith('._'), gt_names)))
    gt_dirs = list(map(lambda x: os.path.join(gt_dir, x), gt_names))

    pred_names = list(mmengine.scandir(pred_dir))
    pred_names = sorted(list(filter(lambda x: '.pth' in x and not x.startswith('._'), pred_names)))
    pred_dirs = list(map(lambda x: os.path.join(pred_dir, x), pred_names))

    print("There are totally {} frames.".format(len(pred_dirs)))

    for metric in eval_metrics:
        assert metric in ["PQ", "VPQ", "STQ", "VPQ6"]
        if metric == "PQ":
            # multi core version
            func = partial(vpq_eval, num_classes=num_classes)
            tasks = [(func, pred, gt) for pred, gt in zip(pred_dirs, gt_dirs)]
            results = track_parallel_progress(
                evaluate_single_core,
                tasks=tasks,
                nproc=128,
            )

            iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)[:num_classes]
            tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)[:num_classes]
            fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)[:num_classes]
            fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)[:num_classes]
            epsilon = 0.
            sq = iou_per_class / (tp_per_class + epsilon)
            rq = tp_per_class / (tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class + epsilon)
            pq = sq * rq
            # for nan cases
            pq = np.nan_to_num(pq)
            print(
                {
                    "PQ": pq,
                    "PQ_all": pq.mean(),
                }
            )
        elif metric == 'STQ':
            state = STQState(num_classes=num_classes)
            for pred, gt in zip(pred_dirs, gt_dirs):
                seq_id = int(os.path.basename(pred).split('_')[0])
                result = evaluate_single_core(
                    stq, pred, gt, num_classes=num_classes, num_things=num_things,
                    label_divisor=state.label_divisor, ins_divisor=state.ins_divisor
                )
                state.update_state_stq(result, seq_id)
            print(state.result())
        elif metric == 'VPQ' or metric == 'VPQ6':
            results = []
            length = len(pred_dirs)

            if metric == 'VPQ':
                windows = [1, 2, 3, 4]
            else:
                windows = [1, 2, 4, 6]

            for k in windows:
                print("evaluate VPQ: k={}".format(k))
                tasks = []
                for idx in range(length):
                    if idx + k - 1 >= length:
                        break
                    seq_id = int(os.path.basename(pred_dirs[idx]).split('_')[0])
                    seq_id_last = int(os.path.basename(pred_dirs[idx + k - 1]).split('_')[0])
                    if seq_id != seq_id_last:
                        continue
                    all_pred = []
                    all_gt = []
                    for j in range(k):
                        pred_cur = pred_dirs[idx + j]
                        gt_cur = gt_dirs[idx + j]
                        all_pred.append(pred_cur)
                        all_gt.append(gt_cur)

                    # multi core version
                    func = partial(vpq_eval, num_classes=num_classes)
                    tasks.append((func, all_pred, all_gt))
                # multi core version
                results = track_parallel_progress(
                    evaluate_clip_single_core,
                    tasks=tasks,
                    nproc=128,
                )
                iou_per_class = np.stack([result[0] for result in results]).sum(axis=0)[:num_classes]
                tp_per_class = np.stack([result[1] for result in results]).sum(axis=0)[:num_classes]
                fn_per_class = np.stack([result[2] for result in results]).sum(axis=0)[:num_classes]
                fp_per_class = np.stack([result[3] for result in results]).sum(axis=0)[:num_classes]

                sq = iou_per_class / (tp_per_class + _EPSILON)
                rq = tp_per_class / (tp_per_class + 0.5 * fn_per_class + 0.5 * fp_per_class + _EPSILON)
                pq = sq * rq
                # remove nan
                pq = np.nan_to_num(pq)
                tpq = pq[:num_things]  # thing
                spq = pq[num_things:]  # stuff
                print(
                    r'PQ : {:.3f} PQ_thing : {:.3f} PQ_stuff : {:.3f}'.format(
                        pq.mean() * 100,
                        tpq.mean() * 100,
                        spq.mean() * 100)
                )
        else:
            raise NotImplementedError


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluation of Dumped Video Clips')
    parser.add_argument('result_path')
    parser.add_argument('--eval_metrics', type=str, nargs='+', help='No description yet')
    parser.add_argument('--launcher', type=str,)
    parser.add_argument('--num_classes', default=124, type=int)
    parser.add_argument('--num_thing_classes', default=58, type=int)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    evaluate(args.result_path, args.eval_metrics, args.num_classes, args.num_thing_classes)
