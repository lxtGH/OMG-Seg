from typing import Optional, Sequence, List

import torch
import random
import numpy as np
from mmcv.transforms import to_tensor
from mmcv.transforms.base import BaseTransform
from mmdet.datasets.transforms import PackDetInputs
from mmdet.structures.bbox import BaseBoxes
from mmengine.structures import InstanceData, PixelData

from mmdet.registry import TRANSFORMS
from mmdet.structures import DetDataSample, TrackDataSample


@TRANSFORMS.register_module()
class PackVidSegInputs(BaseTransform):
    """Pack the inputs data for the multi object tracking and video instance
    segmentation. All the information of images are packed to ``inputs``. All
    the information except images are packed to ``data_samples``. In order to
    get the original annotaiton and meta info, we add `instances` key into meta
    keys.

    Args:
        meta_keys (Sequence[str]): Meta keys to be collected in
            ``data_sample.metainfo``. Defaults to None.
        default_meta_keys (tuple): Default meta keys. Defaults to ('img_id',
            'img_path', 'ori_shape', 'img_shape', 'scale_factor',
            'flip', 'flip_direction', 'frame_id', 'is_video_data',
            'video_id', 'video_length', 'instances').
    """
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_instances_ids': 'instances_ids'
    }

    def __init__(self,
                 meta_keys: Optional[dict] = None,
                 default_meta_keys: tuple = ('img_id', 'img_path', 'ori_shape',
                                             'img_shape', 'scale_factor',
                                             'flip', 'flip_direction',
                                             'frame_id', 'video_id',
                                             'video_length',
                                             'ori_video_length', 'instances')):
        self.meta_keys = default_meta_keys
        if meta_keys is not None:
            if isinstance(meta_keys, str):
                meta_keys = (meta_keys,)
            else:
                assert isinstance(meta_keys, tuple), \
                    'meta_keys must be str or tuple'
            self.meta_keys += meta_keys

    def transform(self, results: dict) -> dict:
        """Method to pack the input data.
        Args:
            results (dict): Result dict from the data pipeline.
        Returns:
            dict:
            - 'inputs' (dict[Tensor]): The forward data of models.
            - 'data_samples' (obj:`TrackDataSample`): The annotation info of
                the samples.
        """
        packed_results = dict()
        packed_results['inputs'] = dict()

        # 1. Pack images
        if 'img' in results:
            imgs = results['img']
            imgs = np.stack(imgs, axis=0)
            imgs = imgs.transpose(0, 3, 1, 2)
            packed_results['inputs'] = to_tensor(imgs)

        # 2. Pack InstanceData
        if 'gt_ignore_flags' in results:
            gt_ignore_flags_list = results['gt_ignore_flags']
            valid_idx_list, ignore_idx_list = [], []
            for gt_ignore_flags in gt_ignore_flags_list:
                valid_idx = np.where(gt_ignore_flags == 0)[0]
                ignore_idx = np.where(gt_ignore_flags == 1)[0]
                valid_idx_list.append(valid_idx)
                ignore_idx_list.append(ignore_idx)

        assert 'img_id' in results, "'img_id' must contained in the results "
        'for counting the number of images'

        num_imgs = len(results['img_id'])
        instance_data_list = [InstanceData() for _ in range(num_imgs)]
        ignore_instance_data_list = [InstanceData() for _ in range(num_imgs)]

        for key in self.mapping_table.keys():
            if key not in results:
                continue
            if key == 'gt_masks' or (isinstance(results[key], List) and isinstance(results[key][0], BaseBoxes)):
                mapped_key = self.mapping_table[key]
                gt_masks_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, gt_mask in enumerate(gt_masks_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][mapped_key] = gt_mask[valid_idx]
                        ignore_instance_data_list[i][mapped_key] = gt_mask[
                            ignore_idx]

                else:
                    for i, gt_mask in enumerate(gt_masks_list):
                        instance_data_list[i][mapped_key] = gt_mask

            else:
                anns_list = results[key]
                if 'gt_ignore_flags' in results:
                    for i, ann in enumerate(anns_list):
                        valid_idx, ignore_idx = valid_idx_list[
                            i], ignore_idx_list[i]
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                            ann[valid_idx])
                        ignore_instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(
                            ann[ignore_idx])
                else:
                    for i, ann in enumerate(anns_list):
                        instance_data_list[i][
                            self.mapping_table[key]] = to_tensor(ann)

        det_data_samples_list = []
        for i in range(num_imgs):
            det_data_sample = DetDataSample()
            det_data_sample.gt_instances = instance_data_list[i]
            det_data_sample.ignored_instances = ignore_instance_data_list[i]

            if 'proposals' in results:
                proposals = InstanceData(
                    bboxes=to_tensor(results['proposals'][i]),
                    scores=to_tensor(results['proposals_scores'][i]))
                det_data_sample.proposals = proposals

            if 'gt_seg_map' in results:
                gt_sem_seg_data = dict(
                    sem_seg=to_tensor(results['gt_seg_map'][i][None, ...].copy()))
                gt_sem_seg_data = PixelData(**gt_sem_seg_data)
                if 'ignore_index' in results:
                    metainfo = dict(ignore_index=results['ignore_index'][i])
                    gt_sem_seg_data.set_metainfo(metainfo)
                det_data_sample.gt_sem_seg = gt_sem_seg_data

            det_data_samples_list.append(det_data_sample)

        # 3. Pack metainfo
        for key in self.meta_keys:
            if key not in results:
                continue
            img_metas_list = results[key]
            for i, img_meta in enumerate(img_metas_list):
                det_data_samples_list[i].set_metainfo({f'{key}': img_meta})

        track_data_sample = TrackDataSample()
        track_data_sample.video_data_samples = det_data_samples_list
        if 'key_frame_flags' in results:
            key_frame_flags = np.asarray(results['key_frame_flags'])
            key_frames_inds = np.where(key_frame_flags)[0].tolist()
            ref_frames_inds = np.where(~key_frame_flags)[0].tolist()
            track_data_sample.set_metainfo(
                dict(key_frames_inds=key_frames_inds))
            track_data_sample.set_metainfo(
                dict(ref_frames_inds=ref_frames_inds))

        packed_results['data_samples'] = track_data_sample
        return packed_results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'meta_keys={self.meta_keys}, '
        repr_str += f'default_meta_keys={self.default_meta_keys})'
        return repr_str


@TRANSFORMS.register_module()
class PackSAMInputs(PackDetInputs):
    mapping_table = {
        'gt_bboxes': 'bboxes',
        'gt_bboxes_labels': 'labels',
        'gt_masks': 'masks',
        'gt_point_coords': 'point_coords',
    }

    def transform(self, results: dict) -> dict:
        if 'feat' in results:
            gt_feats = results['feat']
            results = super().transform(results)
            results['data_samples'].gt_feats = gt_feats
            return results
        else:
            return super().transform(results)


@TRANSFORMS.register_module()
class GeneratePoint(BaseTransform):
    def __init__(self, num_proposals=60, num_mask_tokens=4):
        self.num_proposals = num_proposals
        self.num_mask_tokens = num_mask_tokens

    def transform(self, results):
        data_samples = results['data_samples']
        gt_instances = data_samples.gt_instances

        ori_num_instances = len(gt_instances)
        ori_indices = torch.randperm(ori_num_instances)

        if ori_num_instances < self.num_proposals:
            repeat_cnt = (self.num_proposals // ori_num_instances) + 1
            ori_indices = ori_indices.repeat(repeat_cnt)
        indices = ori_indices[:self.num_proposals]

        masks = gt_instances.masks.to_tensor(torch.bool, 'cpu')
        gt_collected = []
        for instance_idx in indices:
            mask = masks[instance_idx]
            candidate_indices = mask.nonzero()
            assert len(candidate_indices) > 0
            selected_index = random.randint(0, len(candidate_indices) - 1)
            selected_point = candidate_indices[selected_index].flip(0)

            selected_instances_idx = []
            for instance_to_match_idx in range(len(gt_instances)):
                mask_to_match = masks[instance_to_match_idx]
                if mask_to_match[tuple(selected_point.flip(0))]:
                    selected_instances_idx.append(instance_to_match_idx)
            assert len(selected_instances_idx) > 0
            if len(selected_instances_idx) > self.num_mask_tokens:
                random.shuffle(selected_instances_idx)
                selected_instances_idx = selected_instances_idx[:self.num_mask_tokens]
            selected_point = torch.cat([selected_point - 3, selected_point + 3], 0)
            gt_collected.append({
                'point_coords': selected_point,
                'instances': selected_instances_idx,
            })

        data_samples.gt_instances_collected = InstanceData(
            point_coords=torch.stack([itm['point_coords'] for itm in gt_collected]),
            sub_instances=[itm['instances'] for itm in gt_collected],
            idx=indices
        )
        return results
