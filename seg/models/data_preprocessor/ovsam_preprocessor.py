import copy
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from mmdet.models import DetDataPreprocessor
from mmdet.registry import MODELS
from kornia.contrib import distance_transform
from mmengine.structures import InstanceData

from seg.models.data_preprocessor import VideoSegDataPreprocessor


def get_center_coords(gt_instances, rescale_shape=None, device='cpu'):
    if rescale_shape is not None:
        masks = gt_instances.masks
        masks = masks.rescale(rescale_shape)
    else:
        masks = gt_instances.masks
    masks = masks.to_tensor(dtype=torch.bool, device=device)[:, None]
    point_coords = []
    for mask in masks:
        mask = mask[None]
        n, _, h, w = mask.shape
        mask_dt = (
            distance_transform(
                (~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float()
            )[:, :, 1:-1, 1:-1]
        )
        selected_point = torch.tensor([mask_dt.argmax() / w, mask_dt.argmax() % w]).long().flip(0).to(
            device)
        point_coords.append(selected_point)
    if len(point_coords) > 0:
        point_coords = torch.stack(point_coords)[:, None]
    else:
        point_coords = torch.empty((0, 1, 2), dtype=torch.int32).to(device=device)
    return point_coords


def get_random_points(gt_instances, device='cpu'):
    point_coords = []
    for instance_idx in range(len(gt_instances)):
        mask = gt_instances.masks.masks[instance_idx]
        candidate_indices = torch.tensor(mask, device=device).nonzero()
        assert len(candidate_indices) > 0
        selected_point = candidate_indices[torch.randperm(
            len(candidate_indices), dtype=torch.int32, device=device)[0]].flip(0)
        point_coords.append(selected_point)
    if len(point_coords) > 0:
        point_coords = torch.stack(point_coords)[:, None]
    else:
        point_coords = torch.empty((0, 1, 2), dtype=torch.int32).to(device=device)
    return point_coords


@MODELS.register_module()
class OVSAMDataPreprocessor(DetDataPreprocessor):
    def __init__(self, *args,
                 use_det: bool = False,
                 use_point: bool = False,
                 use_center_point: bool = False,
                 use_point_det: bool = False,
                 use_center_point_det: bool = False,
                 use_point_pseudo_box: bool = False,
                 use_img_center: bool = False,
                 use_custom_bbox: Optional[Tuple] = None,
                 use_custom_point: Optional[Tuple] = None,
                 num_proposals: int = 60,
                 default_mode: str = 'sam',
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_proposals = num_proposals
        self.use_det = use_det
        self.use_point = use_point
        self.use_center_point = use_center_point
        self.use_point_det = use_point_det
        self.use_center_point_det = use_center_point_det
        self.use_point_pseudo_box = use_point_pseudo_box
        self.use_img_center = use_img_center
        self.use_custom_bbox = use_custom_bbox
        self.use_custom_point = use_custom_point
        self.default_mode = default_mode

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']
        if 'data_tag' in data_samples[0]:
            data_tag = data_samples[0].data_tag
            for i in range(1, len(data_samples)):
                assert data_samples[i].data_tag == data_tag
        else:
            data_tag = self.default_mode
            for i in range(0, len(data_samples)):
                data_samples[i].data_tag = data_tag
        device = inputs.device

        if data_tag == 'sam_mul':
            for data_sample in data_samples:
                gt_instances_collected = data_sample.gt_instances_collected
                gt_instances = data_sample.gt_instances
                masks_list = []
                for idx in range(len(gt_instances_collected)):
                    gt_ids = gt_instances_collected.sub_instances[idx]
                    masks_list.append(gt_instances.masks[gt_ids])
                gt_instances = InstanceData(
                    labels=torch.zeros_like(gt_instances_collected.idx),
                    masks=masks_list,
                    point_coords=gt_instances_collected.point_coords,
                    bp=torch.zeros_like(gt_instances_collected.idx),  # all box
                )
                # all points
                data_sample.gt_instances = gt_instances
                del data_sample.gt_instances_collected
        elif data_tag == 'sam':
            num_proposals = self.num_proposals if training else 10000000
            if self.use_custom_bbox:
                for data_sample in data_samples:
                    img_shape = data_sample.img_shape
                    data_sample.gt_instances = InstanceData(
                        bboxes=inputs.new_tensor([[img_shape[1] * self.use_custom_bbox[0],
                                                   img_shape[0] * self.use_custom_bbox[1],
                                                   img_shape[1] * self.use_custom_bbox[2],
                                                   img_shape[0] * self.use_custom_bbox[3]]])
                    )
            elif self.use_img_center:
                for data_sample in data_samples:
                    data_sample.gt_instances = InstanceData(
                        point_coords=inputs.new_tensor([[[data_sample.img_shape[1] / 2, data_sample.img_shape[0] / 2]]])
                    )
            elif self.use_custom_point:
                for data_sample in data_samples:
                    data_sample.gt_instances = InstanceData(
                        point_coords=inputs.new_tensor([[[self.use_custom_point[0], self.use_custom_point[1]]]])
                    )
            elif self.use_det:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if not training:
                        bboxes = gt_instances.bboxes
                        scale_factor = bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                        bboxes = bboxes * scale_factor
                        gt_instances.bboxes = bboxes
                        num_ins = len(gt_instances)
                        bp_indicator = torch.zeros((num_ins,))
                        gt_instances.bp = bp_indicator.to(device=device)
                    data_sample.gt_instances = gt_instances
            elif self.use_point_det:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if len(gt_instances) < num_proposals:
                        num_copy = num_proposals // len(gt_instances) + 1
                        gt_instances = InstanceData.cat([copy.deepcopy(gt_instances) for _ in range(num_copy)])
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_random_points(gt_instances, device=device)
                    else:
                        raise NotImplementedError
                    num_ins = len(gt_instances)
                    bp_indicator = torch.arange(2).repeat_interleave((num_ins // 2) + 1)[:num_ins]
                    gt_instances = gt_instances[torch.randperm(num_ins, device=device)]
                    gt_instances.bp = bp_indicator.to(device=device)
                    data_sample.gt_instances = gt_instances
            elif self.use_center_point_det:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_center_coords(gt_instances, device=device)
                    else:
                        gt_instances.point_coords = get_center_coords(
                            gt_instances, rescale_shape=data_sample.img_shape, device=device
                        )
                        bboxes = gt_instances.bboxes
                        scale_factor = bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                        bboxes = bboxes * scale_factor
                        gt_instances.bboxes = bboxes
                    data_sample.gt_instances = gt_instances
            elif self.use_point:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_random_points(gt_instances, device=device)
                    else:
                        raise NotImplementedError
                    data_sample.gt_instances = gt_instances
            elif self.use_center_point:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_center_coords(gt_instances, device=device)
                    else:
                        gt_instances.point_coords = get_center_coords(
                            gt_instances, rescale_shape=data_sample.img_shape, device=device
                        )
                    data_sample.gt_instances = gt_instances
            elif self.use_point_pseudo_box:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if training:
                        if len(gt_instances) < num_proposals:
                            num_copy = num_proposals // len(gt_instances) + 1
                            gt_instances = InstanceData.cat([copy.deepcopy(gt_instances) for _ in range(num_copy)])
                        gt_instances = gt_instances[:num_proposals]
                        points = get_random_points(gt_instances, device=device)
                    else:
                        points = get_center_coords(
                            gt_instances, rescale_shape=data_sample.img_shape, device=device
                        )
                    points = points.squeeze(1)
                    gt_instances.point_coords = torch.cat([points - 3, points + 3], 1)
                    gt_instances.bp = torch.zeros_like(gt_instances.labels)  # bug to match sam_mul
                    data_sample.gt_instances = gt_instances
            else:
                raise NotImplementedError
        elif data_tag == 'coco':
            pass
        elif data_tag == 'img':
            for data_sample in data_samples:
                gt_instances = data_sample.gt_instances
                h, w = data_sample.img_shape
                gt_instances.bboxes = torch.tensor(
                    [[0., 0., h, w]], dtype=torch.float32, device=gt_instances.labels.device
                )
                gt_instances.bp = torch.zeros((1,), dtype=torch.int32, device=gt_instances.labels.device)
        elif data_tag == 'mosaic_img':
            b, three, h, w = inputs.shape
            num_img_per_batch = 4 * 4
            assert b % num_img_per_batch == 0
            target_h, target_w = h * 4, w * 4
            new_b = b // num_img_per_batch
            result_input = inputs.new_empty(b // num_img_per_batch, three, target_h, target_w)
            cnt = 0
            result_data_samples = []
            for id_b in range(new_b):
                cur_data_sample = data_samples[cnt]
                cur_gt_instances = []
                for id_x in range(4):
                    for id_y in range(4):
                        result_input[id_b, :, id_x * h: (id_x + 1) * h, id_y * w: (id_y + 1) * w] = inputs[cnt]
                        img_gt_instances = data_samples[cnt].gt_instances
                        img_gt_instances.bboxes += img_gt_instances.bboxes.new_tensor([
                            id_x * h, id_y * w, id_x * h, id_y * w
                        ])
                        cur_gt_instances.append(img_gt_instances)
                        cnt += 1
                cur_gt_instances = InstanceData.cat(cur_gt_instances)
                cur_data_sample.gt_instances = cur_gt_instances
                result_data_samples.append(cur_data_sample)

            inputs = result_input
            data_samples = result_data_samples
        else:
            raise NotImplementedError
        return dict(inputs=inputs, data_samples=data_samples)


@MODELS.register_module()
class OVSAMVideoSegDataPreprocessor(VideoSegDataPreprocessor):
    def __init__(self, *args,
                 use_det: bool = False,
                 use_point: bool = False,
                 use_center_point: bool = False,
                 use_point_det: bool = False,
                 use_center_point_det: bool = False,
                 use_point_pseudo_box: bool = False,
                 num_proposals: int = 60,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.num_proposals = num_proposals
        self.use_det = use_det
        self.use_point = use_point
        self.use_center_point = use_center_point
        self.use_point_det = use_point_det
        self.use_center_point_det = use_center_point_det
        self.use_point_pseudo_box = use_point_pseudo_box

    def forward(self, data: dict, training: bool = False) -> dict:
        data = super().forward(data, training=training)
        inputs, data_samples = data['inputs'], data['data_samples']
        if 'data_tag' in data_samples[0]:
            data_tag = data_samples[0].data_tag
            for i in range(1, len(data_samples)):
                assert data_samples[i].data_tag == data_tag
        else:
            data_tag = 'sam'
            for i in range(0, len(data_samples)):
                data_samples[i].data_tag = data_tag
        device = inputs.device

        if data_tag == 'sam_mul':
            for data_sample in data_samples:
                gt_instances_collected = data_sample.gt_instances_collected
                gt_instances = data_sample.gt_instances
                masks_list = []
                for idx in range(len(gt_instances_collected)):
                    gt_ids = gt_instances_collected.sub_instances[idx]
                    masks_list.append(gt_instances.masks[gt_ids])
                gt_instances = InstanceData(
                    labels=torch.zeros_like(gt_instances_collected.idx),
                    masks=masks_list,
                    point_coords=gt_instances_collected.point_coords,
                    bp=torch.zeros_like(gt_instances_collected.idx),  # all box
                )
                # all points
                data_sample.gt_instances = gt_instances
                del data_sample.gt_instances_collected
        elif data_tag == 'sam':
            num_proposals = self.num_proposals if training else 10000000
            if self.use_det:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if not training:
                        bboxes = gt_instances.bboxes
                        scale_factor = bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                        bboxes = bboxes * scale_factor
                        gt_instances.bboxes = bboxes
                    data_sample.gt_instances = gt_instances
            elif self.use_point_det:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if len(gt_instances) < num_proposals:
                        num_copy = num_proposals // len(gt_instances) + 1
                        gt_instances = InstanceData.cat([copy.deepcopy(gt_instances) for _ in range(num_copy)])
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_random_points(gt_instances, device=device)
                    else:
                        raise NotImplementedError
                    num_ins = len(gt_instances)
                    bp_indicator = torch.arange(2).repeat_interleave((num_ins // 2) + 1)[:num_ins]
                    gt_instances = gt_instances[torch.randperm(num_ins, device=device)]
                    gt_instances.bp = bp_indicator.to(device=device)
                    data_sample.gt_instances = gt_instances
            elif self.use_center_point_det:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_center_coords(gt_instances, device=device)
                    else:
                        gt_instances.point_coords = get_center_coords(
                            gt_instances, rescale_shape=data_sample.img_shape, device=device
                        )
                        bboxes = gt_instances.bboxes
                        scale_factor = bboxes.new_tensor(data_sample.scale_factor).repeat(2)
                        bboxes = bboxes * scale_factor
                        gt_instances.bboxes = bboxes
                    data_sample.gt_instances = gt_instances
            elif self.use_point:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_random_points(gt_instances, device=device)
                    else:
                        raise NotImplementedError
                    data_sample.gt_instances = gt_instances
            elif self.use_center_point:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    gt_instances = gt_instances[:num_proposals]
                    if training:
                        gt_instances.point_coords = get_center_coords(gt_instances, device=device)
                    else:
                        gt_instances.point_coords = get_center_coords(
                            gt_instances, rescale_shape=data_sample.img_shape, device=device
                        )
                    data_sample.gt_instances = gt_instances
            elif self.use_point_pseudo_box:
                for data_sample in data_samples:
                    gt_instances = data_sample.gt_instances
                    if training:
                        if len(gt_instances) < num_proposals:
                            num_copy = num_proposals // len(gt_instances) + 1
                            gt_instances = InstanceData.cat([copy.deepcopy(gt_instances) for _ in range(num_copy)])
                        gt_instances = gt_instances[:num_proposals]
                        points = get_random_points(gt_instances, device=device)
                    else:
                        points = get_center_coords(
                            gt_instances, rescale_shape=data_sample.img_shape, device=device
                        )
                    points = points.squeeze(1)
                    gt_instances.point_coords = torch.cat([points - 3, points + 3], 1)
                    gt_instances.bp = torch.zeros_like(gt_instances.labels)  # bug to match sam_mul
                    data_sample.gt_instances = gt_instances
            else:
                raise NotImplementedError
        elif data_tag == 'coco':
            pass
        elif data_tag == 'img':
            for data_sample in data_samples:
                gt_instances = data_sample.gt_instances
                h, w = data_sample.img_shape
                gt_instances.bboxes = torch.tensor(
                    [[0., 0., h, w]], dtype=torch.float32, device=gt_instances.labels.device
                )
                gt_instances.bp = torch.zeros((1,), dtype=torch.int32, device=gt_instances.labels.device)
        else:
            raise NotImplementedError
        return dict(inputs=inputs, data_samples=data_samples)
