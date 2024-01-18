# Copyright (c) OpenMMLab. All rights reserved.
import os

import torch
from scipy.optimize import linear_sum_assignment
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import SampleList, TrackDataSample

from seg.models.detectors import Mask2formerVideo
from seg.models.utils import mask_pool

BACKBONE_BATCH = 50


def video_split(total, tube_size, overlap=0):
    assert tube_size > overlap
    total -= overlap
    tube_size -= overlap

    if total % tube_size == 0:
        splits = total // tube_size
    else:
        splits = (total // tube_size) + 1

    ind_list = []
    for i in range(splits):
        ind_list.append((i + 1) * tube_size)

    diff = ind_list[-1] - total

    # currently only supports diff < splits
    if diff < splits:
        for i in range(diff):
            ind_list[splits - 1 - i] -= diff - i
    else:
        ind_list[splits - 1] -= diff
        assert ind_list[splits - 1] > 0
        print("Warning: {} / {}".format(total, tube_size))

    for idx in range(len(ind_list)):
        ind_list[idx] += overlap

    return ind_list


def match_from_embeds(tgt_embds, cur_embds):
    cur_embds = cur_embds / cur_embds.norm(dim=-1, keepdim=True)
    tgt_embds = tgt_embds / tgt_embds.norm(dim=-1, keepdim=True)
    cos_sim = torch.bmm(cur_embds, tgt_embds.transpose(1, 2))

    cost_embd = 1 - cos_sim

    C = 1.0 * cost_embd
    C = C.cpu()

    indices = []
    for i in range(len(cur_embds)):
        indice = linear_sum_assignment(C[i].transpose(0, 1))  # target x current
        indice = indice[1]  # permutation that makes current aligns to target
        indices.append(indice)

    return indices


@MODELS.register_module()
class Mask2formerVideoMinVIS(Mask2formerVideo):
    r"""Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""
    OVERLAPPING = None

    def __init__(self,
                 *args,
                 clip_size=6,
                 clip_size_small=3,
                 whole_clip_thr=0,
                 small_clip_thr=12,
                 overlap=0,
                 **kwargs,
                 ):
        super().__init__(*args, **kwargs)
        self.clip_size = clip_size
        self.clip_size_small = clip_size_small
        self.overlap = overlap
        self.whole_clip_thr = whole_clip_thr
        self.small_clip_thr = small_clip_thr

    def predict(self,
                batch_inputs: Tensor,
                batch_data_samples: SampleList,
                rescale: bool = True) -> SampleList:
        """Predict results from a batch of inputs and data samples with post-
        processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.
            rescale (bool): Whether to rescale the results.
                Defaults to True.

        Returns:
            list[:obj:`DetDataSample`]: Detection results of the
            input images. Each DetDataSample usually contain
            'pred_instances' and `pred_panoptic_seg`. And the
            ``pred_instances`` usually contains following keys.

                - scores (Tensor): Classification scores, has a shape
                    (num_instance, )
                - labels (Tensor): Labels of bboxes, has a shape
                    (num_instances, ).
                - bboxes (Tensor): Has a shape (num_instances, 4),
                    the last dimension 4 arrange as (x1, y1, x2, y2).
                - masks (Tensor): Has a shape (num_instances, H, W).

            And the ``pred_panoptic_seg`` contains the following key

                - sem_seg (Tensor): panoptic segmentation mask, has a
                    shape (1, h, w).
        """
        assert isinstance(batch_data_samples[0], TrackDataSample)

        bs, num_frames, three, h, w = batch_inputs.shape
        assert three == 3, "Only supporting images with 3 channels."
        if num_frames <= self.whole_clip_thr:
            return super().predict(batch_inputs, batch_data_samples, rescale)

        device = batch_inputs.device

        if num_frames > self.small_clip_thr:
            tube_inds = video_split(num_frames, self.clip_size, self.overlap)
        else:
            tube_inds = video_split(num_frames, self.clip_size_small, self.overlap)
        if num_frames > BACKBONE_BATCH:
            feat_bins = [[], [], [], []]
            num_clip = num_frames // BACKBONE_BATCH + 1
            step_size = num_frames // num_clip + 1
            for i in range(num_clip):
                start = i * step_size
                end = min(num_frames, (i + 1) * step_size)
                inputs = batch_inputs[:, start:end].reshape(
                    (bs * (end - start), three, h, w))
                _feats = self.extract_feat(inputs)
                assert len(_feats) == 4
                for idx, item in enumerate(_feats):
                    feat_bins[idx].append(item.to('cpu'))
            feats = []
            for item in feat_bins:
                feat = torch.cat(item, dim=0)
                assert feat.size(0) == bs * num_frames, "{} vs {}".format(feat.size(0), bs * num_frames)
                feats.append(feat)
        else:
            x = batch_inputs.reshape((bs * num_frames, three, h, w))
            feats = self.extract_feat(x)
        assert len(feats[0]) == bs * num_frames

        del batch_inputs

        ind_pre = 0
        cls_list = []
        mask_list = []
        query_list = []
        iou_list = []
        flag = False
        for ind in tube_inds:
            tube_feats = [itm[ind_pre:ind].to(device=device) for itm in feats]
            tube_data_samples = [TrackDataSample(video_data_samples=itm[ind_pre:ind]) for itm in batch_data_samples]
            _mask_cls_results, _mask_pred_results, _query_feat, _iou_results = \
                self.panoptic_head.predict(tube_feats, tube_data_samples, return_query=True)
            cls_list.append(_mask_cls_results)
            if not flag:
                mask_list.append(_mask_pred_results.cpu())
                flag = True
            else:
                mask_list.append(_mask_pred_results[:, self.overlap:].cpu())
            query_list.append(_query_feat.cpu())
            iou_list.append(_iou_results)

            ind_pre = ind
            ind_pre -= self.overlap

        num_tubes = len(tube_inds)

        out_cls = [cls_list[0]]
        out_mask = [mask_list[0]]
        out_embed = [query_list[0]]
        ious = [iou_list[0]]

        for i in range(1, num_tubes):
            indices = match_from_embeds(out_embed[-1], query_list[i])
            indices = indices[0]  # since bs == 1

            out_cls.append(cls_list[i][:, indices])
            out_mask.append(mask_list[i][:, indices])
            out_embed.append(query_list[i][:, indices])
            ious.append(iou_list[i][:, indices])

        del mask_list
        del out_embed
        mask_cls_results = sum(out_cls) / num_tubes
        mask_pred_results = torch.cat(out_mask, dim=2)
        iou_results = sum(ious) / num_tubes

        if self.OVERLAPPING is not None:
            assert len(self.OVERLAPPING) == self.num_classes
            mask_cls_results = self.open_voc_inference(feats, mask_cls_results, mask_pred_results)

        del feats
        mask_cls_results = mask_cls_results.to(device='cpu')
        iou_results = iou_results.to(device='cpu')

        id_assigner = [{} for _ in range(bs)]

        for frame_id in range(num_frames):
            results_list_img = self.panoptic_fusion_head.predict(
                mask_cls_results,
                mask_pred_results[:, :, frame_id],
                [batch_data_samples[idx][frame_id] for idx in range(bs)],
                iou_results=iou_results,
                rescale=rescale
            )
            if frame_id == 0 and 'pro_results' in results_list_img[0]:
                for batch_id in range(bs):
                    mask = results_list_img[batch_id]['pro_results'].to(dtype=torch.int32)
                    mask_gt = torch.tensor(batch_data_samples[batch_id][frame_id].gt_instances.masks.masks, dtype=torch.int32)
                    a, b = mask.flatten(1), mask_gt.flatten(1)
                    intersection = torch.einsum('nc,mc->nm', a, b)
                    union = (a[:, None] + b[None]).clamp(min=0, max=1).sum(-1)
                    iou_cost = intersection / union
                    a_indices, b_indices = linear_sum_assignment(-iou_cost.numpy())

                    for a_ind, b_ind in zip(a_indices, b_indices):
                        id_assigner[batch_id][a_ind] = batch_data_samples[batch_id][frame_id].gt_instances.instances_ids[b_ind].item()

            if 'pro_results' in results_list_img[0]:
                h, w = results_list_img[batch_id]['pro_results'].shape[-2:]
                seg_map = torch.full((h, w), 0, dtype=torch.int32, device='cpu')
                for ind in id_assigner[batch_id]:
                    seg_map[results_list_img[batch_id]['pro_results'][ind]] = id_assigner[batch_id][ind]
                results_list_img[batch_id]['pro_results'] = seg_map.cpu().numpy()

            _ = self.add_track_pred_to_datasample(
                [batch_data_samples[idx][frame_id] for idx in range(bs)], results_list_img
            )
        results = batch_data_samples

        return results

    def open_voc_inference(self, feats, mask_cls_results, mask_pred_results):
        if len(mask_pred_results.shape) == 5:
            batch_size = mask_cls_results.shape[0]
            num_frames = mask_pred_results.shape[2]
            mask_pred_results = mask_pred_results.permute(0, 2, 1, 3, 4).flatten(0, 1)
        else:
            batch_size = mask_cls_results.shape[0]
            num_frames = 0
        clip_feat = self.backbone.get_clip_feature(feats[-1]).to(device=mask_cls_results.device)
        clip_feat_mask = F.interpolate(
            mask_pred_results,
            size=clip_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        ).to(device=mask_cls_results.device)
        if num_frames > 0:
            clip_feat_mask = clip_feat_mask.unflatten(0, (batch_size, num_frames)).permute(0, 2, 1, 3, 4).flatten(2, 3)
            clip_feat = clip_feat.unflatten(0, (batch_size, num_frames)).permute(0, 2, 1, 3, 4).flatten(2, 3)
        instance_feat = mask_pool(clip_feat, clip_feat_mask)
        instance_feat = self.backbone.forward_feat(instance_feat)
        clip_logit = self.panoptic_head.forward_logit(instance_feat)
        clip_logit = clip_logit[..., :-1]
        query_logit = mask_cls_results[..., :-1]

        clip_logit = clip_logit.softmax(-1)
        query_logit = query_logit.softmax(-1)
        overlapping_mask = torch.tensor(self.OVERLAPPING, dtype=torch.float32, device=clip_logit.device)

        valid_masking = ((clip_feat_mask > 0).to(dtype=torch.float32).flatten(-2).sum(-1) > 0).to(
            torch.float32)[..., None]
        alpha = torch.ones_like(clip_logit) * self.alpha * valid_masking
        beta = torch.ones_like(clip_logit) * self.beta * valid_masking

        cls_logits_seen = (
                (query_logit ** (1 - alpha) * clip_logit ** alpha).log()
                * overlapping_mask
        )
        cls_logits_unseen = (
                (query_logit ** (1 - beta) * clip_logit ** beta).log()
                * (1 - overlapping_mask)
        )
        cls_results = cls_logits_seen + cls_logits_unseen
        is_void_prob = F.softmax(mask_cls_results, dim=-1)[..., -1:]
        mask_cls_results = torch.cat([
            cls_results.softmax(-1) * (1.0 - is_void_prob), is_void_prob], dim=-1)
        mask_cls_results = torch.log(mask_cls_results + 1e-8)
        return mask_cls_results
