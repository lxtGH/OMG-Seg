# Copied from OMG-Seg
from typing import Dict, List, Tuple

import torch
from mmengine.structures import InstanceData
from torch import Tensor
import torch.nn.functional as F

from mmdet.registry import MODELS
from mmdet.structures import SampleList, OptSampleList, TrackDataSample
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from mmdet.models.detectors.single_stage import SingleStageDetector

from .utils import mask_pool


@MODELS.register_module()
class Mask2formerVideo(SingleStageDetector):
    r"""Implementation of `Per-Pixel Classification is
    NOT All You Need for Semantic Segmentation
    <https://arxiv.org/pdf/2107.06278>`_."""
    OVERLAPPING = None

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 inference_sam: bool = False,
                 init_cfg: OptMultiConfig = None
                 ):
        super(SingleStageDetector, self).__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)
        if neck is not None:
            self.neck = MODELS.build(neck)

        panoptic_head_ = panoptic_head.deepcopy()
        panoptic_head_.update(train_cfg=train_cfg)
        panoptic_head_.update(test_cfg=test_cfg)
        self.panoptic_head_cfg = panoptic_head_
        self.panoptic_head = MODELS.build(panoptic_head_)

        panoptic_fusion_head_ = panoptic_fusion_head.deepcopy()
        panoptic_fusion_head_.update(test_cfg=test_cfg)
        self.panoptic_fusion_head = MODELS.build(panoptic_fusion_head_)

        self.num_things_classes = self.panoptic_head.num_things_classes
        self.num_stuff_classes = self.panoptic_head.num_stuff_classes
        self.num_classes = self.panoptic_head.num_classes

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self.alpha = 0.4
        self.beta = 0.8

        self.inference_sam = inference_sam

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Dict[str, Tensor]:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        if isinstance(batch_data_samples[0], TrackDataSample):
            bs, num_frames, three, h, w = batch_inputs.shape
            assert three == 3, "Only supporting images with 3 channels."

            x = batch_inputs.reshape((bs * num_frames, three, h, w))
            x = self.extract_feat(x)
        else:
            x = self.extract_feat(batch_inputs)
        losses = self.panoptic_head.loss(x, batch_data_samples)
        return losses

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
        if isinstance(batch_data_samples[0], TrackDataSample):
            bs, num_frames, three, h, w = batch_inputs.shape
            assert three == 3, "Only supporting images with 3 channels."
            x = batch_inputs.reshape((bs * num_frames, three, h, w))
            feats = self.extract_feat(x)
        else:
            num_frames = 0
            bs = batch_inputs.shape[0]
            feats = self.extract_feat(batch_inputs)

        # in case no queries are provided for prompt.
        if self.inference_sam and len(batch_data_samples[0].gt_instances) == 0:
            for idx, data_sample in enumerate(batch_data_samples):
                results = InstanceData()
                data_sample.pred_instances = results
            return batch_data_samples

        mask_cls_results, mask_pred_results, iou_results = self.panoptic_head.predict(feats, batch_data_samples)

        if self.OVERLAPPING is not None:
            assert len(self.OVERLAPPING) == self.num_classes
            mask_cls_results = self.open_voc_inference(feats, mask_cls_results, mask_pred_results)

        if self.inference_sam:
            for idx, data_sample in enumerate(batch_data_samples):
                results = InstanceData()
                mask = mask_pred_results[idx]
                img_height, img_width = data_sample.metainfo['img_shape'][:2]
                mask = mask[:, :img_height, :img_width]
                ori_height, ori_width = data_sample.metainfo['ori_shape'][:2]
                mask = F.interpolate(
                    mask[:, None],
                    size=(ori_height, ori_width),
                    mode='bilinear',
                    align_corners=False)[:, 0]
                results.masks = mask.sigmoid() > 0.5
                data_sample.pred_instances = results
            return batch_data_samples

        if num_frames > 0:
            for frame_id in range(num_frames):
                results_list_img = self.panoptic_fusion_head.predict(
                    mask_cls_results,
                    mask_pred_results[:, :, frame_id],
                    [batch_data_samples[idx][frame_id] for idx in range(bs)],
                    rescale=rescale
                )
                _ = self.add_track_pred_to_datasample(
                    [batch_data_samples[idx][frame_id] for idx in range(bs)], results_list_img
                )
            results = batch_data_samples
        else:
            results_list = self.panoptic_fusion_head.predict(
                mask_cls_results,
                mask_pred_results,
                batch_data_samples,
                iou_results=iou_results,
                rescale=rescale
            )
            results = self.add_pred_to_datasample(batch_data_samples, results_list)

        return results

    def add_pred_to_datasample(self, data_samples: SampleList,
                               results_list: List[dict]) -> SampleList:
        """Add predictions to `DetDataSample`.

        Args:
            data_samples (list[:obj:`DetDataSample`], optional): A batch of
                data samples that contain annotations and predictions.
            results_list (List[dict]): Instance segmentation, segmantic
                segmentation and panoptic segmentation results.

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
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                data_sample.pred_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                data_sample.pred_instances = pred_results['ins_results']

            assert 'sem_results' not in pred_results

        return data_samples

    def add_track_pred_to_datasample(self, data_samples: SampleList, results_list: List[dict]) -> SampleList:
        for data_sample, pred_results in zip(data_samples, results_list):
            if 'pan_results' in pred_results:
                assert self.num_stuff_classes > 0
                pred_results['pan_results'].sem_seg = pred_results['pan_results'].sem_seg.cpu()
                data_sample.pred_track_panoptic_seg = pred_results['pan_results']

            if 'ins_results' in pred_results:
                bboxes = pred_results['ins_results']['bboxes']
                labels = pred_results['ins_results']['labels']
                track_ids = torch.arange(len(bboxes), dtype=labels.dtype, device=bboxes.device) + 1
                pred_results['ins_results']['instances_id'] = track_ids
                data_sample.pred_track_instances = pred_results['ins_results']

            if 'pro_results' in pred_results:
                data_sample.pred_track_proposal = pred_results['pro_results']

            assert 'sem_results' not in pred_results

        return data_samples

    def _forward(
            self,
            batch_inputs: Tensor,
            batch_data_samples: OptSampleList = None) -> Tuple[List[Tensor]]:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

         Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            tuple[List[Tensor]]: A tuple of features from ``panoptic_head``
            forward.
        """
        if isinstance(batch_data_samples[0], TrackDataSample):
            bs, num_frames, three, h, w = batch_inputs.shape
            assert three == 3, "Only supporting images with 3 channels."

            x = batch_inputs.reshape((bs * num_frames, three, h, w))
            feats = self.extract_feat(x)
        else:
            feats = self.extract_feat(batch_inputs)
        results = self.panoptic_head.forward(feats, batch_data_samples)
        return results

    def open_voc_inference(self, feats, mask_cls_results, mask_pred_results):
        if len(mask_pred_results.shape) == 5:
            batch_size = mask_cls_results.shape[0]
            num_frames = mask_pred_results.shape[2]
            mask_pred_results = mask_pred_results.permute(0, 2, 1, 3, 4).flatten(0, 1)
        else:
            batch_size = mask_cls_results.shape[0]
            num_frames = 0
        clip_feat = self.backbone.get_clip_feature(feats[-1])
        clip_feat_mask = F.interpolate(
            mask_pred_results,
            size=clip_feat.shape[-2:],
            mode='bilinear',
            align_corners=False
        )
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
