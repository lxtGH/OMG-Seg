# Copyright (c) OpenMMLab. All rights reserved.
import copy
import os
from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from mmcv.cnn import Conv2d
from mmcv.ops import point_sample
from mmdet.models import Mask2FormerTransformerDecoder, inverse_sigmoid, coordinate_to_encoding
from mmdet.structures.bbox import bbox_xyxy_to_cxcywh
from mmengine.dist import get_dist_info
from mmengine.model import caffe2_xavier_init, ModuleList
from mmengine.structures import InstanceData, PixelData
from torch import Tensor

from mmdet.registry import MODELS, TASK_UTILS
from mmdet.structures import SampleList, TrackDataSample
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptMultiConfig, reduce_mean)
from mmdet.models.layers import SinePositionalEncoding3D
from mmdet.models.utils import multi_apply, preprocess_panoptic_gt, get_uncertain_point_coords_with_randomness
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from .utils import mask_pool
preprocess_video_panoptic_gt = None
from kornia.contrib import distance_transform
from omg_llava.model.modules.projector.modeling_projector import CrossAttentionLayer, FFNLayer
import numpy as np

@MODELS.register_module()
class Mask2FormerVideoSemSamHead(AnchorFreeHead):
    """Implements the Mask2Former head.

    See `Masked-attention Mask Transformer for Universal Image
    Segmentation <https://arxiv.org/pdf/2112.01527>`_ for details.

    Args:
        in_channels (list[int]): Number of channels in the input feature map.
        feat_channels (int): Number of channels for features.
        out_channels (int): Number of channels for output.
        num_things_classes (int): Number of things.
        num_stuff_classes (int): Number of stuff.
        num_queries (int): Number of query in Transformer decoder.
        pixel_decoder (:obj:`ConfigDict` or dict): Config for pixel
            decoder. Defaults to None.
        enforce_decoder_input_project (bool, optional): Whether to add
            a layer to change the embed_dim of tranformer encoder in
            pixel decoder to the embed_dim of transformer decoder.
            Defaults to False.
        transformer_decoder (:obj:`ConfigDict` or dict): Config for
            transformer decoder. Defaults to None.
        positional_encoding (:obj:`ConfigDict` or dict): Config for
            transformer decoder position encoding. Defaults to
            dict(num_feats=128, normalize=True).
        loss_cls (:obj:`ConfigDict` or dict): Config of the classification
            loss. Defaults to None.
        loss_mask (:obj:`ConfigDict` or dict): Config of the mask loss.
            Defaults to None.
        loss_dice (:obj:`ConfigDict` or dict): Config of the dice loss.
            Defaults to None.
        train_cfg (:obj:`ConfigDict` or dict, optional): Training config of
            Mask2Former head.
        test_cfg (:obj:`ConfigDict` or dict, optional): Testing config of
            Mask2Former head.
        init_cfg (:obj:`ConfigDict` or dict or list[:obj:`ConfigDict` or \
            dict], optional): Initialization config dict. Defaults to None.
    """

    def __init__(self,
                 in_channels: List[int],
                 feat_channels: int,
                 out_channels: int,
                 num_things_classes: int = 80,
                 num_stuff_classes: int = 53,
                 num_queries: int = 100,
                 num_transformer_feat_level: int = 3,
                 pixel_decoder: ConfigType = ...,
                 enforce_decoder_input_project: bool = False,
                 transformer_decoder: ConfigType = ...,
                 positional_encoding: ConfigType = None,
                 loss_cls: ConfigType = None,
                 loss_mask: ConfigType = None,
                 loss_dice: ConfigType = None,
                 loss_iou: ConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 init_cfg: OptMultiConfig = None,
                 # ov configs
                 sphere_cls: bool = False,
                 ov_classifier_name: Optional[str] = None,
                 logit: Optional[int] = None,
                 # box sup
                 matching_whole_map: bool = False,
                 ov_path=None,
                 **kwargs) -> None:
        super(AnchorFreeHead, self).__init__(init_cfg=init_cfg)
        enable_box_query = True
        self.feat_channels = feat_channels
        self.out_mask_channel = out_channels
        self.num_things_classes = num_things_classes
        self.num_stuff_classes = num_stuff_classes
        self.num_classes = self.num_things_classes + self.num_stuff_classes
        self.num_queries = num_queries
        self.num_transformer_feat_level = num_transformer_feat_level
        self.num_heads = transformer_decoder.layer_cfg.cross_attn_cfg.num_heads
        self.num_transformer_decoder_layers = transformer_decoder.num_layers
        assert pixel_decoder.encoder.layer_cfg.\
                   self_attn_cfg.num_levels == num_transformer_feat_level
        pixel_decoder_ = copy.deepcopy(pixel_decoder)
        pixel_decoder_.update(
            in_channels=in_channels,
            feat_channels=feat_channels,
            out_channels=out_channels)

        self.pixel_decoder_cfg = pixel_decoder_

        self.pixel_decoder = MODELS.build(pixel_decoder_)
        self.transformer_decoder_cfg = transformer_decoder
        self.transformer_decoder = Mask2FormerTransformerDecoder(
            **transformer_decoder)
        self.decoder_embed_dims = self.transformer_decoder.embed_dims

        self.decoder_input_projs = ModuleList()
        # from low resolution to high resolution
        for _ in range(num_transformer_feat_level):
            if (self.decoder_embed_dims != feat_channels
                    or enforce_decoder_input_project):
                self.decoder_input_projs.append(
                    Conv2d(
                        feat_channels, self.decoder_embed_dims, kernel_size=1))
            else:
                self.decoder_input_projs.append(nn.Identity())
        self.decoder_positional_encoding = SinePositionalEncoding3D(
            **positional_encoding)
        self.query_embed = nn.Embedding(self.num_queries, feat_channels)
        self.query_feat = nn.Embedding(self.num_queries, feat_channels)
        # from low resolution to high resolution
        self.level_embed = nn.Embedding(self.num_transformer_feat_level,
                                        feat_channels)

        if not sphere_cls:
            self.cls_embed = nn.Linear(feat_channels, self.num_classes + 1)
        self.mask_embed = nn.Sequential(
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
            nn.Linear(feat_channels, out_channels))

        if loss_iou is not None:
            self.iou_embed = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, 1))
        else:
            self.iou_embed = None

        self.test_cfg = test_cfg
        self.train_cfg = train_cfg
        if train_cfg:
            self.assigner = TASK_UTILS.build(self.train_cfg['assigner'])
            self.sampler = TASK_UTILS.build(
                self.train_cfg['sampler'], default_args=dict(context=self))
            self.num_points = self.train_cfg.get('num_points', 12544)
            self.oversample_ratio = self.train_cfg.get('oversample_ratio', 3.0)
            self.importance_sample_ratio = self.train_cfg.get(
                'importance_sample_ratio', 0.75)

        self.class_weight = loss_cls.class_weight
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_mask = MODELS.build(loss_mask)
        self.loss_dice = MODELS.build(loss_dice)
        if loss_iou is not None:
            self.loss_iou = MODELS.build(loss_iou)
        else:
            self.loss_iou = None

        # prepare OV things
        # OV cls embed
        if sphere_cls:
            rank, world_size = get_dist_info()
            if ov_classifier_name is None:
                _dim = 1024  # temporally hard code
                cls_embed = torch.empty(self.num_classes, _dim)
                torch.nn.init.orthogonal_(cls_embed)
                cls_embed = cls_embed[:, None]
            else:
                if ov_path is None:
                    ov_path = os.path.join(os.path.expanduser('~/.cache/embd'), f"{ov_classifier_name}.pth")
                else:
                    ov_path = ov_path
                cls_embed = torch.load(ov_path)
                cls_embed_norm = cls_embed.norm(p=2, dim=-1)
                assert torch.allclose(cls_embed_norm, torch.ones_like(cls_embed_norm))
            if self.loss_cls and self.loss_cls.use_sigmoid:
                pass
            else:
                _dim = cls_embed.size(2)
                _prototypes = cls_embed.size(1)

                if rank == 0:
                    back_token = torch.zeros(1, _dim, dtype=torch.float32, device='cuda')
                else:
                    back_token = torch.empty(1, _dim, dtype=torch.float32, device='cuda')
                if world_size > 1:
                    dist.broadcast(back_token, src=0)
                back_token = back_token.to(device='cpu')
                cls_embed = torch.cat([
                    cls_embed, back_token.repeat(_prototypes, 1)[None]
                ], dim=0)
            self.register_buffer('cls_embed', cls_embed.permute(2, 0, 1).contiguous(), persistent=False)

            # cls embd proj
            cls_embed_dim = self.cls_embed.size(0)
            self.cls_proj = nn.Sequential(
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, feat_channels), nn.ReLU(inplace=True),
                nn.Linear(feat_channels, cls_embed_dim)
            )

            # Haobo Yuan:
            # For the logit_scale, I refer to this issue.
            # https://github.com/openai/CLIP/issues/46#issuecomment-945062212
            # https://github.com/openai/CLIP/issues/46#issuecomment-782558799
            # Based on my understanding, it is a mistake of CLIP.
            # Because they mention that they refer to InstDisc (Wu, 2018) paper.
            # InstDisc set a non-learnable temperature to np.log(1 / 0.07).
            # 4.6052 is np.log(1 / 0.01)
            # np.log(1 / 0.07) will be fast converged to np.log(1 / 0.01)
            if logit is None:
                logit_scale = torch.tensor(4.6052, dtype=torch.float32)
            else:
                logit_scale = torch.tensor(logit, dtype=torch.float32)
            self.register_buffer('logit_scale', logit_scale, persistent=False)

            # Mask Pooling
            self.mask_pooling = mask_pool
            self.mask_pooling_proj = nn.Sequential(
                nn.LayerNorm(feat_channels),
                nn.Linear(feat_channels, feat_channels)
            )

        # box inst
        self.matching_whole_map = matching_whole_map

        # enable box query
        self.enable_box_query = enable_box_query
        if self.enable_box_query:
            self.num_mask_tokens = 1
            self.mask_tokens = nn.Embedding(self.num_mask_tokens, feat_channels)
            self.pb_embedding = nn.Embedding(2, feat_channels)
            self.pos_linear = nn.Linear(2 * feat_channels, feat_channels)

        self.transformer_decoder_llm = None
        self.mask_embed_llm = None
        self.pixel_decoder_llm = None

        self.additional_cross_attn_layers = None

    def init_new_decoder(self):
        if self.transformer_decoder_llm is not None:
            return
        dtype = self.query_embed.weight.dtype
        device = self.query_embed.weight.device
        self.transformer_decoder_llm_dtype = dtype
        self.transformer_decoder_llm = Mask2FormerTransformerDecoder(
            **self.transformer_decoder_cfg).to(dtype).to(device)
        self.transformer_decoder_llm.load_state_dict(self.transformer_decoder.state_dict(), strict=True)
        for name, param in self.transformer_decoder_llm.named_parameters():
            param.requires_grad_(True)
        print("Init transformer_decoder_llm and resume omg seg decoder weight and not frozen !!!")

        self.mask_embed_llm =\
            nn.Sequential(
                nn.Linear(self.feat_channels, self.feat_channels), nn.ReLU(inplace=True),
                nn.Linear(self.feat_channels, self.feat_channels), nn.ReLU(inplace=True),
                nn.Linear(self.feat_channels, self.out_mask_channel)).to(dtype).to(device)
        self.mask_embed_llm.load_state_dict(self.mask_embed.state_dict(), strict=True)
        for name, param in self.mask_embed_llm.named_parameters():
            param.requires_grad_(True)
        print("Init mask_embed_llm and resume omg seg weight and not frozen !!!")
        return

    def init_cross_attn_layer(self):
        if self.additional_cross_attn_layers is not None:
            return
        dtype = self.query_embed.weight.dtype
        device = self.query_embed.weight.device
        self.additional_cross_attn_layers = CrossAttentionLayer(
            self.decoder_embed_dims, self.num_heads, dropout=0.0,
            activation="relu", normalize_before=False
        ).to(dtype).to(device)
        self.additional_ffn = FFNLayer(self.decoder_embed_dims)

        for name, param in self.additional_cross_attn_layers.named_parameters():
            param.requires_grad_(True)
        for name, param in self.additional_ffn.named_parameters():
            param.requires_grad_(True)
        print("Init additional cross attn layer and ffn layer !!!")
        return

    def init_weights(self) -> None:
        for m in self.decoder_input_projs:
            if isinstance(m, Conv2d):
                caffe2_xavier_init(m, bias=0)

        self.pixel_decoder.init_weights()

        for p in self.transformer_decoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_normal_(p)

    def forward_logit(self, cls_embd):
        cls_pred = torch.einsum('bnc,ckp->bnkp', F.normalize(cls_embd, dim=-1), self.cls_embed)
        cls_pred = cls_pred.max(-1).values
        cls_pred = self.logit_scale.exp() * cls_pred
        return cls_pred

    def _forward_head_llm(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],
                      num_frames: int = 0) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
                - num_frames: How many frames are there in video.
        """
        if self.transformer_decoder_llm is None:
            decoder_out = self.transformer_decoder.post_norm(decoder_out)
        else:
            self.transformer_decoder_llm.post_norm = self.transformer_decoder_llm.post_norm.to(decoder_out.dtype)
            decoder_out = self.transformer_decoder_llm.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)

        if self.mask_embed_llm is None:
            mask_embed = self.mask_embed(decoder_out)
        else:
            self.mask_embed_llm = self.mask_embed_llm.to(decoder_out.dtype)
            mask_embed = self.mask_embed_llm(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        if num_frames > 0:
            assert len(mask_pred.shape) == 4
            assert mask_pred.shape[2] % num_frames == 0
            frame_h = mask_pred.shape[2] // num_frames
            num_q = mask_pred.shape[1]
            _mask_pred = mask_pred.unflatten(-2, (num_frames, frame_h)).flatten(1, 2)
            attn_mask = F.interpolate(
                _mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
            attn_mask = attn_mask.unflatten(1, (num_q, num_frames)).flatten(2, 3)
        else:
            attn_mask = F.interpolate(
                mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return mask_pred, attn_mask

    def _forward_head(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],
                      num_frames: int = 0) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
                - num_frames: How many frames are there in video.
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        # shape (num_queries, batch_size, c)
        if isinstance(self.cls_embed, nn.Module):
            cls_pred = self.cls_embed(decoder_out)
        # shape (num_queries, batch_size, c)
        mask_embed = self.mask_embed(decoder_out)
        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        if not isinstance(self.cls_embed, nn.Module):
            maskpool_embd = self.mask_pooling(x=mask_feature, mask=mask_pred.detach())
            maskpool_embd = self.mask_pooling_proj(maskpool_embd)
            cls_embd = self.cls_proj(maskpool_embd + decoder_out)
            cls_pred = self.forward_logit(cls_embd)

        if self.iou_embed is not None:
            iou_pred = self.iou_embed(decoder_out)
            cls_pred = torch.cat([cls_pred, iou_pred], dim=-1)

        if num_frames > 0:
            assert len(mask_pred.shape) == 4
            assert mask_pred.shape[2] % num_frames == 0
            frame_h = mask_pred.shape[2] // num_frames
            num_q = mask_pred.shape[1]
            _mask_pred = mask_pred.unflatten(-2, (num_frames, frame_h)).flatten(1, 2)
            attn_mask = F.interpolate(
                _mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
            attn_mask = attn_mask.unflatten(1, (num_q, num_frames)).flatten(2, 3)
        else:
            attn_mask = F.interpolate(
                mask_pred,
                attn_mask_target_size,
                mode='bilinear',
                align_corners=False)
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.sigmoid() < 0.5
        attn_mask = attn_mask.detach()

        return cls_pred, mask_pred, attn_mask

    def forward_llm_seg(self, hidden_states, batch_idxs):
        # hidden_states (N, C) -> (N, 1, C)
        hidden_states = hidden_states.to(self.query_feat.weight.dtype)
        hidden_states = hidden_states.unsqueeze(1)
        C = hidden_states.shape[-1]
        num_frames = 0

        if self.pixel_decoder_llm is not None:
            self.pixel_decoder_llm = self.pixel_decoder_llm.to(hidden_states.dtype)
            mask_features, multi_scale_memorys = self.pixel_decoder(self.image_feat)
            mask_features = mask_features[batch_idxs]
        else:
            mask_features = self.cur_batch_mask_features[batch_idxs]
            multi_scale_memorys = self.cur_batch_multi_scale_memorys

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            decoder_input = decoder_input[batch_idxs]  # (N, hw, c)
            batch_size = len(decoder_input)

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            num_frames_real = 1
            mask = decoder_input.new_zeros(
                (batch_size, num_frames_real) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.transpose(
                1, 2).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        query_feat = hidden_states[:, :, :C//2]
        query_embed = hidden_states[:, :, C//2:]
        self_attn_mask = None

        mask_pred_list = []
        mask_pred, attn_mask = self._forward_head_llm(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            num_frames=num_frames
        )
        if num_frames > 0:
            mask_pred = mask_pred.unflatten(2, (num_frames, -1))
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            if self.transformer_decoder_llm is not None:
                layer = self.transformer_decoder_llm.layers[i]
            else:
                layer = self.transformer_decoder.layers[i]
            layer = layer.to(query_feat.dtype)
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
                query_cache=None,
                query_cache_pos=None,
            )
            mask_pred, attn_mask = self._forward_head_llm(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                num_frames=num_frames
            )

            if num_frames > 0:
                mask_pred = mask_pred.unflatten(2, (num_frames, -1))
            mask_pred_list.append(mask_pred)

        # mask_pred_list [(b, 1, h, w), ...]
        return mask_pred_list

    def sample_points(self, mask_pred, gt_masks):
        gt_masks = gt_masks.unsqueeze(1)
        gt_masks = gt_masks.to(mask_pred)
        # (N, 1, h, w)

        with torch.no_grad():
            points_coords = get_uncertain_point_coords_with_randomness(
                mask_pred, None, self.num_points,
                self.oversample_ratio, self.importance_sample_ratio)
            # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
            mask_point_targets = point_sample(
                gt_masks.float(), points_coords).squeeze(1)
        # shape (num_queries, h, w) -> (num_queries, num_points)
        mask_point_preds = point_sample(
            mask_pred, points_coords).squeeze(1)
        return mask_point_preds, mask_point_targets

    def llm_seg_loss(self, mask_pred_list, gt_masks):

        if gt_masks is None or gt_masks[0] is None:
            ret_loss = 0
            for mask_pred in mask_pred_list:
                ret_loss = ret_loss + mask_pred.sum() * 0.0
            return [ret_loss], [ret_loss]

        # dice loss and ce loss
        all_loss_dice = []
        all_loss_mask = []

        for mask_pred in mask_pred_list:

            sampled_mask_pred, sampled_mask_gt = self.sample_points(mask_pred, gt_masks)
            loss_dice = self.loss_dice(
                sampled_mask_pred,
                sampled_mask_gt, avg_factor=(len(gt_masks) + 1e-4))
            loss_mask = self.loss_mask(
                sampled_mask_pred.reshape(-1),
                sampled_mask_gt.reshape(-1),
                avg_factor=(sampled_mask_pred.shape[0] * sampled_mask_pred.shape[1] + 1e-4))
            all_loss_dice.append(loss_dice)
            all_loss_mask.append(loss_mask)
        return all_loss_dice, all_loss_mask

    def forward(self, x: List[Tensor], batch_data_samples: SampleList,
                return_mask_features=False, save_feat=False,
                return_query_pos=False) -> Tuple[List[Tensor]]:
        """Forward function.

        Args:
            x (list[Tensor]): Multi scale Features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[list[Tensor]]: A tuple contains two elements.

                - cls_pred_list (list[Tensor)]: Classification logits \
                    for each decoder layer. Each is a 3D-tensor with shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred_list (list[Tensor]): Mask logits for each \
                    decoder layer. Each with shape (batch_size, num_queries, \
                    h, w).
        """
        batch_img_metas = []
        if isinstance(batch_data_samples[0], TrackDataSample):
            for track_sample in batch_data_samples:
                cur_list = []
                for det_sample in track_sample:
                    cur_list.append(det_sample.metainfo)
                batch_img_metas.append(cur_list)
            num_frames = len(batch_img_metas[0])
        else:
            for data_sample in batch_data_samples:
                batch_img_metas.append(data_sample.metainfo)
            num_frames = 0
        batch_size = len(batch_img_metas)

        mask_features, multi_scale_memorys = self.pixel_decoder(x)

        if num_frames > 0:
            mask_features = mask_features.unflatten(0, (batch_size, num_frames))
            mask_features = mask_features.transpose(1, 2).flatten(2, 3)

        # save for decode the llm's [SEG] tokens
        if save_feat:
            self.cur_batch_mask_features = mask_features
            self.cur_batch_multi_scale_memorys = multi_scale_memorys
            self.image_feat = x

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            if num_frames > 0:
                decoder_input = decoder_input.unflatten(0, (batch_size, num_frames))
                decoder_input = decoder_input.flatten(1, 2)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            num_frames_real = 1 if num_frames == 0 else num_frames
            mask = decoder_input.new_zeros(
                (batch_size, num_frames_real) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.transpose(
                1, 2).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        # only for encode the image for llm, not support sam mode in this process
        if False and batch_data_samples[0].data_tag in ['sam_mul', 'sam']:
            query_feat, input_query_bbox, self_attn_mask, _ = self.prepare_for_dn_mo(batch_data_samples)
            query_embed = coordinate_to_encoding(input_query_bbox.sigmoid())
            query_embed = self.pos_linear(query_embed)
        else:
            # coco style query generation
            # shape (num_queries, c) -> (batch_size, num_queries, c)
            query_feat = self.query_feat.weight.unsqueeze(0).repeat((batch_size, 1, 1))
            query_embed = self.query_embed.weight.unsqueeze(0).repeat((batch_size, 1, 1))
            self_attn_mask = None

        cls_pred_list = []
        mask_pred_list = []
        cls_pred, mask_pred, attn_mask = self._forward_head(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            num_frames=num_frames
        )
        cls_pred_list.append(cls_pred)
        if num_frames > 0:
            mask_pred = mask_pred.unflatten(2, (num_frames, -1))
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn
            layer = self.transformer_decoder.layers[i]
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            cls_pred, mask_pred, attn_mask = self._forward_head(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                num_frames=num_frames
            )

            cls_pred_list.append(cls_pred)
            if num_frames > 0:
                mask_pred = mask_pred.unflatten(2, (num_frames, -1))
            mask_pred_list.append(mask_pred)
        if return_mask_features:
            if return_query_pos:
                return cls_pred_list, mask_pred_list, query_feat, query_embed, mask_features
            return cls_pred_list, mask_pred_list, query_feat, mask_features
        if return_query_pos:
            return cls_pred_list, mask_pred_list, query_feat, query_embed
        return cls_pred_list, mask_pred_list, query_feat

    def predict(self, x: Tuple[Tensor],
                batch_data_samples: SampleList,
                return_query=False,
                return_mask_features=False,
                save_feat=False,
                return_query_pos=False,
                ) -> Tuple[Tensor, ...]:
        """Test without augmentaton.

        Args:
            return_query:
            x (tuple[Tensor]): Multi-level features from the
                upstream network, each is a 4D-tensor.
            batch_data_samples (List[:obj:`DetDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance`, `gt_panoptic_seg` and `gt_sem_seg`.

        Returns:
            tuple[Tensor]: A tuple contains two tensors.

                - mask_cls_results (Tensor): Mask classification logits,\
                    shape (batch_size, num_queries, cls_out_channels).
                    Note `cls_out_channels` should includes background.
                - mask_pred_results (Tensor): Mask logits, shape \
                    (batch_size, num_queries, h, w).
        """
        data_sample = batch_data_samples[0]
        if isinstance(data_sample, TrackDataSample):
            img_shape = data_sample[0].metainfo['batch_input_shape']
            num_frames = len(data_sample)
        else:
            img_shape = data_sample.metainfo['batch_input_shape']
            num_frames = 0
        if return_mask_features:
            all_cls_scores, all_mask_preds, query_feat, query_pos, mask_features =\
                self(x, batch_data_samples, return_mask_features,
                     save_feat=save_feat, return_query_pos=True)
        else:
            all_cls_scores, all_mask_preds, query_feat, query_pos =\
                self(x, batch_data_samples, save_feat=save_feat, return_query_pos=True)
        if self.iou_embed is not None:
            _all_cls_scores = [cls_score[..., :-1] for cls_score in all_cls_scores]
            iou_results = [cls_score[..., -1:] for cls_score in all_cls_scores]
            all_cls_scores = _all_cls_scores
        else:
            iou_results = None
        mask_cls_results = all_cls_scores[-1]
        mask_pred_results = all_mask_preds[-1]
        if iou_results is not None:
            iou_results = iou_results[-1]

        if num_frames > 0:
            mask_pred_results = mask_pred_results.flatten(1, 2)
        mask_pred_results = F.interpolate(
            mask_pred_results,
            size=(img_shape[0], img_shape[1]),
            mode='bilinear',
            align_corners=False)
        if num_frames > 0:
            num_queries = mask_cls_results.shape[1]
            mask_pred_results = mask_pred_results.unflatten(1, (num_queries, num_frames))

        if iou_results is None:
            return mask_cls_results, mask_pred_results

        if return_query:
            if return_mask_features:
                if return_query_pos:
                    return mask_cls_results, mask_pred_results, query_feat, query_pos, iou_results, mask_features
                return mask_cls_results, mask_pred_results, query_feat, iou_results, mask_features
            else:
                if return_query_pos:
                    return mask_cls_results, mask_pred_results, query_feat, query_pos, iou_results
                return mask_cls_results, mask_pred_results, query_feat, iou_results
        else:
            if return_mask_features:
                return mask_cls_results, mask_pred_results, iou_results, mask_features
            else:
                return mask_cls_results, mask_pred_results, iou_results

    def prepare_for_dn_mo(self, batch_data_samples):
        scalar, noise_scale = 100, 0.4
        gt_instances = [t.gt_instances for t in batch_data_samples]

        point_coords = torch.stack([inst.point_coords for inst in gt_instances])
        pb_labels = torch.stack([inst['bp'] for inst in gt_instances])
        labels = torch.zeros_like(pb_labels).long()

        boxes = point_coords  # + boxes

        factors = []
        for i, data_sample in enumerate(batch_data_samples):
            h, w, = data_sample.metainfo['img_shape']
            factor = boxes[i].new_tensor([w, h, w, h]).unsqueeze(0).repeat(boxes[i].size(0), 1)
            factors.append(factor)
        factors = torch.stack(factors, 0)

        boxes = bbox_xyxy_to_cxcywh(boxes / factors)
        box_start = [len(point) for point in point_coords]

        known_labels = labels
        known_pb_labels = pb_labels
        known_bboxs = boxes

        known_labels_expaned = known_labels.clone()
        known_pb_labels_expaned = known_pb_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale > 0 and self.training:
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :, :2] = known_bbox_expand[:, :, 2:] / 2
            diff[:, :, 2:] = known_bbox_expand[:, :, 2:]
            # add very small noise to input points; no box
            sc = 0.01
            for i, st in enumerate(box_start):
                diff[i, :st] = diff[i, :st] * sc
            known_bbox_expand += torch.mul(
                (torch.rand_like(known_bbox_expand) * 2 - 1.0),
                diff) * noise_scale

            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        input_label_embed = self.pb_embedding(known_pb_labels_expaned)

        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(
            self.num_mask_tokens,
            1) + self.mask_tokens.weight.unsqueeze(0).repeat(
            input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(
            self.num_mask_tokens, 1)

        single_pad = self.num_mask_tokens

        # NOTE scalar is modified to 100, each click cannot see each other
        scalar = int(input_label_embed.shape[1] / self.num_mask_tokens)

        pad_size = input_label_embed.shape[1]

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        tgt_size = pad_size
        attn_mask = torch.ones(tgt_size, tgt_size).to('cuda') < 0
        # match query cannot see the reconstruct
        attn_mask[pad_size:, :pad_size] = True
        # reconstruct cannot see each other
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
        mask_dict = {
            'known_lbs_bboxes': (known_labels, known_bboxs),
            'pad_size': pad_size,
            'scalar': scalar,
        }
        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def prepare_sam_query(self, points, w, h):
        # points N, 1, 2
        tl_points = points - 3
        br_points = points + 3
        boxes = torch.cat([tl_points, br_points], dim=-1)

        labels = torch.zeros((points.shape[0], 1, ), dtype=torch.int64).to(points.device)

        factors = torch.Tensor([[[w, h, w, h]]]).to(boxes)

        boxes = bbox_xyxy_to_cxcywh(boxes / factors)  # xyxy / factor or xywh / factor ????
        print('rela_coords:', boxes)
        known_bboxs = boxes

        known_bbox_expand = known_bboxs.clone()

        input_label_embed = self.pb_embedding(labels)
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)

        input_label_embed = input_label_embed.repeat_interleave(
            self.num_mask_tokens,
            1) + self.mask_tokens.weight.unsqueeze(0).repeat(
            input_label_embed.shape[0], input_label_embed.shape[1], 1)
        input_bbox_embed = input_bbox_embed.repeat_interleave(
            self.num_mask_tokens, 1)

        if input_label_embed.shape[1] > 0:
            input_query_label = input_label_embed
            input_query_bbox = input_bbox_embed

        query_embed = coordinate_to_encoding(input_query_bbox.sigmoid())
        query_embed = self.pos_linear(query_embed)
        return input_query_label, query_embed  # (N, 1, C)

    def forward_visual_prompt(self, regions, batch_idxs):
        # points (N, 2)
        points = get_center_coords(regions)
        points = points.to(torch.float32).to(regions.device)

        query_feat, query_embed = self.prepare_sam_query(
            points.unsqueeze(1), w=regions.shape[-1], h=regions.shape[-2],
        ) # (N, 1, c)

        # hidden_states (N, C) -> (N, 1, C)
        num_frames = 0

        mask_features = self.cur_batch_mask_features[batch_idxs]
        multi_scale_memorys = self.cur_batch_multi_scale_memorys

        query_feat = query_feat.to(mask_features)
        query_embed = query_embed.to(mask_features)

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            decoder_input = decoder_input[batch_idxs]  # (N, hw, c)
            batch_size = len(decoder_input)

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            num_frames_real = 1
            mask = decoder_input.new_zeros(
                (batch_size, num_frames_real) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.transpose(
                1, 2).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        self_attn_mask = None

        mask_pred_list = []
        mask_pred, attn_mask = self._forward_head_sam(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            num_frames=num_frames, regions=regions,
        )
        if num_frames > 0:
            mask_pred = mask_pred.unflatten(2, (num_frames, -1))
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn

            layer = self.transformer_decoder.layers[i]
            layer = layer.to(query_feat.dtype)
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            mask_pred, attn_mask = self._forward_head_sam(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                num_frames=num_frames, regions=regions,
            )

            if num_frames > 0:
                mask_pred = mask_pred.unflatten(2, (num_frames, -1))
            mask_pred_list.append(mask_pred)
        return query_feat, query_embed


    def _forward_head_sam(self, decoder_out: Tensor, mask_feature: Tensor,
                      attn_mask_target_size: Tuple[int, int],
                      num_frames: int = 0, regions=None, bboxes=None
        ) -> Tuple[Tensor]:
        """Forward for head part which is called after every decoder layer.

        Args:
            decoder_out (Tensor): in shape (batch_size, num_queries, c).
            mask_feature (Tensor): in shape (batch_size, c, h, w).
            attn_mask_target_size (tuple[int, int]): target attention
                mask size.

        Returns:
            tuple: A tuple contain three elements.

                - cls_pred (Tensor): Classification scores in shape \
                    (batch_size, num_queries, cls_out_channels). \
                    Note `cls_out_channels` should includes background.
                - mask_pred (Tensor): Mask scores in shape \
                    (batch_size, num_queries,h, w).
                - attn_mask (Tensor): Attention mask in shape \
                    (batch_size * num_heads, num_queries, h, w).
                - num_frames: How many frames are there in video.
        """
        decoder_out = self.transformer_decoder.post_norm(decoder_out)
        mask_embed = self.mask_embed(decoder_out)

        # shape (num_queries, batch_size, h, w)
        mask_pred = torch.einsum('bqc,bchw->bqhw', mask_embed, mask_feature)

        if regions is not None:
            attn_mask = self._get_attn_mask_from_gt_mask(regions,
                                                         attn_mask_target_size)
        else:
            if num_frames > 0:
                assert len(mask_pred.shape) == 4
                assert mask_pred.shape[2] % num_frames == 0
                frame_h = mask_pred.shape[2] // num_frames
                num_q = mask_pred.shape[1]
                _mask_pred = mask_pred.unflatten(-2, (num_frames, frame_h)).flatten(1, 2)
                attn_mask = F.interpolate(
                    _mask_pred,
                    attn_mask_target_size,
                    mode='bilinear',
                    align_corners=False)
                attn_mask = attn_mask.unflatten(1, (num_q, num_frames)).flatten(2, 3)
            else:
                attn_mask = F.interpolate(
                    mask_pred,
                    attn_mask_target_size,
                    mode='bilinear',
                    align_corners=False)

            # set attn maps
            if bboxes is not None:
                cur_scale_bboxes = copy.deepcopy(bboxes.cpu().numpy())
                bs, _, h, w = attn_mask.shape
                assert len(bboxes) == bs
                cur_scale_bboxes = np.clip(cur_scale_bboxes, a_min=0, a_max=1)
                cur_scale_bboxes[:, [0, 2]] *= w
                cur_scale_bboxes[:, [1, 3]] *= h
                cur_scale_bboxes[:, 2:] += 1
                cur_scale_bboxes = torch.Tensor(np.floor(cur_scale_bboxes))
                cur_scale_bboxes = cur_scale_bboxes.to(torch.int64)
                for i in range(bs):
                    sx, sy = cur_scale_bboxes[i][0], cur_scale_bboxes[i][1]
                    ex, ey = cur_scale_bboxes[i][2], cur_scale_bboxes[i][3]
                    attn_mask[i, :, :sy, :] = -100
                    attn_mask[i, :, ey:, :] = -100
                    attn_mask[i, :, :, :sx] = -100
                    attn_mask[i, :, :, ex:] = -100

            # shape (num_queries, batch_size, h, w) ->
            #   (batch_size * num_head, num_queries, h, w)
            attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
                (1, self.num_heads, 1, 1)).flatten(0, 1)
            attn_mask = attn_mask.sigmoid() < 0.5
            attn_mask = attn_mask.detach()

        return mask_pred, attn_mask

    def forward_point_prompt(self, points, batch_idxs, width, height):
        # regions (N, H, W)

        query_feat, query_embed = self.prepare_sam_query(
            points.unsqueeze(1), w=width, h=height,
        ) # (N, 1, c)

        # hidden_states (N, C) -> (N, 1, C)
        num_frames = 0

        mask_features = self.cur_batch_mask_features[batch_idxs]
        multi_scale_memorys = self.cur_batch_multi_scale_memorys

        query_feat = query_feat.to(mask_features)
        query_embed = query_embed.to(mask_features)

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            decoder_input = decoder_input[batch_idxs]  # (N, hw, c)
            batch_size = len(decoder_input)

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            num_frames_real = 1
            mask = decoder_input.new_zeros(
                (batch_size, num_frames_real) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.transpose(
                1, 2).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        self_attn_mask = None

        mask_pred_list = []
        mask_pred, attn_mask = self._forward_head_sam(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            num_frames=num_frames
        )
        # attn_mask = self._get_attn_mask_from_gt_mask(regions, multi_scale_memorys[0].shape[-2:])
        if num_frames > 0:
            mask_pred = mask_pred.unflatten(2, (num_frames, -1))
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn

            layer = self.transformer_decoder.layers[i]
            layer = layer.to(query_feat.dtype)
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            mask_pred, attn_mask = self._forward_head_sam(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                num_frames=num_frames
            )

            if num_frames > 0:
                mask_pred = mask_pred.unflatten(2, (num_frames, -1))
            mask_pred_list.append(mask_pred)

        return query_feat, query_embed

    def forward_box_prompt(self, boxes, batch_idxs, width, height):

        # regions (N, H, W)
        points = (boxes[:, :2] + boxes[:, 2:]) / 2.0
        boxes_rela_coords = copy.deepcopy(boxes) * 1.0
        boxes_rela_coords[:, [0, 2]] /= width
        boxes_rela_coords[:, [1, 3]] /= height

        query_feat, query_embed = self.prepare_sam_query(
            points.unsqueeze(1), w=width, h=height,
        ) # (N, 1, c)

        # hidden_states (N, C) -> (N, 1, C)
        num_frames = 0

        mask_features = self.cur_batch_mask_features[batch_idxs]
        multi_scale_memorys = self.cur_batch_multi_scale_memorys

        query_feat = query_feat.to(mask_features)
        query_embed = query_embed.to(mask_features)

        decoder_inputs = []
        decoder_positional_encodings = []
        for i in range(self.num_transformer_feat_level):
            decoder_input = self.decoder_input_projs[i](multi_scale_memorys[i])
            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            decoder_input = decoder_input.flatten(2).permute(0, 2, 1)
            level_embed = self.level_embed.weight[i].view(1, 1, -1)
            decoder_input = decoder_input + level_embed

            decoder_input = decoder_input[batch_idxs]  # (N, hw, c)
            batch_size = len(decoder_input)

            # shape (batch_size, c, h, w) -> (batch_size, h*w, c)
            num_frames_real = 1
            mask = decoder_input.new_zeros(
                (batch_size, num_frames_real) + multi_scale_memorys[i].shape[-2:],
                dtype=torch.bool)
            decoder_positional_encoding = self.decoder_positional_encoding(
                mask)
            decoder_positional_encoding = decoder_positional_encoding.transpose(
                1, 2).flatten(2).permute(0, 2, 1)
            decoder_inputs.append(decoder_input)
            decoder_positional_encodings.append(decoder_positional_encoding)

        self_attn_mask = None

        mask_pred_list = []
        mask_pred, attn_mask = self._forward_head_sam(
            query_feat, mask_features, multi_scale_memorys[0].shape[-2:],
            num_frames=num_frames, bboxes=boxes_rela_coords,
        )
        # attn_mask = self._get_attn_mask_from_gt_mask(regions, multi_scale_memorys[0].shape[-2:])
        if num_frames > 0:
            mask_pred = mask_pred.unflatten(2, (num_frames, -1))
        mask_pred_list.append(mask_pred)

        for i in range(self.num_transformer_decoder_layers):
            level_idx = i % self.num_transformer_feat_level
            # if a mask is all True(all background), then set it all False.
            attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False

            # cross_attn + self_attn

            layer = self.transformer_decoder.layers[i]
            layer = layer.to(query_feat.dtype)
            query_feat = layer(
                query=query_feat,
                key=decoder_inputs[level_idx],
                value=decoder_inputs[level_idx],
                query_pos=query_embed,
                key_pos=decoder_positional_encodings[level_idx],
                cross_attn_mask=attn_mask,
                self_attn_mask=self_attn_mask,
                query_key_padding_mask=None,
                # here we do not apply masking on padded region
                key_padding_mask=None,
            )
            mask_pred, attn_mask = self._forward_head_sam(
                query_feat, mask_features, multi_scale_memorys[(i + 1) % self.num_transformer_feat_level].shape[-2:],
                num_frames=num_frames, bboxes=boxes_rela_coords,
            )

            if num_frames > 0:
                mask_pred = mask_pred.unflatten(2, (num_frames, -1))
            mask_pred_list.append(mask_pred)

        return query_feat, query_embed

    def _get_attn_mask_from_gt_mask(self, regions, attn_mask_target_size):
        regions = regions.unsqueeze(1) # (N, 1, H, W)
        attn_mask = F.interpolate(
            regions,
            attn_mask_target_size,
            mode='nearest')
        # shape (num_queries, batch_size, h, w) ->
        #   (batch_size * num_head, num_queries, h, w)
        attn_mask = attn_mask.flatten(2).unsqueeze(1).repeat(
            (1, self.num_heads, 1, 1)).flatten(0, 1)
        attn_mask = attn_mask.to(torch.bool)
        attn_mask = attn_mask.detach()
        return ~attn_mask

    def get_targets(self, **kwargs):
        return

    def loss_by_feat(self, **kwargs):
        return


def get_center_coords(masks):
    point_coords = []
    for mask in masks:
        mask = mask[None, None]
        mask = mask.to(torch.bool)
        n, _, h, w = mask.shape
        mask_dt = (
            distance_transform(
                (~F.pad(mask, pad=(1, 1, 1, 1), mode='constant', value=0)).float()
            )[:, :, 1:-1, 1:-1]
        )
        selected_point = torch.tensor([mask_dt.argmax() / w, mask_dt.argmax() % w]).long().flip(0).to(
            mask.device)
        point_coords.append(selected_point)
    if len(point_coords) > 0:
        point_coords = torch.stack(point_coords)[:, None]
    else:
        point_coords = torch.empty((0, 1, 2), dtype=torch.int32).to(device=mask.device)
    return point_coords[:, 0]
