from .mask2former_vid import Mask2formerVideo
from mmdet.structures import DetDataSample
import torch
import torch.nn.functional as F
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig

class OMGSegVisualEncoder(Mask2formerVideo):
    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 panoptic_head: OptConfigType = None,
                 panoptic_fusion_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 inference_sam: bool = False,
                 init_cfg: OptMultiConfig = None,
                 dtype=torch.float32,
                 pixel_shuffle_down_ratio=2,
                 **kwargs,
                 ):
        super().__init__(backbone=backbone, neck=neck, panoptic_head=panoptic_head,
                         panoptic_fusion_head=panoptic_fusion_head, train_cfg=train_cfg,
                         test_cfg=test_cfg, data_preprocessor=data_preprocessor,
                         inference_sam=inference_sam, init_cfg=init_cfg, )
        self.dtype = dtype
        self.enable_output_gradient = False
        self.backbone_type = None
        self.pixel_shuffle_down_ratio = pixel_shuffle_down_ratio

        weight_path = init_cfg['checkpoint']
        state_dict = torch.load(weight_path)["state_dict"]
        self.load_state_dict(state_dict, strict=False)
        print("Loaded omg weight from {} !!!".format(weight_path))

    def init_new_decoder(self):
        self.panoptic_head.init_new_decoder()
        return

    def init_cross_attn_layer(self):
        self.panoptic_head.init_cross_attn_layer()
        return

    def prepare_input(self, image):
        # image (b, 3, h, w)
        h, w = image.shape[-2:]
        datasamples = DetDataSample()
        metainfo = {'batch_input_shape': (h, w),
                    'ori_shape': (h, w),
                    'img_shape': (h, w)}
        datasamples.set_metainfo(metainfo)
        return image, [datasamples for i in range(image.shape[0])]

    def pixel_shuffle_feat(self, feat):
        # feat (b, c, h, w)
        if self.pixel_shuffle_down_ratio is None:
            return feat
        # pixel shuffle
        b, c, h, w = feat.shape
        feat = feat.reshape(
            b, c, h // self.pixel_shuffle_down_ratio, self.pixel_shuffle_down_ratio,
            w // self.pixel_shuffle_down_ratio, self.pixel_shuffle_down_ratio,
        )
        feat = feat.permute(0, 3, 5, 1, 2, 4)  # (bs, rh, rw, c, h_down, w_down)
        feat = feat.flatten(1, 3)  # (bs, rh * rw * c, h_down, w_down)
        return feat

    def llava_visual_feat(self, backbone_feat):
        # get clip feature
        ret = []
        last_outs = backbone_feat[-1]
        # more downsample ratio by pixel shuffle
        last_outs = self.pixel_shuffle_feat(last_outs)
        ret.append(last_outs.flatten(2).permute(0, 2, 1))
        return ret

    def forward_llm_seg(self, hidden_states, batch_idxs):
        # hidden_states (N, 256) batch_idxs (N, )
        hidden_states = hidden_states.to(self.dtype)

        mask_pred_results = self.panoptic_head.forward_llm_seg(
            hidden_states, batch_idxs,
        )
        return mask_pred_results

    def forward_region_sam(self, regions, batch_idxs):
        query_feat, query_embed = self.panoptic_head.forward_visual_prompt(
            regions, batch_idxs
        )
        return torch.cat([query_feat, query_embed], dim=-1)

    def forward_point_sam(self, points, batch_idxs, width, height):
        query_feat, query_embed = self.panoptic_head.forward_point_prompt(
            points, batch_idxs, width=width, height=height
        )
        return torch.cat([query_feat, query_embed], dim=-1)

    def forward_box_sam(self, bboxes, batch_idxs, width, height):
        query_feat, query_embed = self.panoptic_head.forward_box_prompt(
            bboxes, batch_idxs, width=width, height=height
        )
        return torch.cat([query_feat, query_embed], dim=-1)

    def loss_llm_seg(self, mask_pred_results, gt_masks):
        all_loss_dice, all_loss_mask = self.panoptic_head.llm_seg_loss(
            mask_pred_results, gt_masks,
        )
        return sum(all_loss_dice), sum(all_loss_mask)

    def forward(self, images, output_hidden_states=True):
        if self.backbone_type is None:
            self.backbone_type = [p.dtype for p in self.parameters()][0]
            self.to(self.dtype)
        images = images.to(self.dtype)

        img_shape = images.shape[-2:]
        # last scale for ConvNext-L
        feat_shape = [item // 32 for item in img_shape]
        if self.pixel_shuffle_down_ratio is not None:
            feat_shape = [item // self.pixel_shuffle_down_ratio for item in feat_shape]
        batch_inputs, batch_data_samples = self.prepare_input(images)

        # directly for image perception
        num_frames = 0  # only consider image
        bs = batch_inputs.shape[0]
        feats = self.extract_feat(batch_inputs)
        llava_clip_feat = self.llava_visual_feat(feats)

        # directly do panoptic segmentation
        mask_cls_results, mask_pred_results, query_feat, query_pos, iou_results, mask_features =\
            self.panoptic_head.predict(
            feats, batch_data_samples, return_query=True, return_mask_features=True, save_feat=True,
                return_query_pos=True,)

        if self.OVERLAPPING is not None:
            assert len(self.OVERLAPPING) == self.num_classes
            mask_cls_results = self.open_voc_inference(feats, mask_cls_results, mask_pred_results)

        # llava_clip_feat [(bs, hw, c), ]
        # query_feat (bs, q, c), query_pos (bs, q, c)
        # mask_pred (b, q, h, w)
        query_pos_feat = torch.cat([query_feat, query_pos], dim=-1)  # (bs, q, 2c)

        ret_pixel_query = []
        ret_attn_mask = []
        for i in range(bs):
            pixel_query, attn_mask = self.panoptic_postprocess(
                mask_cls_results[i], mask_pred_results[i], query_pos_feat[i],
                feat_size=feat_shape,
            )
            ret_pixel_query.append(pixel_query)  # (h, w, c)
            ret_attn_mask.append(attn_mask)  # (q, hw)
        ret_pixel_query = torch.stack(ret_pixel_query, dim=0)  # (bs, h, w, c)
        ret_attn_mask = torch.stack(ret_attn_mask, dim=0)  # (bs, q, hw)

        ret_pixel_query = ret_pixel_query.flatten(1, 2)  # (bs, hw, c)
        llava_clip_feat[0] = torch.cat([llava_clip_feat[0], ret_pixel_query], dim=-1)
        ret = llava_clip_feat + [query_pos_feat, ret_attn_mask]

        for i in range(len(ret) - 1):
            ret[i] = ret[i].to(self.backbone_type)
        for i in range(len(ret) - 1):
            ret[i] = self.set_output_gradient(ret[i])
        return ret

    def panoptic_postprocess(self, mask_cls, mask_pred, query_feat, feat_size=[320, 320]):
        """assign queries for per pixel.
           mask_cls (q, c)
           mask_pred (q, h, w)
           query_feat (q, c)
        """

        scores_foreground, _ = F.softmax(mask_cls, dim=-1)[..., :-1].max(-1)

        mask_pred = mask_pred.sigmoid()
        cur_scores = scores_foreground
        cur_masks = mask_pred

        cur_query_feat = query_feat
        # smooth_ratio = 0.05
        smooth_ratio = 0.5
        # use 0.5 as the smooth score
        cur_prob_masks = (cur_scores.view(-1, 1, 1) * (1 - smooth_ratio) + smooth_ratio) * cur_masks

        # for visualization
        ori_prob_masks = cur_prob_masks

        cur_prob_masks = F.interpolate(cur_prob_masks.unsqueeze(0),
            size=feat_size,
            mode='bilinear',
            align_corners=False
        )[0]

        cur_mask_ids = cur_prob_masks.argmax(0).unsqueeze(0)  # (1, h, w)

        pixel_query = cur_prob_masks.softmax(dim=0).permute(1, 2, 0) @ \
                      cur_query_feat  # (h, w, c)

        # need attn mask to filter none mask
        attn_mask = cur_mask_ids != torch.arange(0, cur_query_feat.shape[0]).unsqueeze(1).unsqueeze(2).to(cur_mask_ids.device)
        attn_mask = attn_mask.flatten(1)

        # for visualization
        if not self.training:
            valid_mask = attn_mask.sum(dim=-1) < attn_mask.shape[-1]  # (bs, q)
            keep = valid_mask
            ori_prob_masks = ori_prob_masks[keep]

            vis_mask_ids = ori_prob_masks.argmax(0).unsqueeze(0)
            self.vis_binary_masks = (vis_mask_ids == torch.arange(0, ori_prob_masks.shape[0]).unsqueeze(1).unsqueeze(2).to(
                cur_mask_ids.device)).unsqueeze(1).to(torch.float32)
        else:
            self.vis_binary_masks = None
        return pixel_query, attn_mask

    def enable_input_require_grads(self):
        self.enable_output_gradient = True
        return

    def set_output_gradient(self, output):
        if not self.training:
            return output
        output.requires_grad_(self.enable_output_gradient)
        return output

    def requires_grad_(self, state):
        if not self.training:
            return 
        if state:
            print("Not Frozen the Visual Encoder !")
        else:
            print("Frozen the Visual Encoder !")
        for p in self.parameters():
            p.requires_grad_(state)
        return

