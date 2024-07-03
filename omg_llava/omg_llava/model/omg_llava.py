from collections import OrderedDict
import torch
import torch.nn as nn
from mmengine.config import Config, ConfigDict
from mmengine.model import BaseModel
from peft import get_peft_model, prepare_model_for_kbit_training

from xtuner.registry import BUILDER
from .modules import ProjectorConfig_OMG_LLaVA, ProjectorModel_OMG_LLaVA
from xtuner.model.modules import ProjectorModel, ProjectorConfig
from xtuner.model.modules import dispatch_modules
from .utils import (LoadWoInit, find_all_linear_names,
                    get_peft_model_state_dict, guess_load_checkpoint,
                    make_inputs_require_grad,
                    traverse_dict,
                    prepare_inputs_labels_for_multimodal_with_visual_prompts)
from .convnext_clip import OpenCLIPBackbone
from .omg_seg import OMGSegVisualEncoder

class OMG_LLaVA(BaseModel):
    def __init__(self,
                 llm,
                 visual_encoder,
                 visual_select_layer=-2,
                 freeze_llm=False,
                 freeze_visual_encoder=False,
                 require_omg_decoder=False,
                 pretrained_pth=None,
                 llm_lora=None,
                 visual_encoder_lora=None,
                 use_activation_checkpointing=True,
                 projector_depth=2,
                 text2vision_projector=False,
                 tokenizer=None,
                 keep_omg_decoder_frozen=False,
                 add_seg_pretrain=False,
                 additional_cross_attn_layers=False,
                 pixel_shuffle_ratio=None,
                 train_vocabulary=False,
                 freeze_llm_with_lora=False,
                 freeze_visual_projector=False,
                 rm_prior_embedding=False,
                 rm_query=False,
                 clip_feat_channel=1536,
                 ):
        super().__init__()

        self.freeze_llm_with_lora = freeze_llm_with_lora
        self.freeze_visual_projector = freeze_visual_projector

        self.freeze_llm = freeze_llm
        self.freeze_visual_encoder = freeze_visual_encoder
        with LoadWoInit():
            self.llm = self._build_from_cfg_or_module(llm)
            if visual_encoder.type == OpenCLIPBackbone or visual_encoder.type == OMGSegVisualEncoder:
                self.visual_encoder = visual_encoder.type(**visual_encoder)
            else:
                self.visual_encoder = self._build_from_cfg_or_module(
                    visual_encoder)
        self.llm.config.use_cache = False
        dispatch_modules(self.llm)

        projector_config = ProjectorConfig_OMG_LLaVA(
            query_channels=256,
            feat_channels=clip_feat_channel,
            llm_hidden_size=self.llm.config.hidden_size,
            depth=projector_depth,
            pixel_shuffle_ratio=pixel_shuffle_ratio,
        )
        self.projector = ProjectorModel_OMG_LLaVA(projector_config).to(
            self.visual_encoder.dtype)

        self.text2vision_projector = text2vision_projector
        if text2vision_projector:
            projector_config = ProjectorConfig(
                visual_hidden_size=self.llm.config.hidden_size,
                llm_hidden_size=256 * 2,
                depth=projector_depth)
            self.projector_text2vision = ProjectorModel(projector_config).to(
                self.visual_encoder.dtype)


        if rm_query:
            self.projector.model.rm_query = rm_query
        if rm_prior_embedding:
            self.projector.model.rm_prior_embedding = rm_prior_embedding

        if self.freeze_llm:
            self.llm.requires_grad_(False)
        if self.freeze_visual_encoder:
            self.visual_encoder.requires_grad_(False)

        self.use_activation_checkpointing = use_activation_checkpointing
        if use_activation_checkpointing:
            # For backward compatibility
            if hasattr(self.llm, 'enable_input_require_grads'):
                self.llm.enable_input_require_grads()
            else:
                self.llm.get_input_embeddings().register_forward_hook(
                    make_inputs_require_grad)
            if hasattr(self.visual_encoder, 'enable_input_require_grads'):
                self.visual_encoder.enable_input_require_grads()
            else:
                self.visual_encoder.get_input_embeddings(
                ).register_forward_hook(make_inputs_require_grad)
            self.projector.enable_input_require_grads()
            if text2vision_projector:
                self.projector_text2vision.enable_input_require_grads()

            # enable gradient (activation) checkpointing for memory efficiency
            self.gradient_checkpointing_enable()

        # resize input embed before add llm lora
        self.added_special_token = False
        if tokenizer is not None:
            self.tokenizer = tokenizer
            tokenizer_type = self.tokenizer['type']
            del self.tokenizer['type']
            self.tokenizer = tokenizer_type(**self.tokenizer)
            self._add_special_tokens()

        self.use_llm_lora = llm_lora is not None
        self.use_visual_encoder_lora = visual_encoder_lora is not None

        if self.use_llm_lora:
            self._prepare_llm_for_lora(llm_lora, use_activation_checkpointing)
            if self.freeze_llm_with_lora:
                for name, param in self.llm.named_parameters():
                    param.requires_grad_(False)
        else:
            if train_vocabulary:
                # train vocabulary embedding and logit head when pretrain
                for name, param in self.named_parameters():
                    if 'tok_' in name or 'lm_head' in name:
                        print("Unfrozen {} !!!".format(name))
                        param.requires_grad_(True)
                    if 'output.' in name and 'llm' in name and 'lora' not in name:
                        print("Unfrozen {} !!!".format(name))
                        param.requires_grad_(True)

        if self.use_visual_encoder_lora:
            self._prepare_visual_encoder_for_lora(
                visual_encoder_lora, use_activation_checkpointing)

        if pretrained_pth is not None:
            pretrained_state_dict = guess_load_checkpoint(pretrained_pth)
            self.load_state_dict(pretrained_state_dict, strict=False)
            print(f'Load pretrained weight from {pretrained_pth}')

        self.visual_select_layer = visual_select_layer

        self._is_init = True

        self.require_omg_decoder = require_omg_decoder
        if require_omg_decoder:
            self.visual_encoder.init_new_decoder()
            if keep_omg_decoder_frozen:
                for name, param in self.visual_encoder.panoptic_head.transformer_decoder_llm.named_parameters():
                    param.requires_grad_(False)
                print("Frozen all the omg seg decoder !!!")

        self.additional_cross_attn_layers = additional_cross_attn_layers
        if self.additional_cross_attn_layers:
            self.visual_encoder.init_cross_attn_layer()

        if self.freeze_visual_projector:
            for name, param in self.projector.named_parameters():
                param.requires_grad_(False)

        self.add_seg_pretrain = add_seg_pretrain


    def _add_special_tokens(self):
        assert hasattr(self, "tokenizer")

        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        # add for visual prompt
        region_tokens = ['<region>']
        point_tokens = ['<mark>']
        special_tokens = segmentation_tokens + phrase_tokens + region_tokens
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)

        self.seg_token_idx = self.tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
        self.bop_token_idx = self.tokenizer("<p>", add_special_tokens=False).input_ids[0]
        self.eop_token_idx = self.tokenizer("</p>", add_special_tokens=False).input_ids[0]
        self.region_token_idx = self.tokenizer("<region>", add_special_tokens=False).input_ids[0]

        self.llm.resize_token_embeddings(len(self.tokenizer))

        self.tokenizer.add_tokens(point_tokens, special_tokens=True)
        self.mark_token_idx = self.tokenizer("<mark>", add_special_tokens=False).input_ids[0]
        if self.use_activation_checkpointing or self.use_llm_lora or not self.freeze_llm:
            self.llm.enable_input_require_grads()
        self.added_special_token = True
        print("[SEG]: {}, <p>: {}, </p>: {}, <region>: {}, <mark>: {}" \
              .format(self.seg_token_idx, self.bop_token_idx,
                      self.eop_token_idx, self.region_token_idx, self.mark_token_idx))
        print('****************************Add special tokens ********************************************')
        return

    def _parse_lora_config(self, lora_config):
        if isinstance(lora_config, dict) or isinstance(
                lora_config, Config) or isinstance(lora_config, ConfigDict):
            lora_config = BUILDER.build(lora_config)
        return lora_config

    def _prepare_llm_for_lora(self,
                              lora_config,
                              use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        self.llm = prepare_model_for_kbit_training(
            self.llm, use_activation_checkpointing)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.llm)
            lora_config.target_modules = modules
        self.llm = get_peft_model(self.llm, lora_config)
        for name, param in self.named_parameters():
            if 'tok_' in name or 'lm_head' in name:
                print("Unfrozen {} !!!".format(name))
                param.requires_grad_(True)
            if 'output.' in name and 'llm' in name and 'lora' not in name:
                print("Unfrozen {} !!!".format(name))
                param.requires_grad_(True)

    def _prepare_visual_encoder_for_lora(self,
                                         lora_config,
                                         use_activation_checkpointing=True):
        lora_config = self._parse_lora_config(lora_config)
        if lora_config.target_modules is None:
            modules = find_all_linear_names(self.visual_encoder)
            lora_config.target_modules = modules
        self.visual_encoder = get_peft_model(self.visual_encoder, lora_config)

    def gradient_checkpointing_enable(self):
        self.activation_checkpointing_enable()

    def activation_checkpointing_enable(self):
        self.llm.gradient_checkpointing_enable()
        if hasattr(self.visual_encoder, 'gradient_checkpointing_enable'):
            self.visual_encoder.gradient_checkpointing_enable()
        elif hasattr(self.visual_encoder, 'clip_model'):
            if self.visual_encoder.clip_model is not None:
                self.visual_encoder.clip_model.gradient_checkpointing_enable()
        if hasattr(self.projector, 'gradient_checkpointing_enable'):
            self.projector.gradient_checkpointing_enable()
        if self.text2vision_projector and hasattr(self.projector_text2vision, 'gradient_checkpointing_enable'):
            self.projector_text2vision.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self):
        self.activation_checkpointing_disable()

    def activation_checkpointing_disable(self):
        self.llm.gradient_checkpointing_disable()
        if hasattr(self.visual_encoder, 'gradient_checkpointing_disable'):
            self.visual_encoder.gradient_checkpointing_disable()
        if hasattr(self.projector, 'gradient_checkpointing_disable'):
            self.projector.gradient_checkpointing_disable()
        if self.text2vision_projector and hasattr(self.projector_text2vision, 'gradient_checkpointing_disable'):
            self.projector_text2vision.gradient_checkpointing_disable()

    def init_weights(self):
        pass

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)

        to_return = OrderedDict()

        # vocabulary embedding
        to_return.update(
            {k: v for k, v in state_dict.items() if 'tok_' in k}
        )
        # logit head
        to_return.update(
            {k: v for k, v in state_dict.items() if 'output.' in k and 'llm' in k and 'lora' not in k}
        )

        # Step 1. visual_encoder
        if self.use_visual_encoder_lora:
            to_return.update(
                get_peft_model_state_dict(
                    self.visual_encoder, state_dict=state_dict))
        elif not self.freeze_visual_encoder:
            to_return.update({
                k: v
                for k, v in state_dict.items() if 'visual_encoder.' in k
            })
        # Step 2. LLM
        if self.use_llm_lora:
            to_return.update(
                get_peft_model_state_dict(self.llm, state_dict=state_dict))
        elif not self.freeze_llm:
            to_return.update(
                {k: v
                 for k, v in state_dict.items() if 'llm.' in k})
        # Step 3. Projector
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector.' in k})
        # projector text2vision
        to_return.update(
            {k: v
             for k, v in state_dict.items() if 'projector_text2vision' in k})

        # visual_encoder.adapter_proj
        if self.freeze_visual_encoder:
            to_return.update(
                {k: v
                for k, v in state_dict.items() if 'visual_encoder.adapter_proj' in k})

        # git_clip lora
        if hasattr(self.visual_encoder, 'clip_model'):
            if self.visual_encoder.clip_lora is not None:
                to_return.update(
                    get_peft_model_state_dict(self.visual_encoder.clip_model,
                                              state_dict=state_dict))
        # omg decoder for llm
        if self.require_omg_decoder:
            to_return.update(
                {k: v
                for k, v in state_dict.items()
                if 'visual_encoder.panoptic_head.transformer_decoder_llm' in k or
                   'visual_encoder.panoptic_head.mask_embed_llm' in k or
                   'visual_encoder.panoptic_head.pixel_decoder_llm' in k or
                   'visual_encoder.panoptic_head.additional_cross_attn_layers' in k or
                   'visual_encoder.panoptic_head.additional_ffn' in k or
                   'visual_encoder.downsample_layer' in k
                 })

        return to_return

    def _build_from_cfg_or_module(self, cfg_or_mod):
        if isinstance(cfg_or_mod, nn.Module):
            return cfg_or_mod
        elif isinstance(cfg_or_mod, dict):
            traverse_dict(cfg_or_mod)
            return BUILDER.build(cfg_or_mod)
        else:
            raise NotImplementedError

    def forward(self, data, data_samples=None, mode='loss'):

        if 'pixel_values' in data:
            if 'masks' in data:
                masks = data['masks']
                del data['masks']
            else:
                masks = None
            if 'regions' in data:
                regions = data['regions']
                del data['regions']
            else:
                regions = None
            if 'points' in data:
                points = data['points']
                del data['points']
            else:
                points = None

            visual_outputs = self.visual_encoder(
                data['pixel_values'].to(self.visual_encoder.dtype),
                output_hidden_states=True)

            if self.add_seg_pretrain:
                pred_obj_query, gt_obj_query = prepare_seg_pretrain_data(
                    visual_outputs,
                    [self.projector.model.query_proj, self.projector.model.model],
                    self.projector_text2vision.model
                )

            if isinstance(visual_outputs, list) or isinstance(visual_outputs, tuple)\
                    or isinstance(visual_outputs, torch.Tensor):
                pixel_values = self.projector(visual_outputs)
            else:
                pixel_values = self.projector(
                    visual_outputs.hidden_states[self.visual_select_layer][:, 1:])

            if regions is not None:
                region_embeddings, region_success = self.get_region_embeddings(
                    regions, data['input_ids'],
                )
                del regions
            else:
                region_success = True
                region_embeddings = []

            if points is not None:
                points_mark_embedding, mark_success = self.get_points_embeddings(
                    points, data['input_ids'],
                    width=data['pixel_values'].shape[-1],
                    height=data['pixel_values'].shape[-2],
                )
            else:
                points_mark_embedding = []
                mark_success = True

            data['pixel_values'] = pixel_values
            data = prepare_inputs_labels_for_multimodal_with_visual_prompts(
                llm=self.llm, region_id=self.region_token_idx,
                regions_feats=region_embeddings,
                mark_id=self.mark_token_idx,
                mark_feats=points_mark_embedding,
                **data)
        else:
            masks = None

        if mode == 'loss':
            if self.add_seg_pretrain:
                return self.compute_loss(data, data_samples, masks=masks, region_success=region_success,
                                         pred_gt_obj_query=(pred_obj_query, gt_obj_query),
                                         mark_success=mark_success)
            else:
                return self.compute_loss(data, data_samples, masks=masks,
                                         pred_gt_obj_query=None,
                                         region_success=region_success,
                                         mark_success=mark_success)
        elif mode == 'predict':
            return self.predict(data, data_samples)
        elif mode == 'tensor':
            return self._forward(data, data_samples)
        else:
            raise NotImplementedError

    def _forward(self, data, data_samples=None):

        outputs = self.llm(**data)

        return outputs

    def predict(self, data, data_samples=None):
        outputs = self.llm(**data)
        logits_dict = [{'logits': logits} for logits in outputs.logits]
        return logits_dict

    def compute_loss(self, data, data_samples=None, masks=None, pred_gt_obj_query=None,
                     region_success=True, mark_success=True):
        if 'original_labels' in data.keys():
            input_ids = data['original_labels']
            del data['original_labels']
        else:
            input_ids = data['labels']
        outputs = self.llm(**data, output_hidden_states=True)

        loss_dice, loss_mask = self.compute_seg_loss(
            input_ids, outputs.hidden_states[-1], masks)

        if pred_gt_obj_query is not None:
            pred_obj_query, gt_obj_query = pred_gt_obj_query
            proj_loss = torch.mean((pred_obj_query - gt_obj_query) ** 2) * 10
        else:
            proj_loss = 0

        if not region_success:
            loss = outputs.loss * 0
        else:
            loss = outputs.loss

        if not mark_success:
            loss = outputs.loss * 0

        loss = loss + self.get_visual_prompts_projector_zero()

        loss_dict = {'loss': loss, 'loss_dice': outputs.loss* 0 + loss_dice * 0.1,
                     'loss_mask': outputs.loss * 0 + loss_mask * 0.4,
                     'loss_proj': outputs.loss * 0 + proj_loss}
        return loss_dict

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.llm, name)

    def get_region_embeddings(self, regions, input_ids):
        success = True
        if regions is None or len(regions) == 0:
            return [], success
        else:
            region_token_mask = input_ids == self.region_token_idx
            batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
                input_ids.device)
            batch_idxs = batch_idxs[region_token_mask]  # (N, ) batch_size number
            if len(regions) != len(batch_idxs):
                # There is a bug !!! skip it.
                success = False
                if len(regions) > len(batch_idxs):
                    regions = regions[:len(batch_idxs)]
                else:
                    n_pad = len(batch_idxs) - len(regions)
                    pad_region = regions[:1].repeat(n_pad, 1, 1)
                    regions = torch.cat([pad_region, regions])

            regions_embeddings = self.visual_encoder.forward_region_sam(
                regions, batch_idxs
            )[:, 0]  # (N, C)

            regions_embeddings = self.projector.model.forward_visual_prompts_embeddings(
                regions_embeddings, batch_idxs)
            return regions_embeddings, success  # (N, C)

    def get_points_embeddings(self, points, input_ids, width, height):
        success = True
        if points is None or len(points) == 0:
            return []

        mark_token_mask = input_ids == self.mark_token_idx
        batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
            input_ids.device)
        batch_idxs = batch_idxs[mark_token_mask]  # (N, ) batch_size number

        if len(points) != len(batch_idxs):
            # There is a bug !!! skip it.
            success = False
            if len(points) > len(batch_idxs):
                points = points[:len(batch_idxs)]
            else:
                n_pad = len(batch_idxs) - len(points)
                pad_region = points[:1].repeat(n_pad, 1, 1)
                points = torch.cat([pad_region, points])

        marks_embeddings = self.visual_encoder.forward_point_sam(
            points, batch_idxs, width=width, height=height
        )[:, 0]  # (N, C)

        marks_embeddings = self.projector.model.forward_visual_prompts_embeddings(
            marks_embeddings, batch_idxs)
        return marks_embeddings, success  # (N, C)

    def get_visual_prompts_projector_zero(self):
        return self.projector.model.visual_prompt_zero

    def compute_seg_loss(self, input_ids, hidden_states, gt_masks):
        if not self.text2vision_projector or self.add_seg_pretrain:
            return 0.0, 0.0
        success = True
        if gt_masks is None or len(gt_masks) == 0:
            batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
                input_ids.device)
            batch_idxs = batch_idxs[0, :1]  # (N, ) batch_size number
            gt_masks = [None]
            hidden_states = hidden_states[0, :1]
            hidden_states = self.projector_text2vision(hidden_states)  # (N, C)

            pred_masks_list = self.visual_encoder.forward_llm_seg(hidden_states, batch_idxs)
            dice_loss, mask_loss = self.visual_encoder.loss_llm_seg(pred_masks_list, gt_masks)

            return dice_loss * 0.0, mask_loss * 0.0


        seg_tokens_mask = input_ids == self.seg_token_idx
        batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(seg_tokens_mask.device)

        ori_hidden_states = hidden_states
        hidden_states = hidden_states[seg_tokens_mask]
        batch_idxs = batch_idxs[seg_tokens_mask]  # (N, ) batch_size number

        if len(hidden_states) != len(gt_masks) or len(hidden_states) == 0:
            # drop this batch
            print("Drop the batch because the number of [SEG] and masks not equal !!!")
            hidden_states = ori_hidden_states
            batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
                input_ids.device)
            batch_idxs = batch_idxs[0, :1]  # (N, ) batch_size number
            gt_masks = [None]
            hidden_states = hidden_states[0, :1]
            hidden_states = self.projector_text2vision(hidden_states)  # (N, C)

            pred_masks_list = self.visual_encoder.forward_llm_seg(hidden_states, batch_idxs)
            dice_loss, mask_loss = self.visual_encoder.loss_llm_seg(pred_masks_list, gt_masks)

            return dice_loss * 0.0, mask_loss * 0.0

        assert len(hidden_states) == len(gt_masks), "expect [seg] number equal to mask number, but get {} [seg] and {} masks".format(len(hidden_states), len(gt_masks))
        hidden_states = self.projector_text2vision(hidden_states)  # (N, C)

        pred_masks_list = self.visual_encoder.forward_llm_seg(hidden_states, batch_idxs)
        dice_loss, mask_loss = self.visual_encoder.loss_llm_seg(pred_masks_list, gt_masks)

        if not success:
            return dice_loss * 0.0, mask_loss * 0.0

        return dice_loss, mask_loss

def prepare_seg_pretrain_data(visual_outputs,
                              query_in_proj, query_out_proj):
    clip_feature, query_feat, attention_mask = visual_outputs
    # clip feature (bs, hw, c + 2 * q_c)
    # query_feat (bs, q, 2c)
    # attention_mask (bs, q, hw)
    bs, q, _ = query_feat.shape
    pred_query_embed = []
    gt_query_embed = []
    for i in range(bs):
        valid = attention_mask[i].sum(-1) > 0
        valid_query_feat = query_feat[i][valid]  # (n, 2c)
        gt_query_embed.append(valid_query_feat)

        if isinstance(query_in_proj, list):
            llm_query = valid_query_feat
            for proj in query_in_proj:
                llm_query = proj(llm_query)
        else:
            llm_query = query_in_proj(valid_query_feat)

        pred_query_embed.append(query_out_proj(llm_query))

    pred_query_embed = torch.cat(pred_query_embed, dim=0)
    gt_query_embed = torch.cat(gt_query_embed, dim=0)
    return pred_query_embed, gt_query_embed

