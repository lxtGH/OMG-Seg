from typing import Optional, List

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from mmdet.registry import MODELS

from mmengine.model import BaseModule
from mmengine.dist import get_dist_info
from mmengine.logging import MMLogger
from timm.layers import resample_abs_pos_embed

import ext.open_clip as open_clip
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix


def flatten_permute(x):
    x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
    x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]
    return x


@MODELS.register_module()
class OpenCLIPBackbone(BaseModule):
    """OpenCLIPBackbone,
    Please refer to:
    https://github.com/mlfoundations/open_clip/tree/5f7892b672b21e6853d0f6c11b18dda9bcf36c8d#pretrained-model-interface
    for the supported models and checkpoints.
    """
    STAGES = 4

    def __init__(
            self,
            img_size: int = 1024,
            model_name: str = '',
            fix: bool = True,
            fix_layers: Optional[List] = None,
            init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] in ['clip_pretrain', 'image_pretrain', 'Pretrained'], \
            f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()
        rank, world_size = get_dist_info()

        if world_size > 1:
            if rank == 0:
                if init_cfg['type'] == 'clip_pretrain':
                    _ = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained,
                                                               return_transform=False, logger=self.logger)
                elif init_cfg['type'] == 'image_pretrain':
                    _ = open_clip.create_model(model_name, pretrained_image=True, logger=self.logger)

            else:
                pass
            dist.barrier()

        # Get the clip model
        if init_cfg['type'] == 'clip_pretrain':
            clip_model = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained,
                                                                return_transform=False, logger=self.logger)
        elif init_cfg['type'] == 'image_pretrain':
            clip_model = open_clip.create_model(model_name, pretrained_image=True, logger=self.logger)
        elif init_cfg['type'] == 'Pretrained':
            clip_model = open_clip.create_model(model_name, pretrained_image=False, logger=self.logger)
        else:
            raise NotImplementedError

        self.out_indices = (0, 1, 2, 3)
        model_name_lower = model_name.lower()
        if 'convnext_' in model_name_lower:
            model_type = 'convnext'
            if '_base' in model_name_lower:
                output_channels = [128, 256, 512, 1024]
                feat_size = 0
            elif '_large' in model_name_lower:
                output_channels = [192, 384, 768, 1536]
                feat_size = 0
            elif '_xxlarge' in model_name_lower:
                output_channels = [384, 768, 1536, 3072]
                feat_size = 0
            else:
                raise NotImplementedError(f"{model_name} not supported yet.")
        elif 'rn' in model_name_lower:
            model_type = 'resnet'
            if model_name_lower.replace('-quickgelu', '') in ['rn50', 'rn101']:
                output_channels = [256, 512, 1024, 2048]
                feat_size = 7
            elif model_name_lower == 'rn50x4':
                output_channels = [320, 640, 1280, 2560]
                feat_size = 9
            elif model_name_lower == 'rn50x16':
                output_channels = [384, 768, 1536, 3072]
                feat_size = 12
            elif model_name_lower == 'rn50x64':
                output_channels = [512, 1024, 2048, 4096]
                feat_size = 14
            else:
                raise NotImplementedError(f"{model_name} not supported yet.")
        elif "vit" in model_name_lower:
            model_type = 'vit'
            if model_name_lower == 'vit-l-14':
                output_channels = [1024, 1024, 1024, 1024]
                feat_size = 0
                assert not clip_model.visual.input_patchnorm
                assert clip_model.visual.attn_pool is None
            else:
                raise NotImplementedError(f"{model_name} not supported yet.")
        else:
            raise NotImplementedError(f"{model_name} not supported yet.")

        self.model_name = model_name
        self.fix = fix
        self.model_type = model_type
        self.output_channels = output_channels
        self.feat_size = feat_size

        # Get the visual model
        if self.model_type == 'resnet':
            self.stem = nn.Sequential(*[
                clip_model.visual.conv1, clip_model.visual.bn1, clip_model.visual.act1,
                clip_model.visual.conv2, clip_model.visual.bn2, clip_model.visual.act2,
                clip_model.visual.conv3, clip_model.visual.bn3, clip_model.visual.act3,
            ])
        elif self.model_type == 'convnext':
            self.stem = clip_model.visual.trunk.stem
        elif self.model_type == 'vit':
            self.stem = clip_model.visual.conv1
        else:
            raise ValueError

        if self.model_type == 'resnet':
            self.avgpool = clip_model.visual.avgpool
        elif self.model_type == 'convnext':
            self.avgpool = nn.Identity()
        elif self.model_type == 'vit':
            self.avgpool = flatten_permute
        else:
            raise ValueError

        self.res_layers = []
        if self.model_type in ['vit']:
            self.t_class_embedding = clip_model.visual.class_embedding
            self.t_positional_embedding = clip_model.visual.positional_embedding
            self.t_ln_pre_trans = clip_model.visual.ln_pre
            self.t_transformer = clip_model.visual.transformer
        else:
            for i in range(self.STAGES):
                if self.model_type == 'resnet':
                    layer_name = f'layer{i + 1}'
                    layer = getattr(clip_model.visual, layer_name)
                elif self.model_type == 'convnext':
                    layer_name = f'layer{i + 1}'
                    layer = clip_model.visual.trunk.stages[i]
                else:
                    raise ValueError
                self.add_module(layer_name, layer)
                self.res_layers.append(layer_name)

        if self.model_type == 'resnet':
            self.norm_pre = nn.Identity()
        elif self.model_type == 'convnext':
            self.norm_pre = clip_model.visual.trunk.norm_pre
        elif self.model_type == 'vit':
            self.norm_pre = nn.Identity()

        if self.model_type == 'resnet':
            self.head = clip_model.visual.attnpool
        elif self.model_type == 'convnext':
            self.head = nn.Sequential(*[
                clip_model.visual.trunk.head,
                clip_model.visual.head,
            ])
        elif self.model_type == 'vit':
            self.head = clip_model.visual.ln_post

        if self.init_cfg['type'] == 'Pretrained':
            checkpoint_path = pretrained
            state_dict = load_checkpoint_with_prefix(checkpoint_path, prefix=self.init_cfg['prefix'])
            self.load_state_dict(state_dict, strict=True)

        self.fix_layers = fix_layers

        if not self.fix:
            self.train()
            for name, param in self.norm_pre.named_parameters():
                param.requires_grad = False
            for name, param in self.head.named_parameters():
                param.requires_grad = False
            if self.fix_layers is not None:
                for i, layer_name in enumerate(self.res_layers):
                    if i in self.fix_layers:
                        res_layer = getattr(self, layer_name)
                        for name, param in res_layer.named_parameters():
                            param.requires_grad = False
                        if i == 0:
                            for name, param in self.stem.named_parameters():
                                param.requires_grad = False

        if self.fix:
            self.train(mode=False)
            for name, param in self.named_parameters():
                param.requires_grad = False

    def init_weights(self):
        self.logger.info(f"Init Config for {self.model_name}")
        self.logger.info(self.init_cfg)

    def train(self: torch.nn.Module, mode: bool = True) -> torch.nn.Module:
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        if self.fix:
            super().train(mode=False)
        else:
            super().train(mode=mode)
            if self.fix_layers is not None:
                for i, layer_name in enumerate(self.res_layers):
                    if i in self.fix_layers:
                        res_layer = getattr(self, layer_name)
                        res_layer.train(mode=False)
                        if i == 0:
                            self.stem.train(mode=False)
        return self

    def forward_func(self, x):
        x = self.stem(x)
        h, w = x.shape[-2:]
        x = self.avgpool(x)
        outs = []
        if self.model_type == 'vit':
            x = torch.cat(
                [self.t_class_embedding.to(x.dtype) +
                 torch.zeros(x.shape[0], 1, x.shape[-1], dtype=x.dtype, device=x.device),
                 x], dim=1
            )  # shape = [*, grid ** 2 + 1, width]
            new_pos_embed = resample_abs_pos_embed(
                self.t_positional_embedding[None],
                [h, w],
                num_prefix_tokens=1
            )
            x = x + new_pos_embed.to(x.dtype)
            x = self.t_ln_pre_trans(x)

            x = x.permute(1, 0, 2)  # NLD -> LND
            x = self.t_transformer(x)
            x = x.permute(1, 0, 2)  # LND -> NLD
            x = x[:, 1:]
            x = x.permute(0, 2, 1).unflatten(2, (h, w))  # BCHW
            for i in range(self.STAGES):
                outs.append(
                    F.interpolate(
                        x,
                        scale_factor=2 ** (2 - i),
                        mode='bilinear',
                        align_corners=False
                    )
                )
        else:
            for i, layer_name in enumerate(self.res_layers):
                res_layer = getattr(self, layer_name)
                x = res_layer(x).contiguous()
                if i in self.out_indices:
                    outs.append(x)
        return tuple(outs)

    def get_clip_feature(self, backbone_feat):
        if self.model_type == 'resnet':
            return backbone_feat
        elif self.model_type == 'convnext':
            return self.norm_pre(backbone_feat)
        raise NotImplementedError

    def forward_feat(self, features):
        if self.model_type == 'convnext':
            batch, num_query, channel = features.shape
            features = features.reshape(batch * num_query, channel, 1, 1)
            features = self.head(features)
            return features.view(batch, num_query, features.shape[-1])
        elif self.model_type == 'resnet':
            num_query, channel, seven, seven = features.shape
            features = self.head(features)
            return features

    def forward(self, x):
        if self.fix:
            with torch.no_grad():
                outs = self.forward_func(x)
        else:
            outs = self.forward_func(x)
        return outs

    def get_text_model(self):
        return OpenCLIPBackboneText(
            self.model_name,
            init_cfg=self.init_cfg
        )


@MODELS.register_module()
class OpenCLIPBackboneText(BaseModule):
    def __init__(
            self,
            model_name: str = '',
            init_cfg=None,
    ):
        assert init_cfg is not None and init_cfg['type'] == 'clip_pretrain', f"{init_cfg['type']} is not supported."
        pretrained = init_cfg['checkpoint']
        super().__init__(init_cfg=None)
        self.init_cfg = init_cfg
        self.logger = MMLogger.get_current_instance()
        rank, world_size = get_dist_info()

        if world_size > 1:
            if rank == 0:
                _ = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained, return_transform=False,
                                                           logger=self.logger)
            else:
                pass
            dist.barrier()

        # Get the clip model
        clip_model = open_clip.create_model_from_pretrained(model_name, pretrained=pretrained, return_transform=False,
                                                            logger=self.logger)

        # Get the textual model
        self.text_tokenizer = open_clip.get_tokenizer(model_name)
        self.text_transformer = clip_model.transformer
        self.text_token_embedding = clip_model.token_embedding
        self.text_pe = clip_model.positional_embedding
        self.text_ln_final = clip_model.ln_final
        self.text_proj = clip_model.text_projection

        self.register_buffer('text_attn_mask', clip_model.attn_mask)

        self.param_dtype = torch.float32
        self.model_name = model_name

    def init_weights(self):
        self.logger.info(f"Init Config for {self.model_name}")
        self.logger.info(self.init_cfg)

    # Copied from
    # https://github.com/openai/CLIP/blob/d50d76daa670286dd6cacf3bcd80b5e4823fc8e1/clip/model.py#L343
    @torch.no_grad()
    def forward(self, text):
        text_tokens = self.text_tokenizer(text).to(device=self.text_proj.device)
        x = self.text_token_embedding(text_tokens).to(self.param_dtype)
        x = x + self.text_pe.to(self.param_dtype)
        x = x.permute(1, 0, 2)
        x = self.text_transformer(x, attn_mask=self.text_attn_mask)
        x = x.permute(1, 0, 2)
        x = self.text_ln_final(x)  # [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), text_tokens.argmax(dim=-1)] @ self.text_proj
        return x
