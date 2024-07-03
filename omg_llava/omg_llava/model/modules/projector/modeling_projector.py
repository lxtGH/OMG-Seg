# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
from transformers import PreTrainedModel
from transformers.activations import ACT2FN
from typing import Optional
from torch import Tensor
from torch.nn import functional as F

from .configuration_projector import ProjectorConfig_OMG_LLaVA

class Naive_Proj(nn.Module):
    def __init__(self, config, rm_prior_embedding=False, rm_query=False):
        super().__init__()
        query_channels = config.query_channels
        self.query_channels = query_channels
        feat_channels = config.feat_channels
        if isinstance(query_channels, tuple):
            query_channels = query_channels[0]
        if isinstance(feat_channels, tuple):
            feat_channels = feat_channels[0]

        query_channels = query_channels * 2  # feat + embed

        self.query_proj = nn.Linear(query_channels, feat_channels)

        modules = [
            nn.Linear(
                feat_channels,
                config.llm_hidden_size,
                bias=config.bias)
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(
                    config.llm_hidden_size,
                    config.llm_hidden_size,
                    bias=config.bias))
        self.model = nn.Sequential(*modules)

        modules = [
            nn.Linear(
                feat_channels + query_channels,
                config.llm_hidden_size,
                bias=config.bias)
        ]
        for _ in range(1, config.depth):
            modules.append(ACT2FN[config.hidden_act])
            modules.append(
                nn.Linear(
                    config.llm_hidden_size,
                    config.llm_hidden_size,
                    bias=config.bias))
        self.model_feat = nn.Sequential(*modules)

        self.seperate_embed = nn.Embedding(1, config.llm_hidden_size)

        self.rm_prior_embedding = rm_prior_embedding
        self.rm_query = rm_query

    def forward(self, x):
        clip_feature, query_feat, attention_mask = x
        # clip feature (bs, hw, c + 2 * q_c)
        # query_feat (bs, q, c)
        # attention_mask (bs, q, hw)

        if self.rm_prior_embedding:
            clip_feature_feat = clip_feature[:, :, :-512]
            clip_feature_query = clip_feature[:, :, -512:] * 0.0
            clip_feature = torch.cat([clip_feature_feat, clip_feature_query], dim=-1)

        query_feat = self.query_proj(query_feat)

        valid_mask = attention_mask.sum(dim=-1) < attention_mask.shape[-1]  # (bs, q)
        # valid_mask # (bs, q)
        # query_feat (bs, q, c)
        # clip_feature (bs, hw, c)
        # attn_map (bs, q, hw)
        bs, n_q = query_feat.shape[:2]

        layer_outputs = self.model(query_feat)

        # filter
        clip_feature_out = clip_feature
        clip_feature_out = self.model_feat(clip_feature_out)
        ret = []

        valid_queries_embeddings = []
        for layer_output, keep in zip(layer_outputs, valid_mask):
            valid_queries_embeddings.append(layer_output[keep])
        self.valid_queries_embeddings = valid_queries_embeddings

        for clip_feat, layer_output, keep in zip(clip_feature_out, layer_outputs, valid_mask):
            if self.rm_query:
                ret.append(clip_feat + torch.mean(self.seperate_embed.weight) * 0.0 + torch.mean(layer_output[keep]) * 0.0)
            else:
                ret.append(torch.cat([clip_feat, self.seperate_embed.weight, layer_output[keep]], dim=0))
        return ret

class ProjectorModel_OMG_LLaVA(PreTrainedModel):
    _auto_class = 'AutoModel'
    config_class = ProjectorConfig_OMG_LLaVA
    base_model_prefix = 'model'
    supports_gradient_checkpointing = True

    def __init__(self, config: ProjectorConfig_OMG_LLaVA) -> None:
        super().__init__(config)
        self.gradient_checkpointing = False

        self.rm_prior_embedding = False
        self.rm_query = False
        self.model = Naive_Proj(config, )

    def enable_input_require_grads(self):

        def make_inputs_require_grad(module, input, output):
            if isinstance(output, torch.Tensor):
                output.requires_grad_(True)
            else:
                for item in output:
                    item.requires_grad_(True)

        self.model.register_forward_hook(make_inputs_require_grad)

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, ProjectorConfig_OMG_LLaVA):
            module.gradient_checkpointing = value

    def forward(self, x):
        if self.gradient_checkpointing and self.training:
            layer_outputs = torch.utils.checkpoint.checkpoint(self.model, x)
        else:
            layer_outputs = self.model(x)
        return layer_outputs


class SelfAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt,
                     tgt_mask: Optional[Tensor] = None,
                     tgt_key_padding_mask: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        q = k = self.with_pos_embed(tgt, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt,
                    tgt_mask: Optional[Tensor] = None,
                    tgt_key_padding_mask: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2, attn_mask=tgt_mask,
                              key_padding_mask=tgt_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt,
                tgt_mask: Optional[Tensor] = None,
                tgt_key_padding_mask: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, tgt_mask,
                                    tgt_key_padding_mask, query_pos)
        return self.forward_post(tgt, tgt_mask,
                                 tgt_key_padding_mask, query_pos)


class CrossAttentionLayer(nn.Module):

    def __init__(self, d_model, nhead, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)

        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt, memory,
                     memory_mask: Optional[Tensor] = None,
                     memory_key_padding_mask: Optional[Tensor] = None,
                     pos: Optional[Tensor] = None,
                     query_pos: Optional[Tensor] = None):
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)

        return tgt

    def forward_pre(self, tgt, memory,
                    memory_mask: Optional[Tensor] = None,
                    memory_key_padding_mask: Optional[Tensor] = None,
                    pos: Optional[Tensor] = None,
                    query_pos: Optional[Tensor] = None):
        tgt2 = self.norm(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)[0]
        tgt = tgt + self.dropout(tgt2)

        return tgt

    def forward(self, tgt, memory,
                memory_mask: Optional[Tensor] = None,
                memory_key_padding_mask: Optional[Tensor] = None,
                pos: Optional[Tensor] = None,
                query_pos: Optional[Tensor] = None):
        if self.normalize_before:
            return self.forward_pre(tgt, memory, memory_mask,
                                    memory_key_padding_mask, pos, query_pos)
        return self.forward_post(tgt, memory, memory_mask,
                                 memory_key_padding_mask, pos, query_pos)


class FFNLayer(nn.Module):

    def __init__(self, d_model, dim_feedforward=2048, dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm = nn.LayerNorm(d_model)

        self.activation = _get_activation_fn(activation)
        self.normalize_before = normalize_before

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        return tensor if pos is None else tensor + pos

    def forward_post(self, tgt):
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm(tgt)
        return tgt

    def forward_pre(self, tgt):
        tgt2 = self.norm(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout(tgt2)
        return tgt

    def forward(self, tgt):
        if self.normalize_before:
            return self.forward_pre(tgt)
        return self.forward_post(tgt)

def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")