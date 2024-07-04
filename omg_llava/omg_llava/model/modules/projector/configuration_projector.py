# Copyright (c) OpenMMLab. All rights reserved.
from transformers import PretrainedConfig

class ProjectorConfig_OMG_LLaVA(PretrainedConfig):
    model_type = 'projector'
    _auto_class = 'AutoConfig'

    def __init__(
        self,
        visual_hidden_size=4096,
        llm_hidden_size=4096,
        depth=2,
        hidden_act='gelu',
        bias=True,
        query_channels=256,
        feat_channels=1536,
        pixel_shuffle_ratio=None,
        additional_bg_tokens=10,
        visual_prompt_proj=False,
        add_cross_attn_layer=False,
        **kwargs,
    ):
        self.visual_hidden_size = visual_hidden_size
        self.llm_hidden_size = llm_hidden_size
        self.depth = depth
        self.hidden_act = hidden_act
        self.bias = bias
        self.query_channels=query_channels
        self.feat_channels=feat_channels
        if pixel_shuffle_ratio is not None:
            self.feat_channels = self.feat_channels * pixel_shuffle_ratio * pixel_shuffle_ratio
        self.additional_bg_tokens = additional_bg_tokens
        self.visual_prompt_proj = visual_prompt_proj
        self.add_cross_attn_layer = add_cross_attn_layer
        super().__init__(**kwargs)