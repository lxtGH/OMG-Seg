from xtuner.registry import BUILDER
from xtuner.engine.hooks import DatasetInfoHook

class DatasetInfoHook_withSpecoalTokens(DatasetInfoHook):
    def __init__(self, tokenizer, is_intern_repo_dataset=False):
        self.tokenizer = BUILDER.build(tokenizer)
        self.is_intern_repo_dataset = is_intern_repo_dataset
        # add special tokens
        # Adding special tokens for pixel grounding
        segmentation_tokens = ['[SEG]']
        # Adding tokens for GCG
        phrase_tokens = ['<p>', '</p>']
        # add for visual prompt
        region_tokens = ['<region>']
        point_tokens = ['<mark>']
        special_tokens = segmentation_tokens + phrase_tokens + region_tokens + point_tokens
        self.tokenizer.add_tokens(special_tokens, special_tokens=True)