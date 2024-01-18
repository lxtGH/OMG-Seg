from mmengine.optim import LinearLR, MultiStepLR, OptimWrapper
from mmengine.runner import EpochBasedTrainLoop, ValLoop, TestLoop
from torch.optim import AdamW

# training schedule for 50e
train_cfg = dict(
    type=EpochBasedTrainLoop,
    max_epochs=12,
    val_interval=2,
)
val_cfg = dict(type=ValLoop)
test_cfg = dict(type=TestLoop)

# learning rate
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=0.001,
        by_epoch=False,
        begin=0,
        end=500
    ),
    dict(
        type=MultiStepLR,
        begin=0,
        end=12,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1
    )
]

_embed_multi = dict(lr_mult=1.0, decay_mult=0.0)
optim_wrapper = dict(
    type=OptimWrapper,
    optimizer=dict(
        type=AdamW,
        lr=0.0001,
        weight_decay=0.05,
        eps=1e-8,
        betas=(0.9, 0.999)
    ),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1, decay_mult=1.0),
            'query_embed': _embed_multi,
            'query_feat': _embed_multi,
            'level_embed': _embed_multi,
        },
        norm_decay_mult=0.0
    ),
    clip_grad=dict(max_norm=0.01, norm_type=2)
)

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=True, base_batch_size=16)
