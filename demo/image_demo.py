import torch
from mmcv import LoadImageFromFile, Resize
from mmdet.datasets.transforms import PackDetInputs

from mmdet.registry import MODELS
from mmengine import Config
from mmengine.dataset import Compose, default_collate

IMG_SIZE = 1024
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(type=Resize, scale=(IMG_SIZE, IMG_SIZE), keep_ratio=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]
pipeline = Compose(test_pipeline)

if __name__ == '__main__':
    model_cfg = Config.fromfile('demo/configs/m2_convl.py')

    model = MODELS.build(model_cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model = model.eval()
    model.init_weights()

    inputs = pipeline(dict(
        img_path='demo/images/sa_1002.jpg'
    ))
    for key in inputs:
        inputs[key] = inputs[key].to(device=device)

    inputs = default_collate([inputs])
    results = model.val_step(inputs)

