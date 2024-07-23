import copy

import mmcv
import torch
from mmcv import LoadImageFromFile, Resize
from mmdet.datasets.transforms import PackDetInputs

from mmdet.registry import MODELS
from mmengine import Config
from mmengine.dataset import Compose, default_collate

from seg.evaluation.hooks.visual_hook import SAMLocalVisualizer
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix

IMG_SIZE = 1024
test_pipeline = [
    dict(type=LoadImageFromFile, backend_args=None),
    dict(type=Resize, scale=(IMG_SIZE, IMG_SIZE), keep_ratio=True),
    dict(
        type=PackDetInputs,
        meta_keys=('img_path', 'ori_shape', 'img_shape', 'scale_factor')
    )
]
IMG_PATH = 'demo/images/sa_1002.jpg'
MODEL_PATH = 'models/omg_seg_convl.pth'
pipeline = Compose(test_pipeline)

if __name__ == '__main__':
    model_cfg = Config.fromfile('demo/configs/m2_convl.py')

    model = MODELS.build(model_cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model = model.eval()
    model.init_weights()
    model_states = load_checkpoint_with_prefix(MODEL_PATH)
    incompatible_keys = model.load_state_dict(model_states, strict=False)
    print(incompatible_keys)

    inputs = pipeline(dict(
        img_path=IMG_PATH
    ))
    for key in inputs:
        inputs[key] = inputs[key].to(device=device)

    inputs = default_collate([inputs])
    with torch.no_grad():
        results = model.val_step(inputs)

    print("Starting to visualize results...")

    classes = copy.deepcopy(model_cfg.get('CLASSES', None))
    assert classes is not None, "You need to provide classes for visualization."
    for idx, cls in enumerate(classes):
        classes[idx] = cls.split(',')[0]

    # Visualization
    visualizer = SAMLocalVisualizer()
    visualizer.dataset_meta = dict(
        classes=classes
    )
    result = results[0]
    img = mmcv.imread(result.img_path, channel_order='rgb')
    visualizer.add_datasample(
        'test_img',
        img,
        data_sample=result,
        draw_gt=False,
        show=False,
        wait_time=0,
        pred_score_thr=0.,
        out_file='test_img.jpg',
        step=0
    )

    print("Done!")