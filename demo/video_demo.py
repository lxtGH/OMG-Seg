import copy
import os.path

import mmcv
import mmengine
import torch
from mmcv import LoadImageFromFile, Resize, TransformBroadcaster

from mmdet.registry import MODELS
from mmengine import Config
from mmengine.dataset import Compose, default_collate

from seg.datasets.pipelines.formatting import PackVidSegInputs
from seg.evaluation.hooks.visual_hook import VidSegLocalVisualizer
from seg.models.utils.load_checkpoint import load_checkpoint_with_prefix

VID_SIZE = (1280, 736)
test_pipeline = [
    dict(
        type=TransformBroadcaster,
        transforms=[
            dict(type=LoadImageFromFile, backend_args=None),
            dict(type=Resize, scale=VID_SIZE, keep_ratio=True),
        ]),
    dict(
        type=PackVidSegInputs,
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                   'frame_id', 'video_length', 'ori_video_length'),
        default_meta_keys=()
    )
]
VID_PATH = 'demo/images/350_6L1vA-xJt-M'
MODEL_PATH = 'models/omg_seg_convl.pth'
pipeline = Compose(test_pipeline)

if __name__ == '__main__':
    model_cfg = Config.fromfile('demo/configs/m2_convl_vid.py')

    model = MODELS.build(model_cfg.model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device=device)
    model = model.eval()
    model.init_weights()
    model_states = load_checkpoint_with_prefix(MODEL_PATH)
    incompatible_keys = model.load_state_dict(model_states, strict=False)
    print(incompatible_keys)

    imgs = sorted(list(mmengine.list_dir_or_file(VID_PATH)))
    img_list = []
    img_id_list = []
    for idx, img in enumerate(imgs):
        img_list.append(os.path.join(VID_PATH, img))
        img_id_list.append(idx + 100)
    inputs = pipeline(dict(
        img_path=img_list,
        img_id=img_id_list,
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
    visualizer = VidSegLocalVisualizer()
    visualizer.dataset_meta = dict(
        classes=classes
    )
    result = results[0]
    mmengine.mkdir_or_exist('test_video')
    for data_sample in result:
        img = mmcv.imread(data_sample.img_path, channel_order='rgb')
        visualizer.add_datasample(
            'test_video',
            img,
            data_sample=data_sample,
            draw_gt=False,
            show=False,
            wait_time=0,
            pred_score_thr=0.,
            out_file=os.path.join('test_video', os.path.basename(data_sample.img_path)),
            step=0
        )

    print("Done!")