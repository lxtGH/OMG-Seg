import copy

import numpy as np
from mmcv import BaseTransform
from mmdet.registry import TRANSFORMS

from seg.models.utils import NO_OBJ


@TRANSFORMS.register_module()
class ImageCopy(BaseTransform):
    """Copy an image several times to build a video seq.
    """
    DIVISOR = 10000

    def __init__(
            self,
            num_frames: int = 1,
    ) -> None:
        assert num_frames > 1
        self.num_frames = num_frames

    def transform(self, results: dict) -> dict:
        for key in results:
            value = results[key]
            results[key] = []
            for _ in range(self.num_frames):
                results[key].append(copy.deepcopy(value))

        num_instances = len(results['gt_bboxes_labels'][0])
        num_frames = len(results['gt_bboxes_labels'])
        gt_instance_ids = results['gt_bboxes_labels'][0] * self.DIVISOR + np.arange(num_instances) + 1
        results['gt_instances_ids'] = [copy.deepcopy(gt_instance_ids) for _ in range(num_frames)]
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'(num_frames={self.num_frames})'
        return repr_str


@TRANSFORMS.register_module()
class AddSemSeg(BaseTransform):
    """Add dummy semantic segmentation map.
    """

    def __init__(self, ) -> None:
        pass

    def transform(self, results: dict) -> dict:
        gt_seg = np.zeros(results['img'].shape[:2], dtype=np.int32) + NO_OBJ
        results['gt_seg_map'] = gt_seg
        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        return repr_str
