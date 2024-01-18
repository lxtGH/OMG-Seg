from typing import Optional, Tuple, Union

import mmcv
import mmengine
import numpy as np
import pycocotools.mask as maskUtils
import torch

from mmcv.transforms.base import BaseTransform
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import LoadAnnotations as MMDET_LoadAnnotations
from mmdet.structures.bbox import autocast_box_type
from mmdet.structures.mask import BitmapMasks
from mmdet.datasets.transforms import LoadPanopticAnnotations
from mmengine.fileio import get

from seg.models.utils import NO_OBJ


@TRANSFORMS.register_module()
class LoadPanopticAnnotationsHB(LoadPanopticAnnotations):
    def _load_masks_and_semantic_segs(self, results: dict) -> None:
        """Private function to load mask and semantic segmentation annotations.

        In gt_semantic_seg, the foreground label is from ``0`` to
        ``num_things - 1``, the background label is from ``num_things`` to
        ``num_things + num_stuff - 1``, 255 means the ignored label (``VOID``).

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.
        """
        # seg_map_path is None, when inference on the dataset without gts.
        if results.get('seg_map_path', None) is None:
            return

        img_bytes = get(
            results['seg_map_path'], backend_args=self.backend_args)
        pan_png = mmcv.imfrombytes(
            img_bytes, flag='color', channel_order='rgb').squeeze()
        pan_png = self.rgb2id(pan_png)

        gt_masks = []
        gt_seg = np.zeros_like(pan_png).astype(np.int32) + NO_OBJ  # 255 as ignore

        for segment_info in results['segments_info']:
            mask = (pan_png == segment_info['id'])
            gt_seg = np.where(mask, segment_info['category'], gt_seg)

            # The legal thing masks
            if segment_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['ori_shape']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks

        if self.with_seg:
            results['gt_seg_map'] = gt_seg


@TRANSFORMS.register_module()
class LoadVideoSegAnnotations(LoadPanopticAnnotations):

    def __init__(
            self,
            **kwargs
    ) -> None:
        super().__init__(**kwargs)

    def _load_instances_ids(self, results: dict) -> None:
        """Private function to load instances id annotations.

        Args:
            results (dict): Result dict from :obj :obj:``mmcv.BaseDataset``.

        Returns:
            dict: The dict containing instances id annotations.
        """
        gt_instances_ids = []
        for instance in results['instances']:
            gt_instances_ids.append(instance['instance_id'])
        results['gt_instances_ids'] = np.array(
            gt_instances_ids, dtype=np.int32)

    def _load_masks_and_semantic_segs(self, results: dict) -> None:
        h, w = results['ori_shape']
        gt_masks = []
        gt_seg = np.zeros((h, w), dtype=np.int32) + NO_OBJ

        for segment_info in results['segments_info']:
            mask = maskUtils.decode(segment_info['mask'])
            gt_seg = np.where(mask, segment_info['category'], gt_seg)

            # The legal thing masks
            if segment_info.get('is_thing'):
                gt_masks.append(mask.astype(np.uint8))

        if self.with_mask:
            h, w = results['ori_shape']
            gt_masks = BitmapMasks(gt_masks, h, w)
            results['gt_masks'] = gt_masks

        if self.with_seg:
            results['gt_seg_map'] = gt_seg

    def transform(self, results: dict) -> dict:
        """Function to load multiple types panoptic annotations.

        Args:
            results (dict): Result dict from :obj:``mmdet.CustomDataset``.

        Returns:
            dict: The dict contains loaded bounding box, label, mask and
                semantic segmentation annotations.
        """

        super().transform(results)
        self._load_instances_ids(results)
        return results


@TRANSFORMS.register_module()
class LoadJSONFromFile(BaseTransform):
    """Load an json from file.

    Required Keys:

    - info_path

    Modified Keys:

    Args:
        backend_args (dict, optional): Instantiates the corresponding file
            backend. It may contain `backend` key to specify the file
            backend. If it contains, the file backend corresponding to this
            value will be used and initialized with the remaining values,
            otherwise the corresponding file backend will be selected
            based on the prefix of the file path. Defaults to None.
            New in version 2.0.0rc4.
    """

    def __init__(self, backend_args: Optional[dict] = None) -> None:
        self.backend_args: Optional[dict] = None
        if backend_args is not None:
            self.backend_args = backend_args.copy()

    def transform(self, results: dict) -> Optional[dict]:
        """Functions to load image.

        Args:
            results (dict): Result dict from
                :class:`mmengine.dataset.BaseDataset`.

        Returns:
            dict: The dict contains loaded image and meta information.
        """

        filename = results['info_path']
        data_info = mmengine.load(filename, backend_args=self.backend_args)

        results['height'] = data_info['image']['height']
        results['width'] = data_info['image']['width']

        # The code here are similar to `parse_data_info` in coco
        instances = []
        for ann in sorted(data_info['annotations'], key=lambda x: -x['area']):
            instance = {}

            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, results['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, results['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]

            instance['ignore_flag'] = 0
            instance['bbox'] = bbox
            instance['bbox_label'] = 0

            if ann.get('segmentation', None):
                instance['mask'] = ann['segmentation']

            if ann.get('point_coords', None):
                instance['point_coords'] = ann['point_coords']

            instances.append(instance)

        results['instances'] = instances
        return results

    def __repr__(self):
        repr_str = (f'{self.__class__.__name__}('
                    f'backend_args={self.backend_args})')

        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotationsSAM(MMDET_LoadAnnotations):

    def __init__(self, *args, with_point_coords=False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.with_point_coords = with_point_coords

    def _load_point_coords(self, results: dict) -> None:
        assert self.with_point_coords
        gt_point_coords = []
        for instance in results.get('instances', []):
            gt_point_coords.append(instance['point_coords'])
        results['gt_point_coords'] = np.array(gt_point_coords, dtype=np.float32)

    def transform(self, results: dict) -> Optional[dict]:
        super().transform(results)
        if self.with_point_coords:
            self._load_point_coords(results)
        return results


@TRANSFORMS.register_module()
class FilterAnnotationsHB(BaseTransform):
    """Filter invalid annotations.

    Required Keys:

    - gt_bboxes (BaseBoxes[torch.float32]) (optional)
    - gt_bboxes_labels (np.int64) (optional)
    - gt_masks (BitmapMasks | PolygonMasks) (optional)
    - gt_ignore_flags (bool) (optional)

    Modified Keys:

    - gt_bboxes (optional)
    - gt_bboxes_labels (optional)
    - gt_masks (optional)
    - gt_ignore_flags (optional)

    Args:
        min_gt_bbox_wh (tuple[float]): Minimum width and height of ground truth
            boxes. Default: (1., 1.)
        min_gt_mask_area (int): Minimum foreground area of ground truth masks.
            Default: 1
        by_box (bool): Filter instances with bounding boxes not meeting the
            min_gt_bbox_wh threshold. Default: True
        by_mask (bool): Filter instances with masks not meeting
            min_gt_mask_area threshold. Default: False
        keep_empty (bool): Whether to return None when it
            becomes an empty bbox after filtering. Defaults to True.
    """

    def __init__(self,
                 min_gt_bbox_wh: Tuple[int, int] = (1, 1),
                 min_gt_mask_area: int = 1,
                 by_box: bool = True,
                 by_mask: bool = False) -> None:
        assert by_box or by_mask
        self.min_gt_bbox_wh = min_gt_bbox_wh
        self.min_gt_mask_area = min_gt_mask_area
        self.by_box = by_box
        self.by_mask = by_mask

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        assert 'gt_bboxes' in results
        gt_bboxes = results['gt_bboxes']
        if gt_bboxes.shape[0] == 0:
            return None

        tests = []
        if self.by_box:
            tests.append(
                ((gt_bboxes.widths > self.min_gt_bbox_wh[0]) &
                 (gt_bboxes.heights > self.min_gt_bbox_wh[1])).numpy())
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks']
            tests.append(gt_masks.areas >= self.min_gt_mask_area)

        keep = tests[0]
        for t in tests[1:]:
            keep = keep & t

        results['gt_ignore_flags'] = np.logical_or(results['gt_ignore_flags'], np.logical_not(keep))
        if results['gt_ignore_flags'].all():
            return None
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class GTNMS(BaseTransform):

    def __init__(self,
                 by_box: bool = True,
                 by_mask: bool = False
                 ) -> None:
        assert by_box or by_mask and not (by_box and by_mask)
        self.by_box = by_box
        self.by_mask = by_mask

    @autocast_box_type()
    def transform(self, results: dict) -> Union[dict, None]:
        """Transform function to filter annotations.

        Args:
            results (dict): Result dict.

        Returns:
            dict: Updated result dict.
        """
        gt_ignore_flags = results['gt_ignore_flags']
        if self.by_box:
            raise NotImplementedError
        if self.by_mask:
            assert 'gt_masks' in results
            gt_masks = results['gt_masks'].masks
            tot_mask = np.zeros_like(gt_masks[0], dtype=np.uint8)
            for idx, mask in enumerate(gt_masks):
                if gt_ignore_flags[idx]:
                    continue
                overlapping = mask * tot_mask
                ratio = overlapping.sum() / sum(mask).sum()
                if ratio > 0.8:
                    # ignore with overlapping
                    gt_ignore_flags[idx] = True
                    continue
                tot_mask = (tot_mask + mask).clip(max=1)

        results['gt_ignore_flags'] = gt_ignore_flags
        return results

    def __repr__(self):
        return self.__class__.__name__


@TRANSFORMS.register_module()
class LoadFeatFromFile(BaseTransform):

    def __init__(self, model_name='vit_h'):
        self.cache_suffix = f'_{model_name}_cache.pth'

    def transform(self, results: dict) -> Optional[dict]:
        img_path = results['img_path']
        feat_path = img_path.replace('.jpg', self.cache_suffix)
        assert mmengine.exists(feat_path)
        feat = torch.load(feat_path)
        results['feat'] = feat
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'

        return repr_str


@TRANSFORMS.register_module()
class ResizeOri(BaseTransform):

    def __init__(
            self,
            backend: str = 'cv2',
            interpolation='bilinear'
    ):
        self.backend = backend
        self.interpolation = interpolation

    def transform(self, results: dict) -> Optional[dict]:
        results['ori_shape'] = results['img_shape']
        results['scale_factor'] = (1., 1.)
        return results

    def __repr__(self):
        repr_str = f'{self.__class__.__name__}'
        return repr_str
