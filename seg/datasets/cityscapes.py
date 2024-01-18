from mmdet.registry import DATASETS
from mmdet.datasets.coco_panoptic import CocoPanopticDataset
import os.path as osp

@DATASETS.register_module()
class CityscapesPanopticDataset(CocoPanopticDataset):
    """Cityscapes dataset for Panoptic segmentation.
    The class names are changed.
    """

    METAINFO = {
        'classes':
            (
                'person', 'rider', 'car', 'truck', 'bus',
                'train', 'motorcycle', 'bicycle',

                'road', 'sidewalk', 'building', 'wall', 'fence',
                'pole', 'traffic light', 'traffic sign', 'vegetation',
                'terrain', 'sky'
            ),
        'thing_classes':
            (
                'person', 'rider', 'car', 'truck', 'bus',
                'train', 'motorcycle', 'bicycle'
            ),
        'stuff_classes':
            (
                'road', 'sidewalk', 'building', 'wall', 'fence',
                'pole', 'traffic light', 'traffic sign', 'vegetation',
                'terrain', 'sky'
            ),
    }

    def parse_data_info(self, raw_data_info: dict) -> dict:
        """Parse raw annotation to target format.

        Args:
            raw_data_info (dict): Raw data information load from ``ann_file``.

        Returns:
            dict: Parsed annotation.
        """
        img_info = raw_data_info['raw_img_info']
        ann_info = raw_data_info['raw_ann_info']
        # filter out unmatched annotations which have
        # same segment_id but belong to other image
        ann_info = [
            ann for ann in ann_info if ann['image_id'] == img_info['img_id']
        ]
        data_info = {}

        img_path = osp.join(self.data_prefix['img'], img_info['file_name'])
        if self.data_prefix.get('seg', None):
            seg_map_path = osp.join(
                self.data_prefix['seg'],
                img_info['file_name'].replace("_leftImg8bit.png", "_panoptic.png")
            )
        else:
            seg_map_path = None
        data_info['img_path'] = img_path
        data_info['img_id'] = img_info['img_id']
        data_info['seg_map_path'] = seg_map_path
        data_info['height'] = img_info['height']
        data_info['width'] = img_info['width']

        if self.return_classes:
            data_info['text'] = self.metainfo['thing_classes']
            data_info['stuff_text'] = self.metainfo['stuff_classes']
            data_info['custom_entities'] = True  # no important

        instances = []
        segments_info = []
        for ann in ann_info:
            instance = {}
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            category_id = ann['category_id']
            contiguous_cat_id = self.cat2label[category_id]

            is_thing = self.coco.load_cats(ids=category_id)[0]['isthing']
            if is_thing:
                is_crowd = ann.get('iscrowd', False)
                instance['bbox'] = bbox
                instance['bbox_label'] = contiguous_cat_id
                if not is_crowd:
                    instance['ignore_flag'] = 0
                else:
                    instance['ignore_flag'] = 1
                    is_thing = False

            segment_info = {
                'id': ann['id'],
                'category': contiguous_cat_id,
                'is_thing': is_thing
            }
            segments_info.append(segment_info)
            if len(instance) > 0 and is_thing:
                instances.append(instance)
        data_info['instances'] = instances
        data_info['segments_info'] = segments_info
        return data_info
