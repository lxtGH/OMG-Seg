import os
from typing import Tuple, List

import pycocotools.mask as maskUtils

import mmcv
import numpy as np
from mmdet.registry import DATASETS
from mmdet.datasets.base_video_dataset import BaseVideoDataset
from mmengine import fileio, join_path, scandir, track_parallel_progress, dump, list_from_file, print_log, exists, load
from mmengine.dist import master_only, dist


def mask2bbox(mask):
    bbox = np.zeros((4,), dtype=np.float32)
    x_any = np.any(mask, axis=0)
    y_any = np.any(mask, axis=1)
    x = np.where(x_any)[0]
    y = np.where(y_any)[0]
    if len(x) > 0 and len(y) > 0:
        bbox = np.array((x[0], y[0], x[-1], y[-1]), dtype=np.float32)
    return bbox


def video_parser(params):
    seq_id, vid_folder, ann_folder = params
    images = []
    assert os.path.basename(vid_folder) == os.path.basename(ann_folder)
    _tmp_img_id = -1
    imgs_cur = sorted(list(map(
        lambda x: str(x), scandir(vid_folder, recursive=False, suffix='.jpg')
    )))
    pans_cur = sorted(list(map(
        lambda x: str(x), scandir(ann_folder, recursive=False, suffix='.png')
    )))
    for img_cur, pan_cur in zip(imgs_cur, pans_cur):
        assert img_cur.split('.')[0] == pan_cur.split('.')[0]
        _tmp_img_id += 1
        img_id = _tmp_img_id
        item_full = os.path.join(vid_folder, img_cur)
        inst_map = os.path.join(ann_folder, pan_cur)
        img_dict = {
            'img_path': item_full,
            'ann_path': inst_map,
        }
        assert os.path.exists(img_dict['img_path'])
        assert os.path.exists(img_dict['ann_path'])
        instances = []
        ann_map = mmcv.imread(img_dict['ann_path'], flag='unchanged').astype(np.uint32)
        ann_map = ann_map[..., 0] * 1000000 + ann_map[..., 1] * 1000 + ann_map[..., 2]
        img_dict['height'], img_dict['width'] = ann_map.shape

        for pan_seg_id in np.unique(ann_map):
            if pan_seg_id == 0:
                continue
            instance = {}
            mask = (ann_map == pan_seg_id).astype(np.uint8)
            instance['instance_id'] = pan_seg_id
            instance['bbox'] = mask2bbox(mask)
            instance['bbox_label'] = 0
            instance['ignore_flag'] = 0
            instance['mask'] = maskUtils.encode(np.asfortranarray(mask))
            instance['mask']['counts'] = instance['mask']['counts'].decode()
            instances.append(instance)
        img_dict['instances'] = instances
        img_dict['video_id'] = seq_id
        img_dict['frame_id'] = img_id
        img_dict['img_id'] = seq_id * 10000 + img_id
        images.append(img_dict)
    return {
        'video_id': seq_id,
        'images': images,
        'video_length': len(images)
    }


@DATASETS.register_module()
class DAVIS(BaseVideoDataset):
    METAINFO = {
        'classes': {},
        'palette': {},
    }

    def __init__(self, dataset_version: str, *args, **kwargs):
        self.__class__.__name__ = f'DVAIS_{dataset_version}'
        super().__init__(*args, **kwargs)

    @master_only
    def build_cache(self, ann_json_path, video_folders, ann_folders) -> None:
        vid_ids = range(len(video_folders))

        data_list = track_parallel_progress(
            video_parser,
            tasks=list(zip(vid_ids, video_folders, ann_folders)),
            nproc=20,
            keep_order=False,
        )
        data_list = sorted(data_list, key=lambda x: x['video_id'])
        dump(data_list, ann_json_path)

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``.

        Returns:
            tuple(list[dict], list): A list of annotation and a list of
            valid data indices.
        """
        with fileio.get_local_path(self.ann_file) as local_path:
            video_folders = list_from_file(local_path, prefix=self.data_prefix['img'])
            ann_folders = list_from_file(local_path, prefix=self.data_prefix['ann'])
        assert len(video_folders) == len(ann_folders)
        print_log(f"#videos : {len(video_folders)} ", logger='current')

        split = os.path.basename(self.ann_file).split('.')[0]
        ann_json_path = f"{split}_annotations.json"
        ann_json_path = join_path(self.data_root, ann_json_path)
        if not exists(ann_json_path):
            self.build_cache(ann_json_path, video_folders, ann_folders)
        dist.barrier()
        raw_data_list = load(ann_json_path)
        data_list = []
        for raw_data_info in raw_data_list:
            data_info = self.parse_data_info(raw_data_info)
            data_list.append(data_info)
        vid_len_list = [itm['video_length'] for itm in data_list]
        max_vid_len = max(vid_len_list)
        min_vid_len = min(vid_len_list)
        print_log(
            f"Max video len : {max_vid_len}; "
            f"Min video len : {min_vid_len}."
            ,
            logger='current',
        )
        return data_list

    def parse_data_info(self, raw_data_info: dict) -> dict:
        data_info = {
            'video_id': raw_data_info['video_id'],
            'video_length': raw_data_info['video_length']
        }
        images = []
        for raw_img_data_info in raw_data_info['images']:
            img_data_info = {
                'img_path': raw_img_data_info['img_path'],
                'height': raw_img_data_info['height'],
                'width': raw_img_data_info['width'],
                'video_id': raw_img_data_info['video_id'],
                'frame_id': raw_img_data_info['frame_id'],
                'img_id': raw_img_data_info['img_id']
            }
            instances = []
            segments_info = []
            for ann in raw_img_data_info['instances']:
                instance = {}
                category_id = ann['bbox_label']
                bbox = ann['bbox']
                is_thing = 1
                if is_thing:
                    instance['bbox'] = bbox
                    instance['bbox_label'] = category_id
                    instance['ignore_flag'] = ann['ignore_flag']
                    instance['instance_id'] = ann['instance_id']

                segment_info = {
                    'mask': ann['mask'],
                    'category': category_id,
                    'is_thing': is_thing
                }
                segments_info.append(segment_info)
                if len(instance) > 0 and is_thing:
                    instances.append(instance)
            img_data_info['instances'] = instances
            img_data_info['segments_info'] = segments_info
            images.append(img_data_info)
        data_info['images'] = images
        return data_info

    def filter_data(self) -> List[dict]:
        """Filter image annotations according to filter_cfg.

        Returns:
            list[int]: Filtered results.
        """
        if self.test_mode:
            return self.data_list

        num_imgs_before_filter = sum([len(info['images']) for info in self.data_list])
        num_imgs_after_filter = num_imgs_before_filter

        new_data_list = self.data_list

        print_log(
            'The number of samples before and after filtering: '
            f'{num_imgs_before_filter} / {num_imgs_after_filter}', 'current')
        return new_data_list
