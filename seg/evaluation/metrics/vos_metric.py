# Copyright (c) OpenMMLab. All rights reserved.
import os.path
from collections import defaultdict
from typing import Dict, List, Optional, Sequence

import mmcv
import numpy as np
from mmengine import mkdir_or_exist
from mmengine.dist import barrier
from mmdet.registry import METRICS
from mmdet.evaluation.metrics.base_video_metric import BaseVideoMetric

PALETTE = {
    'davis': b'\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0',
    'mose': b'\x00\x00\x00\xe4\x1a\x1c7~\xb8M\xafJ\x98N\xa3\xff\x7f\x00\xff\xff3\xa6V(\xf7\x81\xbf\x99\x99\x99f\xc2\xa5\xfc\x8db\x8d\xa0\xcb\xe7\x8a\xc3\xa6\xd8T\xff\xd9/\xe5\xc4\x94\xb3\xb3\xb3\x8d\xd3\xc7\xff\xff\xb3\xbe\xba\xda\xfb\x80r\x80\xb1\xd3\xfd\xb4b\xb3\xdei\xfc\xcd\xe5\xd9\xd9\xd9\xbc\x80\xbd\xcc\xeb\xc5\xff\xedo',
}


@METRICS.register_module()
class VOSMetric(BaseVideoMetric):
    """mAP evaluation metrics for the VIS task.

    Args:
        metric (str | list[str]): Metrics to be evaluated.
            Default value is `youtube_vis_ap`..
        outfile_prefix (str | None): The prefix of json files. It includes
            the file path and the prefix of filename, e.g., "a/b/prefix".
            If not specified, a temp file will be created. Defaults to None.
        collect_device (str): Device name used for collecting results from
            different ranks during distributed training. Must be 'cpu' or
            'gpu'. Defaults to 'cpu'.
        prefix (str, optional): The prefix that will be added in the metric
            names to disambiguate homonyms metrics of different evaluators.
            If prefix is not provided in the argument, self.default_prefix
            will be used instead. Default: None
        format_only (bool): If True, only formatting the results to the
            official format and not performing evaluation. Defaults to False.
    """

    default_prefix: Optional[str] = 'vip_seg'

    def __init__(self,
                 collect_device: str = 'cpu',
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 palette: Optional[str] = None,
                 results_path: str = 'DAVIS'
                 ) -> None:
        super().__init__(collect_device=collect_device, prefix=prefix)
        self.format_only = format_only
        if palette is not None:
            self.palette = PALETTE[palette]
        else:
            self.palette = None
        self.results_path = results_path

        self.per_video_res = []
        self.categories = {}
        self._vis_meta_info = defaultdict(list)  # record video and image infos

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        for track_data_sample in data_samples:
            video_data_samples = track_data_sample['video_data_samples']
            if 'pred_track_proposal' not in video_data_samples[0]:
                continue
            ori_video_len = video_data_samples[0].ori_video_length
            if ori_video_len == len(video_data_samples):
                # video process
                self.process_video(video_data_samples)
            else:
                # image process
                raise NotImplementedError

    def process_video(self, data_samples):
        video_length = len(data_samples)
        mkdir_or_exist(self.results_path)
        for frame_id in range(video_length):
            img_data_sample = data_samples[frame_id].to_dict()
            pred = img_data_sample['pred_track_proposal']
            h, w = pred.shape
            pred_map = np.zeros((h, w, 3), dtype=np.uint8)
            for ins_id in np.unique(pred):
                if ins_id == 0:
                    continue
                r = ins_id // 1000000
                g = (ins_id % 1000000) // 1000
                b = ins_id % 1000
                pred_map[pred == ins_id] = np.array([r, g, b], dtype=np.uint8)
            ori_img_path = data_samples[frame_id].img_path
            folder_name = os.path.basename(os.path.dirname(ori_img_path))
            file_name = os.path.basename(ori_img_path)
            file_name = file_name.replace('.jpg', '.png')
            if self.palette is not None:
                from PIL import Image
                pred_map = mmcv.bgr2rgb(pred_map)
                pil_image = Image.fromarray(pred_map)
                pil_image = pil_image.convert('P', palette=self.palette)
                out_path = os.path.join(self.results_path, folder_name, file_name)
                mkdir_or_exist(os.path.dirname(out_path))
                pil_image.save(out_path)
            else:
                mmcv.imwrite(pred_map, os.path.join(self.results_path, folder_name, file_name))

    def compute_metrics(self, results: List) -> Dict[str, float]:
        return {}

    def evaluate(self, size: int) -> dict:
        """Evaluate the model performance of the whole dataset after processing
        all batches.

        Args:
            size (int): Length of the entire validation dataset.

        Returns:
            dict: Evaluation metrics dict on the val dataset. The keys are the
            names of the metrics, and the values are corresponding results.
        """
        # wait for all processes to complete prediction.
        barrier()
        metrics = self.compute_metrics([])
        return metrics
