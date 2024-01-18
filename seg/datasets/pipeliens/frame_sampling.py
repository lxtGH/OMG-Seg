import random
from typing import Dict, List, Optional

import numpy as np
from mmdet.registry import TRANSFORMS
from mmdet.datasets.transforms import BaseFrameSample


@TRANSFORMS.register_module()
class VideoClipSample(BaseFrameSample):
    def __init__(self,
                 num_selected: int = 1,
                 interval: int = 1,
                 collect_video_keys: List[str] = ['video_id', 'video_length']):
        self.num_selected = num_selected
        self.interval = interval
        super().__init__(collect_video_keys=collect_video_keys)

    def transform(self, video_infos: dict) -> Optional[Dict[str, List]]:
        """Transform the video information.

        Args:
            video_infos (dict): The whole video information.

        Returns:
            dict: The data information of the sampled frames.
        """
        len_with_interval = self.num_selected + (self.num_selected - 1) * (self.interval - 1)
        len_video = video_infos['video_length']
        if len_with_interval > len_video:
            return None

        first_frame_id = random.sample(range(len_video - len_with_interval + 1), 1)[0]

        sampled_frames_ids = first_frame_id + np.arange(self.num_selected) * self.interval
        results = self.prepare_data(video_infos, sampled_frames_ids)

        return results

    def __repr__(self) -> str:
        repr_str = self.__class__.__name__
        repr_str += f'num_selected=({self.num_selected}'
        repr_str += f'interval={self.interval}'
        repr_str += f'collect_video_keys={self.collect_video_keys})'
        return repr_str
