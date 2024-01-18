from typing import Sequence

import torch
import torch.distributed as torch_dist
from mmengine.dist import get_dist_info, get_default_group, get_comm_device
from torch._C._distributed_c10d import ReduceOp
from torch.utils.data import Sampler, BatchSampler

from mmdet.datasets.samplers.batch_sampler import AspectRatioBatchSampler
from mmdet.registry import DATA_SAMPLERS


@DATA_SAMPLERS.register_module()
class VideoSegAspectRatioBatchSampler(AspectRatioBatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (int): Size of mini-batch.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
            its size would be less than ``batch_size``.
    """

    def __iter__(self) -> Sequence[int]:
        for idx in self.sampler:
            # hard code to solve TrackImgSampler
            video_idx = idx
            # video_idx
            data_info = self.sampler.dataset.get_data_info(video_idx)
            # data_info {video_id, images, video_length}
            if 'images' in data_info:
                img_data_info = data_info['images'][0]
            else:
                img_data_info = data_info
            width, height = img_data_info['width'], img_data_info['height']
            bucket_id = 0 if width < height else 1
            bucket = self._aspect_ratio_buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size:
                yield bucket[:]
                del bucket[:]

        # yield the rest data and reset the bucket
        left_data = self._aspect_ratio_buckets[0] + self._aspect_ratio_buckets[
            1]
        self._aspect_ratio_buckets = [[] for _ in range(2)]
        while len(left_data) > 0:
            if len(left_data) <= self.batch_size:
                if not self.drop_last:
                    yield left_data[:]
                left_data = []
            else:
                yield left_data[:self.batch_size]
                left_data = left_data[self.batch_size:]


@DATA_SAMPLERS.register_module()
class MultiDataAspectRatioBatchSampler(BatchSampler):
    """A sampler wrapper for grouping images with similar aspect ratio (< 1 or.

    >= 1) into a same batch for multi-source datasets.

    Args:
        sampler (Sampler): Base sampler.
        batch_size (Sequence(int)): Size of mini-batch for multi-source
        datasets.
        num_datasets(int): Number of multi-source datasets.
        drop_last (bool): If ``True``, the sampler will drop the last batch if
        its size would be less than ``batch_size``.
    """

    def __init__(self,
                 sampler: Sampler,
                 batch_size: Sequence[int],
                 num_datasets: int,
                 drop_last: bool = True) -> None:
        if not isinstance(sampler, Sampler):
            raise TypeError('sampler should be an instance of ``Sampler``, '
                            f'but got {sampler}')
        self.sampler = sampler
        if isinstance(batch_size, int):
            self.batch_size = [batch_size] * num_datasets
        else:
            self.batch_size = batch_size
        self.num_datasets = num_datasets
        self.drop_last = drop_last
        # two groups for w < h and w >= h for each dataset --> 2 * num_datasets
        self._buckets = [[] for _ in range(2 * self.num_datasets)]

    def __iter__(self) -> Sequence[int]:
        num_batch = torch.tensor(len(self), device='cpu')
        rank, world_size = get_dist_info()
        if world_size > 1:
            group = get_default_group()
            backend_device = get_comm_device(group)
            num_batch = num_batch.to(device=backend_device)
            torch_dist.all_reduce(num_batch, op=ReduceOp.MIN, group=group)
        num_batch = num_batch.to('cpu').item()

        for idx in self.sampler:
            data_info = self.sampler.dataset.get_data_info(idx)
            width, height = data_info.get('width', 0), data_info.get('height', 0)
            dataset_source_idx = self.sampler.dataset.get_dataset_source(idx)
            aspect_ratio_bucket_id = 0 if width < height else 1
            bucket_id = dataset_source_idx * 2 + aspect_ratio_bucket_id
            bucket = self._buckets[bucket_id]
            bucket.append(idx)
            # yield a batch of indices in the same aspect ratio group
            if len(bucket) == self.batch_size[dataset_source_idx]:
                yield bucket[:]
                num_batch -= 1
                if num_batch <= 0:
                    return
                del bucket[:]

        # yield the rest data and reset the bucket
        for i in range(self.num_datasets):
            left_data = self._buckets[i * 2 + 0] + self._buckets[i * 2 + 1]
            while len(left_data) > 0:
                if len(left_data) < self.batch_size[i]:
                    if not self.drop_last:
                        yield left_data[:]
                        num_batch -= 1
                        if num_batch <= 0:
                            return
                    left_data = []
                else:
                    yield left_data[:self.batch_size[i]]
                    num_batch -= 1
                    if num_batch <= 0:
                        return
                    left_data = left_data[self.batch_size[i]:]

        self._buckets = [[] for _ in range(2 * self.num_datasets)]

    def __len__(self) -> int:
        sizes = [0 for _ in range(self.num_datasets)]
        for idx in self.sampler:
            dataset_source_idx = self.sampler.dataset.get_dataset_source(idx)
            sizes[dataset_source_idx] += 1

        if self.drop_last:
            lens = 0
            for i in range(self.num_datasets):
                lens += sizes[i] // self.batch_size[i]
            return lens
        else:
            lens = 0
            for i in range(self.num_datasets):
                lens += (sizes[i] + self.batch_size[i] - 1) // self.batch_size[i]
            return lens
