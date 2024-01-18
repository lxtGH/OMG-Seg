from abc import ABC
import logging
from typing import Sequence, Union, Optional, Tuple

from mmengine.dataset import ConcatDataset, RepeatDataset, ClassBalancedDataset
from mmengine.logging import print_log
from mmengine.registry import DATASETS
from mmengine.dataset.base_dataset import BaseDataset

from mmdet.structures import TrackDataSample

from seg.models.utils import NO_OBJ


@DATASETS.register_module()
class ConcatOVDataset(ConcatDataset, ABC):
    _fully_initialized: bool = False

    def __init__(self,
                 datasets: Sequence[Union[BaseDataset, dict]],
                 lazy_init: bool = False,
                 data_tag: Optional[Tuple[str]] = None,
                 ):
        for i, dataset in enumerate(datasets):
            if isinstance(dataset, dict):
                dataset.update(lazy_init=lazy_init)
                if 'times' in dataset:
                    dataset['dataset'].update(lazy_init=lazy_init)
        super().__init__(datasets, lazy_init=lazy_init,
                         ignore_keys=['classes', 'thing_classes', 'stuff_classes', 'palette'])
        self.data_tag = data_tag
        if self.data_tag is not None:
            assert len(self.data_tag) == len(datasets)

        cls_names = []
        for dataset in self.datasets:
            if isinstance(dataset, RepeatDataset) or isinstance(dataset, ClassBalancedDataset):
                if hasattr(dataset.dataset, 'dataset_name'):
                    name = dataset.dataset.dataset_name
                else:
                    name = dataset.dataset.__class__.__name__
            else:
                if hasattr(dataset, 'dataset_name'):
                    name = dataset.dataset_name
                else:
                    name = dataset.__class__.__name__
            cls_names.append(name)

        thing_classes = []
        thing_mapper = []
        stuff_classes = []
        stuff_mapper = []
        for idx, dataset in enumerate(self.datasets):
            if 'classes' not in dataset.metainfo or (self.data_tag is not None and self.data_tag[idx] in ['sam']):
                # class agnostic dataset
                _thing_mapper = {}
                _stuff_mapper = {}
                thing_mapper.append(_thing_mapper)
                stuff_mapper.append(_stuff_mapper)
                continue
            _thing_classes = dataset.metainfo['thing_classes'] \
                if 'thing_classes' in dataset.metainfo else dataset.metainfo['classes']
            _stuff_classes = dataset.metainfo['stuff_classes'] if 'stuff_classes' in dataset.metainfo else []
            _thing_mapper = {}
            _stuff_mapper = {}
            for idy, cls in enumerate(_thing_classes):
                flag = False
                cls = cls.replace('_or_', ',')
                cls = cls.replace('/', ',')
                cls = cls.replace('_', ' ')
                cls = cls.lower()
                for all_idx, all_cls in enumerate(thing_classes):
                    if set(cls.split(',')).intersection(set(all_cls.split(','))):
                        _thing_mapper[idy] = all_idx
                        flag = True
                        break
                if not flag:
                    thing_classes.append(cls)
                    _thing_mapper[idy] = len(thing_classes) - 1
            thing_mapper.append(_thing_mapper)

            for idy, cls in enumerate(_stuff_classes):
                flag = False
                cls = cls.replace('_or_', ',')
                cls = cls.replace('/', ',')
                cls = cls.replace('_', ' ')
                cls = cls.lower()
                for all_idx, all_cls in enumerate(stuff_classes):
                    if set(cls.split(',')).intersection(set(all_cls.split(','))):
                        _stuff_mapper[idy] = all_idx
                        flag = True
                        break
                if not flag:
                    stuff_classes.append(cls)
                    _stuff_mapper[idy] = len(stuff_classes) - 1
            stuff_mapper.append(_stuff_mapper)

        cls_name = ""
        cnt = 0
        dataset_idx = 0
        classes = [*thing_classes, *stuff_classes]
        mapper = []
        meta_cls_names = []
        for _thing_mapper, _stuff_mapper in zip(thing_mapper, stuff_mapper):
            if not _thing_mapper and not _stuff_mapper:
                # class agnostic dataset
                _mapper = dict()
                for idx in range(1000):
                    _mapper[idx] = -1
            else:
                _mapper = {**_thing_mapper}
                _num_thing = len(_thing_mapper)
                for key, value in _stuff_mapper.items():
                    assert value < len(stuff_classes)
                    _mapper[key + _num_thing] = _stuff_mapper[key] + len(thing_classes)
                assert len(_mapper) == len(_thing_mapper) + len(_stuff_mapper)
                cnt += 1
                cls_name = cls_name + cls_names[dataset_idx] + "_"
                meta_cls_names.append(cls_names[dataset_idx])
            _mapper[NO_OBJ] = NO_OBJ
            mapper.append(_mapper)
            dataset_idx += 1
        if cnt > 1:
            cls_name = "Concat_" + cls_name
        cls_name = cls_name[:-1]
        self.dataset_name = cls_name

        self._metainfo.update({
            'classes': classes,
            'thing_classes': thing_classes,
            'stuff_classes': stuff_classes,
            'mapper': mapper,
            'dataset_names': meta_cls_names
        })
        print_log(
            f"------------{self.dataset_name}------------",
            logger='current',
            level=logging.INFO
        )

        for idx, dataset in enumerate(self.datasets):
            dataset_type = cls_names[idx]
            if isinstance(dataset, RepeatDataset):
                times = dataset.times
            else:
                times = 1
            print_log(
                f"|---dataset#{idx + 1} --> name: {dataset_type}; length: {len(dataset)}; repeat times: {times}",
                logger='current',
                level=logging.INFO
            )

        print_log(
            f"------num_things : {len(thing_classes)}; num_stuff : {len(stuff_classes)}------",
            logger='current',
            level=logging.INFO
        )

    def get_dataset_source(self, idx: int) -> int:
        dataset_idx, _ = self._get_ori_dataset_idx(idx)
        return dataset_idx

    def __getitem__(self, idx):
        if not self._fully_initialized:
            print_log(
                'Please call `full_init` method manually to '
                'accelerate the speed.',
                logger='current',
                level=logging.WARNING)
            self.full_init()
        dataset_idx, sample_idx = self._get_ori_dataset_idx(idx)
        results = self.datasets[dataset_idx][sample_idx]
        _mapper = self.metainfo['mapper'][dataset_idx]

        data_samples = results['data_samples']
        if isinstance(data_samples, TrackDataSample):
            for det_sample in data_samples:
                if 'gt_sem_seg' in det_sample:
                    det_sample.gt_sem_seg.sem_seg.apply_(lambda x: _mapper.__getitem__(x))
                if 'gt_instances' in det_sample:
                    det_sample.gt_instances.labels.apply_(lambda x: _mapper.__getitem__(x))
        else:
            if 'gt_sem_seg' in data_samples:
                data_samples.gt_sem_seg.sem_seg.apply_(lambda x: _mapper.__getitem__(x))
            if 'gt_instances' in data_samples:
                data_samples.gt_instances.labels.apply_(lambda x: _mapper.__getitem__(x))

        if self.data_tag is not None:
            data_samples.data_tag = self.data_tag[dataset_idx]
        return results
