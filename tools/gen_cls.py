# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import copy
import os
import os.path as osp
from typing import List, Optional

import torch
import torch.distributed as dist

import mmengine
from mmengine import MMLogger
from mmengine.config import Config, DictAction

from mmdet.engine.hooks.utils import trigger_visualization_hook
from mmdet.registry import MODELS, DATASETS
from mmdet.utils import setup_cache_size_limit_of_dynamo
from mmengine.dist import get_dist_info

from ext.templates import VILD_PROMPT


def split_label(x: str) -> List[str]:
    x = x.replace('_or_', ',')
    x = x.replace('/', ',')
    x = x.replace('_', ' ')
    x = x.lower()
    x = x.split(',')
    x = [_x.strip() for _x in x]
    return x


NUM_BATCH = 256


# TODO: support fuse_conv_bn and format_only
def parse_args():
    parser = argparse.ArgumentParser(
        description='MMDet test (and eval) a model')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('--checkpoint', default=None, help='checkpoint file')
    parser.add_argument(
        '--work-dir',
        help='the directory to save the file containing evaluation metrics')
    parser.add_argument(
        '--out',
        type=str,
        help='dump predictions to a pickle file for offline evaluation')
    parser.add_argument(
        '--show', action='store_true', help='show prediction results')
    parser.add_argument(
        '--show-dir',
        help='directory where painted images will be saved. '
             'If specified, it will be automatically saved '
             'to the work_dir/timestamp/show_dir')
    parser.add_argument(
        '--wait-time', type=float, default=2, help='the interval of show (s)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--tta', action='store_true')
    # When using PyTorch version >= 2.0.0, the `torch.distributed.launch`
    # will pass the `--local-rank` parameter to `tools/train.py` instead
    # of `--local_rank`.
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
    args = parse_args()

    # Reduce the number of repeated compilations and improve
    # testing speed.
    setup_cache_size_limit_of_dynamo()

    # load config
    cfg = Config.fromfile(args.config)
    cfg.launcher = args.launcher
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # work_dir is determined in this priority: CLI > segment in file > filename
    if args.work_dir is not None:
        # update configs according to CLI args if args.work_dir is not None
        cfg.work_dir = args.work_dir
    elif cfg.get('work_dir', None) is None:
        # use config filename as default work_dir if cfg.work_dir is None
        cfg.work_dir = osp.join('./work_dirs',
                                osp.splitext(osp.basename(args.config))[0])

    cfg.load_from = args.checkpoint

    if args.show or args.show_dir:
        cfg = trigger_visualization_hook(cfg, args)

    logger = MMLogger.get_current_instance()

    model = MODELS.build(cfg.model.backbone)
    model = model.get_text_model()
    model.init_weights()
    dataset_cfg = copy.deepcopy(cfg.train_dataloader.dataset)
    dataset_cfg.update(lazy_init=True)
    dataset = DATASETS.build(dataset_cfg)
    classes = dataset.metainfo['classes']
    logger.info(f"Dataset classes:\n{classes}")

    descriptions = []
    candidates = []
    for cls_name in classes:
        labels_per_cls = split_label(cls_name)
        candidates.append(len(labels_per_cls))
        for label in labels_per_cls:
            for template in VILD_PROMPT:
                description = template.format(label)
                descriptions.append(description)

    rank, world_size = get_dist_info()
    if world_size > 1:
        bs = len(descriptions)
        local_bs = bs // dist.get_world_size()
        if bs % dist.get_world_size() != 0:
            local_bs += 1
        local_descriptions = descriptions[rank * local_bs: (rank + 1) * local_bs]
        local_feat = model(local_descriptions)
        feat_list: List[Optional[torch.Tensor]] = [None for _ in range(world_size)]
        dist.all_gather_object(feat_list, local_feat.to(device='cpu'))
        features = torch.cat(feat_list)
    else:
        bs = len(descriptions)
        local_bs = bs // NUM_BATCH
        if bs % NUM_BATCH != 0:
            local_bs += 1
        feat_list = []
        for i in range(NUM_BATCH):
            local_descriptions = descriptions[i * local_bs: (i + 1) * local_bs]
            local_feat = model(local_descriptions).to(device='cpu')
            feat_list.append(local_feat)
        features = torch.cat(feat_list)

    if rank == 0:
        dim = features.shape[-1]
        candidate_tot = sum(candidates)
        candidate_max = max(candidates)
        features = features.reshape(candidate_tot, len(VILD_PROMPT), dim)
        features = features / features.norm(dim=-1, keepdim=True)
        features = features.mean(dim=1, keepdims=False)
        features = features / features.norm(dim=-1, keepdim=True)

        cur_pos = 0
        classifier = []
        for candidate in candidates:
            cur_feat = features[cur_pos:cur_pos + candidate]
            if candidate < candidate_max:
                cur_feat = torch.cat([cur_feat, cur_feat[0].repeat(candidate_max - candidate, 1)])
            classifier.append(cur_feat)
            cur_pos += candidate
        classifier = torch.stack(classifier)

        embd_path = os.path.join(os.path.expanduser('~/.cache'), 'embd')
        dataset_name = dataset.dataset_name if hasattr(dataset, 'dataset_name') else dataset.__class__.__name__
        save_path = os.path.join(embd_path, f'{model.model_name}_{dataset_name}.pth')
        mmengine.mkdir_or_exist(os.path.dirname(save_path))
        classifier_to_save = classifier
        torch.save(classifier_to_save, save_path)
        logger.info(f"The size of classifier is:\n{classifier_to_save.size()}")
        logger.info(f"Saved to {save_path}")
    else:
        pass
    if world_size > 1:
        dist.barrier()
    logger.info("Done!")


if __name__ == '__main__':
    main()
