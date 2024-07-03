# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import math
import os
import os.path as osp
import numpy as np
import torch
import tqdm
from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.model.utils import prepare_inputs_labels_for_multimodal
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)
from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict

import logging
from mmengine import print_log
from PIL import Image
from pycocotools import mask
import torch.nn.functional as F
from omg_llava.dataset.utils import expand2square
from omg_llava.dataset.utils.refcoco_refer import REFER
from omg_llava.tools.utils_refcoco import AverageMeter, Summary, intersectionAndUnionGPU


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument(
        '--dataset',
        choices=DATASETS_ATTRIBUTES.keys(),
        default='refcoco',
        help='Specify a ref dataset')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default='internlm2_chat',
        help='Specify a prompt template')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
        'a specific `dtype`.')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
        'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=100,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    args = parser.parse_args()
    return args

DATASETS_ATTRIBUTES = {
    'refcoco': {'splitBy': "unc", 'dataset_name': 'refcoco'},
    'refcoco_plus': {'splitBy': "unc", 'dataset_name': 'refcoco+'},
    'refcocog': {'splitBy': "umd", 'dataset_name': 'refcocog'},
}

@master_only
def master_print(msg):
    print(msg)

class RefcocoReferringSegDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 dataset_name,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 pad_image_to_square=False,
                 debug=False,
                 repeats=1,
                 split='val',
        ):
        self.split = split
        self._set_attribute(dataset_name)
        self.debug = debug
        if offline_processed_text_folder and data_path:
            print_log(
                'Both `offline_processed_text_folder` and '
                '`data_path` are set, and we load dataset from'
                '`offline_processed_text_folder` '
                f'({offline_processed_text_folder})',
                logger='current',
                level=logging.WARNING)

        if offline_processed_text_folder is not None:
            raise NotImplementedError
        else:
            json_datas = self.json_file_preprocess(data_path)
            self.json_datas = json_datas
            # json_datas = self.only_get_hf_map_infos()
            # json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            # self.text_data = process_hf_dataset(
            #     dataset=json_data,
            #     tokenizer=tokenizer,
            #     max_length=max_length,
            #     dataset_map_fn=dataset_map_fn,
            #     template_map_fn=template_map_fn,
            #     split='train',
            #     max_dataset_length=max_dataset_length,
            #     remove_unused_columns=False,
            #     pack_to_max_length=False,
            #     with_image_token=True,
            #     map_num_proc=num_proc,  # because limited mem
            # )

        self.image_folder = image_folder
        size = image_processor.crop_size
        if isinstance(size, int):
            self.image_h, self.image_w = size, size
        else:
            self.image_w, self.image_h = size

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.down_ratio = 1
        self.repeats = repeats

    def _set_attribute(self, dataset_name):
        attr_dict = DATASETS_ATTRIBUTES[dataset_name]

        self.splitBy = attr_dict['splitBy']
        self.dataset_name = attr_dict['dataset_name']

    def __len__(self):
        return len(self.json_datas) * self.repeats

    def real_len(self):
        return len(self.json_datas)

    def json_file_preprocess(self, data_path):
        splitBy = self.splitBy
        dataset_name = self.dataset_name
        refer_api = REFER(data_path, dataset_name, splitBy)
        ref_ids_train = refer_api.getRefIds(split=self.split)
        images_ids_train = refer_api.getImgIds(ref_ids=ref_ids_train)
        refs_train = refer_api.loadRefs(ref_ids=ref_ids_train)
        self.img2refs = self.create_img_to_refs_mapping(refs_train)

        image_infos = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        for item in loaded_images:
            item = item.copy()
            image_infos.append(item)

        self.annotations = refer_api.Anns
        # self.img2refs = self.create_img_to_refs_mapping(refs_train)

        refs = [self.img2refs[image_info['id']] for image_info in image_infos]

        ret = []
        for image_info, ref in zip(image_infos, refs):
            if len(ref) == 0:
                continue

            sents = []
            ann_ids = []
            for _ref in ref:
                for sent in _ref["sentences"]:
                    text = sent["sent"]
                    sents.append(text)
                    ann_ids.append(_ref["ann_id"])

            # if len(sents) >= 3:
            #     sampled_inds = np.random.choice(
            #         list(range(len(sents))), 3, replace=False
            #     )
            # else:
            #     sampled_inds = list(range(len(sents)))

            sampled_inds = list(range(len(sents)))
            sampled_sents = np.vectorize(sents.__getitem__)(sampled_inds).tolist()
            # sampled_ann_ids = np.vectorize(ann_ids.__getitem__)(sampled_inds).tolist()
            sampled_ann_ids = [ann_ids[ind] for ind in sampled_inds]
            selected_labels = sampled_sents
            ret.append(
                {'image_info': image_info,
                 'sampled_ann_id': sampled_ann_ids,
                 'selected_labels': selected_labels,
                 'image': image_info['file_name']
                 }
            )
        if self.debug:
            return ret[:10]
        return ret

    def load_images(self, refer_api, images_ids_train, dataset_dir, dataset_name, inference=False):
        images = []
        loaded_images = refer_api.loadImgs(image_ids=images_ids_train)
        for item in loaded_images:
            item = item.copy()
            images.append(item)
        return images

    def create_img_to_refs_mapping(self, refs_train):
        img2refs = {}
        for ref in refs_train:
            img2refs[ref["image_id"]] = img2refs.get(ref["image_id"], []) + [ref, ]
        return img2refs

    def decode_mask(self, annotations_ids, image_info):
        flag = False
        masks = []

        for ann_id in annotations_ids:
            if isinstance(ann_id, list):
                flag = True
                if -1 in ann_id:
                    assert len(ann_id) == 1
                    m = np.zeros((image_info["height"], image_info["width"])).astype(
                        np.uint8
                    )
                else:
                    m_final = np.zeros(
                        (image_info["height"], image_info["width"])
                    ).astype(np.uint8)
                    for ann_id_i in ann_id:
                        ann = self.annotations[ann_id_i]

                        if len(ann["segmentation"]) == 0:
                            m = np.zeros(
                                (image_info["height"], image_info["width"])
                            ).astype(np.uint8)
                        else:
                            if type(ann["segmentation"][0]) == list:  # polygon
                                rle = mask.frPyObjects(
                                    ann["segmentation"], image_info["height"], image_info["width"], )
                            else:
                                rle = ann["segmentation"]
                                for i in range(len(rle)):
                                    if not isinstance(rle[i]["counts"], bytes):
                                        rle[i]["counts"] = rle[i]["counts"].encode()
                            m = mask.decode(rle)
                            m = np.sum(
                                m, axis=2
                            )  # sometimes there are multiple binary map (corresponding to multiple segs)
                            m = m.astype(np.uint8)  # convert to np.uint8
                        m_final = m_final | m
                    m = m_final
                masks.append(m)
                continue

            ann = self.annotations[ann_id]

            if len(ann["segmentation"]) == 0:
                m = np.zeros((image_info["height"], image_info["width"])).astype(
                    np.uint8
                )
                masks.append(m)
                continue

            if type(ann["segmentation"][0]) == list:  # polygon
                rle = mask.frPyObjects(
                    ann["segmentation"], image_info["height"], image_info["width"]
                )
            else:
                rle = ann["segmentation"]
                for i in range(len(rle)):
                    if not isinstance(rle[i]["counts"], bytes):
                        rle[i]["counts"] = rle[i]["counts"].encode()
            m = mask.decode(rle)
            m = np.sum(m, axis=2)  # sometimes there are multiple binary map (corresponding to multiple segs)
            m = m.astype(np.uint8)  # convert to np.uint8
            masks.append(m)
        masks = np.stack(masks, axis=0)

        # if self.pad_image_to_square:
            # masks = expand2square_mask(masks)
        masks = torch.from_numpy(masks)

        # masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio,
        #                                                 self.image_w // self.down_ratio), mode='nearest').squeeze(0)

        # print(image_info['file_name'])
        # print(masks.shape)
        # save_masks = torch.stack([masks[0], masks[0], masks[0]], dim=-1)
        # save_masks = save_masks.numpy() * 255
        # save_masks = Image.fromarray(save_masks.astype(np.uint8))
        # save_masks.save("/root/mask.png")
        # print(kkk)
        return masks

    def only_get_text_infos(self, json_data):
        return {'sampled_sents': json_data['selected_labels']}

    def get_questions(self, text_require_infos):
        sampled_sents = text_require_infos['sampled_sents']
        ret = []
        for sent in sampled_sents:
            ret.append("Please segment {} in this image.".format(sent))
        return ret

    def filter_data_dict(self, data_dict):
        names = ['pixel_values', 'masks', 'ori_size', 'questions']
        ret = {name: data_dict[name] for name in names}
        return ret

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = self.json_datas[index]
        text_require_infos = self.only_get_text_infos(data_dict)
        questions = self.get_questions(text_require_infos)

        assert data_dict.get('image', None) is not None
        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image_file = os.path.join(self.image_folder, image_file)
            # print(image_file)
            image = Image.open(image_file).convert('RGB')
            ori_width, ori_height = image.size
            if self.pad_image_to_square:
                image = expand2square(
                    image,
                    tuple(
                        int(x * 255) for x in self.image_processor.image_mean))
            image = self.image_processor.preprocess(
                image, return_tensors='pt')['pixel_values'][0]
            data_dict['pixel_values'] = image

            # process and get masks
            masks = self.decode_mask(data_dict['sampled_ann_id'], data_dict['image_info'])
            data_dict['masks'] = masks
            data_dict['ori_size'] = (ori_width, ori_height)
            data_dict['questions'] = questions
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['masks'] = None
        # pixel_values, binary masks, conversation/input ids
        return self.filter_data_dict(data_dict)

def main():
    args = parse_args()

    torch.manual_seed(args.seed)

    if args.launcher != 'none':
        set_multi_processing(distributed=True)
        init_dist(args.launcher)

        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)
    # if args.cfg_options is not None:
        # cfg.merge_from_dict(args.cfg_options)

    model_name = cfg.model.type if isinstance(cfg.model.type,
                                              str) else cfg.model.type.__name__
    if 'LLaVAModel' or 'OMG' in model_name:
        cfg.model.pretrained_pth = None

    model = BUILDER.build(cfg.model)
    backend = get_file_backend(args.pth_model)

    if os.path.exists(cfg.pretrained_pth):
        if isinstance(backend, PetrelBackend):
            from xtuner.utils.fileio import patch_fileio
            with patch_fileio():
                state_dict = guess_load_checkpoint(cfg.pretrained_pth)
        else:
            state_dict = guess_load_checkpoint(cfg.pretrained_pth)

        # del state_dict['llm.base_model.model.model.tok_embeddings.weight']
        model.load_state_dict(state_dict, strict=False)
        print(f'Load pre PTH model from {cfg.pretrained_pth}')

    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    # del state_dict['llm.base_model.model.model.tok_embeddings.weight']
    # print(state_dict.keys())
    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')
    # image_processor_cfg = copy.deepcopy(cfg.image_processor)
    image_processor = cfg.image_processor
    image_processor_type = image_processor['type']
    del image_processor['type']
    image_processor = image_processor_type(**image_processor)

    llm = model.llm
    tokenizer = model.tokenizer

    model.cuda()
    model.eval()
    llm.eval()
    visual_encoder = model.visual_encoder
    projector = model.projector
    projector_text2vision = model.projector_text2vision

    projector.cuda()
    projector.eval()

    visual_encoder.cuda()
    visual_encoder.eval()

    stop_words = args.stop_words
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    # # work_dir
    # if args.work_dir is not None:
    #     # update configs according to CLI args if args.work_dir is not None
    #     save_dir = args.work_dir
    # else:
    #     # use config filename as default work_dir
    #     save_dir = osp.join('./work_dirs',
    #                         osp.splitext(osp.basename(args.data_path))[0])
    # timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))
    # save_dir = osp.join(save_dir, timestamp)

    # if rank == 0:
        # mkdir_or_exist(osp.abspath(save_dir))
        # print('=======================================================')
        # print(f'Dataset path: {osp.abspath(args.data_path)}\n'
        #       f'Results will be saved to {osp.abspath(save_dir)}')
        # print('=======================================================')

        # args_path = osp.join(save_dir, 'args.json')
        # with open(args_path, 'w', encoding='utf-8') as f:
        #     json.dump(args.__dict__, f, indent=2)

    # results_xlsx_path = osp.join(save_dir, 'mmbench_result.xlsx')
    # results_json_path = osp.join(save_dir, 'mmbench_result.json')

    dataset = RefcocoReferringSegDataset(
        dataset_name=args.dataset,
        image_folder='./data/glamm_data/' + 'images/coco2014/train2014/',
        image_processor=image_processor,
        data_path="./data/ref_seg/",
        tokenizer=tokenizer,
        pad_image_to_square=True,
        debug=False,
        split=args.split,
        # debug=True,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))

    trackers = {
        "intersection": AverageMeter("Intersec", ":6.3f", Summary.SUM),
        "union": AverageMeter("Union", ":6.3f", Summary.SUM),
        "gIoU": AverageMeter("gIoU", ":6.3f", Summary.SUM)
    }

    for i in tqdm.tqdm(per_rank_ids, desc=f'Rank {rank}'):
        data_sample = dataset[i]
        questions = data_sample['questions']
        texts = []
        for question in questions:
            texts.append(DEFAULT_IMAGE_TOKEN + '\n' + question)

        # if data_sample['context'] is not None:
        #     text = data_sample['context'] + '\n' + data_sample[
        #         'question'] + '\n' + data_sample['options']
        # else:
        #     text = data_sample['question'] + '\n' + data_sample['options']
        #
        # text = DEFAULT_IMAGE_TOKEN + '\n' + text
        #
        # if is_cn_string(text):
        #     text = text + '请直接回答选项字母。'
        # else:
        #     text = text + ("Answer with the option's letter from the "
        #                    'given choices directly.')
        prompt_texts = []

        if args.prompt_template:
            for text in texts:
                prompt_text = ''
                template = PROMPT_TEMPLATE[args.prompt_template]
                prompt_text += template['INSTRUCTION'].format(
                    input=text, round=1, bot_name=args.bot_name)
                prompt_texts.append(prompt_text)
        else:
            prompt_texts = texts

        batch_inputs = prompt_texts

        image = data_sample['pixel_values']  # ()
        image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)
        visual_outputs = visual_encoder(image, output_hidden_states=True)
        if isinstance(visual_outputs, list) or isinstance(visual_outputs, tuple)\
                or isinstance(visual_outputs, torch.Tensor):
            pixel_values = projector(visual_outputs)
        else:
            pixel_values = projector(
                visual_outputs.hidden_states[args.visual_select_layer][:, 1:])
        # pixel_values = projector(
        #     visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

        ori_size = data_sample['ori_size']
        target_masks = data_sample['masks'].cuda().to(torch.uint8)

        intersection, union, accuracy_iou = 0.0, 0.0, 0.0

        for idx_inp, inputs in enumerate(batch_inputs):
            # print("Question: ", inputs)
            chunk_encode = []
            for idx, chunk in enumerate(inputs.split(DEFAULT_IMAGE_TOKEN)):
                if idx == 0:
                    cur_encode = tokenizer.encode(chunk)
                else:
                    cur_encode = tokenizer.encode(chunk, add_special_tokens=False)
                chunk_encode.append(cur_encode)
            assert len(chunk_encode) == 2
            ids = []
            for idx, cur_chunk_encode in enumerate(chunk_encode):
                ids.extend(cur_chunk_encode)
                if idx != len(chunk_encode) - 1:
                    ids.append(IMAGE_TOKEN_INDEX)
            ids = torch.tensor(ids).cuda().unsqueeze(0)
            mm_inputs = prepare_inputs_labels_for_multimodal(
                llm=llm, input_ids=ids, pixel_values=pixel_values)

            # mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(torch.float16)

            generate_output = llm.generate(
                **mm_inputs,
                generation_config=gen_config,
                streamer=None,
                bos_token_id=tokenizer.bos_token_id,
                stopping_criteria=stop_criteria,
                output_hidden_states=True,
                return_dict_in_generate=True
            )
            predict = tokenizer.decode(
                # generate_output.sequences[0], skip_special_tokens=True).strip()
                generate_output.sequences[0]).strip()
            # print("Answer:", predict)

            hidden_states = generate_output.hidden_states
            last_hidden_states = [item[-1][-1] for item in hidden_states]
            last_hidden_states = torch.cat(last_hidden_states, dim=0)
            seg_hidden_states = get_seg_hidden_states(
                # last_hidden_states, generate_output.sequences[0],
                last_hidden_states, generate_output.sequences[0][:-1],
                seg_id=model.seg_token_idx
            )
            # seg_hidden_states = seg_hidden_states.to(torch.float32)
            # print("Mask num: ", len(seg_hidden_states))
            if len(seg_hidden_states) == 0:
                print("Warning, no [SEG] tokens !!!")
                continue
            elif len(seg_hidden_states) > 1:
                print("Warning, {} [SEG] tokens !!!".format(len(seg_hidden_states)))
                seg_hidden_states = seg_hidden_states[:1]

            seg_hidden_states = projector_text2vision(seg_hidden_states)
            batch_idxs = torch.zeros((seg_hidden_states.shape[0],),
                                      dtype=torch.int64).to(seg_hidden_states.device)
            pred_masks_list = model.visual_encoder.forward_llm_seg(seg_hidden_states, batch_idxs)
            pred_masks = pred_masks_list[-1]
            w, h = ori_size
            masks = F.interpolate(pred_masks, size=(max(w, h), max(w, h)),
                                  mode='bilinear', align_corners=False)
            masks = masks[:, 0]
            # remove padding
            if w == h:
                pass
            elif w > h:
                n_pad = w - h
                n_pad_1 = n_pad // 2
                n_pad_2 = n_pad - n_pad_1
                masks = masks[:, n_pad_1: w - n_pad_2]
            else:
                n_pad = h - w
                n_pad_1 = n_pad // 2
                n_pad_2 = n_pad - n_pad_1
                masks = masks[:, :, n_pad_1: h - n_pad_2]
            # binary
            masks = masks.sigmoid() > 0.5
            masks = masks.int()
            _target = target_masks[idx_inp:idx_inp+1].int()

            # intersection, union, accuracy_iou = 0.0, 0.0, 0.0
            for target, prediction in zip(masks, _target):
                intersect, union_, _ = intersectionAndUnionGPU(
                    prediction.contiguous().clone(), target.contiguous(), 2, ignore_index=255
                )
                intersection += intersect
                union += union_
                accuracy_iou += intersect / (union_ + 1e-5)
                # print(intersect / (union_ + 1e-5))
                # handles no-object targets
                accuracy_iou[union_ == 0] += 1.0

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        accuracy_iou = accuracy_iou.cpu().numpy() / target_masks.shape[0]
        trackers["intersection"].update(intersection)
        trackers["union"].update(union)
        trackers["gIoU"].update(accuracy_iou, n=target_masks.shape[0])

    # for meter in trackers.values():
    #     meter.all_reduce()
    # print(trackers["intersection"].sum, '       ', trackers["union"].sum, '    ',
    #       trackers["gIoU"].avg, '          ', trackers["gIoU"].count)
    cur_results = {'pixel_intersection': trackers["intersection"].sum[1],
                   'pixel_union': trackers["union"].sum[1],
                   'gIoU': trackers["gIoU"].avg[1],
                   'mask_counts': trackers["gIoU"].count,
                   }
    results.append(cur_results)
    # iou_per_class = trackers["intersection"].sum / (trackers["union"].sum + 1e-10)
    # class_iou = iou_per_class[1]
    # global_iou = trackers["gIoU"].avg[1]
    #
    # print("ciou: ", class_iou)
    # print("giou: ", global_iou)

    results = collect_results(results, n_samples)

    if get_rank() == 0:
        pixel_intersection = [cur_result['pixel_intersection'] for cur_result in results]
        pixel_union = [cur_result['pixel_union'] for cur_result in results]
        gIoUs = [cur_result['gIoU'] for cur_result in results]
        mask_counts = [cur_result['mask_counts'] for cur_result in results]

        class_iou = sum(pixel_intersection) / (sum(pixel_union) + 1e-10)
        global_iou = sum([giou * n_masks for giou, n_masks in zip(gIoUs, mask_counts)]) / sum(mask_counts)
        print("ciou: ", class_iou)
        print("giou: ", global_iou)

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]


if __name__ == '__main__':

    main()
