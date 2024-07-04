# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import json
import math
import os
import os.path as osp
import re
import torch
import tqdm

from mmengine.dist import (collect_results, get_dist_info, get_rank, init_dist,
                           master_only)
from mmengine.utils.dl_utils import set_multi_processing
from torch.utils.data import Dataset
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from xtuner.model.utils import LoadWoInit
from omg_llava.model.utils import prepare_inputs_labels_for_multimodal_with_visual_prompts
from xtuner.tools.utils import get_stop_criteria, is_cn_string
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE)

from xtuner.registry import BUILDER
from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from mmengine.config import Config
from mmengine.fileio import PetrelBackend, get_file_backend
from mmengine.config import ConfigDict

from PIL import Image
import torch.nn.functional as F
from omg_llava.dataset.utils import expand2square, expand2square_mask
from pycocotools import mask

from pycocotools.coco import COCO
import numpy as np

def bbox_to_x1y1x2y2(bbox):
    x1, y1, w, h = bbox
    bbox = [x1, y1, x1 + w, y1 + h]

    return bbox

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
        '--output-path', type=str, default='./work_dirs/region_cap_pred.json', help='Name for Bot')
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
        default=300,
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


@master_only
def master_print(msg):
    print(msg)

class RegionCap_Inference_Dataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 pad_image_to_square=True,
                 annotation_file=None,
                 debug=False,
                 ):
        self.debug = debug
        self.image_folder = image_folder
        size = image_processor.crop_size
        # if isinstance(size, int):
        #     self.image_h, self.image_w = size, size
        # else:
        #     self.image_w, self.image_h = size
        self.image_h, self.image_w = 1024, 1024

        if isinstance(image_processor, dict) or isinstance(
                image_processor, Config) or isinstance(image_processor,
                                                       ConfigDict):
            self.image_processor = BUILDER.build(image_processor)
        else:
            self.image_processor = image_processor
        self.pad_image_to_square = pad_image_to_square
        self.down_ratio = 1

        self.coco = COCO(annotation_file)
        self.image_dict = self.coco.imgs
        self.ann_dict = self.coco.anns
        self.image_dict_keys = list(self.image_dict.keys())

    def __len__(self):
        return len(self.image_dict_keys)

    def decode_mask(self, annotation, image_info):
        flag = False
        masks = []

        for ann_id in range(1):

            ann = {"segmentation": annotation}

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

        if self.pad_image_to_square:
            masks = expand2square_mask(masks)
        masks = torch.from_numpy(masks)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio,
                                                        self.image_w // self.down_ratio), mode='nearest').squeeze(0)

        return masks

    def get_questions(self):
        question = "Can you provide me with a detailed description of the region in the picture marked by region1 <mark>?"
        return question

    def __getitem__(self, index):

        data_dict = {}

        image_id = self.image_dict_keys[index]
        image_file = self.image_dict[image_id]['file_name']

        questions = self.get_questions()

        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
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
        data_dict['ori_size'] = (ori_width, ori_height)
        data_dict['questions'] = questions

        masks = self.ann_dict[image_id]['segmentation']
        image_info = self.image_dict[image_id]
        masks = self.decode_mask(masks, image_info)

        data_dict['regions'] = masks
        data_dict['image_id'] = image_id

        return data_dict

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
    if isinstance(backend, PetrelBackend):
        from xtuner.utils.fileio import patch_fileio
        with patch_fileio():
            state_dict = guess_load_checkpoint(args.pth_model)
    else:
        state_dict = guess_load_checkpoint(args.pth_model)

    # del state_dict['llm.base_model.model.model.tok_embeddings.weight']
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

    dataset = RegionCap_Inference_Dataset(
        annotation_file='./data/region_caption/refcocog/finetune_refcocog_val_with_mask.json',
        image_folder='./data/glamm_data/images/coco2014/train2014/',
        image_processor=image_processor,
        pad_image_to_square=True,
        debug=False,
        # debug=True,
    )
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))
    results = []
    for i in tqdm.tqdm(per_rank_ids, desc=f'Rank {rank}'):
        # pixel feature
        data_sample = dataset[i]
        image = data_sample['pixel_values']  # ()
        image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)
        visual_outputs = visual_encoder(image, output_hidden_states=True)
        if isinstance(visual_outputs, list) or isinstance(visual_outputs, tuple) \
                or isinstance(visual_outputs, torch.Tensor):
            pixel_values = projector(visual_outputs)
        else:
            pixel_values = projector(
                visual_outputs.hidden_states[args.visual_select_layer][:, 1:])


        questions = data_sample['questions']
        regions = data_sample['regions']
        texts = DEFAULT_IMAGE_TOKEN + '\n' + questions

        if args.prompt_template:
            prompt_text = ''
            template = PROMPT_TEMPLATE[args.prompt_template]
            prompt_text += template['INSTRUCTION'].format(
                input=texts, round=1, bot_name=args.bot_name)
        else:
            prompt_text = texts

        batch_inputs = prompt_text

        predict = forward_model(
            batch_inputs, pixel_values,
            tokenizer, model, llm,
            projector_text2vision,
            gen_config, stop_criteria, points=regions,
            mark_token_id=model.mark_token_idx,
            width=image.shape[-1], height=image.shape[-2],
            visual_encoder=visual_encoder, projector=projector
        )

        text_output = predict.replace("<s>", "").replace("\n", "")\
            .replace("region1", '').replace("Region1", '')\
            .replace(':', '').replace("   ", " ").replace("  ", " ")
        text_output = text_output.split("ASSISTANT: ")[-1]

        cleaned_str = re.sub(r'<.*?>', '', text_output)

        # Remove the [SEG] token
        cleaned_str = cleaned_str.replace('[SEG]', '')

        # only select 1 setence for eval
        # cleaned_str = cleaned_str.split('.')[0]

        # Strip unnecessary spaces
        cleaned_str = ' '.join(cleaned_str.split()).strip("'")
        cleaned_str = cleaned_str.strip()

        result_dict = {}
        result_dict["image_id"] = data_sample['image_id']
        result_dict["caption"] = cleaned_str
        result_dict["image_file"] = data_sample['image_file']
        result_dict["prediction"] = cleaned_str
        results.append(result_dict)
        print(cleaned_str)

    results = collect_results(results, n_samples)

    if get_rank() == 0:
        with open(args.output_path, 'w') as json_file:
            json.dump(results, json_file, indent=2)

def forward_model(question, pixel_values,
                  tokenizer, model, llm,
                  projector_text2vision,
                  gen_config, stop_criteria,
                  mark_token_id=None,
                  points=None, width=None, height=None,
                  visual_encoder=None, projector=None):
    # pixel_values = projector(
    #     visual_outputs.hidden_states[args.visual_select_layer][:, 1:])

    inputs = question
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
    points = points.cuda()

    points_mark_embedding = get_points_embeddings(
        points, ids, width, height,
        mark_token_id, visual_encoder,
        projector)

    mm_inputs = prepare_inputs_labels_for_multimodal_with_visual_prompts(
        llm=llm, input_ids=ids, pixel_values=pixel_values,
        mark_id=mark_token_id,
        mark_feats=points_mark_embedding, region_id=-9999)

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
        generate_output.sequences[0], skip_special_tokens=True).strip()
    return predict


def get_points_embeddings(points, input_ids, width, height,
                          mark_token_idx, visual_encoder,
                          projector):
    if points is None or len(points) == 0:
        return []

    mark_token_mask = input_ids == mark_token_idx
    batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
            input_ids.device)
    batch_idxs = batch_idxs[mark_token_mask]  # (N, ) batch_size number

    # points = points.to(torch.float32)
    # print(points.dtype, batch_idxs.dtype)
    # marks_embeddings = visual_encoder.forward_point_sam(
    #         points, batch_idxs, width=width, height=height
    #     )[:, 0]  # (N, C)

    marks_embeddings = visual_encoder.forward_region_sam(
        points, batch_idxs
    )[:, 0]  # (N, C)

    marks_embeddings = marks_embeddings.to(projector.model.query_proj.weight.dtype)
    marks_embeddings = projector.model.query_proj(marks_embeddings)
    marks_embeddings = projector.model.model(marks_embeddings)
    return marks_embeddings  # (N, C)

if __name__ == '__main__':

    main()
