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

from xtuner.model.utils import LoadWoInit, prepare_inputs_labels_for_multimodal
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
from xtuner.dataset.utils import expand2square
from pycocotools import mask as mask_utils


def convert_dict2config_dict(input):
    input = ConfigDict(**input)
    for key in input.keys():
        if isinstance(input[key], dict):
            input[key] = convert_dict2config_dict(input[key])
    return input

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')


GCG_QUESTIONS = [
    'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Can you provide a thorough description of the this image? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Please describe in detail the contents of the image. Please respond with interleaved segmentation masks for the corresponding parts of the answer.',
    'Could you give a comprehensive explanation of what can be found within this picture? Please output with interleaved segmentation masks for the corresponding phrases.',
    'Could you give me an elaborate explanation of this picture? Please respond with interleaved segmentation masks for the corresponding phrases.',
    'Could you provide me with a detailed analysis of this photo? Please output with interleaved segmentation masks for the corresponding parts of the answer.',
]

def parse_args():
    parser = argparse.ArgumentParser(description='RefCocoSeg')
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')
    parser.add_argument(
        '--output-name', type=str, default='gcg', help='save folder name')
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


@master_only
def master_print(msg):
    print(msg)

class GCD_Inference_Dataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 debug=False,
                 pad_image_to_square=True,
                 ):
        self.debug = debug
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

        self.images = os.listdir(image_folder)
        if debug:
            self.images = self.images[:20]

    def __len__(self):
        return len(self.images)

    def get_questions(self):
        question = "Could you please give me a detailed description of the image? Please respond with interleaved \
    segmentation masks for the corresponding parts of the answer."
        return question

    def __getitem__(self, index):

        data_dict = {}

        questions = self.get_questions()
        image_file = self.images[index]
        data_dict['image_file'] = image_file
        image_file = os.path.join(self.image_folder, image_file)
        print(image_file)
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
    if 'LLaVAModel' in model_name:
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

    dataset = GCD_Inference_Dataset(
        image_folder='./data/glamm_data/images/grandf/val_test/',
        image_processor=image_processor,
        pad_image_to_square=True,
        debug=False,
        # debug=True,
    )
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size)

    per_rank_ids = range(per_rank_samples * rank,
                         min(n_samples, per_rank_samples * (rank + 1)))

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


        # questions = data_sample['questions']
        questions = GCG_QUESTIONS
        for question in questions:
            # print(question)
            texts = DEFAULT_IMAGE_TOKEN + '\n' + question

            if args.prompt_template:
                prompt_text = ''
                template = PROMPT_TEMPLATE[args.prompt_template]
                prompt_text += template['INSTRUCTION'].format(
                    input=texts, round=1, bot_name=args.bot_name)
            else:
                prompt_text = texts

            batch_inputs = prompt_text

            predict, seg_hidden_states = forward_model(
                batch_inputs, pixel_values,
                tokenizer, model, llm,
                projector_text2vision,
                gen_config, stop_criteria)
            if len(seg_hidden_states) != 0:
                break


        ori_size = data_sample['ori_size']
        # print("Answer:", predict)
        # print("Mask num: ", len(seg_hidden_states))

        if len(seg_hidden_states) == 0:
            print("Warnning !!! No mask Pred !!!")
            w, h = ori_size
            masks = torch.zeros((0, h, w), dtype=torch.bool)
        else:
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
        process_and_save_output(
            "./work_dirs/{}/".format(args.output_name),
            data_sample['image_file'],
            predict,
            masks
        )

def forward_model(question, pixel_values,
                  tokenizer, model, llm,
                  projector_text2vision,
                  gen_config, stop_criteria):
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
    print("Answer:", predict)

    hidden_states = generate_output.hidden_states
    last_hidden_states = [item[-1][0] for item in hidden_states]
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    seg_hidden_states = get_seg_hidden_states(
        last_hidden_states, generate_output.sequences[0][:-1],
        seg_id=model.seg_token_idx
    )
    # seg_hidden_states = seg_hidden_states.to(torch.float32)
    # print("Mask num: ", len(seg_hidden_states))

    # seg_hidden_states = projector_text2vision(seg_hidden_states)
    return predict, seg_hidden_states

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)

    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for m in uncompressed_mask_rles:
        rle_masks.append(coco_encode_rle(m))

    # Create results dictionary
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "pred_masks": rle_masks
    }

    # print(cleaned_str)
    # print(phrases)

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json

    return rle

if __name__ == '__main__':

    main()
