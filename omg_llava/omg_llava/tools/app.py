import gradio as gr
import numpy as np

import sys
from omg_llava.tools.app_utils import process_markdown, show_mask_pred, parse_visual_prompts, description

import torch
from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)
from transformers.generation.streamers import TextStreamer

from xtuner.dataset.utils import expand2square, load_image
from omg_llava.dataset.utils import expand2square_bbox, expand2square_mask, expand2square_points
from omg_llava.model.utils import prepare_inputs_labels_for_multimodal_with_visual_prompts
from xtuner.tools.utils import get_stop_criteria
from xtuner.utils import (DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX,
                          PROMPT_TEMPLATE, SYSTEM_TEMPLATE)

import argparse
import os.path as osp

from mmengine.config import Config, DictAction
from mmengine.fileio import PetrelBackend, get_file_backend

from xtuner.configs import cfgs_name_path
from xtuner.model.utils import guess_load_checkpoint
from xtuner.registry import BUILDER

from gradio_image_prompter import ImagePrompter

TORCH_DTYPE_MAP = dict(
    fp16=torch.float16, bf16=torch.bfloat16, fp32=torch.float32, auto='auto')

def parse_args(args):
    parser = argparse.ArgumentParser(description="OMG-LLaVA Demo")
    parser.add_argument('config', help='config file name or path.')
    parser.add_argument('pth_model', help='pth model file')

    parser.add_argument(
        '--torch-dtype',
        default='fp16',
        choices=TORCH_DTYPE_MAP.keys(),
        help='Override the default `torch.dtype` and load the model under '
             'a specific `dtype`.')
    parser.add_argument(
        '--prompt-template',
        choices=PROMPT_TEMPLATE.keys(),
        default="internlm2_chat",
        help='Specify a prompt template')
    system_group = parser.add_mutually_exclusive_group()
    system_group.add_argument(
        '--system', default=None, help='Specify the system text')
    system_group.add_argument(
        '--system-template',
        choices=SYSTEM_TEMPLATE.keys(),
        default=None,
        help='Specify a system template')
    parser.add_argument(
        '--bits',
        type=int,
        choices=[4, 8, None],
        default=None,
        help='LLM bits')
    parser.add_argument(
        '--bot-name', type=str, default='BOT', help='Name for Bot')
    parser.add_argument(
        '--with-plugins',
        nargs='+',
        choices=['calculate', 'solve', 'search'],
        help='Specify plugins to use')
    parser.add_argument(
        '--no-streamer', action='store_true', help='Whether to with streamer')
    parser.add_argument(
        '--lagent', action='store_true', help='Whether to use lagent')
    parser.add_argument(
        '--stop-words', nargs='+', type=str, default=[], help='Stop words')
    parser.add_argument(
        '--offload-folder',
        default=None,
        help='The folder in which to offload the model weights (or where the '
             'model weights are already offloaded).')
    parser.add_argument(
        '--max-new-tokens',
        type=int,
        default=256,
        help='Maximum number of new tokens allowed in generated text')
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.1,
        help='The value used to modulate the next token probabilities.')
    parser.add_argument(
        '--top-k',
        type=int,
        default=40,
        help='The number of highest probability vocabulary tokens to '
             'keep for top-k-filtering.')
    parser.add_argument(
        '--top-p',
        type=float,
        default=0.75,
        help='If set to float < 1, only the smallest set of most probable '
             'tokens with probabilities that add up to top_p or higher are '
             'kept for generation.')
    parser.add_argument(
        '--repetition-penalty',
        type=float,
        default=1.0,
        help='The parameter for repetition penalty. 1.0 means no penalty.')
    parser.add_argument(
        '--seed',
        type=int,
        default=0,
        help='Random seed for reproducible text generation')
    return parser.parse_args(args)

def get_points_embeddings(points, input_ids, width, height,
                          mark_token_idx, mode='point'):
    if points is None or len(points) == 0:
        return []

    mark_token_mask = input_ids == mark_token_idx
    batch_idxs = torch.arange(input_ids.shape[0]).unsqueeze(1).repeat(1, input_ids.shape[1]).to(
            input_ids.device)
    batch_idxs = batch_idxs[mark_token_mask]  # (N, ) batch_size number

    points = points.to(torch.float32)

    if mode == 'point':
        marks_embeddings = visual_encoder.forward_point_sam(
                points, batch_idxs, width=width, height=height
        )[:, 0]  # (N, C)
    elif mode == 'box':
        marks_embeddings = visual_encoder.forward_box_sam(
            points, batch_idxs, width=width, height=height
        )[:, 0]  # (N, C)
    else:
        raise NotImplementedError

    marks_embeddings = marks_embeddings.to(projector.model.query_proj.weight.dtype)
    marks_embeddings = projector.model.query_proj(marks_embeddings)
    marks_embeddings = projector.model.model(marks_embeddings)
    return marks_embeddings  # (N, C)

def get_visual_prompts_embeddings(
        height, width, input_ids,
):
    points_prompts = global_infos.point_prompts
    boxes_prompts = global_infos.box_prompts

    if len(points_prompts) == 0:
        points_mark_embedding = []
    else:
        points = np.array(points_prompts)
        points = expand2square_points(points, height=height, width=width)
        points[:, 0] = points[:, 0] / max(height, width) * 1024
        points[:, 1] = points[:, 1] / max(height, width) * 1024
        points = torch.from_numpy(points)
        points = points.cuda()
        mark_token_id = omg_llava.mark_token_idx

        points_mark_embedding = get_points_embeddings(
            points, input_ids,
            1024, 1024,
            mark_token_id)


    if len(boxes_prompts) == 0:
        boxes_mark_embedding = []
    else:
        boxes_prompts = np.array(boxes_prompts)

        boxes_prompts = expand2square_bbox(boxes_prompts, height=height, width=width)
        boxes_prompts[:, [0, 2]] = boxes_prompts[:, [0, 2]] / max(height, width) * 1024
        boxes_prompts[:, [1, 3]] = boxes_prompts[:, [1, 3]] / max(height, width) * 1024
        boxes_prompts = torch.from_numpy(boxes_prompts)
        boxes_prompts = boxes_prompts.cuda()
        # using <region> token
        region_token_id = omg_llava.region_token_idx

        boxes_mark_embedding = get_points_embeddings(
            boxes_prompts, input_ids,
            1024, 1024,
            region_token_id,
            mode='box'
        )
    return points_mark_embedding, boxes_mark_embedding

def inference(input_str, all_inputs, follow_up):
    input_str = input_str.replace('<point>', '<mark>')\
        .replace('<box>', '<region>')

    prompts = all_inputs['points']
    visual_prompts = parse_visual_prompts(prompts)
    input_image = all_inputs['image']

    if not follow_up:
        # reset
        print('Log: History responses have been removed!')
        global_infos.n_turn = 0
        global_infos.inputs = ''
        # reset prompts
        global_infos.point_prompts = []
        global_infos.box_prompts = []
        global_infos.mask_prompts = []

        # first conversation, add image tokens
        text = DEFAULT_IMAGE_TOKEN + '\n' + input_str

        # prepare image
        image = load_image(input_image)
        width, height = image.size
        global_infos.image_width = width
        global_infos.image_height = height
        image = expand2square(
            image, tuple(int(x * 255) for x in image_processor.image_mean))
        global_infos.image_for_show = image
        image = image_processor.preprocess(
            image, return_tensors='pt')['pixel_values'][0]
        image = image.cuda().unsqueeze(0).to(visual_encoder.dtype)
        visual_outputs = visual_encoder(image, output_hidden_states=True)
        pixel_values = projector(visual_outputs)
        global_infos.panoptic_masks = omg_llava.visual_encoder.vis_binary_masks
        global_infos.pixel_values = pixel_values

        # for remove padding
        if width == height:
            sx, ex, sy, ey = 0, width, 0, height
        elif width > height:
            sy = int((width - height) / 2.0)
            ey = width - sy
            sx, ex = 0, width
        else:
            sx = int((height - width) / 2.0)
            ex = height - sx
            sy, ey = 0, height

        global_infos.sx = sx
        global_infos.sy = sy
        global_infos.ex = ex
        global_infos.ey = ey

    else:
        text = input_str
        pixel_values = global_infos.pixel_values

    # add cur prompts into global prompts
    global_infos.point_prompts += visual_prompts['points']
    global_infos.box_prompts += visual_prompts['boxes']

    if args.prompt_template:
        prompt_text = ''
        template = PROMPT_TEMPLATE[args.prompt_template]
        if 'SYSTEM' in template and global_infos.n_turn == 0:
            system_text = None
            if args.system_template is not None:
                system_text = SYSTEM_TEMPLATE[
                    args.system_template].format(
                    round=global_infos.n_turn + 1, bot_name=args.bot_name)
            elif args.system is not None:
                system_text = args.system
            if system_text is not None:
                prompt_text += template['SYSTEM'].format(
                    system=system_text,
                    round=global_infos.n_turn + 1,
                    bot_name=args.bot_name)
        prompt_text += template['INSTRUCTION'].format(
            input=text, round=global_infos.n_turn + 1, bot_name=args.bot_name)
    else:
        prompt_text = text

    global_infos.inputs += prompt_text

    # encode prompt text
    chunk_encode = []
    for idx, chunk in enumerate(global_infos.inputs.split(DEFAULT_IMAGE_TOKEN)):
        if idx == 0 and global_infos.n_turn == 0:
            cur_encode = tokenizer.encode(chunk)
        else:
            cur_encode = tokenizer.encode(
                chunk, add_special_tokens=False)
        chunk_encode.append(cur_encode)
    assert len(chunk_encode) == 2
    ids = []
    for idx, cur_chunk_encode in enumerate(chunk_encode):
        ids.extend(cur_chunk_encode)
        if idx != len(chunk_encode) - 1:
            ids.append(IMAGE_TOKEN_INDEX)
    ids = torch.tensor(ids).cuda().unsqueeze(0)

    points_mark_embeddings, boxes_mark_embeddings = get_visual_prompts_embeddings(
        height=global_infos.image_height,
        width=global_infos.image_width, input_ids=ids
    )

    mark_embeddings = points_mark_embeddings

    mark_token_id = omg_llava.mark_token_idx
    mm_inputs = prepare_inputs_labels_for_multimodal_with_visual_prompts(
        llm=llm, input_ids=ids, pixel_values=pixel_values,
        mark_id=mark_token_id,
        mark_feats=mark_embeddings, region_id=omg_llava.region_token_idx,
        regions_feats=boxes_mark_embeddings,
    )

    # mm_inputs['inputs_embeds'] = mm_inputs['inputs_embeds'].to(torch.float16)

    generate_output = llm.generate(
        **mm_inputs,
        generation_config=gen_config,
        streamer=streamer,
        bos_token_id=tokenizer.bos_token_id,
        stopping_criteria=stop_criteria,
        output_hidden_states=True,
        return_dict_in_generate=True
    )

    predict = tokenizer.decode(
        generate_output.sequences[0])

    global_infos.inputs += predict
    predict = predict.strip()
    global_infos.n_turn += 1
    global_infos.inputs += sep
    if len(generate_output.sequences[0]) >= args.max_new_tokens:
        print(
            'Remove the memory of history responses, since '
            f'it exceeds the length limitation {args.max_new_tokens}.')
        global_infos.n_turn = 0
        global_infos.inputs = ''

    hidden_states = generate_output.hidden_states
    last_hidden_states = [item[-1][0] for item in hidden_states]
    last_hidden_states = torch.cat(last_hidden_states, dim=0)
    seg_hidden_states = get_seg_hidden_states(
        last_hidden_states, generate_output.sequences[0][:-1],
        seg_id=omg_llava.seg_token_idx
    )
    # seg_hidden_states = seg_hidden_states.to(torch.float32)
    if len(seg_hidden_states) != 0:
        seg_hidden_states = projector_text2vision(seg_hidden_states)
        batch_idxs = torch.zeros((seg_hidden_states.shape[0],),
                                 dtype=torch.int64).to(seg_hidden_states.device)
        pred_masks_list = omg_llava.visual_encoder.forward_llm_seg(seg_hidden_states, batch_idxs)

        image_mask_show, selected_colors = show_mask_pred(
            global_infos.image_for_show, pred_masks_list[-1],
            crop_range = (global_infos.sx, global_infos.ex, global_infos.sy, global_infos.ey)
        )
    else:
        image_mask_show = global_infos.image_for_show.crop(
            (global_infos.sx, global_infos.sy, global_infos.ex, global_infos.ey))
        selected_colors = []

    panoptic_show, _ = show_mask_pred(
        global_infos.image_for_show, global_infos.panoptic_masks,
        crop_range=(global_infos.sx, global_infos.ex, global_infos.sy, global_infos.ey)
    )

    predict = process_markdown(predict, selected_colors)
    # return panoptic_show, image_mask_show, predict
    return image_mask_show, predict

def init_models(args):
    torch.manual_seed(args.seed)

    # parse config
    if not osp.isfile(args.config):
        try:
            args.config = cfgs_name_path[args.config]
        except KeyError:
            raise FileNotFoundError(f'Cannot find {args.config}')

    # load config
    cfg = Config.fromfile(args.config)

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

    model.load_state_dict(state_dict, strict=False)
    print(f'Load PTH model from {args.pth_model}')

    image_processor = cfg.image_processor
    image_processor_type = image_processor['type']
    del image_processor['type']
    image_processor = image_processor_type(**image_processor)

    # build llm
    quantization_config = None
    load_in_8bit = False
    if args.bits == 4:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')
    elif args.bits == 8:
        load_in_8bit = True
    model_kwargs = {
        'quantization_config': quantization_config,
        'load_in_8bit': load_in_8bit,
        'device_map': 'auto',
        'offload_folder': args.offload_folder,
        'trust_remote_code': True,
        'torch_dtype': TORCH_DTYPE_MAP[args.torch_dtype]
    }

    inner_thoughts_open = False
    calculate_open = False
    solve_open = False
    search_open = False

    # build llm
    llm = model.llm
    tokenizer = model.tokenizer

    model.cuda()
    model.eval()
    llm.eval()
    visual_encoder = model.visual_encoder
    projector = model.projector
    projector_text2vision = model.projector_text2vision

    visual_encoder.eval()
    projector.eval()
    projector_text2vision.eval()

    return model, llm, tokenizer, image_processor, visual_encoder, projector, projector_text2vision

def get_seg_hidden_states(hidden_states, output_ids, seg_id):
    seg_mask = output_ids == seg_id
    n_out = len(seg_mask)
    return hidden_states[-n_out:][seg_mask]

class global_infos:
    inputs = ''
    n_turn = 0
    image_width = 0
    image_height = 0

    image_for_show = None
    pixel_values = None
    panoptic_masks = None

    sx, sy, ex, ey = 0, 0 ,1024, 1024

    point_prompts = []
    box_prompts = []
    mask_prompts = []

if __name__ == "__main__":
    # get parse args and set models
    args = parse_args(sys.argv[1:])

    omg_llava, llm, tokenizer, image_processor, visual_encoder, projector, projector_text2vision = \
        init_models(args)

    stop_words = args.stop_words
    sep = ''
    if args.prompt_template:
        template = PROMPT_TEMPLATE[args.prompt_template]
        stop_words += template.get('STOP_WORDS', [])
        sep = template.get('SEP', '')
    stop_criteria = get_stop_criteria(
        tokenizer=tokenizer, stop_words=stop_words)

    if args.no_streamer:
        streamer = None
    else:
        streamer = TextStreamer(tokenizer, skip_prompt=True)

    gen_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id
        if tokenizer.pad_token_id is not None else tokenizer.eos_token_id,
    )

    demo = gr.Interface(
        inference, inputs=[gr.Textbox(lines=1, placeholder=None, label="Text Instruction"), ImagePrompter(
            type='filepath', label='Input Image (Please click points or draw bboxes)', interactive=True,
            elem_id='image_upload', height=360, visible=True, render=True
            ),
            gr.Checkbox(label="Follow up Question")],
        outputs=[
            # gr.Image(type="pil", label="Panoptic Segmentation", height=360),
            gr.Image(type="pil", label="Output Image"),
            gr.Markdown()],
        theme=gr.themes.Soft(), allow_flagging="auto", description=description,
        title='OMG-LLaVA'
    )

    demo.queue()
    demo.launch(share=True)