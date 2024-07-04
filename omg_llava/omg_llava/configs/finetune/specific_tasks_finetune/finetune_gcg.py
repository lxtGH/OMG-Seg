# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmengine.hooks import (CheckpointHook, DistSamplerSeedHook, IterTimerHook,
                            LoggerHook, ParamSchedulerHook)
from mmengine.optim import AmpOptimWrapper, CosineAnnealingLR, LinearLR
from peft import LoraConfig
from torch.optim import AdamW
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel)

from omg_llava.dataset import LLaVADataset
from omg_llava.dataset.collect_fns import omg_llava_collate_fn
from xtuner.dataset.map_fns import llava_map_fn, template_map_fn_factory
from omg_llava.dataset import GranDfGCGDataset, FlickrGCGDataset, OpenPsgGCGDataset, RefCOCOgGCGDataset,\
    CombineDataset, glamm_refcocog_map_fn, glamm_openpsg_map_fn, glamm_flickr_map_fn, glamm_granf_map_fn,\
    ADE20kSemanticSegDataset, COCOStuffSemanticSegDataset, semantic_seg_map_fn, MapillarySemanticSegDataset,\
    PascalPartSemanticSegDataset, pascal_part_map_fn, PacoSemanticSegDataset,\
    RefcocoReferringSegDataset, referring_seg_map_fn, Refcoco_plus_ReferringSegDataset,\
    Refcocog_ReferringSegDataset, Refclef_ReferringSegDataset,\
    OspreyRegionCaptionDataset, osprey_region_caption_map_fn,\
    OspreyRegionConversationDataset, osprey_region_conversation_map_fn,\
    MDPVPointDetailedCaptionDataset, mdpv_points_map_fn, MDPVPointBriefCaptionDataset,\
    semantic_seg_gcg_format_map_fn, pascal_part_gcg_format_map_fn,\
    referring_seg_gcg_format_map_fn, osprey_region_caption_gcg_format_map_fn
from xtuner.dataset.samplers import LengthGroupedSampler
from omg_llava.engine import DatasetInfoHook_withSpecoalTokens, EvaluateChatHook_withSpecialTokens
from xtuner.engine.runner import TrainLoop
from omg_llava.model import OMG_LLaVA
from xtuner.utils import PROMPT_TEMPLATE
from omg_llava.model import OpenCLIPBackbone_omgseg
from omg_llava.model import OMGSegVisualEncoder, Mask2FormerVideoSemSamHead

from torch.nn import GroupNorm, ReLU

from mmdet.models import BatchFixedSizePad, MSDeformAttnPixelDecoder, CrossEntropyLoss, \
    DiceLoss, MaskFormerFusionHead, FocalLoss
from mmdet.models.task_modules.assigners import HungarianAssigner, CrossEntropyLossCost, DiceCost
from mmdet.models.task_modules.samplers import MaskPseudoSampler

#######################################################################
#                          PART 1  Settings                           #
#######################################################################
# Model
llm_name_or_path = './pretrained/omg_llava/internlm2-chat-7b'  # Please change to your own path
pretrained_pth = './work_dirs/omg_llava_7b_finetune_8gpus.pth'
omg_ov_class_embed_path='./pretrained/omg_llava/convnext_large_d_320_CocoPanopticOVDataset.pth' # Please change to your own path
omg_head_pretrain_pth_path = './pretrained/omg_llava/omg_seg_convl.pth'  # Please change to your own path

# Data
data_root = './data/llava_data/'
data_path = data_root + 'LLaVA-Instruct-150K/llava_v1_5_mix665k.json'
image_folder = data_root + 'llava_images'

glamm_data_root = './data/glamm_data/'

refcocog_image_path = glamm_data_root + 'images/coco2014/train2014/'
refcocog_ann_file = glamm_data_root + 'annotations/RefCOCOg_GCG_train.json'

grandf_image_path = glamm_data_root + 'images/grandf/train/'
grandf_ann_file = glamm_data_root + 'annotations/GranDf_HA_GCG_train.json'

flickr_image_path = glamm_data_root + 'images/flickr30k/Flickr30K/'
flickr_ann_file = glamm_data_root + 'annotations/flickr_mergedGT_GCG_train.json'

psg_image_path = glamm_data_root + 'images/coco2017/'
psg_ann_file = glamm_data_root + 'annotations/OpenPsgGCG_train.json'

ade20k_image_path = './data/semantic_seg/ADEChallengeData2016/images/training/'
ade20k_class_file = './omg_llava/dataset/utils/ade20k_classes.json'

cocostuff_image_path = './data/glamm_data/images/coco2017/train2017/'
cocostuff_class_file = './omg_llava/dataset/utils/cocostuff_classes.txt'
cocostuff_label_path = './data/semantic_seg/coco_stuff/stuffthingmaps_trainval2017/train2017/'

mapillary_image_path = './data/semantic_seg/mapillary/training/images/'
mapillary_class_file = './data/semantic_seg/mapillary/config_v2.0.json'
mapillary_label_path = './data/semantic_seg/mapillary/training/v2.0/labels/'

pascal_part_image_path = './data/semantic_seg/pascal_part/VOCdevkit/VOC2010/JPEGImages/'
pascal_file = './data/semantic_seg/pascal_part/train.json'

paco_image_path = './data/glamm_data/images/coco2017/'
paco_file = './data/semantic_seg/paco_lvis/paco_lvis_v1_train.json'

referring_refcoco_image_path = refcocog_image_path
referring_refcoco_data_path = "./data/ref_seg/"

referring_refcoco_plus_image_path = refcocog_image_path
referring_refcoco_plus_data_path = "./data/ref_seg/"

referring_refcocog_image_path = refcocog_image_path
referring_refcocog_data_path = "./data/ref_seg/"

referring_refclef_image_path = "./data/ref_seg/saiapr_tc-12/"
referring_refclef_data_path = "./data/ref_seg/"

region_cap_osprey_image_path = glamm_data_root + 'images/coco2014/train2014/'
region_cap_osprey_data_path = "./data/region_caption/osprey/osprey_detail_description.json"

region_conversation_osprey_image_path = glamm_data_root + 'images/coco2014/train2014/'
region_conversation_osprey_data_path = "./data/region_caption/osprey/osprey_conversation.json"

mdpv_detailed_caption_ade20k_image_path = './data/semantic_seg/ADEChallengeData2016/images/training/'
mdpv_detailed_caption_ade20k_data_path = './data/mdpv_point/gpt4v_ade20k_detailed_caption_point.json'

mdpv_detailed_caption_cocostuff_10k_image_path = glamm_data_root + 'images/coco2014/train2014/'
mdpv_detailed_caption_cocostuff_10k_data_path = './data/mdpv_point/gpt4v_cocostuff_10k_detailed_caption_point.json'

mdpv_detailed_caption_cocostuff_164k_image_path = './data/glamm_data/images/coco2017/train2017'
mdpv_detailed_caption_cocostuff_164k_data_path = './data/mdpv_point/gpt4v_cocostuff_164k_detailed_caption_point.json'

mdpv_detailed_caption_vg_image_path = './data/llava_data/llava_images/vg/VG_100K'
mdpv_detailed_caption_vg_data_path = './data/mdpv_point/gpt4v_vg_detailed_caption_point.json'

mdpv_brief_caption_cocostuff_10k_image_path = glamm_data_root + 'images/coco2014/train2014/'
mdpv_brief_caption_cocostuff_10k_data_path = './data/mdpv_point/gpt4v_cocostuff_10k_brief_caption_point.json'

mdpv_brief_caption_ade20k_image_path = './data/semantic_seg/ADEChallengeData2016/images/training/'
mdpv_brief_caption_ade20k_data_path = './data/mdpv_point/gpt4v_ade20k_brief_caption_point.json'

mdpv_brief_caption_cocostuff_164k_image_path = './data/glamm_data/images/coco2017/train2017'
mdpv_brief_caption_cocostuff_164k_data_path = './data/mdpv_point/gpt4v_cocostuff_164k_brief_caption_point.json'

mdpv_brief_caption_vg_image_path = './data/llava_data/llava_images/vg/VG_100K'
mdpv_brief_caption_vg_data_path = './data/mdpv_point/gpt4v_vg_brief_caption_point.json'

mdpv_brief_caption_lvis_image_path = './data/glamm_data/images/coco2017/train2017'
mdpv_brief_caption_lvis_data_path = './data/mdpv_point/gpt4v_lvis_brief_caption_point.json'

mdpv_qa_vg_image_path = './data/llava_data/llava_images/vg/VG_100K'
mdpv_qa_vg_data_path = './data/mdpv_point/gpt4v_vg_QA_point.json'

mdpv_qa_ade20k_image_path = './data/semantic_seg/ADEChallengeData2016/images/training/'
mdpv_qa_ade20k_data_path = './data/mdpv_point/gpt4v_ade20k_QA_point.json'

mdpv_qa_cocostuff164k_image_path = './data/glamm_data/images/coco2017/train2017'
mdpv_qa_cocostuff164k_data_path = './data/mdpv_point/gpt4v_cocostuff_164k_QA_point.json'

mdpv_qa_lvis_image_path = './data/glamm_data/images/coco2017/train2017'
mdpv_qa_lvis_data_path = './data/mdpv_point/gpt4v_lvis_QA_point.json'

mdpv_qa_cocostuff10k_image_path = glamm_data_root + 'images/coco2014/train2014/'
mdpv_qa_cocostuff10k_data_path = './data/mdpv_point/gpt4v_cocostuff_10k_QA_point.json'

mdpv_multi_points_flicker30k_image_path = './data/glamm_data/images/flickr30k/Flickr30K/'
mdpv_multi_points_flicker30k_data_path = './data/mdpv_point/Flicker30K_multi_points_to_caption.json'

mdpv_multi_points_openpsg_image_path = glamm_data_root + 'images/coco2017/train2017'
mdpv_multi_points_openpsg_data_path = './data/mdpv_point/OpenPsgGCG_train_multi_points_to_caption.json'

prompt_template = PROMPT_TEMPLATE.internlm2_chat
max_length = int(2048 - (1024 / 64)**2 - 100)

# Scheduler & Optimizer
batch_size = 8  # per_device
accumulative_counts = 2
dataloader_num_workers = 4
max_epochs = 1
optim_type = AdamW
lr = 2e-4
betas = (0.9, 0.999)
weight_decay = 0
max_norm = 1  # grad clip
warmup_ratio = 0.03


# Save
save_steps = 2000
save_total_limit = 4  # Maximum checkpoints to keep (-1 means unlimited)

# Evaluate the generation performance during the training
evaluation_freq = 2000
SYSTEM = ''
evaluation_images = './work_dirs/test.jpg'
evaluation_inputs = ['请描述一下这张照片', 'Please describe this picture',
                     'Could you please give me a detailed description of the image? Please respond with interleaved segmentation masks for the corresponding parts of the answer.']

#######################################################################
#            PART 2  Model & Tokenizer & Image Processor              #
#######################################################################
tokenizer = dict(
    type=AutoTokenizer.from_pretrained,
    pretrained_model_name_or_path=llm_name_or_path,
    trust_remote_code=True,
    padding_side='right')

image_processor = dict(
    type=CLIPImageProcessor,
    do_resize=True,
    size=1024,
    resample=3,
    do_center_crop=True,
    crop_size=1024,
    do_rescale=True,
    do_normalize=True,
    image_mean=[0.4814, 0.4578, 0.4082],
    image_std=[0.2686, 0.2613, 0.2757],
    do_convert_rgb=True
)

class_embed = 'convnext_large_d_320_CocoPanopticOVDataset'
num_things_classes = 80
num_stuff_classes = 53
num_classes = num_things_classes + num_stuff_classes

omgseg_model = dict(
    type=OMGSegVisualEncoder,
    data_preprocessor=None,
    pixel_shuffle_down_ratio=2,
    backbone=dict(
        type=OpenCLIPBackbone_omgseg,
        model_name='convnext_large_d_320',
        fix=True,
        init_cfg=dict(
            type='clip_pretrain',
            checkpoint='laion2b_s29b_b131k_ft_soup'
        )
    ),
    panoptic_head=dict(
        type=Mask2FormerVideoSemSamHead,
        sphere_cls=True,
        ov_path=omg_ov_class_embed_path,
        enable_box_query=False,
        ov_classifier_name=class_embed,
        logit=None,
        in_channels=[192, 384, 768, 1536],  # pass to pixel_decoder inside
        strides=[4, 8, 16, 32],
        feat_channels=256,
        out_channels=256,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        num_queries=300,
        num_transformer_feat_level=3,
        pixel_decoder=dict(
            type=MSDeformAttnPixelDecoder,
            num_outs=3,
            norm_cfg=dict(type=GroupNorm, num_groups=32),
            act_cfg=dict(type=ReLU),
            encoder=dict(  # DeformableDetrTransformerEncoder
                num_layers=6,
                layer_cfg=dict(  # DeformableDetrTransformerEncoderLayer
                    self_attn_cfg=dict(  # MultiScaleDeformableAttention
                        embed_dims=256,
                        num_heads=8,
                        num_levels=3,
                        num_points=4,
                        dropout=0.0,
                        batch_first=True),
                    ffn_cfg=dict(
                        embed_dims=256,
                        feedforward_channels=1024,
                        num_fcs=2,
                        ffn_drop=0.0,
                        act_cfg=dict(type=ReLU, inplace=True)))),
        positional_encoding=dict(num_feats=128, normalize=True)),
        enforce_decoder_input_project=False,
        positional_encoding=dict(num_feats=128, normalize=True),
        transformer_decoder=dict(  # Mask2FormerTransformerDecoder
            return_intermediate=True,
            num_layers=9,
            layer_cfg=dict(  # Mask2FormerTransformerDecoderLayer
                self_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                cross_attn_cfg=dict(  # MultiheadAttention
                    embed_dims=256,
                    num_heads=8,
                    dropout=0.0,
                    batch_first=True),
                ffn_cfg=dict(
                    embed_dims=256,
                    feedforward_channels=2048,
                    num_fcs=2,
                    ffn_drop=0.0,
                    act_cfg=dict(type='ReLU', inplace=True))),
            init_cfg=None),
        loss_cls=dict(
            type=CrossEntropyLoss,
            use_sigmoid=False,
            loss_weight=2.0,
            reduction='mean',
            class_weight=[1.0] * 240 + [0.1]),
        loss_mask=dict(
            type=CrossEntropyLoss,
            use_sigmoid=True,
            reduction='mean',
            loss_weight=5.0),
        loss_dice=dict(
            type=DiceLoss,
            use_sigmoid=True,
            activate=True,
            reduction='mean',
            naive_dice=True,
            eps=1.0,
            loss_weight=5.0),
        loss_iou=dict(
            type=FocalLoss,
            use_sigmoid=True,
            loss_weight=2.0,
            reduction='mean')
    ),
    panoptic_fusion_head=dict(
        type=MaskFormerFusionHead,
        num_things_classes=num_things_classes,
        num_stuff_classes=num_stuff_classes,
        loss_panoptic=None,
        init_cfg=None),
    train_cfg=dict(
        num_points=12544,
        oversample_ratio=3.0,
        importance_sample_ratio=0.75,
        assigner=dict(
            type=HungarianAssigner,
            match_costs=[
                # dict(type=FlexibleClassificationCost, weight=2.0),
                dict(type=CrossEntropyLossCost, weight=5.0, use_sigmoid=True),
                dict(type=DiceCost, weight=5.0, pred_act=True, eps=1.0)
            ]),
        sampler=dict(type=MaskPseudoSampler)),
    test_cfg=dict(
        panoptic_on=True,
        # For now, the dataset does not support
        # evaluating semantic segmentation metric.
        semantic_on=False,
        instance_on=True,
        # max_per_image is for instance segmentation.
        max_per_image=100,
        iou_thr=0.8,
        # In Mask2Former's panoptic postprocessing,
        # it will filter mask area where score is less than 0.5 .
        filter_low_score=True),
    init_cfg=dict(
        type='Pretrained',
        checkpoint=omg_head_pretrain_pth_path,
    )
)

model = dict(
    type=OMG_LLaVA,
    freeze_llm=True,
    freeze_visual_encoder=True,
    require_omg_decoder=True,
    pretrained_pth=pretrained_pth,
    text2vision_projector=True,
    pixel_shuffle_ratio=2,
    llm=dict(
        type=AutoModelForCausalLM.from_pretrained,
        pretrained_model_name_or_path=llm_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        quantization_config=dict(
            type=BitsAndBytesConfig,
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4')),
    llm_lora=dict(
        type=LoraConfig,
        r=512,
        lora_alpha=256,
        lora_dropout=0.05,
        bias='none',
        task_type='CAUSAL_LM'),
    visual_encoder=omgseg_model,
    tokenizer=tokenizer,
)

#######################################################################
#                      PART 3  Dataset & Dataloader                   #
#######################################################################
debug=False
llava_dataset = dict(
    type=LLaVADataset,
    data_path=data_path,
    image_folder=image_folder,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=llava_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True)

glamm_refcocog_dataset = dict(
    type=RefCOCOgGCGDataset,
    data_path=refcocog_ann_file,
    image_folder=refcocog_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=glamm_refcocog_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

glamm_grandf_dataset = dict(
    type=GranDfGCGDataset,
    data_path=grandf_ann_file,
    image_folder=grandf_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=glamm_granf_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=debug,
    repeats=10,
)

glamm_psg_dataset = dict(
    type=OpenPsgGCGDataset,
    data_path=psg_ann_file,
    image_folder=psg_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=glamm_openpsg_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=debug,
    repeats=1,
)

glamm_flickr_dataset = dict(
    type=FlickrGCGDataset,
    data_path=flickr_ann_file,
    image_folder=flickr_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=glamm_flickr_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=debug,
    repeats=1,
)

semantic_seg_ade20k_dataset = dict(
    type=ADE20kSemanticSegDataset,
    data_path=ade20k_class_file,
    image_folder=ade20k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=semantic_seg_gcg_format_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
    gcg_format=True,
)

semantic_seg_cocostuff_dataset = dict(
    type=COCOStuffSemanticSegDataset,
    data_path=cocostuff_class_file,
    image_folder=cocostuff_image_path,
    label_path=cocostuff_label_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=semantic_seg_gcg_format_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
    gcg_format=True,
)

referring_seg_refcoco_dataset = dict(
    type=RefcocoReferringSegDataset,
    data_path=referring_refcoco_data_path,
    image_folder=referring_refcoco_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=referring_seg_gcg_format_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

referring_seg_refcoco_plus_dataset = dict(
    type=Refcoco_plus_ReferringSegDataset,
    data_path=referring_refcoco_plus_data_path,
    image_folder=referring_refcoco_plus_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=referring_seg_gcg_format_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

referring_seg_refcocog_dataset = dict(
    type=Refcocog_ReferringSegDataset,
    data_path=referring_refcocog_data_path,
    image_folder=referring_refcocog_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=referring_seg_gcg_format_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

referring_seg_refclef_dataset = dict(
    type=Refclef_ReferringSegDataset,
    data_path=referring_refclef_data_path,
    image_folder=referring_refclef_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=referring_seg_gcg_format_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

region_cap_osprey_dataset = dict(
    type=OspreyRegionCaptionDataset,
    data_path=region_cap_osprey_data_path,
    image_folder=region_cap_osprey_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=osprey_region_caption_gcg_format_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

region_conversation_osprey_dataset = dict(
    type=OspreyRegionConversationDataset,
    data_path=region_conversation_osprey_data_path,
    image_folder=region_conversation_osprey_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=osprey_region_conversation_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_detailed_description_ade20k_dataset = dict(
    type=MDPVPointDetailedCaptionDataset,
    data_path=mdpv_detailed_caption_ade20k_data_path,
    image_folder=mdpv_detailed_caption_ade20k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_detailed_description_cocostuff_10k_dataset = dict(
    type=MDPVPointDetailedCaptionDataset,
    data_path=mdpv_detailed_caption_cocostuff_10k_data_path,
    image_folder=mdpv_detailed_caption_cocostuff_10k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_detailed_description_cocostuff_164k_dataset = dict(
    type=MDPVPointDetailedCaptionDataset,
    data_path=mdpv_detailed_caption_cocostuff_164k_data_path,
    image_folder=mdpv_detailed_caption_cocostuff_164k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_detailed_description_vg_dataset = dict(
    type=MDPVPointDetailedCaptionDataset,
    data_path=mdpv_detailed_caption_vg_data_path,
    image_folder=mdpv_detailed_caption_vg_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_brief_description_vg_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_brief_caption_vg_data_path,
    image_folder=mdpv_brief_caption_vg_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_brief_description_cocostuff10k_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_brief_caption_cocostuff_10k_data_path,
    image_folder=mdpv_brief_caption_cocostuff_10k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_brief_description_cocostuff164k_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_brief_caption_cocostuff_164k_data_path,
    image_folder=mdpv_brief_caption_cocostuff_164k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_brief_description_ade20k_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_brief_caption_ade20k_data_path,
    image_folder=mdpv_brief_caption_ade20k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_brief_description_lvis_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_brief_caption_lvis_data_path,
    image_folder=mdpv_brief_caption_lvis_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_qa_vg_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_qa_vg_data_path,
    image_folder=mdpv_qa_vg_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_qa_ade20k_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_qa_ade20k_data_path,
    image_folder=mdpv_qa_ade20k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_qa_lvis_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_qa_lvis_data_path,
    image_folder=mdpv_qa_lvis_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_qa_cocostuff10k_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_qa_cocostuff10k_data_path,
    image_folder=mdpv_qa_cocostuff10k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_qa_cocostuff164k_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_qa_cocostuff164k_data_path,
    image_folder=mdpv_qa_cocostuff164k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_multi_points_openpsg_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_multi_points_openpsg_data_path,
    image_folder=mdpv_multi_points_openpsg_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

mdpv_multi_points_flicker30k_dataset = dict(
    type=MDPVPointBriefCaptionDataset,
    data_path=mdpv_multi_points_flicker30k_data_path,
    image_folder=mdpv_multi_points_flicker30k_image_path,
    tokenizer=tokenizer,
    image_processor=image_processor,
    dataset_map_fn=mdpv_points_map_fn,
    template_map_fn=dict(
        type=template_map_fn_factory, template=prompt_template),
    max_length=max_length,
    pad_image_to_square=True,
    debug=False,
    repeats=1,
)

train_dataset = dict(
    type=CombineDataset,
    datasets_cfgs=[glamm_flickr_dataset, glamm_refcocog_dataset,
                   glamm_grandf_dataset, glamm_psg_dataset,],
)

train_dataloader = dict(
    batch_size=batch_size,
    num_workers=dataloader_num_workers,
    dataset=train_dataset,
    sampler=dict(
        type=LengthGroupedSampler,
        length_property='modality_length',
        per_device_batch_size=batch_size * accumulative_counts),
    collate_fn=dict(type=omg_llava_collate_fn))

#######################################################################
#                    PART 4  Scheduler & Optimizer                    #
#######################################################################
# optimizer
optim_wrapper = dict(
    type=AmpOptimWrapper,
    optimizer=dict(
        type=optim_type, lr=lr, betas=betas, weight_decay=weight_decay),
    clip_grad=dict(max_norm=max_norm, error_if_nonfinite=False),
    accumulative_counts=accumulative_counts,
    loss_scale='dynamic',
    dtype='float16')

# learning policy
# More information: https://github.com/open-mmlab/mmengine/blob/main/docs/en/tutorials/param_scheduler.md  # noqa: E501
param_scheduler = [
    dict(
        type=LinearLR,
        start_factor=1e-5,
        by_epoch=True,
        begin=0,
        end=warmup_ratio * max_epochs,
        convert_to_iter_based=True),
    dict(
        type=CosineAnnealingLR,
        eta_min=0.0,
        by_epoch=True,
        begin=warmup_ratio * max_epochs,
        end=max_epochs,
        convert_to_iter_based=True)
]

# train, val, test setting
train_cfg = dict(type=TrainLoop, max_epochs=max_epochs)

#######################################################################
#                           PART 5  Runtime                           #
#######################################################################
# Log the dialogue periodically during the training process, optional
custom_hooks = [
    dict(type=DatasetInfoHook_withSpecoalTokens, tokenizer=tokenizer),
    dict(
        type=EvaluateChatHook_withSpecialTokens,
        tokenizer=tokenizer,
        image_processor=image_processor,
        every_n_iters=evaluation_freq,
        evaluation_inputs=evaluation_inputs,
        evaluation_images=evaluation_images,
        system=SYSTEM,
        prompt_template=prompt_template)
]

# configure default hooks
default_hooks = dict(
    # record the time of every iteration.
    timer=dict(type=IterTimerHook),
    # print log every 10 iterations.
    logger=dict(type=LoggerHook, log_metric_by_epoch=False, interval=10),
    # enable the parameter scheduler.
    param_scheduler=dict(type=ParamSchedulerHook),
    # save checkpoint per `save_steps`.
    checkpoint=dict(
        type=CheckpointHook,
        by_epoch=False,
        interval=save_steps,
        max_keep_ckpts=save_total_limit),
    # set sampler seed in distributed evrionment.
    sampler_seed=dict(type=DistSamplerSeedHook),
)

# configure environment
env_cfg = dict(
    # whether to enable cudnn benchmark
    cudnn_benchmark=False,
    # set multi process parameters
    mp_cfg=dict(mp_start_method='fork', opencv_num_threads=0),
    # set distributed parameters
    dist_cfg=dict(backend='nccl'),
)

# set visualizer
visualizer = None

# set log level
log_level = 'INFO'

# load from which checkpoint
load_from = None

# whether to resume training from the loaded checkpoint
resume = False

# Defaults to use random seed and disable `deterministic`
randomness = dict(seed=None, deterministic=False)

# set log processor
log_processor = dict(by_epoch=False)
