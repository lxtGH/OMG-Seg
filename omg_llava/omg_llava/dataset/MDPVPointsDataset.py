import json
import logging
import os
import torch
from datasets import Dataset as HFDataset
from datasets import DatasetDict, load_from_disk
from mmengine import print_log
from mmengine.config import Config, ConfigDict
from PIL import Image
from torch.utils.data import Dataset
from pycocotools import mask
import numpy as np
import torch.nn.functional as F

from xtuner.registry import BUILDER
from omg_llava.dataset.utils import expand2square, expand2square_mask, expand2square_points
from xtuner.dataset.huggingface import process_hf_dataset
import copy

class MDPVPointDetailedCaptionDataset(Dataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 num_proc=32,
                 debug=False,
                 repeats=1):
        super().__init__()

        assert offline_processed_text_folder or (data_path and tokenizer)
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
            json_data = self.json_file_preprocess(data_path)
            self.json_data = json_data
            hf_json_data = self.filter_hf_require_infos(json_data)
            hf_json_data = DatasetDict({'train': HFDataset.from_list(hf_json_data)})
            self.text_data = process_hf_dataset(
                dataset=hf_json_data,
                tokenizer=tokenizer,
                max_length=max_length,
                dataset_map_fn=dataset_map_fn,
                template_map_fn=template_map_fn,
                split='train',
                max_dataset_length=max_dataset_length,
                remove_unused_columns=False,
                pack_to_max_length=False,
                with_image_token=True,
                map_num_proc=num_proc,  # because limited mem
            )

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

    def filter_hf_require_infos(self, dataset_infos):
        ret = []
        for dataset_info in dataset_infos:
            conversations = dataset_info["conversations"]
            image = dataset_info['image'].split('/')[-1]
            num_marks = len(dataset_info['points'])
            required_info = {'image': image,
                             'conversations': conversations,
                             'num_marks': num_marks}
            ret.append(required_info)
        return ret

    def json_file_preprocess(self, data_path):
        with open(data_path, 'r') as f:
            json_file = json.load(f)
        if self.debug:
            json_file = json_file[:10000]
        return json_file

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.text_data:
            cur_len = len(data_dict['input_ids'])
            if data_dict.get('image', None) is None:
                cur_len = -cur_len
            length_list.append(cur_len)
        length_list = length_list * self.repeats
        return length_list

    def __len__(self):
        return len(self.text_data) * self.repeats

    def real_len(self):
        return len(self.text_data)

    def decode_mask(self, object_masks, ori_height, ori_width):
        binary_masks = []
        for object_mask in object_masks:
            binary_mask = np.zeros((ori_height, ori_width), dtype=np.uint8)
            for seg in object_mask:
                rles = mask.frPyObjects([seg], ori_height, ori_width)
                m = mask.decode(rles)
                m = m.astype(np.uint8)
                binary_mask += m.squeeze()

            binary_masks.append(binary_mask)
        if len(binary_masks) == 0:
            return None
        masks = np.stack(binary_masks, axis=0)
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)
        masks = torch.from_numpy(masks)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio, self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.json_data[index])
        data_dict.update(self.text_data[index])

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
            image_path = os.path.join(self.image_folder, image_file)
            if not os.path.exists(image_path) and "VG" in self.image_folder:
                image_path =  os.path.join(self.image_folder + "_2", image_file)
            image = Image.open(image_path).convert('RGB')
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
            points = data_dict["points"]
            points = np.array(points)
            if self.pad_image_to_square:
                points = expand2square_points(points, height=ori_height, width=ori_width)
                points[:, 0] = points[:, 0] / max(ori_height, ori_width) * self.image_w
                points[:, 1] = points[:, 1] / max(ori_height, ori_width) * self.image_h
            else:
                points[:, 0] = points[:, 0] / ori_width * self.image_w
                points[:, 1] = points[:, 1] / ori_height * self.image_h
            data_dict['points'] = torch.from_numpy(points)
            if data_dict['points'] is None:
                return self.__getitem__(0)
            data_dict['masks'] = None
            data_dict['regions'] = None
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['masks'] = None
            data_dict['regions'] = None
            data_dict['points'] = None
        return data_dict

class MDPVPointBriefCaptionDataset(MDPVPointDetailedCaptionDataset):
    def __init__(self,
                 image_folder,
                 image_processor,
                 data_path=None,
                 tokenizer=None,
                 offline_processed_text_folder=None,
                 max_dataset_length=None,
                 dataset_map_fn=None,
                 template_map_fn=None,
                 max_length=2048,
                 pad_image_to_square=False,
                 num_proc=32,
                 debug=False,
                 repeats=1):
        super().__init__(
            image_folder=image_folder,
            image_processor=image_processor,
            data_path=data_path,
            tokenizer=tokenizer,
            offline_processed_text_folder=offline_processed_text_folder,
            max_dataset_length=max_dataset_length,
            dataset_map_fn=dataset_map_fn,
            template_map_fn=template_map_fn,
            max_length=max_length,
            pad_image_to_square=pad_image_to_square,
            num_proc=num_proc,
            debug=debug,
            repeats=repeats
            )
