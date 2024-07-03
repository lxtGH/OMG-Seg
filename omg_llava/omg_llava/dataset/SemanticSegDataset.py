import random
import glob
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
import numpy as np
import torch.nn.functional as F
from pycocotools.coco import COCO

from xtuner.registry import BUILDER
from omg_llava.dataset.utils import expand2square, expand2square_mask
from xtuner.dataset.huggingface import process_hf_dataset
from omg_llava.dataset.process_functions.semantic_seg_process import semantic_seg_conversations, semantic_seg_gcg_format_conversations
import copy

class SemanticSegDataset(Dataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 gcg_format=False):
        super().__init__()
        self.tokenizer = tokenizer
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
            self.image_label_datas = self.json_file_preprocess(data_path, image_folder)
            if gcg_format:
                conversations_datas = semantic_seg_gcg_format_conversations(self.classes)
            else:
                conversations_datas = semantic_seg_conversations(self.classes)
            json_data = DatasetDict({'train': HFDataset.from_list(conversations_datas)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
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

        self.clsid2convs = self.construct_cls2convs_dict()
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

    def construct_cls2convs_dict(self):
        ret = {}
        for conv_item in self.text_data:
            cls_id = conv_item['class_id']
            if cls_id in ret.keys():
                ret[cls_id].append(conv_item)
            else:
                ret[cls_id] = [conv_item]
        return ret

    def json_file_preprocess(self, data_path, image_folder):
        # ade20k
        with open(data_path, 'r') as file:
            ade20k_classes = json.load(file)
        ade20k_image_dir = image_folder
        ade20k_images = [os.path.join(ade20k_image_dir, img) for img in os.listdir(ade20k_image_dir) if
                         img.endswith('.jpg')]
        ade20k_labels = [img.replace(".jpg", ".png").replace("images", "annotations") for img in ade20k_images]
        self.classes = np.array(ade20k_classes)

        ret = []
        for image, label in zip(ade20k_images, ade20k_labels):
            ret.append({"image": image, "label": label})
        if self.debug:
            return ret[:1000]
        return ret

    def __len__(self):
        return len(self.image_label_datas) * self.repeats

    @property
    def modality_length(self):
        length_list = []
        for data_dict in self.image_label_datas:
            length_list.append(-100)
        length_list = length_list * self.repeats
        return length_list

    def real_len(self):
        return len(self.image_label_datas)

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))

        # ade 20k
        label = np.where(label == 0, 255, label - 1)
        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            return None, None

        # only choose 1
        selected_labels = np.random.choice(
            unique_labels, 1, replace=False
        )
        label = torch.from_numpy(label).long()
        masks = torch.stack([label == class_id for class_id in selected_labels], dim=0)

        masks = masks.numpy()
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)

        masks = torch.from_numpy(masks).to(torch.float32)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio,
                                                        self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks, selected_labels[0]

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.image_label_datas[index])

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
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
            data_dict['masks'], class_id = self.decode_mask(data_dict['label'])
            if class_id is None:
                return self.__getitem__(0)
            conv_datas = self.clsid2convs[class_id]
            selected_idx = np.random.randint(0, len(conv_datas))
            data_dict.update(conv_datas[selected_idx])
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['masks'] = None
        return data_dict

class ADE20kSemanticSegDataset(SemanticSegDataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 gcg_format=False):
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
            repeats=repeats,
            gcg_format=gcg_format,
        )

class COCOStuffSemanticSegDataset(SemanticSegDataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 label_path=None,
                 gcg_format=False,):
        self.label_path = label_path
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
            repeats=repeats,
            gcg_format=gcg_format,
        )
        self.cocostuff_class2index = {c: i for i, c in enumerate(self.classes)}

    def json_file_preprocess(self, data_path, image_folder):
        # coco stuff
        assert self.label_path is not None
        with open(data_path, 'r') as file:
            cocostuff_classes = [line.strip().split(": ")[-1] for line in file.readlines()[1:]]
        coco_stuff_image_dir = image_folder
        coco_stuff_label_dir = self.label_path
        coco_stuff_labels = glob.glob(os.path.join(coco_stuff_label_dir, "*.png"))

        coco_stuff_images = [label.replace(".png", ".jpg").replace(coco_stuff_label_dir, coco_stuff_image_dir)
            for label in coco_stuff_labels]

        self.classes = np.array(cocostuff_classes)

        ret = []
        for image, label in zip(coco_stuff_images, coco_stuff_labels):
            ret.append({"image": image, "label": label})
        if self.debug:
            return ret[:1000]
        return ret

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))

        # coco stuff
        ignored_classes = [index for class_name, index in self.cocostuff_class2index.items() if
                           "-" in class_name]
        label = np.where(np.isin(label, ignored_classes), 255, label)

        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            print("No valid label !!!")
            return None, None

        # only choose 1
        selected_labels = np.random.choice(
            unique_labels, 1, replace=False
        )
        label = torch.from_numpy(label).long()
        masks = torch.stack([label == class_id for class_id in selected_labels], dim=0)

        masks = masks.numpy()
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)

        masks = torch.from_numpy(masks).to(torch.float32)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio,
                                                        self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks, selected_labels[0]

class MapillarySemanticSegDataset(SemanticSegDataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,
                 label_path=None,
                 gcg_format=False,):
        self.label_path = label_path
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
            repeats=repeats,
            gcg_format=gcg_format,
        )

    def json_file_preprocess(self, data_path, image_folder):
        assert self.label_path is not None
        # mapillary
        with open(data_path, 'r') as file:
            mapillary_classes = json.load(file)["labels"]
        mapillary_classes = [cls["readable"].lower() for cls in mapillary_classes]

        mapillary_labels = sorted(
            glob.glob(os.path.join(self.label_path, "*.png")))
        mapillary_images = [
            label.replace(".png", ".jpg").replace(self.label_path, image_folder)
            for label in mapillary_labels]

        self.classes = np.array(mapillary_classes)

        ret = []
        for image, label in zip(mapillary_images, mapillary_labels):
            ret.append({"image": image, "label": label})
        if self.debug:
            return ret[:1000]
        return ret

    def decode_mask(self, label_path):
        label = np.array(Image.open(label_path))

        ignored_classes = [index for index, class_name in enumerate(self.classes) if
                           "-" in class_name or '(' in class_name or
                           'unlabeled' in class_name]
        label = np.where(np.isin(label, ignored_classes), 255, label)
        unique_labels = [lbl for lbl in np.unique(label) if lbl != 255]
        if not unique_labels:
            print("No valid label !!!")
            return None, None
        # only choose 1
        selected_labels = np.random.choice(
            unique_labels, 1, replace=False
        )
        label = torch.from_numpy(label).long()
        masks = torch.stack([label == class_id for class_id in selected_labels], dim=0)

        masks = masks.numpy()
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)

        masks = torch.from_numpy(masks).to(torch.float32)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio,
                                                        self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks, selected_labels[0]

class PascalPartSemanticSegDataset(Dataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1):
        super().__init__()
        self.tokenizer = tokenizer
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
            json_datas = self.json_file_preprocess(data_path)
            json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
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

    def json_file_preprocess(self, data_path):
        pascal_part_api = COCO(data_path)
        all_classes = pascal_part_api.loadCats(pascal_part_api.getCatIds())
        class_map_pascal_part = {}
        for cat in all_classes:
            cat_main, cat_part = cat["name"].strip().split(":")
            name = (cat_main, cat_part)
            class_map_pascal_part[cat["id"]] = name
        img_ids = pascal_part_api.getImgIds()
        self.classes = class_map_pascal_part
        self.coco_api = pascal_part_api

        img_infos = [self.coco_api.loadImgs([img_id])[0] for img_id in img_ids]
        valid_img_infos = []
        for img_info in img_infos:
            annotation_ids = self.coco_api.getAnnIds(imgIds=img_info["id"])
            annotations = self.coco_api.loadAnns(annotation_ids)
            if not annotations:
                continue

            # sampled to max number as 5
            sampled_anns = np.random.choice(annotations, 5, replace=False) if len(
                annotations
            ) >= 5 else annotations

            selected_labels = []
            for ann in sampled_anns:
                category_id = ann["category_id"]
                sampled_cls = self.classes[category_id]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
                else:
                    name = sampled_cls
                selected_labels.append(name)

            img_info.update({"annotations": sampled_anns,
                             "selected_labels": selected_labels})
            valid_img_infos.append(img_info)

        if self.debug:
            return valid_img_infos[:1000]
        return valid_img_infos

    def __len__(self):
        return len(self.text_data) * self.repeats

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

    def real_len(self):
        return len(self.text_data)

    def decode_mask(self, annotations):

        try:
            masks = [self.coco_api.annToMask(ann) for ann in annotations]
        except Exception as e:
            print(f"Error generating mask: {e}")
            return None

        masks = np.stack(masks, axis=0)
        if self.pad_image_to_square:
            masks = expand2square_mask(masks)
        masks = torch.from_numpy(masks)
        masks = F.interpolate(masks.unsqueeze(0), size=(self.image_h // self.down_ratio,
                                                        self.image_w // self.down_ratio), mode='nearest').squeeze(0)
        return masks

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
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

            # process and get masks
            data_dict['masks'] = self.decode_mask(data_dict['annotations'])
            if data_dict['masks'] is None:
                return self.__getitem__(0)
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['masks'] = None
        return data_dict

class PacoSemanticSegDataset(PascalPartSemanticSegDataset):
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
                 num_proc=8,
                 debug=False,
                 repeats=1,):
        self.tokenizer = tokenizer
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
            json_datas = self.json_file_preprocess(data_path)
            self.json_datas = json_datas
            json_datas = self.only_get_hf_map_infos()
            json_data = DatasetDict({'train': HFDataset.from_list(json_datas)})
            self.text_data = process_hf_dataset(
                dataset=json_data,
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

    def only_get_hf_map_infos(self):
        ret = []
        for json_data in self.json_datas:
            ret.append({'file_name': json_data['file_name'],
                        'selected_labels': json_data['selected_labels']})
        return ret

    def json_file_preprocess(self, data_path):
        paco_api = COCO(data_path)
        all_classes = paco_api.loadCats(paco_api.getCatIds())
        class_map_paco = {}
        for cat in all_classes:
            cat_split = cat["name"].strip().split(":")
            if len(cat_split) == 1:
                name = cat_split[0].split("_(")[0]
            else:
                assert len(cat_split) == 2
                obj, part = cat_split
                obj = obj.split("_(")[0]
                part = part.split("_(")[0]
                name = (obj, part)
            class_map_paco[cat["id"]] = name

        img_ids = paco_api.getImgIds()
        self.classes = class_map_paco
        self.coco_api = paco_api

        img_infos = [self.coco_api.loadImgs([img_id])[0] for img_id in img_ids]
        valid_img_infos = []
        for img_info in img_infos:
            annotation_ids = self.coco_api.getAnnIds(imgIds=img_info["id"])
            annotations = self.coco_api.loadAnns(annotation_ids)
            if not annotations:
                continue

            # sampled to max number as 5
            sampled_anns = np.random.choice(annotations, 5, replace=False) if len(
                annotations
            ) >= 5 else annotations

            selected_labels = []
            for ann in sampled_anns:
                category_id = ann["category_id"]
                sampled_cls = self.classes[category_id]
                if isinstance(sampled_cls, tuple):
                    obj, part = sampled_cls
                    name = f"{obj} {part}" if random.random() < 0.5 else f"the {part} of the {obj}"
                else:
                    name = sampled_cls
                selected_labels.append(name)

            img_info.update({"annotations": sampled_anns,
                             "selected_labels": selected_labels})
            valid_img_infos.append(img_info)

        if self.debug:
            return valid_img_infos[:1000]
        return valid_img_infos

    def __getitem__(self, index):
        index = index % self.real_len()
        data_dict = copy.deepcopy(self.text_data[index])
        data_dict.update(self.json_datas[index])

        if data_dict.get('image', None) is not None:
            image_file = data_dict['image']
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

            # process and get masks
            data_dict['masks'] = self.decode_mask(data_dict['annotations'])
            if data_dict['masks'] is None:
                return self.__getitem__(0)
        else:
            if hasattr(self.image_processor, 'crop_size'):
                crop_size = self.image_processor.crop_size
            else:
                crop_size = self.image_processor.size
            data_dict['pixel_values'] = torch.zeros(3, crop_size['height'],
                                                    crop_size['width'])
            data_dict['masks'] = None
        return data_dict

