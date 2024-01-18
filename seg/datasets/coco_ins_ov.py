import copy
from typing import List

from mmdet.registry import DATASETS
from mmdet.datasets.coco import CocoDataset
from mmengine import get_local_path, print_log

CLASSES_ORIGINAL = (
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep',
    'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
    'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
    'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
    'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
    'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
    'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
    'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
    'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
    'scissors', 'teddy bear', 'hair drier', 'toothbrush'
)

CLASSES_48 = (
    'person', 'bicycle', 'car', 'motorcycle', 'truck', 'boat', 'bench',
    'bird', 'horse', 'sheep', 'zebra', 'giraffe', 'backpack',
    'handbag', 'skis', 'kite', 'surfboard', 'bottle', 'spoon',
    'bowl', 'banana', 'apple', 'orange', 'broccoli', 'carrot',
    'pizza', 'donut', 'chair', 'bed', 'tv', 'laptop',
    'remote', 'microwave', 'oven', 'refrigerator', 'book',
    'clock', 'vase', 'toothbrush', 'train', 'bear', 'suitcase',
    'frisbee', 'fork', 'sandwich', 'toilet', 'mouse', 'toaster'
)

CLASSES_17 = (
    'bus', 'dog', 'cow', 'elephant', 'umbrella', 'tie',
    'skateboard', 'cup', 'knife', 'cake',
    'couch', 'keyboard', 'sink', 'scissors',
    'airplane', 'cat', 'snowboard'
)

CLASSES_IDS_48 = [0, 1, 2, 3, 7, 8, 13, 14, 17, 18, 22, 23, 24, 26, 30, 33, 37, 39, 44, 45, 46, 47, 49, 50, 51, 53, 54,
                  56, 59, 62, 63, 65, 68, 69, 72, 73, 74, 75, 79, 6, 21, 28, 29, 42, 48, 61, 64, 70]
CLASSES_IDS_17 = [5, 16, 19, 20, 25, 27, 36, 41, 43, 55, 57, 66, 71, 76, 4, 15, 31]


@DATASETS.register_module()
class CocoOVDataset(CocoDataset):
    """Coco Open Vocabulary dataset for Instance segmentation.
    The class names are changed.
    """
    METAINFO = {
        'classes':
            ('person,child,girl,boy,woman,man,people,children,girls,boys,women,men,lady,guy,ladies,guys,clothes',
             'bicycle,bicycles,bike,bikes',
             'car,cars',
             'motorcycle,motorcycles',
             'airplane,airplanes',
             'bus,buses',
             'train,trains,locomotive,locomotives,freight train',
             'truck,trucks',
             'boat,boats',
             'traffic light',
             'fire hydrant',
             'stop sign',
             'parking meter',
             'bench,benches',
             'bird,birds',
             'cat,cats,kitties,kitty',
             'dog,dogs,puppy,puppies',
             'horse,horses,foal',
             'sheep',
             'cow,cows,calf',
             'elephant,elephants',
             'bear,bears',
             'zebra,zebras',
             'giraffe,giraffes',
             'backpack,backpacks',
             'umbrella,umbrellas',
             'handbag,handbags',
             'tie',
             'suitcase,suitcases',
             'frisbee',
             'skis',
             'snowboard',
             'sports ball',
             'kite,kites',
             'baseball bat',
             'baseball glove',
             'skateboard',
             'surfboard',
             'tennis racket',
             'bottle,bottles,water bottle',
             'wine glass,wine glasses,wineglass',
             'cup,cups,water cup,water glass',
             'fork,forks',
             'knife,knives',
             'spoon,spoons',
             'bowl,bowls',
             'banana,bananas',
             'apple,apples,apple fruit',
             'sandwich,sandwiches',
             'orange fruit',
             'broccoli',
             'carrot,carrots',
             'hot dog',
             'pizza',
             'donut,donuts',
             'cake,cakes',
             'chair,chairs',
             'couch,sofa,sofas',
             'potted plant,potted plants,pottedplant,pottedplants,planter,planters',
             'bed,beds',
             'dining table,dining tables,diningtable,diningtables,plate,plates,diningtable tablecloth',
             'toilet',
             'tv',
             'laptop',
             'mouse',
             'tv remote,remote control',
             'keyboard',
             'cell phone,mobile',
             'microwave',
             'oven,ovens',
             'toaster',
             'sink,sinks',
             'refrigerator,fridge',
             'book,books',
             'clock',
             'vase,vases',
             'scissor,scissors',
             'teddy bear,teddy bears',
             'hair drier',
             'toothbrush,toothbrushes',
             ),

        'palette':
            [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230), (106, 0, 228),
             (0, 60, 100), (0, 80, 100), (0, 0, 70), (0, 0, 192), (250, 170, 30),
             (100, 170, 30), (220, 220, 0), (175, 116, 175), (250, 0, 30),
             (165, 42, 42), (255, 77, 255), (0, 226, 252), (182, 182, 255),
             (0, 82, 0), (120, 166, 157), (110, 76, 0), (174, 57, 255),
             (199, 100, 0), (72, 0, 118), (255, 179, 240), (0, 125, 92),
             (209, 0, 151), (188, 208, 182), (0, 220, 176), (255, 99, 164),
             (92, 0, 73), (133, 129, 255), (78, 180, 255), (0, 228, 0),
             (174, 255, 243), (45, 89, 255), (134, 134, 103), (145, 148, 174),
             (255, 208, 186), (197, 226, 255), (171, 134, 1), (109, 63, 54),
             (207, 138, 255), (151, 0, 95), (9, 80, 61), (84, 105, 51),
             (74, 65, 105), (166, 196, 102), (208, 195, 210), (255, 109, 65),
             (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
             (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
             (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
             (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
             (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
             (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
             (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
             (246, 0, 122), (191, 162, 208)]
    }

    def load_data_list(self) -> List[dict]:
        """Load annotations from an annotation file named as ``self.ann_file``

        Returns:
            List[dict]: A list of annotation.
        """  # noqa: E501
        with get_local_path(
                self.ann_file, backend_args=self.backend_args) as local_path:
            self.coco = self.COCOAPI(local_path)
        # The order of returned `cat_ids` will not
        # change with the order of the `classes`
        self.cat_ids = self.coco.get_cat_ids(
            cat_names=CLASSES_ORIGINAL)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.cat_img_map = copy.deepcopy(self.coco.cat_img_map)

        img_ids = self.coco.get_img_ids()
        data_list = []
        total_ann_ids = []
        for img_id in img_ids:
            raw_img_info = self.coco.load_imgs([img_id])[0]
            raw_img_info['img_id'] = img_id

            ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
            raw_ann_info = self.coco.load_anns(ann_ids)
            total_ann_ids.extend(ann_ids)

            parsed_data_info = self.parse_data_info({
                'raw_ann_info':
                    raw_ann_info,
                'raw_img_info':
                    raw_img_info
            })
            data_list.append(parsed_data_info)
        if self.ANN_ID_UNIQUE:
            assert len(set(total_ann_ids)) == len(
                total_ann_ids
            ), f"Annotation ids in '{self.ann_file}' are not unique!"

        del self.coco

        return data_list

    def filter_data(self) -> List[dict]:
        valid_data_infos = super().filter_data()

        if self.filter_cfg is None:
            return valid_data_infos

        sub_split = self.filter_cfg.get('sub_split', None)
        if sub_split is None:
            return valid_data_infos

        if sub_split == '48_17':
            with_cat_ids = []
            wo_cat_ids = []
            classes = list(CLASSES_ORIGINAL)
            if self.test_mode:
                for cls in CLASSES_17:
                    with_cat_ids.append(classes.index(cls))
                for cls in CLASSES_48:
                    with_cat_ids.append(classes.index(cls))
            else:
                for cls in CLASSES_48:
                    with_cat_ids.append(classes.index(cls))
                for cls in CLASSES_17:
                    wo_cat_ids.append(classes.index(cls))
        else:
            raise ValueError(f"{sub_split} does not support")

        keep_w_novel = True
        filtered_data_infos = []
        for data_info in valid_data_infos:
            instances = data_info['instances']
            filtered_instances = []
            flag = False
            for ins in instances:
                if ins['bbox_label'] in with_cat_ids:
                    filtered_instances.append(ins)
                    flag = True
            if not flag:
                continue
            if not keep_w_novel:
                for ins in instances:
                    if ins['bbox_label'] in wo_cat_ids:
                        filtered_instances.append(ins)
                        flag = False
                        break
            if flag:
                data_info['instances'] = filtered_instances
                filtered_data_infos.append(data_info)

        print_log(
            f"There are totally {len(filtered_data_infos)} images in the filtered dataset.",
            logger='current',
        )
        return filtered_data_infos
