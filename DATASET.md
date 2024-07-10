Please prepare dataset in the following format.


For easier use, we suggest to use the datasets we have pre-processed in Hugging Face [OMG-Seg dataset](https://huggingface.co/HarborYuan/omgseg_data)
and [OMG-LLaVA](https://huggingface.co/datasets/zhangtao-whu/OMG-LLaVA).

## OMG-Seg datasets

### [PS/IS] COCO dataset

The default setting as mmdetection.

```
├── coco
│   ├── annotations
│   │   ├── panoptic_{train,val}2017.json
│   │   ├── instance_{train,val}2017.json
│   ├── train2017
│   ├── val2017
│   ├── panoptic_{train,val}2017/  # png annotations
```


### [PS] Cityscapes dataset

Please use the scripts in ext/cityscapes_scripts/createPanopticImgs.py to generate COCO-style cityscape panoptic segmentation format.

```commandline
python ext/cityscapes_scripts/createPanopticImgs.py --dataset-folder ./data/cityscapes --output-folder ./data/cityscapes
```


```
├── cityscapes
│   ├── annotations
│   │   ├── cityscapes_panoptic_train_trainId.json  # panoptic json file 
│   │   ├── cityscapes_panoptic_val_trainId.json 
│   │   ├── cityscapes_panoptic_train_trainId # panoptic png file
│   │   ├── cityscapes_panoptic_val_trainId # panoptic png file
│   ├── leftImg8bit # training images
│   ├── gtFine # origin gt files 
│   │   ├──
│   │   
```


### [VIS] Youtube-VIS (2019/2021) dataset 


Use the scripts tools/dataset_convert/vis_to_coco.py to convert origin json to COCO-style.

```commandline
python tools/dataset_convert/vis_to_coco.py -i ./data/youtubevis2019 --version 2019
```

```commandline
python tools/dataset_convert/vis_to_coco.py -i ./data/youtubevis2021 --version 2021
```


The final results are shown here:

```
├── youtubevis2019
│   ├── annotations
│   │   ├── youtube_vis_2019_train.json
│   │   ├── youtube_vis_2019_valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video folders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video folders
```

```
├── youtubevis2021
│   ├── annotations
│   │   ├── youtube_vis_2021_train.json
│   │   ├── youtube_vis_2021_valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video folders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video folders
```


### [VPS] VIPSeg dataset

Download the origin dataset from the official repo.\
Following official repo, we use resized videos for training and evaluation (The short size of the input is set to 720).

```
├── VIPSeg
│   ├──  imgs
│   │   ├── 1241_qYvEuwrSiXc
        │      ├──*.jpg
│   ├──  panomasks 
│   │   ├── 1241_qYvEuwrSiXc
        │      ├──*.png
│   ├──  panomasksRGB 
```

### [SS/PS] ADE dataset

The default setting as mmdet, note that please use our pre-processed ADE annotations.

```
├── ade
│   ├──  ADEChallengeData2016
│   │   ├── images/
│   │   ├── annotations/
│   │   ├── ade20k_panoptic_train/
│   │   ├── ade20k_panoptic_val/
│   │   ├── ade20k_panoptic_train.json
│   │   ├── ade20k_panoptic_val.json
```

### [VOS] DAVIS dataset

Please download DAVIS datasets as default.


Finally, link the download the dataset into the data folder as 

```
root
├── ext
├── figs
├── seg
├── omg_llava
├── tools
├── data
│   ├──coco
│   ├──ade
│   ├──cityscapes
│   ├──VIPSeg
│   ├──youtube_vis_2019
│   ├──youtube_vis_2021
│   ├──DAVIS
```

## OMG-LLaVA datasets

Please download OMG-LLaVA dataset from HuggingFace webpage.

```
├── data
│   ├──glamm_data
│   ├──llava_data
│   ├──mdpv_point
│   ├──ref_seg
│   ├──region_caption
│   ├──semantic_seg
```