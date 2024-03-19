Please prepare dataset in the following format.

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

```
├── cityscapes
│   ├── annotations
│   │   ├── instancesonly_filtered_gtFine_train.json # coco instance annotation file(COCO format)
│   │   ├── instancesonly_filtered_gtFine_val.json
│   │   ├── cityscapes_panoptic_train.json  # panoptic json file 
│   │   ├── cityscapes_panoptic_val.json  
│   ├── leftImg8bit
│   ├── gtFine
│   │   ├──cityscapes_panoptic_{train,val}/  # png annotations
│   │   
```


### [VIS] Youtube-VIS (2019/2021) dataset 

```
├── youtubevis2019
│   ├── annotations
│   │   ├── train.json
│   │   ├── valid.json
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
│   │   ├── train.json
│   │   ├── valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video folders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video folders
```

### [VPS] VIPSeg dataset

Download the origin dataset from the official repo.\
Following official repo, we use resized videos for training and evaluation (The short size of the input is set to 720 while the ratio is keeped).

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

The default setting as mmdet

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

Finally, link the download the dataset into the data folder as 

```
root
├── ext
├── figs
├── seg
├── tools
├── data
│   ├──coco
│   ├──ade
│   ├──cityscapes
│   ├──VIPSeg
│   ├──youtube_vis_2019
│   ├──youtube_vis_2021
│   ├──DAVIS
