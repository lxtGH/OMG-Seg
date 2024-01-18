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



### [VIS] Youtube-VIS (2019/2021)

```
├── youtubevis
│   ├── annotations
│   │   ├── train.json
│   │   ├── valid.json
│   ├── train
│   │   ├──JPEGImages
│   │   │   ├──video floders
│   ├── valid
│   │   ├──JPEGImages
│   │   │   ├──video floders
```

### [VPS] VIPSeg

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

### [SS/PS] ADE  

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

