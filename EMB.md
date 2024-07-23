## Generate text embedding for each dataset and Download the pretrained models.

### For Separate Dataset. (Mainly For Evaluation)

We adopt the separate dataset embedding for testing.


```commandline
./tools/dist.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_coco.py 1
```
```commandline
./tools/dist.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_ade.py 1
```

```commandline
./tools/dist.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_cityscapes.py 1
```

```commandline
./tools/dist.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_vipseg.py 1
```

```commandline
./tools/dist.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_y19.py 1
```

```commandline
./tools/dist.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_y21.py 1
```

### For Merged Dataset Training. (Mainly For Co-Training)

We adopt the merged dataset embedding for training. 

```commandline
./tools/dist.sh gen_cls seg/configs/m2ov_train/omg_convl_vlm_fix_24e_ov_coco_vid_yt19_vip_city_cocopansam.py 1
```

Once you finish converting the embedding, you will obtain the embedding file in your cache folder.

### Download Pre-trained Open-ClIP models.

When generating the class embedding classifier, the scripts will automatically download the pre-trained CLIP models.

If you are in China, you can use [HF-Mirror](https://hf-mirror.com/). Follow the step to set the default path.

```commandline
pip install -U huggingface_hub
```

```commandline
export HF_ENDPOINT=https://hf-mirror.com
```


