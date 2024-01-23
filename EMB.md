## Generate text embedding for each dataset 

### For Separate Dataset: 

We adopt the separate dataset embedding for testing.


```commandline
./tools/dish.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_coco.py 1
```
```commandline
./tools/dish.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_ade.py 1
```

```commandline
./tools/dish.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_cityscapes.py 1
```

```commandline
./tools/dish.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_vipseg.py 1
```

```commandline
./tools/dish.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_y19.py 1
```

```commandline
./tools/dish.sh gen_cls seg/configs/m2ov_val/eval_m2_convl_300q_ov_y21.py 1
```

### For Merged Dataset 

We adopt the merged dataset embedding for training. 

To be released