
<br />
<p align="center">
  <h1 align="center">OMG-Seg: Is One Model Good Enough For All Segmentation?</h1>
  <p align="center">
    CVPR, 2024
    <br />
    <a href="https://lxtgh.github.io/"><strong>Xiangtai Li </strong></a>
    ·
    <a href="https://yuanhaobo.me/"><strong>Haobo Yuan</strong></a>
    .
    <a href="https://weivision.github.io/"><strong>Wei Li</strong></a>
    .
    <a href="https://henghuiding.github.io/"><strong>Henghui Ding</strong></a>
    ·
    <a href="https://wusize.github.io/"><strong>Size Wu</strong></a>
    ·
    <a href="https://zhangwenwei.cn/"><strong>Wenwei Zhang</strong></a>
    ·
    <br />
    <a href="https://scholar.google.com.hk/citations?user=y_cp1sUAAAAJ&hl=en"><strong>Yining Li</strong></a>
    .
    <a href="https://hellock.github.io/"><strong>Kai Chen</strong></a>
    .
    <a href="https://www.mmlab-ntu.com/person/ccloy/"><strong>Chen Change Loy</strong></a>
  </p>
  
  <p align="center">
    S-Lab, MMlab@NTU, Shanghai AI Laboratory
  </p>
  
   <p align="center">
    Xiangtai is the project leader and corresponding author.
  </p>
  
  <p align="center">
    <a href='https://arxiv.org/abs/2401.10229'>
      <img src='https://img.shields.io/badge/Paper-PDF-green?style=flat&logo=arXiv&logoColor=green' alt='arXiv PDF'> </a>
    <a href='https://lxtgh.github.io/project/omg_seg/' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Project-Page-blue?style=flat&logo=Google%20chrome&logoColor=blue' alt='Project Page'> </a>
    <a href='https://huggingface.co/LXT/OMG_Seg' style='padding-left: 0.5rem;'>
      <img src='https://img.shields.io/badge/Huggingface%20Model-8A2BE2' alt='Project Page'> </a>
    <a href="https://huggingface.co/spaces/LXT/OMG_Seg">
    <img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-App-blue' alt='HuggingFace Model'> </a>
  </p>
<br />

![avatar](./figs/omg_teaser.jpg)


### Short Introduction

In this work, we address various segmentation tasks, each traditionally tackled by distinct or partially unified models. 
We propose OMG-Seg, One Model that is Good enough to efficiently and effectively handle all the Segmentation tasks, including image semantic, instance, and panoptic segmentation, as well as their video counterparts, open vocabulary settings, prompt-driven, interactive segmentation like SAM, and video object segmentation.
To our knowledge, this is **the first model** to fill all these tasks in one model and achieve good enough performance.

We show that OMG-Seg, a transformer-based encoder-decoder architecture with task-specific queries and outputs, can support over ten distinct segmentation tasks and yet significantly reduce computational and parameter overhead across various tasks and datasets. 
We rigorously evaluate the inter-task influences and correlations during co-training. Both the code and models will be publicly available.

Short introduction on VALSE of OMG-Seg with other related work, can be found [here](https://www.bilibili.com/video/BV1PZ421b7U7/?spm_id_from=333.337.search-card.all.click&vd_source=6bb672e5bcff6f43a998d1ba30743967), in Chinese.


## Features of OMG-Seg

### $\color{#2F6EBA}{Universal\ Image\, Video\, Open-Vocabulary\, Segmentation\ Model}$ 

- A **new unified** solution for **over ten different segmentation tasks**: PS, IS, VSS, VIS, VPS, Open-Vocabulary Seg, and Interactive Segmentation.
- A novel unified view for solving multiple segmentation tasks in one model with extremely less parameters.

### $\color{#2F6EBA}{Good\ Enough\ Performance}$  

- OMG-Seg achieves **good enough performance** on in one shared architecture, on multiple datasets. (**only 70M trainable parameters**)

### $\color{#2F6EBA}{The\ First\ OpenSourced\ Universal\ Segmentation\ Codebase}$  

- Our codebase support **joint image/video/multi-dataset co-training**.
- The first open-sourced codebase, including training, inference and demo.

### $\color{#2F6EBA}{Easy\ \ Followed\ By\ Academic\ Lab}$  

- OMG-Seg can be reproduced by only **one 32GB V100 or 40GB A100 machine**, which can be followed by Academic Labs.

## To-Do Plans

- Release Strong Models. (To be Done)
- Release training code. (done)
- Release CKPTs.（done）
- Support HuggingFace. (done)


## Experiment Set Up

### Dataset 

See [DATASET.md](./DATASET.md)


### Install

Our codebase is built with [MMdetection-3.0](https://github.com/open-mmlab/mmdetection) tools.

See [INSTALL.md](./INSTALL.md)



### Quick Start

#### Experiment Preparation

1. First set up the [dataset](./DATASET.md) and [environment](./INSTALL.md). Make sure you have fixed and corresponding versions. 

2. Download pre-trained CLIP backbone. The scripts will automatically download the pre-trained CLIP models.

3. Generate CLIP text embedding for each dataset and joint merged dataset for co-training. See the [embedding generation](EMB.md). 

4. Run the train/test scripts below to carry out experiments on model training and testing.






## Citation

If you think our codebases and works are useful for your research, please consider referring us:

```bibtex



@inproceedings{OMGSeg,
author       = {Xiangtai Li and
                  Haobo Yuan and
                  Wei Li and
                  Henghui Ding and
                  Size Wu and
                  Wenwei Zhang and
                  Yining Li and
                  Kai Chen and
                  Chen Change Loy},
  title        = {OMG-Seg: Is One Model Good Enough For All Segmentation?},
booktitle={CVPR},
  year={2024}
}


```

## License

S-Lab [LICENSE](LICENSE).
