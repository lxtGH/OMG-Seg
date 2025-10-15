# Install

The correct functioning of this repository depends on the right versioning of some of its libraries, especially OpenMMLab. Below you'll see a step-by-step on how to set them up. 

## Requirements

- [CUDA](https://developer.nvidia.com/cuda-toolkit)
- [Anaconda](https://www.anaconda.com/docs/getting-started/anaconda/install) or [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install)

## Setup

1. Create a `conda` environment:
```bash
conda create -n omg-seg python=3.10
conda activate omg-seg
```

2. Install PyTorch:
```bash
# CUDA 11.8
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.8 -c pytorch -c nvidia
# CUDA 11.7
conda install pytorch==2.0.1 torchvision==0.15.2 pytorch-cuda=11.7 -c pytorch -c nvidia
```

3. Install OpenMMLab libraries

```bash
pip install -U openmim
mim install mmcv==2.0.1 mmengine==0.8.5 mmdet==3.1.0 mmsegmentation==1.1.1 mmpretrain==1.0.1
```

4. Install the remaining libraries
```bash
python -m pip install git+https://github.com/cocodataset/panopticapi.git \
                      git+https://github.com/HarborYuan/lvis-api.git \
                      "numpy<2" \
                      opencv-python==4.8.1.78 \
                      tqdm terminaltables pycocotools scipy tqdm ftfy regex timm scikit-image kornia
```

After all the steps, you should have a functional environment for executing the codebase.