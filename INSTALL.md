Install CUDA
```commandline
conda install --channel "nvidia/label/cuda-12.1.0" cuda-toolkit
```

Install PyTorch:
```commandline
conda install pytorch torchvision torchaudio pytorch-cuda==12.1 -c pytorch -c nvidia
```

Or install with CUDA11.8
```commandline
conda install pytorch torchvision torchaudio cuda-toolkit pytorch-cuda==11.8 -c pytorch -c "nvidia/label/cuda-11.8.0"
```

Install mmengine:
```commandline
python -m pip install https://github.com/open-mmlab/mmengine/archive/refs/tags/v0.8.5.zip
```

Install mmcv: (Make sure you use the correct mmcv version as our default setting)

Please see the doc here to find your matched mmcv version. [doc](https://mmcv.readthedocs.io/en/latest/get_started/installation.html).
You can install pre-build mmcv for easier usage. Make sure your mmdetection is v3.1.0 version.

Here, we provide our experimental mmcv version.
```commandline
TORCH_CUDA_ARCH_LIST="8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=$(dirname $(dirname $(which nvcc))) LD_LIBRARY_PATH=$(dirname $(dirname $(which nvcc)))/lib MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@4f65f91db6502d990ce2ee5de0337441fb69dd10
```

Install other openmmlab packages:
```commandline
python -m pip install \
https://github.com/open-mmlab/mmdetection/archive/refs/tags/v3.1.0.zip \
https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v1.1.1.zip \
https://github.com/open-mmlab/mmpretrain/archive/refs/tags/v1.0.1.zip
```

Install extra packages:
```commandline
python -m pip install git+https://github.com/cocodataset/panopticapi.git \
git+https://github.com/HarborYuan/lvis-api.git \
tqdm terminaltables pycocotools scipy tqdm ftfy regex timm scikit-image kornia
```

The default environment is openmmlab on Shanghai AI server.

We suggest using mmcv 2.x with mmdet3.x version for this repo.