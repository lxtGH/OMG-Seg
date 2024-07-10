# 🛠️ Installation

## Python Environments
```commandline
conda create --name omg-llava python=3.10 -y
conda activate omg-llava

# install pytorch with cuda 11.8
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu118

# install omg-seg requirements
python -m pip install https://github.com/open-mmlab/mmengine/archive/refs/tags/v0.8.5.zip
TORCH_CUDA_ARCH_LIST="8.0" TORCH_NVCC_FLAGS="-Xfatbin -compress-all" CUDA_HOME=$(dirname $(dirname $(which nvcc))) LD_LIBRARY_PATH=$(dirname $(dirname $(which nvcc)))/lib MMCV_WITH_OPS=1 FORCE_CUDA=1 python -m pip install git+https://github.com/open-mmlab/mmcv.git@4f65f91db6502d990ce2ee5de0337441fb69dd10

python -m pip install \
https://github.com/open-mmlab/mmdetection/archive/refs/tags/v3.1.0.zip \
https://github.com/open-mmlab/mmsegmentation/archive/refs/tags/v1.1.1.zip \
https://github.com/open-mmlab/mmpretrain/archive/refs/tags/v1.0.1.zip

# install other requirements
pip install -e '.[all]'
```

## Prepare Datas

```shell
# 1. download the data for https://huggingface.co/datasets/zhangtao-whu/OMG-LLaVA/tree/main
# 2. merge the llava and glamm data to zip
  cd glamm_data_zip 
  cat ./*.zip.* > glamm_data.zip
  
  cd ../llava_data_zip
  cat ./*.zip.* > llava_data.zip
# 3. unzip all the datas
# 4. construct datas as follows
 |--- data
    |--- glamm_data
    |--- llava_data
    |--- mdpv_point
    |--- region_cap
    |--- ref_seg
    |--- semantic_seg
```

## Prepare Pretrained Weights

Download pretrained weights from [Hugging Face](https://huggingface.co/zhangtao-whu/OMG-LLaVA/tree/main).
Put the weights to `./pretrained/omg_llava/`
```commandline
 |--- pretrained
    |--- omg_llava
        internlm2-chat-7b
        convnext_large_d_320_CocoPanopticOVDataset.pth
        omg_seg_convl.pth
        omg_llava_7b_pretrain_8gpus.pth
        omg_llava_7b_finetune_8gpus.pth
        finetuned_refseg.pth
        finetuned_gcg.pth
```


